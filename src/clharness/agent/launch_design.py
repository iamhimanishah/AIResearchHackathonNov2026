"""Launch the pre-registered runs on Modal: the screen and the published-claims family.

    python -m clharness.agent.launch_design                 # dry run: the plan and its cost
    python -m clharness.agent.launch_design --go            # launch, wait, collect rows
    python -m clharness.agent.launch_design --go --only screen

What it is. A deterministic launcher for the part of the study that the pre-registration fixes in
advance (`analysis/PREREG.md` sections 6.2 and 6.5; the specs come from
`clharness.analysis.design`). There is nothing for a model to decide in this step: the runs are
written down before any of them is read. It uses the agent system's own parts for everything
else, so the Conductor can take over afterwards from the same files:

  * `ModalExecutor` starts each spec as a call of the deployed `clharness` app, remembers the
    call ids next to the rows file, and is idempotent by spec hash. Running this command again
    resumes: nothing that has a row or is in flight is started twice.
  * `DecisionLog` records the plan, every wave launched and every batch of rows read, hash-chained.

What it is not. It does not choose what to replicate or combine. That is the Conductor's job
(Planner, Gatekeeper, Analyst), reading `<out>/rows.jsonl` when the screen is in.

Rules it enforces:
  * the working tree must be clean and the deployed app must hold this exact commit (the app
    refuses a caller at another commit; this command stops at the first such refusal);
  * a spec that already has a finished row in `--have` files is not run again;
  * at most `--concurrency` runs in flight (the workspace limit is 10 GPUs);
  * the estimated cost of everything still to launch must fit `--cap-usd`, and no wave is
    launched once the money estimated so far would pass it;
  * a run lost to the machine is launched again once; a run that failed for its own reasons is not.

Needs: `uv sync --extra modal`, `modal setup`, and `modal deploy src/clharness/executor/modal_app.py`
at the commit being launched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .. import spec as a1
from ..analysis import design
from .contracts import DecisionLog, a1_row_hash
from .modal_executor import ModalBackend, ModalExecutor, ModalFunctionBackend

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HAVE = (
    "results/rows.phase0.remeasured.jsonl", "results/rows.phase0.jsonl",
    "results/rows.reference.remeasured.jsonl",
)
MINUTES_PER_RUN = 20.0  # measured on the frozen bench: 18 to 23 minutes on an H100
MAX_LAUNCHES = 2  # the first attempt plus one relaunch after a machine failure
STALE_MARKERS = ("stale deployment", "row was built from git")


class Halt(RuntimeError):
    """Stop launching. The message says what a person has to do."""


def planned_specs(only: str = "all") -> list[a1.RunSpec]:
    """The pre-registered runs, screen first. Spec hashes are unique."""
    specs: list[a1.RunSpec] = []
    if only in ("all", "screen"):
        specs += design.screen()
    if only in ("all", "claims"):
        specs += design.claims_new()
    seen: set[str] = set()
    return [s for s in specs if not (s.spec_hash() in seen or seen.add(s.spec_hash()))]


def finished_hashes(paths: Sequence[str | Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if Path(path).exists():
            done |= {r.spec_hash for r in a1.read_rows(path) if r.status == "finished"}
    return done


def code_version(repo: Path = REPO_ROOT) -> str:
    """HEAD, or raise if tracked code differs from it (the deployed app could not match)."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs", "manifests"], text=True, cwd=repo
    ).strip()
    if dirty:
        raise Halt("uncommitted changes under src, configs or manifests; commit them and redeploy first")
    return sha


def launch(
    specs: Sequence[a1.RunSpec],
    out: Path,
    backend: ModalBackend,
    git_sha: str,
    *,
    have: Sequence[str | Path] = (),
    concurrency: int = 10,
    cap_usd: float = 80.0,
    gpu: str = "H100",
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = print,
    tag: str = "design",
) -> dict[str, int]:
    """Run every spec that has no finished row yet. Returns counts by final status."""
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    log = DecisionLog(out / "decisions.jsonl")
    executor = ModalExecutor(rows_path, backend, git_sha=git_sha, gpu=gpu, minutes_per_run=MINUTES_PER_RUN)
    by_hash = {s.spec_hash(): s for s in specs}

    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def latest() -> dict[str, a1.ResultRow]:
        rows = a1.read_rows(rows_path) if rows_path.exists() else []
        return {r.spec_hash: r for r in rows}  # later lines win

    already = finished_hashes(have)
    todo = [h for h in by_hash if h not in already]
    per_run = executor.estimate_cost(specs[0]) if specs else 0.0
    open_cost = per_run * len([h for h in todo if latest().get(h) is None or latest()[h].status != "finished"])
    log.append(
        key=f"{tag}/plan", state="screen/plan", actor="launch_design", ts=now(),
        reason=("The pre-registered screen and published-claims runs (analysis/PREREG.md 6.2, 6.5). "
                "Nothing is chosen here: the list is clharness.analysis.design.") if tag == "design"
        else f"Runs accepted by the gate for {tag}.",
        data={"planned": len(by_hash), "already_finished_elsewhere": len(by_hash) - len(todo),
              "to_run": len(todo), "estimated_usd": round(open_cost, 2), "cap_usd": cap_usd,
              "git_sha": git_sha, "spec_hashes": sorted(todo)},
    )
    if open_cost > cap_usd:
        raise Halt(f"estimated ${open_cost:.2f} for {len(todo)} runs is over the cap of ${cap_usd:.2f}")
    if not backend.healthy():
        raise Halt("the deployed Modal app is not reachable; run: modal deploy src/clharness/executor/modal_app.py")

    spent_estimate, wave = 0.0, 0
    while True:
        rows = latest()
        stale = [r for r in rows.values() if r.reason and any(m in r.reason for m in STALE_MARKERS)]
        if stale:
            raise Halt(f"the deployed app holds other code than {git_sha[:10]}: {stale[0].reason[:160]}. "
                       "Redeploy at this commit, delete the failed rows' lines, and run this again.")
        in_flight = set(executor.in_flight())
        waiting = []
        for h in todo:
            row = rows.get(h)
            lost = row is not None and row.status == "failed" and executor.launches.get(h, 0) < MAX_LAUNCHES \
                and (row.reason or "").startswith(("infrastructure", "timeout"))
            if h not in in_flight and (row is None or lost):
                waiting.append(h)
        if not waiting and not in_flight:
            break
        room = concurrency - len(in_flight)
        batch = waiting[:max(room, 0)]
        if batch and spent_estimate + per_run * len(batch) > cap_usd:
            raise Halt(f"launching {len(batch)} more would pass the cap of ${cap_usd:.2f}")
        if batch:
            wave += 1
            executor.submit([by_hash[h] for h in batch])
            spent_estimate += per_run * len(batch)
            log.append(
                key=f"{tag}/launch#{wave}", state="screen/launch", actor="launch_design", ts=now(),
                reason=f"Wave {wave}: {len(batch)} runs, keeping at most {concurrency} in flight.",
                data={"spec_hashes": batch, "names": [by_hash[h].name for h in batch],
                      "estimated_usd_so_far": round(spent_estimate, 2)},
            )
            say(f"wave {wave}: launched {len(batch)} ({len(waiting) - len(batch)} waiting)")
        landed = executor.poll()
        if landed:
            fresh = {h: r for h, r in latest().items() if h in {x.spec_hash for x in landed}}
            log.append(
                key=f"{tag}/read#{len(log.entries())}", state="screen/read", actor="launch_design", ts=now(),
                reason=f"{len(fresh)} rows landed.",
                input_row_hashes=[a1_row_hash(r) for r in fresh.values()],
                data={"rows": [{"name": r.spec.name, "status": r.status, "reason": (r.reason or "")[:200]}
                               for r in fresh.values()]},
            )
            for r in fresh.values():
                say(f"  {r.status:9s} {r.spec.name} {r.reason or ''}"[:160])
        elif not batch:
            sleep(poll_s)

    counts: dict[str, int] = {}
    final = latest()
    for h in todo:
        status = final[h].status if h in final else "missing"
        counts[status] = counts.get(status, 0) + 1
    log.append(
        key=f"{tag}/done", state="screen/advance", actor="launch_design", ts=now(),
        reason="Every planned run has a row. Replication and combination are the Conductor's to plan.",
        data={"counts": counts, "rows": str(rows_path)},
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--go", action="store_true", help="launch; without it, print the plan only")
    parser.add_argument("--only", choices=["all", "screen", "claims"], default="all")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "agent")
    parser.add_argument("--have", nargs="*", default=[str(REPO_ROOT / p) for p in DEFAULT_HAVE],
                        help="row files whose finished specs are not run again")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--cap-usd", type=float, default=80.0)
    parser.add_argument("--gpu", default="H100")
    args = parser.parse_args(argv)

    specs = planned_specs(args.only)
    have = [*args.have, args.out / "rows.jsonl"]
    done = finished_hashes(have)
    todo = [s for s in specs if s.spec_hash() not in done]
    per_run = MINUTES_PER_RUN / 60 * 3.95
    waves = -(-len(todo) // args.concurrency)
    print(f"planned {len(specs)} runs ({args.only}); {len(specs) - len(todo)} already have a finished row; "
          f"{len(todo)} to run")
    print(f"about ${per_run * len(todo):.0f} at ${per_run:.2f} a run; {waves} wave(s) of up to "
          f"{args.concurrency}, about {waves * MINUTES_PER_RUN:.0f} minutes; cap ${args.cap_usd:.0f}")
    for s in todo:
        print(f"  {s.spec_hash()[:12]}  {s.name}")
    if not args.go:
        print("dry run: nothing launched. Add --go to launch.")
        return 0
    try:
        counts = launch(specs, args.out, ModalFunctionBackend(), code_version(), have=args.have,
                        concurrency=args.concurrency, cap_usd=args.cap_usd, gpu=args.gpu)
    except Halt as stop:
        print(f"HALTED: {stop}", file=sys.stderr)
        return 2
    print(f"done: {counts}; rows in {args.out / 'rows.jsonl'}")
    return 0 if set(counts) <= {"finished", "invalid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
