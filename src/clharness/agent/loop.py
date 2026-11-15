"""Entry point for the agent loop.

    python -m clharness.agent.loop --executor sim --out results/agent-sim
    python -m clharness.agent.loop --executor modal --prereg thresholds.yaml \
        --out results/agent/loop --rows results/agent/rows.jsonl          # dry run: prints the setup
    ... --go                                                               # runs on Modal GPUs

`--executor sim` runs the whole loop against the simulator (fake result rows with planted
effects, on a fake clock, so it finishes in seconds and spends nothing). `--scout exa` adds the
Scout after replicate: one live Exa search, a ranked shortlist of claim cards (tested by the
loop in combine) and candidates (for human review), every source in the decision log (needs
`--llm openai`).

`--executor modal` is the real loop (docs/agent/SCOUT-HANDOFF.md, precondition 2): justin's
`ModalExecutor` on the deployed app, the system clock, the real plugin catalog
(`catalog_real.real_catalog`), the bench check in the Gatekeeper, the frozen fast-bench stream,
and the PREREG thresholds from `--prereg`. Point `--rows` at the rows file `launch_design` is
filling, so the pre-registered grid's rows count and nothing is run twice (`ModalExecutor` is
idempotent by spec hash and shares the call ids next to that file). Without `--go` it builds the
loop and prints what it would use; nothing is launched. Running the same command again resumes
from state.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .agents import FileMemory, LLMAnalyst, LLMPlanner, LLMScout
from .catalog_real import bench_problems, design_claim_arms, real_catalog
from .catalog_real import stream as bench_stream
from .conductor import Conductor
from .config import load_agent_config
from .contracts import Prereg, PreregError, load_prereg, read_jsonl
from .gatekeeper import Gatekeeper
from .llm import BudgetedLLM, LLMClient, OpenAIClient
from .operator import Operator
from .roles import Clock, FakeClock, Scout, StubAnalyst, StubPlanner, SystemClock
from .search import ExaSearch
from .sim import SimExecutor, World, default_catalog, default_prereg, default_stream


def build_sim(out: Path, seed: int, compute_cap_usd: float, concurrency: int = 8,
              clock: FakeClock | None = None, fault=None, infra_fail_rate: float = 0.1,
              gatekeeper: Gatekeeper | None = None, llm: LLMClient | None = None,
              memory: bool = True, scout: Scout | None = None) -> Conductor:
    clock = clock or FakeClock()
    world = World(seed, infra_fail_rate=infra_fail_rate)
    rows_path = out / "rows.jsonl"
    executor = SimExecutor(world, rows_path, clock, concurrency=concurrency)
    return Conductor(
        out, rows_path, executor=executor, clock=clock,
        planner=LLMPlanner(llm) if llm else StubPlanner(),
        analyst=LLMAnalyst(llm, clock=clock) if llm else StubAnalyst(clock),
        memory=FileMemory(out / "memory") if memory else None,
        gatekeeper=gatekeeper or Gatekeeper(), operator=Operator(executor, clock, rows_path),
        prereg=default_prereg(world), catalog=default_catalog(), stream=default_stream(),
        compute_cap_usd=compute_cap_usd, concurrency=concurrency, fault=fault, scout=scout,
    )


def build_real(out: Path, prereg: Prereg, *, backend, git_sha: str, compute_cap_usd: float,
               rows_path: Path | None = None, concurrency: int = 10, gpu: str = "H100",
               clock: Clock | None = None, llm: LLMClient | None = None,
               scout: Scout | None = None, memory: bool = True,
               deadline: datetime | None = None, poll_seconds: float = 60.0,
               fault=None) -> Conductor:
    """The loop on real GPUs. `backend` is a `ModalBackend` (`ModalFunctionBackend()` in
    production, a fake in tests). Builds everything; launches nothing until `.run()`."""
    from .launch_design import MINUTES_PER_RUN
    from .modal_executor import ModalExecutor

    clock = clock or SystemClock()
    rows_path = rows_path or out / "rows.jsonl"
    catalog = real_catalog()
    executor = ModalExecutor(rows_path, backend, git_sha=git_sha, gpu=gpu,
                             minutes_per_run=MINUTES_PER_RUN, catalog=catalog)
    return Conductor(
        out, rows_path, executor=executor, clock=clock,
        planner=LLMPlanner(llm) if llm else StubPlanner(),
        analyst=LLMAnalyst(llm, clock=clock) if llm else StubAnalyst(clock),
        memory=FileMemory(out / "memory") if memory else None,
        gatekeeper=Gatekeeper(runnable=bench_problems),
        operator=Operator(executor, clock, rows_path, poll_seconds=poll_seconds),
        prereg=prereg, catalog=catalog, stream=bench_stream(),
        compute_cap_usd=compute_cap_usd, concurrency=concurrency, deadline=deadline,
        minutes_per_run=MINUTES_PER_RUN, fault=fault, scout=scout,
        design_arms=design_claim_arms(), explore_inbox=out / "explore" / "literature.jsonl",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executor", choices=["sim", "modal"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for rows, state, log")
    parser.add_argument("--rows", type=Path, default=None,
                        help="modal: the rows file to read and append (default <out>/rows.jsonl)")
    parser.add_argument("--prereg", type=Path, default=None,
                        help="modal: PREREG thresholds (a .yaml, or a .md with an "
                             "`agent_thresholds` block)")
    parser.add_argument("--go", action="store_true",
                        help="modal: launch for real; without it, print the setup only")
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--seed", type=int, default=0, help="simulator world seed")
    parser.add_argument("--compute-cap-usd", type=float, default=700.0)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--llm", choices=["none", "openai"], default="none",
                        help="model for the Planner and Analyst; none = deterministic roles")
    parser.add_argument("--scout", choices=["none", "exa"], default="none",
                        help="exa = after replicate, search the web with Exa for one technique "
                             "not in the catalog (needs --llm openai and EXA_API_KEY)")
    parser.add_argument("--config", type=Path, default=None,
                        help="agent config YAML (models, prices, model budget) for --llm openai")
    args = parser.parse_args(argv)

    if args.scout == "exa" and args.llm != "openai":
        parser.error("--scout exa needs --llm openai (the Scout is a model step)")
    llm = None
    if args.llm == "openai":
        if args.config is None:
            parser.error("--llm openai needs --config")
        cfg = load_agent_config(args.config)
        agents = ("planner", "analyst") + (("scout",) if args.scout == "exa" else ())
        models = cfg.require_models(agents)
        llm = BudgetedLLM(
            OpenAIClient(models, max_output_tokens=cfg.llm_max_output_tokens),
            cap_usd=cfg.model_budget_usd, log_path=args.out / "llm_calls.jsonl",
            prices=cfg.prices, max_output_tokens=cfg.llm_max_output_tokens,
        )
    if args.executor == "modal":
        return _main_modal(args, llm)
    scout = LLMScout(llm, ExaSearch()) if args.scout == "exa" and llm else None
    conductor = build_sim(args.out, args.seed, args.compute_cap_usd, llm=llm, scout=scout)
    return _run_and_print(conductor, args)


def _main_modal(args: argparse.Namespace, llm: LLMClient | None) -> int:
    from .launch_design import Halt, code_version
    from .modal_executor import ModalFunctionBackend

    if args.prereg is None:
        print("--executor modal needs --prereg (analysis/PREREG.md has no machine-readable "
              "`agent_thresholds` block yet)")
        return 2
    try:
        prereg = load_prereg(args.prereg)
    except PreregError as exc:
        print(f"PREREG: {exc}")
        return 2
    scout = (LLMScout(llm, ExaSearch(), runnable=bench_problems)
             if args.scout == "exa" and llm else None)
    if not args.go:
        catalog = real_catalog()
        print(f"dry run: nothing launched. catalog {list(catalog.names())}; "
              f"replicate seeds {list(prereg.replicate_seeds)}; cap ${args.compute_cap_usd:.0f}; "
              f"scout {'on' if scout else 'off'}; rows {args.rows or args.out / 'rows.jsonl'}. "
              "Add --go to launch.")
        return 0
    try:
        sha = code_version()
    except Halt as stop:
        print(f"HALTED: {stop}")
        return 2
    conductor = build_real(args.out, prereg, backend=ModalFunctionBackend(), git_sha=sha,
                           compute_cap_usd=args.compute_cap_usd, rows_path=args.rows,
                           gpu=args.gpu, llm=llm, scout=scout)
    return _run_and_print(conductor, args)


def _run_and_print(conductor: Conductor, args: argparse.Namespace) -> int:
    state = conductor.run(args.max_transitions)
    decisions = read_jsonl(conductor.log_path).records
    for d in decisions:
        data = d.get("data", {})
        accepted = {True: " accepted", False: " REJECTED", None: ""}[data.get("accepted")]
        choice = json.dumps(data.get("choice"))[:110]
        print(f"{d['ts'][11:19]}  {d['state']:<24}{accepted}  {choice}")
    print(f"\nstate: {state.state.id}   decisions: {len(decisions)}   "
          f"compute spent: ${state.compute_usd_spent:.2f}")
    if state.halted_reason:
        print(f"halted: {state.halted_reason}")
    print(f"log: {conductor.log_path}\nrows: {conductor.rows_path}")
    return 0 if state.state.id in ("done",) or args.max_transitions else 1


if __name__ == "__main__":
    raise SystemExit(main())
