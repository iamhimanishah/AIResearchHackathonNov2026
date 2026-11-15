"""Freeze the "fragile core": the questions the plain run forgets in almost every seed.

Found on Saturday night of the hackathon (exploratory). On the selection half, a small set of
questions the base model answers correctly is forgotten in at least 4 of the 5 no-mitigation
runs at the bench learning rate (1e-4), while most known questions are never forgotten. This
script writes that set down once, so later runs can be scored on "how much of the core did this
run lose" (clharness/analysis/item_profile.py).

Rule, per protected task, selection half only:

    base-correct  = the base model (checkpoint 0) scores above 0.5 on the item in every source run
    forgotten     = base-correct and at or below 0.5 at the last checkpoint of a run
    core          = forgotten in at least MIN_SEEDS of the source runs

Source runs: the finished rows in results/rows.phase0.remeasured.jsonl with no technique and no
optimizer override, i.e. the five no-mitigation seeds. Positions are positions in the task's full
item list (the order of items_<task>.npy), so they can be used to index those files directly.

The report half is never read: only positions from `half_indices(task, n)[0]` are loaded into
anything. Because these runs define the set, their own core-forgetting rate is high by
construction (at least 80% each on average) and must not be compared with other arms as if it
were measured independently.

    uv run python scripts/make_core_items.py            # writes manifests/eval-protected-v1-core.json
    uv run python scripts/make_core_items.py --check    # verifies the committed file matches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from clharness.executor.rows import half_indices
from clharness.spec import read_rows

ROWS = "results/rows.phase0.remeasured.jsonl"
SPLIT = "manifests/eval-protected-v1-split.json"
OUT = REPO / "manifests" / "eval-protected-v1-core.json"
TASKS = ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal")
MIN_SEEDS = 4
DERIVED_ON = "2026-11-14"  # fixed, so re-running reproduces the file byte for byte


def build() -> dict:
    runs = sorted((r for r in read_rows(REPO / ROWS)
                   if r.status == "finished" and not r.spec.techniques and not r.spec.optim and r.items_path),
                  key=lambda r: r.spec.seed)
    if len(runs) != 5:
        raise SystemExit(f"expected the five no-mitigation seeds in {ROWS}, found {len(runs)}")
    tasks: dict[str, dict] = {}
    for task in TASKS:
        base_ok = None
        times_forgotten = None
        for row in runs:
            ckpts = sorted((REPO / row.items_path).glob("ckpt_*"))
            base, final = (np.load(c / f"items_{task}.npy") for c in (ckpts[0], ckpts[-1]))
            selection = half_indices(task, len(base))[0]
            keep = np.zeros(len(base), dtype=bool)
            keep[selection] = True  # everything outside the selection half is dropped here
            b, f = (base > 0.5) & keep, (final > 0.5) & keep
            base_ok = b if base_ok is None else base_ok & b
            forgot = b & ~f
            times_forgotten = forgot.astype(int) if times_forgotten is None else times_forgotten + forgot
        core = np.flatnonzero(base_ok & (times_forgotten >= MIN_SEEDS))
        tasks[task] = {
            "n_items": len(base_ok),
            "n_selection": len(selection),
            "n_base_correct": int(base_ok.sum()),
            "n_core": len(core),
            "core_positions": [int(p) for p in core],
        }
    n_core = sum(t["n_core"] for t in tasks.values())
    n_base = sum(t["n_base_correct"] for t in tasks.values())
    return {
        "suite": "protected-v1",
        "what": "fragile core: selection-half questions the base model answers correctly and the "
                "no-mitigation run at lr 1e-4 forgets in at least 4 of 5 seeds",
        "status": "exploratory; derived after the pre-registration, not part of the confirmatory analysis",
        "derived_by": "scripts/make_core_items.py",
        "derived_on": DERIVED_ON,
        "rule": {"half": "selection", "base_correct": "checkpoint 0 score > 0.5 in every source run",
                 "forgotten": "base-correct and last-checkpoint score <= 0.5", "min_seeds": MIN_SEEDS,
                 "of_seeds": len(runs)},
        "rows_file": ROWS,
        "source_runs": [{"name": r.spec.name, "seed": r.spec.seed, "spec_hash": r.spec_hash,
                         "items_path": r.items_path} for r in runs],
        "split": SPLIT,
        "split_sha256": hashlib.sha256((REPO / SPLIT).read_bytes()).hexdigest(),
        "caveat": "the source runs define the set, so their own core-forgetting rate is inflated by "
                  "construction; do not use the no-mitigation arm at lr 1e-4 as a control for it",
        "n_core": n_core,
        "n_base_correct": n_base,
        "n_noncore_base_correct": n_base - n_core,
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the committed file; write nothing")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(REPO)} does not match what this script derives", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(REPO)} matches")
        return 0
    OUT.write_text(text)
    made = json.loads(text)
    print(f"wrote {OUT.relative_to(REPO)}: {made['n_core']} core items "
          f"({ {t: v['n_core'] for t, v in made['tasks'].items()} }), "
          f"{made['n_noncore_base_correct']} other base-correct items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
