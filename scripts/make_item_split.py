"""Split each protected multiple-choice task into a selection half and a report half.

Pre-registered in analysis/PREREG.md (section "Item split"). The split is stratified by whether
the base model answers the item correctly, so both halves have the same forgetting headroom.

Source of base correctness: checkpoint 0 (the base model, before any update) of a frozen
September run. Positions refer to the order of `indices` in manifests/eval-protected-v1.json,
which is also the order of items_<task>.npy, so `position` p means manifest index indices[p].

Deterministic: the same seed and source give byte-identical output. Re-running is a check, not
a change; the committed manifest is the one every reported number uses.

    uv run python scripts/make_item_split.py            # writes manifests/eval-protected-v1-split.json
    uv run python scripts/make_item_split.py --check    # verifies the committed file matches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTECTED = REPO / "manifests" / "eval-protected-v1.json"
OUT = REPO / "manifests" / "eval-protected-v1-split.json"
SOURCE_RUN = "results/runs/pilot-chosen-200M"  # frozen; its ckpt_0000 is the base model
TASKS = ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal")  # MBPP is secondary, not split
SEED = 20261114


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    manifest = json.loads(PROTECTED.read_text())
    rng = np.random.default_rng(SEED)
    out: dict = {
        "suite": manifest["suite"],
        "derived_from": "manifests/eval-protected-v1.json",
        "derived_from_sha256": _sha256(PROTECTED),
        "seed": SEED,
        "stratify_by": "base_correct",
        "base_source": f"{SOURCE_RUN}/checkpoints/ckpt_0000",
        "tasks": {},
    }
    for task in TASKS:
        src = REPO / SOURCE_RUN / "checkpoints" / "ckpt_0000" / f"items_{task}.npy"
        correct = np.load(src).astype(bool)
        indices = manifest["tasks"][task]["indices"]
        if len(correct) != len(indices):
            raise ValueError(f"{task}: {len(correct)} outcomes vs {len(indices)} manifest items")
        selection: list[int] = []
        report: list[int] = []
        # Within each stratum, shuffle and deal alternately. An odd stratum gives its extra item
        # to whichever half is currently smaller, so the halves end exactly 250/250.
        for stratum in (True, False):
            pos = np.flatnonzero(correct == stratum)
            rng.shuffle(pos)
            for p in pos.tolist():
                (selection if len(selection) <= len(report) else report).append(p)
        selection.sort()
        report.sort()
        out["tasks"][task] = {
            "n_items": len(indices),
            "source_sha256": _sha256(src),
            "selection_positions": selection,
            "report_positions": report,
            "selection_base_acc": round(float(correct[selection].mean()), 4),
            "report_base_acc": round(float(correct[report].mean()), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the committed file instead of writing")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"MISMATCH: {OUT.relative_to(REPO)} differs from a fresh build", file=sys.stderr)
            return 1
        print(f"OK: {OUT.relative_to(REPO)} matches a fresh build")
        return 0
    OUT.write_text(text)
    split = json.loads(text)["tasks"]
    for task, t in split.items():
        print(f"{task:14s} selection {len(t['selection_positions'])} (base {t['selection_base_acc']:.3f})"
              f" | report {len(t['report_positions'])} (base {t['report_base_acc']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
