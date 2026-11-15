"""Print the "Exploratory runs" table of docs/TECHNIQUES.md from the result rows.

    uv run python scripts/techniques_table.py

Reads the row files below, keeps finished fast-bench rows, and groups three kinds of arm by
technique settings and learning rate: no technique, the reference arm and its warmup-only
variant, and LoRA on its own. Every number in the table is a mean over the seeds of one arm,
with the standard deviation across seeds in brackets. Nothing is typed in.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# the reference arm is whatever its spec file says, so this table cannot drift from it
REFERENCE = json.loads((ROOT / "configs/bench/specs/reference-s0.json").read_text())["techniques"]
# later files win when two hold the same spec, so a re-measured row replaces the original
FILES = (
    "rows.phase0.jsonl",
    "rows.phase0.remeasured.jsonl",
    "rows.reference.jsonl",
    "rows.reference.remeasured.jsonl",
    "agent/rows.jsonl",
    "rows.explore.jsonl",
)
BENCH_LR = 1e-4
LORA_DEFAULT = {"rank": 8, "lr": 1e-3}  # the plugin's defaults, for specs that leave them out


def arm(spec: dict) -> tuple[tuple, str] | None:
    """(sort key, label) for the arms this table shows; None for every other spec."""
    techniques = {t["name"]: t.get("params") or {} for t in spec["techniques"]}
    lr = (spec.get("optim") or {}).get("lr") or BENCH_LR
    if not techniques:
        return (0, -lr), f"no mitigation, lr {lr:g}"
    if set(techniques) == {"lora"}:
        p = {**LORA_DEFAULT, **techniques["lora"]}
        rate = p["lr"] if p["lr"] is not None else lr
        return (2, p["rank"], -rate), f"LoRA rank {p['rank']}, lr {rate:g}"
    if lr != BENCH_LR:
        return None
    if spec["techniques"] == REFERENCE:
        return (1, 0), f"reference arm (lr_rewarm + replay), lr {lr:g}"
    if set(techniques) == {"lr_rewarm"} and techniques["lr_rewarm"].get("schedule") == "constant":
        return (1, 1), f"lr_rewarm, warmup only (constant schedule), lr {lr:g}"
    return None


def main() -> None:
    by_hash: dict[str, tuple[dict, str]] = {}
    for name in FILES:
        path = RESULTS / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["status"] == "finished" and row["spec"].get("bench") == "fast":
                by_hash[row["spec_hash"]] = (row, name)

    groups: dict[tuple, tuple[str, list[dict], set[str]]] = {}
    for row, name in by_hash.values():
        found = arm(row["spec"])
        if found:
            key, label = found
            _, rows, sources = groups.setdefault(key, (label, [], set()))
            rows.append(row)
            sources.add(name)

    def spread(xs: list[float]) -> str:
        return f"{st.mean(xs):+.2f}" + (f" ({st.stdev(xs):.2f})" if len(xs) > 1 else "")

    def mean(xs: list[float]) -> str:
        return f"{st.mean(xs):.3f}"

    print("| Arm | Seeds | Retention, report half | Retention, selection half | Legal perplexity ratio "
          "| Code perplexity ratio, end of stream | General perplexity ratio | Rows in |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        label, rows, sources = groups[key]
        acq = [r["acquisition"] for r in rows]
        print(
            f"| {label} | {len(rows)} "
            f"| {spread([100 * r['retention']['report']['mean_delta'] for r in rows])} "
            f"| {spread([100 * r['retention']['selection']['mean_delta'] for r in rows])} "
            f"| {mean([a['ppl_ratio']['legal'] for a in acq])} "
            f"| {mean([a['ppl_ratio']['code'] for a in acq])} "
            f"| {mean([a['general_ppl_ratio'] for a in acq])} "
            f"| {', '.join(f'`{s}`' for s in sorted(sources))} |"
        )


if __name__ == "__main__":
    main()
