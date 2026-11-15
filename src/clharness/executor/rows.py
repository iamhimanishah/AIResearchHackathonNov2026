"""Turn a finished run directory into the measured parts of a result row.

Reads only the files the training loop already writes under <run>/checkpoints/ckpt_NNNN/:
meta.json, metrics.json and items_<task>.npy (per-item correctness, in manifest order). No torch.
"""

from __future__ import annotations

import functools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from clharness.spec import Acquisition, BoundaryRetention, HalfRetention, Retention, TaskRetention

# The protected items of each task are split in two: decisions use the selection half, reported
# numbers use the report half. analysis/PREREG.md (section 3) pins the split for the protected
# set in manifests/eval-protected-v1-split.json (stratified by base-model correctness). Any other
# task or item count (smoke sets, tests) falls back to a fixed shuffle seeded per task.
SPLIT_SEED = 20261114
SPLIT_MANIFEST = Path(__file__).resolve().parents[3] / "manifests" / "eval-protected-v1-split.json"
COLLAPSE_SHARE = 0.9  # most-common-answer share above this flags a task as collapsed (not invalid)


@functools.lru_cache(maxsize=1)
def _pinned_split() -> dict[str, Any]:
    if not SPLIT_MANIFEST.exists():
        return {}
    return json.loads(SPLIT_MANIFEST.read_text())["tasks"]


def half_indices(task: str, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """(selection, report) item positions for a task. Deterministic; the halves are disjoint."""
    pinned = _pinned_split().get(task)
    if pinned is not None and pinned["n_items"] == n_items:
        return np.array(pinned["selection_positions"]), np.array(pinned["report_positions"])
    order = list(range(n_items))
    random.Random(f"{SPLIT_SEED}:{task}").shuffle(order)
    half = n_items // 2
    return np.array(sorted(order[:half])), np.array(sorted(order[half:]))


def _checkpoints(run_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    out = []
    for d in sorted((run_dir / "checkpoints").glob("ckpt_*")):
        if (d / "meta.json").exists() and (d / "metrics.json").exists():
            out.append((json.loads((d / "meta.json").read_text()),
                        json.loads((d / "metrics.json").read_text()), d))
    return out


def _half(base_dir: Path, cur_dir: Path, tasks: list[str], which: int) -> HalfRetention:
    per_task = {}
    for task in tasks:
        base = np.load(base_dir / f"items_{task}.npy")
        cur = np.load(cur_dir / f"items_{task}.npy")
        idx = half_indices(task, len(base))[which]
        b, c = float(np.nanmean(base[idx])), float(np.nanmean(cur[idx]))
        per_task[task] = TaskRetention(base=b, final=c, delta=c - b, n_items=len(idx))
    mean = sum(t.delta for t in per_task.values()) / len(per_task)
    return HalfRetention(per_task=per_task, mean_delta=mean)


def _answer_share(metrics: dict[str, Any], ckpt_dir: Path, task: str) -> float | None:
    v = metrics.get(f"eval/lm_eval/{task}/answer_share_max")
    if v is not None:
        return float(v)
    p = ckpt_dir / f"choices_{task}.npy"
    if not p.exists():
        return None
    ch = np.load(p)
    ch = ch[ch >= 0]
    return float(np.bincount(ch).max() / ch.size) if ch.size else None


def measure_run(run_dir: str | Path, trained_domains: list[str]) -> dict[str, Any]:
    """Retention, acquisition and the answer-share check for a finished run.

    Returns the fields of a ResultRow that come from measurement, plus `invalid_reason`
    (None when the run passes the sanity checks) and `gradient_tokens`.
    """
    run_dir = Path(run_dir)
    ckpts = _checkpoints(run_dir)
    if len(ckpts) < 2:
        raise ValueError(f"{run_dir}: need the base checkpoint and at least one more, found {len(ckpts)}")
    (_, base_metrics, base_dir), (final_meta, final_metrics, final_dir) = ckpts[0], ckpts[-1]
    # A run that died part-way leaves a directory that looks finished up to its last checkpoint.
    # Refuse it: the last checkpoint must be the boundary of the last trained domain, and every
    # trained domain must have its boundary, in stream order.
    done = [meta["domain"] for meta, _, _ in ckpts[1:] if meta["is_domain_boundary"]]
    if done != list(trained_domains) or not final_meta["is_domain_boundary"]:
        raise ValueError(f"{run_dir}: incomplete run: domain boundaries {done}, expected {list(trained_domains)}")
    tasks = sorted(p.stem.removeprefix("items_") for p in base_dir.glob("items_*.npy")
                   if (final_dir / p.name).exists())
    if not tasks:
        raise ValueError(f"{run_dir}: no per-item eval files in the base and final checkpoints")

    selection = _half(base_dir, final_dir, tasks, 0)
    report = _half(base_dir, final_dir, tasks, 1)
    boundaries = [
        BoundaryRetention(
            domain=meta["domain"],
            gradient_tokens=meta["gradient_tokens"],
            selection_mean_delta=_half(base_dir, d, tasks, 0).mean_delta,
            report_mean_delta=_half(base_dir, d, tasks, 1).mean_delta,
        )
        for meta, _, d in ckpts[1:]
        if meta["is_domain_boundary"]
    ]

    def ratio(domain: str, metrics: dict[str, Any] = final_metrics) -> float | None:
        key = f"eval/ppl_doc/{domain}"
        if key not in base_metrics or key not in metrics:
            return None
        return float(metrics[key]) / float(base_metrics[key])

    def geomean(values: dict[str, float]) -> float | None:
        return math.exp(sum(math.log(v) for v in values.values()) / len(values)) if values else None

    ratios = {d: r for d in trained_domains if (r := ratio(d)) is not None}
    if not ratios:
        raise ValueError(f"{run_dir}: no held-out perplexity for the trained domains {trained_domains}")
    mean_ratio = geomean(ratios)

    # Acquisition at each domain's own boundary: the last boundary checkpoint of that domain.
    own_metrics = {meta["domain"]: m for meta, m, _ in ckpts[1:] if meta["is_domain_boundary"]}
    own = {d: r for d in trained_domains if d in own_metrics and (r := ratio(d, own_metrics[d])) is not None}

    # Answer collapse (share > COLLAPSE_SHARE) is recorded in `answer_share` and flagged by the
    # analysis, but it does not invalidate the run: collapse is itself forgetting (PREREG A9).
    shares = {t: s for t in tasks if (s := _answer_share(final_metrics, final_dir, t)) is not None}
    reasons = []
    if not all(math.isfinite(v) for v in [*ratios.values(), selection.mean_delta, report.mean_delta]):
        reasons.append("non-finite score or perplexity")

    return {
        "retention": Retention(selection=selection, report=report, boundaries=boundaries),
        "acquisition": Acquisition(ppl_ratio=ratios, mean_ppl_ratio=mean_ratio,
                                   general_ppl_ratio=ratio("general"),
                                   own_boundary_ppl_ratio=own, mean_own_boundary_ppl_ratio=geomean(own)),
        "answer_share": shares,
        "gradient_tokens": int(final_meta["gradient_tokens"]),
        "invalid_reason": "; ".join(reasons) or None,
    }
