"""`clh tinker-analyze` — the C5 metrics over the rows in results/tinker/rows.jsonl.

Per arm: the final average accuracy (mean over tasks at the last stage), gap closed =
(method - naive) / (joint - naive), and average forgetting = mean over the earlier tasks of
(best stage accuracy - final accuracy). Intervals come from a paired bootstrap over seeds
and eval items (numpy, 10k resamples): the same seed draw and the same item draw are applied
to every arm, so the pairing honest rows share is kept. If joint - naive < 0.03 the gap
closed carries an "unstable denominator" warning (analysis/PREREG.md section 10).

Writes results/tinker/analysis.json and analysis.md. `clh tinker-figures` renders the
R[stage, task] heatmap from the same rows. Rows with smoke=True never reach a metric —
they are pipeline checks, not data — but they still count toward the budget in rows.py.
"""

from __future__ import annotations

import html
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT
from clharness.tinker.data import TASK_ORDER
from clharness.tinker.rows import ROWS_PATH, TinkerRow, read_tinker_rows

DENOMINATOR_WARN = 0.03  # PREREG 10: below this, gap closed's denominator is unstable
CHUNK = 500  # bootstrap resamples processed at a time (bounds the fancy-index memory)


def final_avg_acc(row: TinkerRow) -> float:
    """Mean task accuracy at the last stage."""
    last = row.r_matrix[-1]
    vals = [v for v in last if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def forgetting(row: TinkerRow) -> float:
    """Mean over tasks of (best stage accuracy - final stage accuracy). One-stage rows
    (joint) forget nothing by construction."""
    per_task = []
    for t in range(len(TASK_ORDER)):
        col = [r[t] for r in row.r_matrix if t < len(r) and r[t] is not None]
        if col:
            per_task.append(max(col) - col[-1])
    return float(np.mean(per_task)) if per_task else float("nan")


def _items_path(row: TinkerRow) -> Path | None:
    if not row.items_path:
        return None
    p = Path(row.items_path)
    p = p if p.is_absolute() else REPO_ROOT / p
    return p if p.exists() else None


def _final_items(row: TinkerRow) -> np.ndarray | None:
    """Per-item correctness at the final stage, shape (n_tasks, n_items), or None."""
    p = _items_path(row)
    if p is None:
        return None
    arr = np.load(p)
    return arr[-1]  # (n_tasks, n_items)


def _full_items(row: TinkerRow) -> np.ndarray | None:
    p = _items_path(row)
    return np.load(p) if p is not None else None


def _by_arm(rows: list[TinkerRow]) -> dict[str, dict[int, TinkerRow]]:
    out: dict[str, dict[int, TinkerRow]] = {}
    for r in rows:
        if r.status == "finished" and not r.smoke:
            out.setdefault(r.arm, {})[r.seed] = r
    return out


def _stat_block(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5))}


def _boot_final_acc(by_seed: dict[int, np.ndarray], seed_draws: np.ndarray,
                    item_draws: np.ndarray) -> np.ndarray:
    """(n_boot,) mean final accuracy over resampled seeds, tasks and items.

    by_seed: seed -> (n_tasks, n_items) final-stage correctness. seed_draws: (n_boot, k)
    indices into sorted(by_seed); item_draws: (n_boot, n_tasks, n_items).
    """
    seeds = sorted(by_seed)
    stack = np.stack([by_seed[s] for s in seeds])  # (k, T, N)
    n_boot = seed_draws.shape[0]
    out = np.empty(n_boot)
    t_idx = np.arange(stack.shape[1])
    for lo in range(0, n_boot, CHUNK):
        sd, idr = seed_draws[lo:lo + CHUNK], item_draws[lo:lo + CHUNK]
        # (b, k, T, N) -> mean over seeds, tasks and items -> (b,)
        sel = stack[sd[:, :, None, None], t_idx[None, None, :, None], idr[:, None, :, :]]
        out[lo:lo + sd.shape[0]] = np.nanmean(sel, axis=(1, 2, 3))
    return out


def analyze(
    rows_path: str | Path = ROWS_PATH,
    out_dir: str | Path = REPO_ROOT / "results" / "tinker",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    rows = read_tinker_rows(rows_path)
    arms = _by_arm(rows)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": str(rows_path), "n_boot": n_boot, "seed": seed, "tasks": TASK_ORDER,
        "smoke_rows_skipped": sum(1 for r in rows if r.smoke and r.status == "finished"),
        "arms": {}, "warnings": [],
    }

    # Per-arm point estimates. Item files, where present, let the bootstrap resample items.
    item_arrays: dict[str, dict[int, np.ndarray]] = {}
    for arm, by_seed in sorted(arms.items()):
        seeds = sorted(by_seed)
        finals = np.array([final_avg_acc(by_seed[s]) for s in seeds])
        forget = np.array([forgetting(by_seed[s]) for s in seeds])
        entry: dict[str, Any] = {
            "seeds": seeds,
            "final_avg_acc": {"mean": float(finals.mean()), "per_seed": finals.tolist()},
            "forgetting": {"mean": float(forget.mean()), "per_seed": forget.tolist()},
            "est_usd_total": round(sum(r.est_usd or 0 for r in by_seed.values()), 4),
            "n_rows": len(by_seed),
        }
        loaded = {s: _final_items(by_seed[s]) for s in seeds}
        item_arrays[arm] = {s: a for s, a in loaded.items() if a is not None}
        if finals.size:
            boot = finals[rng.integers(0, len(seeds), size=(n_boot, len(seeds)))].mean(axis=1)
            entry["final_avg_acc"].update(lo=float(np.percentile(boot, 2.5)),
                                        hi=float(np.percentile(boot, 97.5)))
        out["arms"][arm] = entry

    # Gap closed, paired on the seeds every needed arm ran, resampling seeds AND items.
    method_arms = [a for a in arms if a not in ("naive", "joint")]
    if {"naive", "joint"} <= set(arms):
        naive, joint = arms["naive"], arms["joint"]
        common = sorted(set(naive) & set(joint))
        denom_point = (np.array([final_avg_acc(joint[s]) for s in common]).mean()
                       - np.array([final_avg_acc(naive[s]) for s in common]).mean())
        if denom_point < DENOMINATOR_WARN:
            out["warnings"].append(
                f"unstable denominator: joint - naive = {denom_point:.3f} < {DENOMINATOR_WARN}; "
                "read gap closed with care (analysis/PREREG.md section 10)")
        out["denominator"] = {"joint_minus_naive": float(denom_point), "seeds": common,
                              "unstable": bool(denom_point < DENOMINATOR_WARN)}
        for m in method_arms:
            common_m = sorted(set(common) & set(arms[m]))
            if not common_m:
                continue
            n_seeds = len(common_m)
            have_items = all(
                s in item_arrays.get(a, {}) for a in ("naive", "joint", m) for s in common_m)
            seed_draws = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
            if have_items:
                n_items = item_arrays["naive"][common_m[0]].shape[1]
                item_draws = rng.integers(0, n_items, size=(n_boot, len(TASK_ORDER), n_items))
                acc = {a: _boot_final_acc({s: item_arrays[a][s] for s in common_m},
                                          seed_draws, item_draws)
                       for a in ("naive", "joint", m)}
            else:  # fall back to a seed-only resample of the row-level scores
                acc = {}
                for a in ("naive", "joint", m):
                    finals = np.array([final_avg_acc(arms[a][s]) for s in common_m])
                    acc[a] = finals[seed_draws].mean(axis=1)
            denom = acc["joint"] - acc["naive"]
            gap = np.divide(acc[m] - acc["naive"], denom,
                            out=np.full(n_boot, np.nan), where=np.abs(denom) > 1e-9)
            gap = gap[np.isfinite(gap)]
            out["arms"][m]["gap_closed"] = _stat_block(gap)

    # Forgetting intervals, item-resampled where the files exist.
    for arm, entry in out["arms"].items():
        seeds = entry["seeds"]
        full = {s: _full_items(arms[arm][s]) for s in seeds}
        if all(a is not None for a in full.values()):
            k = len(seeds)
            n_items = full[seeds[0]].shape[2]
            seed_draws = rng.integers(0, k, size=(n_boot, k))
            item_draws = rng.integers(0, n_items, size=(n_boot, len(TASK_ORDER), n_items))
            stack = np.stack([full[s] for s in seeds])  # (k, S, T, N)
            boot = np.empty(n_boot)
            t_idx = np.arange(len(TASK_ORDER))
            for lo in range(0, n_boot, CHUNK):
                sd, idr = seed_draws[lo:lo + CHUNK], item_draws[lo:lo + CHUNK]
                # (b, k, S, T, N): items resampled per task, shared across stages
                sel = stack[sd[:, :, None, None, None],
                            np.arange(stack.shape[1])[None, None, :, None, None],
                            t_idx[None, None, None, :, None], idr[:, None, None, :, :]]
                with warnings.catch_warnings():  # unseen (stage, task) cells are all-NaN
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    acc = np.nanmean(sel, axis=4)  # (b, k, S, T)
                forg = np.nanmax(acc, axis=2) - acc[:, :, -1, :]  # (b, k, T)
                boot[lo:lo + sd.shape[0]] = np.nanmean(forg, axis=(1, 2))
            entry["forgetting"].update(lo=float(np.percentile(boot, 2.5)),
                                       hi=float(np.percentile(boot, 97.5)))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    (out_dir / "analysis.md").write_text(_markdown(out))
    return out


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# C5 — second task sequence (Tinker, exploratory)",
        "",
        f"generated {analysis['generated_at']} · tasks: {' → '.join(analysis['tasks'])}"
        + (f" · {analysis['smoke_rows_skipped']} smoke row(s) skipped"
           if analysis.get("smoke_rows_skipped") else ""),
        "",
        "| arm | seeds | final avg acc | 95% CI | gap closed | 95% CI | forgetting | 95% CI | est $ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, e in analysis["arms"].items():
        fa, gc, fg = e["final_avg_acc"], e.get("gap_closed"), e["forgetting"]
        lines.append(
            f"| {arm} | {','.join(map(str, e['seeds']))} | {fa['mean']:.3f} | "
            f"[{fa.get('lo', float('nan')):.3f}, {fa.get('hi', float('nan')):.3f}] | "
            + (f"{gc['mean']:.3f} | [{gc['lo']:.3f}, {gc['hi']:.3f}] |" if gc else "— | — |")
            + f" {fg['mean']:.3f} | [{fg.get('lo', float('nan')):.3f}, {fg.get('hi', float('nan')):.3f}] | "
              f"{e['est_usd_total']:.2f} |")
    for w in analysis["warnings"]:
        lines += ["", f"**warning:** {w}"]
    lines.append("")
    return "\n".join(lines)


# --- the R[stage, task] heatmap (clh tinker-figures) ---------------------------------------------


def _cell_color(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "#e5e7eb"
    v = min(1.0, max(0.0, v))
    lo = np.array([255, 255, 255])
    hi = np.array([29, 78, 216])  # ACCENT blue, same palette as analysis/figures.py
    c = (lo + (hi - lo) * v).astype(int)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def r_heatmap_svg(rows_path: str | Path = ROWS_PATH) -> str:
    """Mean R[stage, task] per arm: rows = stage reached, columns = task. The forgetting
    pattern shows as a fade down each task column."""
    arms = _by_arm([r for r in read_tinker_rows(rows_path)])
    cell_w, cell_h, pad_l, pad_t, gap = 64, 30, 120, 46, 30
    panel_w = pad_l + 3 * cell_w + 10
    width, height = int(3 * panel_w + 2 * gap), int(pad_t + 3 * cell_h + 60)
    parts = [(f"<text x='{width / 2}' y='20' text-anchor='middle' font-size='13' "
              f"fill='#111827'>C5 accuracy matrix R[stage, task] — mean over seeds</text>")]
    for ai, arm in enumerate(sorted(arms)):
        by_seed = arms[arm]
        mats = [np.array([[np.nan if v is None else v for v in row] for row in r.r_matrix])
                for r in by_seed.values()]
        with warnings.catch_warnings():  # unseen cells are all-NaN across seeds
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_r = np.nanmean(np.stack(mats), axis=0) if mats else np.full((0, 3), np.nan)
        ox = gap + ai * panel_w
        parts.append(f"<text x='{ox + pad_l + 1.5 * cell_w}' y='{pad_t - 24}' "
                     f"text-anchor='middle' font-size='11' fill='#111827'>{html.escape(arm)}"
                     f" ({len(mats)} seed{'s' if len(mats) != 1 else ''})</text>")
        for t, task in enumerate(TASK_ORDER):
            parts.append(f"<text x='{ox + pad_l + t * cell_w + cell_w / 2}' y='{pad_t - 8}' "
                         f"text-anchor='middle' font-size='9' fill='#6b7280'>{task}</text>")
        for s in range(mean_r.shape[0]):
            parts.append(f"<text x='{ox + pad_l - 6}' "
                         f"y='{pad_t + s * cell_h + cell_h / 2 + 3}' text-anchor='end' "
                         f"font-size='9' fill='#6b7280'>after stage {s + 1}</text>")
            for t in range(mean_r.shape[1]):
                v = mean_r[s, t]
                x, y = ox + pad_l + t * cell_w, pad_t + s * cell_h
                parts.append(f"<rect x='{x}' y='{y}' width='{cell_w - 2}' "
                             f"height='{cell_h - 2}' fill='{_cell_color(v)}' "
                             f"stroke='#ffffff'/>")
                txt = "—" if not np.isfinite(v) else f"{v:.2f}"
                ink = "#ffffff" if np.isfinite(v) and v > 0.55 else "#111827"
                parts.append(f"<text x='{x + (cell_w - 2) / 2}' y='{y + cell_h / 2 + 3}' "
                             f"text-anchor='middle' font-size='10' fill='{ink}'>{txt}</text>")
    parts.append(f"<text x='{width / 2}' y='{height - 12}' text-anchor='middle' "
                 f"font-size='9' fill='#6b7280'>verbalizer accuracy; darker is better; "
                 f"'—' = task not yet trained</text>")
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            f"font-family='ui-monospace, Menlo, monospace' font-size='10'>" + "".join(parts)
            + "</svg>\n")


def write_heatmap(rows_path: str | Path = ROWS_PATH,
                  out: str | Path = REPO_ROOT / "results" / "figures" / "tinker_r_heatmap.svg",
                  ) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r_heatmap_svg(rows_path))
    return out
