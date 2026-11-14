"""Attribution, uncertainty and the pre-registered decision rule, over many result rows.

Every rule here is written down in analysis/PREREG.md; section numbers below point to it.

What it computes
  * technique effects: least squares on 0/1 technique indicators (+ estimable pairwise
    interactions), so a coefficient is "what switching this on changes" (section 7)
  * the noise floor sigma_noise: SD of the primary outcome over the no-mitigation seeds (6.1)
  * promotion: the top configurations on the selection half that pass the acquisition gate (6.3)
  * the paired hierarchical bootstrap between two arms, Holm across the confirmatory family,
    and the three-part "beats the reference arm" rule (5, 7)
  * the retention-acquisition Pareto frontier and a cost-adjusted ranking (5, 9)

Halves. Decisions use the selection half. The report half is read only by `analyze(...,
confirmatory=True)`, which is for after every replication run is finished (section 3). The
agent must call `analyze` without `confirmatory`.

Item level. When a row's per-item correctness files are available (`items_path` points at the
run's checkpoints directory), the bootstrap resamples the same items for both arms. When they
are not (fake rows, or a row read away from the volume), it falls back to parametric item noise
with binomial variance per task, drawn independently for each arm. That ignores the positive
correlation between arms on shared items, so its intervals are wider: conservative.

Imports only numpy and clharness.spec, so it is cheap to import from the agent and the page.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from clharness.spec import ResultRow, canonical_json

COLLAPSE_SHARE = 0.90  # PREREG 8: flag above this (provisional: flag, do not invalidate)
ACQ_GATE = 1.05  # PREREG 5, rule 3: acquisition(X) <= 1.05 x acquisition(reference)
SIZE_GATE_SIGMAS = 2.0  # PREREG 5, rule 2: delta > 2 x sigma_noise
MIN_PAIRS = 3  # PREREG 6.3: confirmatory comparisons are over 3 paired seeds
ALPHA = 0.05
REFERENCE = frozenset({"replay", "lr_rewarm"})
NO_MITIGATION: frozenset[str] = frozenset()
Half = str  # "selection" | "report"


# ---------------------------------------------------------------------------------------------
# Grouping rows into configurations
# ---------------------------------------------------------------------------------------------


def config_key(row: ResultRow) -> str:
    """Everything that defines a configuration except the seed and the labels."""
    payload = row.spec.model_dump(mode="json", exclude={"name", "notes", "parent_spec_hash", "seed"})
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]


@dataclass
class Config:
    key: str
    techniques: frozenset[str]
    rows: list[ResultRow] = field(default_factory=list)
    label: str = ""

    @property
    def seeds(self) -> list[int]:
        return sorted(r.spec.seed for r in self.rows)

    def by_seed(self) -> dict[int, ResultRow]:
        return {r.spec.seed: r for r in self.rows}

    def primary(self, half: Half) -> np.ndarray:
        return np.array([_half(r, half).mean_delta for r in self.rows])

    def acquisition(self) -> np.ndarray:
        return np.array([r.acquisition.mean_ppl_ratio for r in self.rows])


def _half(row: ResultRow, half: Half):
    if half not in ("selection", "report"):
        raise ValueError(f"unknown half {half!r}")
    return getattr(row.retention, half)


def collapse_flags(row: ResultRow) -> list[str]:
    return sorted(t for t, s in row.answer_share.items() if s > COLLAPSE_SHARE)


def group(rows: Iterable[ResultRow]) -> dict[str, Config]:
    """Finished rows grouped by configuration. Failed and invalid rows are left out here and
    counted by `analyze`. Labels are the technique names, with a key suffix only when two
    configurations share the same techniques but differ in parameters."""
    configs: dict[str, Config] = {}
    for row in rows:
        if row.status != "finished":
            continue
        key = config_key(row)
        cfg = configs.setdefault(key, Config(key=key, techniques=frozenset(row.spec.technique_names())))
        cfg.rows.append(row)
    names: dict[frozenset[str], list[Config]] = {}
    for cfg in configs.values():
        names.setdefault(cfg.techniques, []).append(cfg)
    for techs, same in names.items():
        base = "+".join(sorted(techs)) or "no-mitigation"
        if len(same) == 1:
            same[0].label = base
            continue
        # name the parameters that differ within the group, e.g. replay+select[select.signal=loss]
        params = [{f"{t.name}.{k}": v for t in c.rows[0].spec.techniques for k, v in t.params.items()}
                  for c in same]
        differing = sorted({k for p in params for k in p if len({json.dumps(q.get(k)) for q in params}) > 1})
        for cfg, p in zip(same, params):
            suffix = ",".join(f"{k}={p.get(k)}" for k in differing)
            cfg.label = f"{base}[{suffix}]" if suffix else f"{base}@{cfg.key[:6]}"
    return configs


def find(configs: dict[str, Config], techniques: frozenset[str]) -> Config | None:
    """The configuration with exactly these techniques. With several (different parameters),
    the one with the most seeds, which is the replicated one."""
    hits = [c for c in configs.values() if c.techniques == techniques]
    return max(hits, key=lambda c: len(c.rows)) if hits else None


# ---------------------------------------------------------------------------------------------
# Noise floor and technique effects
# ---------------------------------------------------------------------------------------------


def noise_sd(cfg: Config, half: Half) -> float | None:
    """SD over seeds of the primary outcome (ddof=1). None with fewer than 2 seeds."""
    vals = cfg.primary(half)
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else None


def fit_effects(
    rows: list[ResultRow],
    outcome: Callable[[ResultRow], float],
    sigma: float | None = None,
    interactions: bool = True,
) -> dict[str, Any]:
    """Least squares of `outcome` on an intercept, one 0/1 column per technique and, if asked,
    one column per pair of techniques that appear together in some row. Columns that would make
    the design rank-deficient are dropped and listed. Standard errors use the residual variance
    when there are spare degrees of freedom, otherwise the known `sigma` (PREREG 7)."""
    techs = sorted({t for r in rows for t in r.spec.technique_names()})
    cols: list[tuple[str, frozenset[str]]] = [(t, frozenset({t})) for t in techs]
    if interactions:
        seen = {frozenset(p) for r in rows for p in itertools.combinations(sorted(r.spec.technique_names()), 2)}
        cols += [("×".join(sorted(p)), p) for p in sorted(seen, key=sorted)]
    present = [frozenset(r.spec.technique_names()) for r in rows]
    y = np.array([outcome(r) for r in rows])

    def column(members: frozenset[str]) -> np.ndarray:
        return np.array([1.0 if members <= p else 0.0 for p in present])

    X = np.ones((len(rows), 1))
    kept, dropped = ["(baseline)"], []
    for name, members in cols:  # greedy: main effects first, then pairs
        trial = np.column_stack([X, column(members)])
        if np.linalg.matrix_rank(trial) > X.shape[1]:
            X, kept = trial, kept + [name]
        else:
            dropped.append(name)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - X.shape[1]
    resid = y - X @ beta
    s2 = float(resid @ resid / df) if df > 0 else None
    if s2 is None and sigma is None:
        raise ValueError("no residual degrees of freedom and no sigma given")
    var = s2 if s2 is not None else sigma**2
    cov = var * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    z = 1.959963984540054
    effects = {
        name: {"estimate": float(b), "se": float(s), "lo": float(b - z * s), "hi": float(b + z * s)}
        for name, b, s in zip(kept, beta, se)
    }
    return {
        "effects": effects, "dropped": dropped, "n_rows": len(rows), "df": df,
        "residual_sd": math.sqrt(s2) if s2 is not None else None,
        "se_source": "residual" if s2 is not None else "sigma_noise",
    }


# ---------------------------------------------------------------------------------------------
# Paired hierarchical bootstrap, Holm, decision rule
# ---------------------------------------------------------------------------------------------

ItemLoader = Callable[[ResultRow, str, Half], "tuple[np.ndarray, np.ndarray] | None"]


def checkpoint_items(row: ResultRow, task: str, half: Half) -> tuple[np.ndarray, np.ndarray] | None:
    """(base, final) per-item correctness on one half, from the run's checkpoint files, or None
    when they are not on this machine. Positions come from the executor's `half_indices`."""
    if not row.items_path:
        return None
    ckpts = sorted(Path(row.items_path).glob("ckpt_*"))
    if len(ckpts) < 2 or not (ckpts[0] / f"items_{task}.npy").exists():
        return None
    from clharness.executor.rows import half_indices  # numpy-only module

    base = np.load(ckpts[0] / f"items_{task}.npy")
    final = np.load(ckpts[-1] / f"items_{task}.npy")
    idx = half_indices(task, len(base))[0 if half == "selection" else 1]
    return base[idx].astype(float), final[idx].astype(float)


def paired_bootstrap(
    x: Config, r: Config, half: Half, n_boot: int = 10_000, seed: int = 0,
    items: ItemLoader | None = checkpoint_items,
) -> dict[str, Any]:
    """Delta = primary(x) - primary(r) on `half`, paired by seed. Resample paired seeds, then
    items within each task (the same items for both arms). Percentile 95% interval and a
    two-sided bootstrap p-value. PREREG 7."""
    xs, rs = x.by_seed(), r.by_seed()
    pairs = sorted(set(xs) & set(rs))
    if not pairs:
        raise ValueError(f"no shared seeds between {x.label} and {r.label}")
    tasks = sorted(_half(xs[pairs[0]], half).per_task)
    rng = np.random.default_rng(seed)
    k = len(pairs)

    # observed delta: mean over paired seeds of the difference in the 4-task mean
    observed = float(np.mean([_half(xs[s], half).mean_delta - _half(rs[s], half).mean_delta for s in pairs]))

    loaded = None
    if items is not None:
        loaded = {}
        for s in pairs:
            for t in tasks:
                a, b = items(xs[s], t, half), items(rs[s], t, half)
                if a is None or b is None or len(a[0]) != len(b[0]):
                    loaded = None
                    break
                loaded[(s, t)] = (a, b)
            if loaded is None:
                break

    seed_draws = rng.integers(0, k, size=(n_boot, k))
    boot = np.empty(n_boot)
    if loaded is not None:
        mode = "items"
        for b in range(n_boot):
            diffs = []
            for j in seed_draws[b]:
                s = pairs[j]
                per_task = []
                for t in tasks:
                    (bx, fx), (br, fr) = loaded[(s, t)]
                    idx = rng.integers(0, len(bx), size=len(bx))
                    per_task.append((fx[idx].mean() - bx[idx].mean()) - (fr[idx].mean() - br[idx].mean()))
                diffs.append(np.mean(per_task))
            boot[b] = np.mean(diffs)
    else:
        mode = "parametric"
        # per seed, per task: delta and binomial SE of the final score on n items
        dx = np.array([[_half(xs[s], half).per_task[t].delta for t in tasks] for s in pairs])
        dr = np.array([[_half(rs[s], half).per_task[t].delta for t in tasks] for s in pairs])

        def se(cfg_rows: dict[int, ResultRow]) -> np.ndarray:
            out = []
            for s in pairs:
                h = _half(cfg_rows[s], half)
                out.append([math.sqrt(max(h.per_task[t].final * (1 - h.per_task[t].final), 1e-6) / h.per_task[t].n_items)
                            for t in tasks])
            return np.array(out)

        sx, sr = se(xs), se(rs)
        nx = rng.standard_normal((n_boot, k, len(tasks))) * sx[seed_draws]
        nr = rng.standard_normal((n_boot, k, len(tasks))) * sr[seed_draws]
        boot = ((dx[seed_draws] + nx) - (dr[seed_draws] + nr)).mean(axis=2).mean(axis=1)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return {"delta": observed, "lo": float(lo), "hi": float(hi), "p": p, "n_pairs": k,
            "seeds": pairs, "mode": mode, "n_boot": n_boot}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def decide(x: Config, ref: Config, boot: dict[str, Any], p_adj: float, sigma: float | None) -> dict[str, Any]:
    """The three-part rule of PREREG 5. `sigma` is sigma_noise on the same half."""
    acq_x, acq_r = float(x.acquisition().mean()), float(ref.acquisition().mean())
    rule1 = p_adj < ALPHA and (boot["lo"] > 0 or boot["hi"] < 0)
    rule2 = sigma is not None and boot["delta"] > SIZE_GATE_SIGMAS * sigma
    rule3 = acq_x <= ACQ_GATE * acq_r
    enough = boot["n_pairs"] >= MIN_PAIRS
    return {
        "config": x.label, "delta": boot["delta"], "ci": [boot["lo"], boot["hi"]],
        "p": boot["p"], "p_holm": p_adj, "bootstrap_mode": boot["mode"], "n_pairs": boot["n_pairs"],
        "size_threshold": None if sigma is None else SIZE_GATE_SIGMAS * sigma,
        "acquisition": acq_x, "acquisition_limit": ACQ_GATE * acq_r,
        "enough_seeds": enough,
        "rule1_significant": bool(rule1), "rule2_large_enough": bool(rule2), "rule3_learns_enough": bool(rule3),
        # Fewer than MIN_PAIRS paired seeds: not a confirmatory test yet, whatever the numbers say.
        "beats_reference": bool(enough and rule1 and rule2 and rule3 and boot["delta"] > 0),
    }


# ---------------------------------------------------------------------------------------------
# Promotion, frontier, cost
# ---------------------------------------------------------------------------------------------


def promote(configs: dict[str, Config], ref: Config | None, k: int = 3) -> list[dict[str, Any]]:
    """PREREG 6.3: rank non-control configurations by mean selection-half primary outcome,
    among those passing the acquisition gate against the reference arm. Uses the selection half
    only, so the agent may call it."""
    controls = {NO_MITIGATION, REFERENCE}
    limit = ACQ_GATE * float(ref.acquisition().mean()) if ref is not None else math.inf
    ranked = []
    for cfg in configs.values():
        if cfg.techniques in controls:
            continue
        acq = float(cfg.acquisition().mean())
        ranked.append({
            "config": cfg.label, "key": cfg.key, "selection_primary": float(cfg.primary("selection").mean()),
            "n_seeds": len(cfg.rows), "acquisition": acq, "passes_acquisition_gate": acq <= limit,
        })
    ranked.sort(key=lambda d: (not d["passes_acquisition_gate"], -d["selection_primary"]))
    for i, d in enumerate(ranked):
        d["rank"] = i + 1
        d["promoted"] = d["passes_acquisition_gate"] and i < k
    return ranked


def pareto(configs: dict[str, Config], half: Half) -> list[str]:
    """Configurations not dominated on (higher retention, lower acquisition ratio)."""
    pts = {c.label: (float(c.primary(half).mean()), float(c.acquisition().mean())) for c in configs.values()}
    front = []
    for name, (ret, acq) in pts.items():
        dominated = any(r2 >= ret and a2 <= acq and (r2 > ret or a2 < acq)
                        for n2, (r2, a2) in pts.items() if n2 != name)
        if not dominated:
            front.append(name)
    return sorted(front, key=lambda n: -pts[n][0])


def cost_table(configs: dict[str, Config], base: Config | None, half: Half) -> list[dict[str, Any]]:
    """Retention gained over no-mitigation, against what it cost (PREREG 9)."""
    if base is None:
        return []
    base_ret = float(base.primary(half).mean())
    base_h = float(np.mean([r.cost.wall_clock_s for r in base.rows])) / 3600
    out = []
    for cfg in configs.values():
        if cfg is base:
            continue
        gain = float(cfg.primary(half).mean()) - base_ret
        hours = float(np.mean([r.cost.wall_clock_s for r in cfg.rows])) / 3600
        extra = hours - base_h
        fwd = float(np.mean([r.cost.extra_forward_tokens for r in cfg.rows]))
        upd = [r.cost.update_bytes for r in cfg.rows if r.cost.update_bytes is not None]
        out.append({
            "config": cfg.label, "retention_gain": gain, "extra_gpu_hours": extra,
            "gain_per_extra_gpu_hour": gain / extra if extra > 1e-9 else None,
            "extra_forward_tokens": fwd, "update_bytes": int(np.mean(upd)) if upd else None,
            "est_usd": float(np.mean([r.cost.est_usd or 0.0 for r in cfg.rows])),
        })
    # cheaper-or-equal configurations first (no extra cost), then by gain per extra hour
    out.sort(key=lambda d: (d["gain_per_extra_gpu_hour"] is not None,
                            -(d["gain_per_extra_gpu_hour"] if d["gain_per_extra_gpu_hour"] is not None else d["retention_gain"])))
    return out


# ---------------------------------------------------------------------------------------------
# The whole analysis
# ---------------------------------------------------------------------------------------------


def analyze(rows: list[ResultRow], confirmatory: bool = False, n_boot: int = 10_000, seed: int = 0,
            items: ItemLoader | None = checkpoint_items) -> dict[str, Any]:
    """Screening-stage analysis on the selection half; with `confirmatory`, also the
    pre-registered tests on the report half (PREREG 3, 5, 6, 7)."""
    statuses = {s: sum(r.status == s for r in rows) for s in ("finished", "failed", "invalid")}
    finished = [r for r in rows if r.status == "finished"]
    configs = group(finished)
    base, ref = find(configs, NO_MITIGATION), find(configs, REFERENCE)
    sigma_sel = noise_sd(base, "selection") if base else None

    out: dict[str, Any] = {
        "halves_read": ["selection"] + (["report"] if confirmatory else []),
        "rows": {"total": len(rows), **statuses},
        "not_finished": [{"name": r.spec.name, "status": r.status, "reason": r.reason}
                         for r in rows if r.status != "finished"],
        "collapse_flags": {r.spec.name: collapse_flags(r) for r in finished if collapse_flags(r)},
        "noise_floor": {"config": base.label if base else None, "n_seeds": len(base.rows) if base else 0,
                        "sigma_selection": sigma_sel},
        "reference": ref.label if ref else None,
        "effects_retention_selection": fit_effects(
            finished, lambda r: r.retention.selection.mean_delta, sigma=sigma_sel),
        "effects_log_acquisition": fit_effects(
            finished, lambda r: math.log(r.acquisition.mean_ppl_ratio), sigma=None),
        "promotion": promote(configs, ref),
        "pareto_selection": pareto(configs, "selection"),
        "cost_selection": cost_table(configs, base, "selection"),
    }
    if not confirmatory:
        return out

    sigma_rep = noise_sd(base, "report") if base else None
    out["noise_floor"]["sigma_report"] = sigma_rep
    promoted = [d["key"] for d in out["promotion"] if d["promoted"]]
    boots = {configs[k].label: paired_bootstrap(configs[k], ref, "report", n_boot, seed, items) for k in promoted}
    adj = holm({name: b["p"] for name, b in boots.items()})
    out["confirmatory"] = [decide(configs[k], ref, boots[configs[k].label], adj[configs[k].label], sigma_rep)
                           for k in promoted]
    sanity = paired_bootstrap(ref, base, "report", n_boot, seed, items)
    out["sanity_reference_vs_no_mitigation"] = {k: sanity[k] for k in ("delta", "lo", "hi", "p", "n_pairs", "mode")}
    out["pareto_report"] = pareto(configs, "report")
    out["cost_report"] = cost_table(configs, base, "report")
    return out


def check_planted(report: dict[str, Any], planted: dict[str, Any]) -> list[dict[str, Any]]:
    """Fake rows only: is each planted effect inside the fitted 95% interval? (PREREG 7)."""
    checks = []
    for fit_key, plant_key, transform in (
        ("effects_retention_selection", "retention_main_effects", None),
        ("effects_log_acquisition", "acquisition_main_effects", None),
    ):
        eff = report[fit_key]["effects"]
        for name, value in planted[plant_key].items():
            if name in eff:
                e = eff[name]
                checks.append({"fit": fit_key, "effect": name, "planted": value, "estimate": e["estimate"],
                               "lo": e["lo"], "hi": e["hi"], "inside": e["lo"] <= value <= e["hi"]})
    eff = report["effects_retention_selection"]["effects"]
    planted_pairs = {"×".join(sorted(k.split("+"))): v for k, v in planted["retention_interactions"].items()}
    for name, e in eff.items():
        if "×" in name:
            value = planted_pairs.get(name, 0.0)
            checks.append({"fit": "effects_retention_selection", "effect": name, "planted": value,
                           "estimate": e["estimate"], "lo": e["lo"], "hi": e["hi"],
                           "inside": e["lo"] <= value <= e["hi"]})
    return checks


def to_markdown(report: dict[str, Any]) -> str:
    """A short human summary of `analyze`'s output."""
    pct = lambda v: "—" if v is None else f"{100 * v:+.2f}"  # noqa: E731
    pts = lambda v: "—" if v is None else f"{100 * v:.2f}"  # noqa: E731
    lines = ["# Grid analysis\n", f"Halves read: {', '.join(report['halves_read'])}\n",
             f"Rows: {report['rows']}\n"]
    nf = report["noise_floor"]
    lines.append(f"Noise floor: `{nf['config']}` over {nf['n_seeds']} seeds, "
                 f"σ(selection) = {pts(nf['sigma_selection'])} points"
                 + (f", σ(report) = {pts(nf.get('sigma_report'))} points" if "sigma_report" in nf else "") + "\n")
    lines.append("## Retention effects (selection half, points)\n")
    lines.append("| Effect | Estimate | 95% CI |\n|---|---|---|")
    for name, e in report["effects_retention_selection"]["effects"].items():
        lines.append(f"| {name} | {pct(e['estimate'])} | {pct(e['lo'])} to {pct(e['hi'])} |")
    if report["effects_retention_selection"]["dropped"]:
        lines.append(f"\nNot estimable from these rows: {', '.join(report['effects_retention_selection']['dropped'])}\n")
    lines.append("\n## Promotion (selection half)\n")
    lines.append("| Rank | Config | Selection primary | Acq. ratio | Gate | Promoted |\n|---|---|---|---|---|---|")
    for d in report["promotion"]:
        lines.append(f"| {d['rank']} | {d['config']} | {pct(d['selection_primary'])} | {d['acquisition']:.3f} | "
                     f"{'pass' if d['passes_acquisition_gate'] else 'fail'} | {'yes' if d['promoted'] else ''} |")
    if "confirmatory" in report:
        lines.append("\n## Confirmatory tests vs the reference arm (report half)\n")
        lines.append("| Config | Seeds | Δ | 95% CI | p (Holm) | > 2σ | Acq. gate | Beats reference |\n"
                     "|---|---|---|---|---|---|---|---|")
        for d in report["confirmatory"]:
            seeds = f"{d['n_pairs']}" + ("" if d["enough_seeds"] else " (too few)")
            lines.append(f"| {d['config']} | {seeds} | {pct(d['delta'])} | {pct(d['ci'][0])} to {pct(d['ci'][1])} | "
                         f"{d['p_holm']:.3f} | {'yes' if d['rule2_large_enough'] else 'no'} | "
                         f"{'pass' if d['rule3_learns_enough'] else 'fail'} | **{'yes' if d['beats_reference'] else 'no'}** |")
        s = report["sanity_reference_vs_no_mitigation"]
        lines.append(f"\nSanity, reference vs no-mitigation: Δ = {pct(s['delta'])} "
                     f"({pct(s['lo'])} to {pct(s['hi'])}), {s['mode']} bootstrap over {s['n_pairs']} seeds.")
    lines.append(f"\nPareto frontier (selection): {', '.join(report['pareto_selection'])}")
    if report["not_finished"]:
        lines.append("\n## Failed and invalid runs\n")
        lines += [f"- `{d['name']}`: {d['status']}, {d['reason']}" for d in report["not_finished"]]
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / "grid_analysis.json", out_dir / "grid_analysis.md"
    j.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    m.write_text(to_markdown(report))
    return j, m
