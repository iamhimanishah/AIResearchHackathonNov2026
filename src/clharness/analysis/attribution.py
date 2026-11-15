"""Attribution, uncertainty and the pre-registered decision rule, over many result rows.

Every rule here is written down in analysis/PREREG.md; section numbers below point to it.

What it computes
  * technique effects: least squares on 0/1 technique indicators (+ estimable pairwise
    interactions), so a coefficient is "what switching this on changes" (section 7)
  * the frozen noise floor sigma_noise = 1.15 points (0.4, 6.1); the measured SD over the
    no-mitigation seeds is reported for comparison
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
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from clharness.analysis import design as D
from clharness.analysis.design import (
    HALVES,
    alias_chains,
    factorial,
    is_design_spec,
    lr_of,
    selection_of,
)
from clharness.spec import ResultRow, canonical_json

COLLAPSE_SHARE = 0.90  # PREREG 8: flag above this; collapse does not invalidate a run (A9)
ACQ_GATE = 1.05  # PREREG 5, rule 3: acquisition(X) <= 1.05 x acquisition(reference)
SIZE_GATE_SIGMAS = 2.0  # PREREG 5, rule 2: delta > 2 x sigma_noise
MIN_PAIRS = 3  # PREREG 6.3: confirmatory comparisons are over 3 paired seeds
ALPHA = 0.05
REFERENCE = frozenset({"replay", "lr_rewarm"})
NO_MITIGATION: frozenset[str] = frozenset()
Half = str  # "selection" | "report"
# PREREG 0.4/6.1: sigma_noise is FROZEN at 1.15 points (report half, phase-0). It is never
# recomputed from the rows at hand — a stale or duplicated noise file must not move the tests.
SIGMA_NOISE = 1.15 / 100  # mean_delta is a fraction; 1.15 points
SCREEN_SEED = 0  # design.screen(): the screening grid is seed 0; replication is seeds 1-2
REAL_BACKENDS = frozenset({"modal", "tinker"})  # rows from local/fake runs never reach the tables
# two-sided 95% t criticals, df 1..10 (claims read 2-4 paired seeds; bootstrap understates there)
T_CRIT_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


# ---------------------------------------------------------------------------------------------
# Grouping rows into configurations
# ---------------------------------------------------------------------------------------------


def spec_key(spec: Any) -> str:
    """Everything that defines a configuration except the seed and the labels."""
    payload = spec.model_dump(mode="json", exclude={"name", "notes", "parent_spec_hash", "seed"})
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]


def config_key(row: ResultRow) -> str:
    return spec_key(row.spec)


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
        return np.array([acquisition_of(r) for r in self.rows])

    @property
    def selection(self) -> tuple[str, str] | None:
        """(signal, keep) when replay selects by a signal; None for plain replay or no replay."""
        return selection_of(self.rows[0].spec)

    @property
    def lr(self) -> float | None:
        """The learning rate of a lower-LR no-mitigation control (PREREG A13), else None."""
        return lr_of(self.rows[0].spec)


def domain_acquisition(row: ResultRow) -> dict[str, float]:
    """Per trained domain: perplexity ratio at the domain's own boundary (PREREG 4), falling back
    to the end-of-stream ratio for rows written before that field existed."""
    a = row.acquisition
    return dict(a.own_boundary_ppl_ratio) if a.own_boundary_ppl_ratio else dict(a.ppl_ratio)


def acquisition_margins(base: Config | None) -> dict[str, float]:
    """PREREG 5, rule 3: the allowed factor per domain, max(ACQ_GATE, exp(2 x SD)), where SD is
    the spread of log own-boundary perplexity ratio over the no-mitigation noise-floor seeds. So
    the margin is never inside the run-to-run noise. Domains without 2+ seeds get ACQ_GATE."""
    if base is None:
        return {}
    by_domain: dict[str, list[float]] = {}
    for r in base.rows:
        for d, v in domain_acquisition(r).items():
            by_domain.setdefault(d, []).append(math.log(v))
    return {d: max(ACQ_GATE, math.exp(2 * float(np.std(v, ddof=1)))) if len(v) >= 2 else ACQ_GATE
            for d, v in by_domain.items()}


def acquisition_gate(x: Config, ref: Config | None, margins: dict[str, float] | None = None) -> dict[str, Any]:
    """Rule 3, per domain: for every trained domain, mean ratio(x) <= margin_d x mean ratio(ref).
    Every domain must pass (code and legal), so one cannot hide behind the other."""
    if ref is None:
        return {"passes": True, "domains": {}}
    margins = margins or {}

    def means(cfg: Config) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for r in cfg.rows:
            for d, v in domain_acquisition(r).items():
                acc.setdefault(d, []).append(v)
        return {d: float(np.mean(v)) for d, v in acc.items()}

    mx, mr = means(x), means(ref)
    domains = {}
    for d in sorted(set(mx) | set(mr)):
        if d not in mx or d not in mr:
            # a domain the run never measured cannot pass the gate; an absent measurement is not
            # evidence the run learned it
            domains[d] = {"ratio": mx.get(d), "reference": mr.get(d), "limit": None, "passes": False}
            continue
        limit = margins.get(d, ACQ_GATE) * mr[d]
        domains[d] = {"ratio": mx[d], "reference": mr[d], "limit": limit, "passes": mx[d] <= limit}
    return {"passes": all(v["passes"] for v in domains.values()) and bool(domains),
            "domains": domains}


def acquisition_of(row: ResultRow) -> float:
    """PREREG 4: acquisition is the perplexity ratio at each domain's own boundary (what was
    learned before later domains could erase it). Rows written before that field existed fall
    back to the final ratio; `analyze` reports how many rows did."""
    a = row.acquisition
    return a.mean_own_boundary_ppl_ratio if a.mean_own_boundary_ppl_ratio is not None else a.mean_ppl_ratio


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
    names: dict[tuple[frozenset[str], tuple[str, str] | None, float | None], list[Config]] = {}
    for cfg in configs.values():
        names.setdefault((cfg.techniques, cfg.selection, cfg.lr), []).append(cfg)
    for (techs, sel, lr), same in names.items():
        base = "+".join(sorted(techs)) or "no-mitigation"
        if sel:
            base += f"+select-{sel[0]}-{sel[1]}"
        if lr is not None:
            base += f"-lr{lr:g}"
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
    """The configuration with exactly these techniques and plain (random) replay. With several
    (different parameters), the one with the most seeds, which is the replicated one."""
    hits = [c for c in configs.values() if c.techniques == techniques and c.selection is None and c.lr is None]
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
    interactions: Iterable[list[tuple[str, str]]] = (),
) -> dict[str, Any]:
    """Least squares of `outcome` on an intercept, one 0/1 column per technique, and one column
    per entry of `interactions`. An entry is a chain of pairs whose columns coincide in the
    design: the first pair names the column and the rest are reported as `aliased_with`, so the
    estimate is read as the sum of the chain. Columns that would make the design rank-deficient
    are dropped and listed. Standard errors use the residual variance when there are spare
    degrees of freedom, otherwise the known `sigma` (PREREG 7)."""
    techs = sorted({t for r in rows for t in r.spec.technique_names()})
    cols: list[tuple[str, frozenset[str]]] = [(t, frozenset({t})) for t in techs]
    chain_of: dict[str, list[str]] = {}
    for chain in interactions:
        name = "×".join(sorted(chain[0]))
        cols.append((name, frozenset(chain[0])))
        chain_of[name] = ["×".join(sorted(p)) for p in chain[1:]]
    if not rows:
        return {"effects": {}, "dropped": [n for n, _ in cols], "aliases": {}, "n_rows": 0,
                "df": 0, "residual_sd": None,
                "se_source": "sigma_noise" if sigma is not None else "residual"}
    present = [frozenset(r.spec.technique_names()) for r in rows]
    y = np.array([outcome(r) for r in rows])

    def column(members: frozenset[str]) -> np.ndarray:
        return np.array([1.0 if members <= p else 0.0 for p in present])

    X = np.ones((len(rows), 1))
    kept, dropped = ["(baseline)"], []
    vectors: dict[str, np.ndarray] = {}
    aliases: dict[str, list[str]] = {}
    for name, members in cols:  # greedy: main effects first, then pairs
        vec = column(members)
        trial = np.column_stack([X, vec])
        if np.linalg.matrix_rank(trial) > X.shape[1]:
            X, kept = trial, kept + [name]
            vectors[name] = vec
        else:
            dropped.append(name)
            # identical column on these rows: the kept estimate is the sum of both effects
            twin = next((k for k, v in vectors.items() if np.array_equal(v, vec)), None)
            if twin is not None:
                aliases.setdefault(twin, []).append(name)
    for name, partners in chain_of.items():
        if name in vectors:
            aliases.setdefault(name, []).extend(partners)

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
        name: {"estimate": float(b), "se": float(s), "lo": float(b - z * s), "hi": float(b + z * s),
               "aliased_with": aliases.get(name, [])}
        for name, b, s in zip(kept, beta, se)
    }
    return {
        "effects": effects, "dropped": dropped, "aliases": aliases, "n_rows": len(rows), "df": df,
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


def decide(x: Config, ref: Config, boot: dict[str, Any], p_adj: float, sigma: float | None,
           lr_control: Config | None = None, lr_boot: dict[str, Any] | None = None,
           lr_p_adj: float | None = None, margins: dict[str, float] | None = None,
           lr_missing: bool = False) -> dict[str, Any]:
    """The rule of PREREG 5. `sigma` is sigma_noise on the same half. Rules 1-3 compare x with
    the reference arm; rule 4 (A13) asks the same significance and size of x against the chosen
    lower-LR control, and holds trivially only when lower-LR arms exist but none passes the
    acquisition gate. `lr_missing` says no lower-LR rows exist at all: that is missing evidence,
    not a trivial hold, so rule 4 fails."""
    acq_x, acq_r = float(x.acquisition().mean()), float(ref.acquisition().mean())
    rule1 = p_adj < ALPHA and (boot["lo"] > 0 or boot["hi"] < 0)
    rule2 = sigma is not None and boot["delta"] > SIZE_GATE_SIGMAS * sigma
    gate = acquisition_gate(x, ref, margins)
    rule3 = gate["passes"]
    enough = boot["n_pairs"] >= MIN_PAIRS
    if lr_missing:
        rule4, lr_info = False, None
    elif lr_control is None or lr_boot is None:
        rule4, lr_info = True, None
    else:
        rule4 = (lr_p_adj is not None and lr_p_adj < ALPHA and lr_boot["lo"] > 0
                 and sigma is not None and lr_boot["delta"] > SIZE_GATE_SIGMAS * sigma
                 and lr_boot["n_pairs"] >= MIN_PAIRS)
        lr_info = {"config": lr_control.label, "delta": lr_boot["delta"], "ci": [lr_boot["lo"], lr_boot["hi"]],
                   "p": lr_boot["p"], "p_holm": lr_p_adj, "n_pairs": lr_boot["n_pairs"]}
    return {
        "config": x.label, "delta": boot["delta"], "ci": [boot["lo"], boot["hi"]],
        "p": boot["p"], "p_holm": p_adj, "bootstrap_mode": boot["mode"], "n_pairs": boot["n_pairs"],
        "size_threshold": None if sigma is None else SIZE_GATE_SIGMAS * sigma,
        "acquisition": acq_x, "acquisition_reference": acq_r, "acquisition_gate": gate["domains"],
        "enough_seeds": enough,
        "rule1_significant": bool(rule1), "rule2_large_enough": bool(rule2), "rule3_learns_enough": bool(rule3),
        "rule4_beats_lr_control": bool(rule4), "vs_lr_control": lr_info,
        # Fewer than MIN_PAIRS paired seeds: not a confirmatory test yet, whatever the numbers say.
        "beats_reference": bool(enough and rule1 and rule2 and rule3 and rule4 and boot["delta"] > 0),
    }


# ---------------------------------------------------------------------------------------------
# Promotion, frontier, cost
# ---------------------------------------------------------------------------------------------


def promote(configs: dict[str, Config], ref: Config | None, k: int = 3,
            margins: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """PREREG 6.3: rank non-control configurations by mean selection-half primary outcome,
    among those passing the acquisition gate against the reference arm. Uses the selection half
    only, so the agent may call it."""
    excluded = {NO_MITIGATION, REFERENCE, *HALVES}  # controls, and the halves of the reference recipe
    ranked = []
    for cfg in configs.values():
        if (cfg.techniques in excluded or cfg.selection is not None or not is_design_spec(cfg.rows[0].spec)
                or cfg.techniques not in set(factorial())):
            continue
        acq = float(cfg.acquisition().mean())
        ranked.append({
            "config": cfg.label, "key": cfg.key, "selection_primary": float(cfg.primary("selection").mean()),
            "n_seeds": len(cfg.rows), "acquisition": acq,
            "passes_acquisition_gate": acquisition_gate(cfg, ref, margins)["passes"],
        })
    ranked.sort(key=lambda d: (not d["passes_acquisition_gate"], -d["selection_primary"]))
    for i, d in enumerate(ranked):
        d["rank"] = i + 1
        d["promoted"] = d["passes_acquisition_gate"] and i < k
    return ranked


def choose_lr_control(configs: dict[str, Config], ref: Config | None,
                      margins: dict[str, float] | None = None) -> dict[str, Any]:
    """PREREG A13: the lower-LR no-mitigation arm with the best selection-half outcome among
    those whose acquisition passes the 5% gate against the reference arm. Selection half only."""
    arms = []
    for cfg in configs.values():
        if cfg.lr is None:
            continue
        acq = float(cfg.acquisition().mean())
        arms.append({"config": cfg.label, "key": cfg.key, "lr": cfg.lr, "n_seeds": len(cfg.rows),
                     "selection_primary": float(cfg.primary("selection").mean()),
                     "acquisition": acq, "passes_acquisition_gate": acquisition_gate(cfg, ref, margins)["passes"]})
    eligible = [a for a in arms if a["passes_acquisition_gate"]]
    chosen = max(eligible, key=lambda a: a["selection_primary"], default=None)
    return {"arms": sorted(arms, key=lambda a: -a["lr"]), "chosen": chosen["key"] if chosen else None,
            "chosen_label": chosen["config"] if chosen else None}


def promote_selection(configs: dict[str, Config], ref: Config | None,
                      margins: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """PREREG 6.3b: for each selection signal, the `keep` direction with the better mean
    selection-half outcome is promoted, if it passes the acquisition gate. Selection half only."""
    ref_sel = float(ref.primary("selection").mean()) if ref is not None else float("nan")
    rows = []
    for cfg in configs.values():
        if cfg.selection is None or (ref is not None and cfg.techniques != ref.techniques):
            continue
        acq = float(cfg.acquisition().mean())
        sel = float(cfg.primary("selection").mean())
        rows.append({"config": cfg.label, "key": cfg.key, "signal": cfg.selection[0], "keep": cfg.selection[1],
                     "selection_primary": sel, "vs_reference": sel - ref_sel, "n_seeds": len(cfg.rows),
                     "acquisition": acq, "passes_acquisition_gate": acquisition_gate(cfg, ref, margins)["passes"]})
    for signal in sorted({r["signal"] for r in rows}):
        options = [r for r in rows if r["signal"] == signal and r["passes_acquisition_gate"]]
        best = max(options, key=lambda r: r["selection_primary"], default=None)
        for r in rows:
            if r["signal"] == signal:
                r["promoted"] = r is best
    return sorted(rows, key=lambda r: (r["signal"], r["keep"]))


def paired_seed_bootstrap(x: dict[int, float], r: dict[int, float], n_boot: int = 10_000,
                          seed: int = 0) -> dict[str, Any]:
    """Delta = mean(x - r) over paired seeds, with a percentile interval from resampling seeds.
    Used for per-domain acquisition, which has no item level."""
    pairs = sorted(set(x) & set(r))
    if not pairs:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_pairs": 0}
    d = np.array([x[s] - r[s] for s in pairs])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n_pairs": len(pairs)}


def paired_seed_t(x: dict[int, float], r: dict[int, float]) -> dict[str, Any]:
    """Paired t interval on the seed differences (PREREG 6.5 acquisition claims). On 2-4 paired
    seeds a bootstrap percentile is far too narrow; the t interval is the honest small-sample
    one. NOTE: this is the interval PREREG 6.5's amendment entry names — it differs from the
    6.5 text written at the freeze, so the amendment log must record it before claims are read."""
    pairs = sorted(set(x) & set(r))
    if not pairs:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_pairs": 0}
    d = np.array([x[s] - r[s] for s in pairs])
    m = float(d.mean())
    if len(d) < 2:
        return {"delta": m, "lo": m, "hi": m, "n_pairs": len(d)}
    half = T_CRIT_975.get(len(d) - 1, 1.96) * float(d.std(ddof=1)) / math.sqrt(len(d))
    return {"delta": m, "lo": m - half, "hi": m + half, "n_pairs": len(d)}


def claims_table(configs: dict[str, Config], half: Half, lr_control: Config | None = None,
                 n_boot: int = 10_000, seed: int = 0, items: ItemLoader | None = None) -> list[dict[str, Any]]:
    """PREREG 6.5, exploratory: each published claim's tests, arm against comparator.
    A directional test reproduces if its interval excludes 0 in the published direction."""
    def cfg_of(name: str) -> Config | None:
        return configs.get(spec_key(D.claim_spec(name, 0)))

    def log_acq(cfg: Config, domain: str) -> dict[int, float]:
        return {r.spec.seed: math.log(domain_acquisition(r)[domain]) for r in cfg.rows
                if domain in domain_acquisition(r)}

    out = []
    for c in D.CLAIMS:
        for arm, vs, outcome, sign, kind in c["tests"]:
            x, r = cfg_of(arm), cfg_of(vs)
            row: dict[str, Any] = {"id": c["id"], "claim": c["claim"], "source": c["source"], "arm": arm,
                                   "comparator": vs, "outcome": outcome, "published_sign": sign, "kind": kind}
            if x is None or r is None:
                out.append({**row, "verdict": "missing runs"})
                continue
            if outcome == "retention":
                if not set(x.seeds) & set(r.seeds):
                    out.append({**row, "verdict": "missing runs"})
                    continue
                b = paired_bootstrap(x, r, half, n_boot, seed, items)
            else:
                b = paired_seed_t(log_acq(x, outcome[4:]), log_acq(r, outcome[4:]))
            row.update(delta=b["delta"], lo=b["lo"], hi=b["hi"], n_pairs=b["n_pairs"])
            if kind == "magnitude":
                row["verdict"] = "reported"
            elif b["n_pairs"] < 2:
                row["verdict"] = "too few seeds"
            elif (sign > 0 and b["lo"] > 0) or (sign < 0 and b["hi"] < 0):
                row["verdict"] = "reproduced"
            elif (sign > 0 and b["hi"] < 0) or (sign < 0 and b["lo"] > 0):
                row["verdict"] = "opposite"
            else:
                row["verdict"] = "not reproduced (interval includes 0)"
            # Justin's caution: say whether the arm also beats simply lowering the learning rate
            if outcome == "retention" and lr_control is not None and x is not lr_control and x.techniques:
                lb = paired_bootstrap(x, lr_control, half, n_boot, seed, items)
                row["vs_lr_control"] = {"config": lr_control.label, "delta": lb["delta"], "lo": lb["lo"], "hi": lb["hi"]}
            out.append(row)
    return out


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
            # the ranking metric (PREREG 9): extra forward tokens are a measured quantity; a
            # wall-clock difference of noisy short runs is not
            "gain_per_extra_forward_token": gain / fwd if fwd > 0 else None,
            "extra_forward_tokens": fwd, "update_bytes": int(np.mean(upd)) if upd else None,
            "est_usd": float(np.mean([r.cost.est_usd or 0.0 for r in cfg.rows])),
        })
    # cheaper-or-equal configurations first (no extra cost), then by gain per extra forward token
    out.sort(key=lambda d: (d["gain_per_extra_forward_token"] is not None,
                            -(d["gain_per_extra_forward_token"]
                              if d["gain_per_extra_forward_token"] is not None else d["retention_gain"])))
    return out


# ---------------------------------------------------------------------------------------------
# The whole analysis
# ---------------------------------------------------------------------------------------------


def _claim_keys() -> set[str]:
    return {spec_key(x) for x in D.claims_specs()}


def _row_rank(r: ResultRow) -> tuple[bool, bool]:
    """Tiebreak between two rows of the same spec: prefer a finished row, then one measured the
    current way (own-boundary acquisition, pinned item split — the remeasured files), then the
    later row (a rerun rewrites the same run)."""
    a = r.acquisition
    return (r.status == "finished", a is not None and a.mean_own_boundary_ppl_ratio is not None)


def dedupe_rows(rows: list[ResultRow]) -> tuple[list[ResultRow], int]:
    """One row per spec hash: loading an old-split noise file next to the remeasured one must not
    double a configuration's seeds. Returns the deduplicated rows in input order and the number
    dropped."""
    seen: dict[str, int] = {}
    out: list[ResultRow] = []
    dropped = 0
    for r in rows:
        i = seen.get(r.spec_hash)
        if i is None:
            seen[r.spec_hash] = len(out)
            out.append(r)
            continue
        dropped += 1
        if _row_rank(r) >= _row_rank(out[i]):
            out[i] = r
    return out, dropped


def analyze(rows: list[ResultRow], confirmatory: bool = False, n_boot: int = 10_000, seed: int = 0,
            items: ItemLoader | None = checkpoint_items, allow_simulated: bool = False) -> dict[str, Any]:
    """Screening-stage analysis on the selection half; with `confirmatory`, also the
    pre-registered tests on the report half (PREREG 3, 5, 6, 7). Finished rows from non-real
    backends (local, fake) are excluded unless `allow_simulated` — a simulator's rows must not
    reach the reported tables. Decisions are re-derived only from the SCREEN_SEED rows: once
    replication rows exist, re-ranking on all seeds would test a different family than the one
    the screen promoted."""
    rows, n_dup = dedupe_rows(rows)
    finished = [r for r in rows if r.status == "finished"]
    simulated = sorted(r.spec.name for r in finished if r.backend not in REAL_BACKENDS)
    if not allow_simulated:
        finished = [r for r in finished if r.backend in REAL_BACKENDS]
    # "finished" counts the rows that are actually used; excluded simulated rows are counted apart
    statuses = {"finished": len(finished),
                **{s: sum(r.status == s for r in rows) for s in ("failed", "invalid")}}
    # selection variants are their own family (PREREG 6.3b), not a seventh factor of the fit
    design_rows = [r for r in finished
                   if selection_of(r.spec) is None and lr_of(r.spec) is None and is_design_spec(r.spec)]
    in_factorial = set(factorial())
    factorial_rows = [r for r in design_rows if frozenset(r.spec.technique_names()) in in_factorial]
    configs = group(finished)
    base, ref = find(configs, NO_MITIGATION), find(configs, REFERENCE)

    # every screen-time decision (family, selection variants, lower-LR control, margins) is
    # computed on the seed-0 rows only — the rows the screen actually produced (PREREG 6.3)
    screen_configs = group([r for r in finished if r.spec.seed == SCREEN_SEED])
    screen_base, screen_ref = find(screen_configs, NO_MITIGATION), find(screen_configs, REFERENCE)
    margins = acquisition_margins(screen_base)

    # the alias chain that replay x lr_rewarm sits in: each partner pair gets its own column —
    # the half runs break the aliasing, so the main estimate is the pair's own effect (PREREG 7)
    rw_pair = ("lr_rewarm", "replay")
    rw_chain = next((c for c in alias_chains() if rw_pair in c), [rw_pair])

    out: dict[str, Any] = {
        "halves_read": ["selection"] + (["report"] if confirmatory else []),
        "rows": {"total": len(rows), **statuses, "duplicates_dropped": n_dup,
                 "excluded_backend": len(simulated)},
        "excluded_backend_rows": simulated,
        "not_finished": [{"name": r.spec.name, "status": r.status, "reason": r.reason}
                         for r in rows if r.status != "finished"],
        "collapse_flags": {r.spec.name: collapse_flags(r) for r in finished if collapse_flags(r)},
        "noise_floor": {"config": base.label if base else None, "n_seeds": len(base.rows) if base else 0,
                        "sigma_noise": SIGMA_NOISE,
                        "sigma_selection_measured": noise_sd(base, "selection") if base else None},
        "reference": ref.label if ref else None,
        "off_design_rows": sorted(r.spec.name for r in finished
                                  if not is_design_spec(r.spec) and spec_key(r.spec) not in _claim_keys()),
        # PREREG 7: six main effects plus replay x lr_rewarm and its alias partners, which the
        # reference halves make separately estimable; sigma is the frozen phase-0 value
        "effects_retention_selection": fit_effects(
            design_rows, lambda r: r.retention.selection.mean_delta, sigma=SIGMA_NOISE,
            interactions=[[p] for p in rw_chain]),
        # exploratory: one contrast per alias chain of the factorial (PREREG 6.2), factorial runs only
        "alias_chains_retention_selection": fit_effects(
            factorial_rows, lambda r: r.retention.selection.mean_delta, sigma=SIGMA_NOISE,
            interactions=alias_chains()),
        "acquisition_measure": {
            "own_boundary": sum(r.acquisition.mean_own_boundary_ppl_ratio is not None for r in finished),
            "final_fallback": sum(r.acquisition.mean_own_boundary_ppl_ratio is None for r in finished),
        },
        "effects_log_acquisition": fit_effects(
            design_rows, lambda r: math.log(acquisition_of(r)), sigma=None),
        "acquisition_margins": margins,
        "promotion": promote(screen_configs, screen_ref, margins=margins),
        "selection_screen": promote_selection(screen_configs, screen_ref, margins),
        "lr_control": choose_lr_control(screen_configs, screen_ref, margins),
        "pareto_selection": pareto(configs, "selection"),
        "cost_selection": cost_table(configs, base, "selection"),
    }
    if not confirmatory:
        return out

    out["noise_floor"]["sigma_report_measured"] = noise_sd(base, "report") if base else None
    promoted = [d["key"] for d in out["promotion"] if d["promoted"]]
    boots = {configs[k].label: paired_bootstrap(configs[k], ref, "report", n_boot, seed, items) for k in promoted}
    adj = holm({name: b["p"] for name, b in boots.items()})
    lr_ctrl = configs.get(out["lr_control"]["chosen"]) if out["lr_control"]["chosen"] else None
    lr_missing = not out["lr_control"]["arms"]
    lr_boots = {configs[k].label: paired_bootstrap(configs[k], lr_ctrl, "report", n_boot, seed, items)
                for k in promoted} if lr_ctrl is not None else {}
    lr_adj = holm({name: b["p"] for name, b in lr_boots.items()}) if lr_boots else {}
    out["confirmatory"] = [
        decide(configs[k], ref, boots[configs[k].label], adj[configs[k].label], SIGMA_NOISE,
               lr_ctrl, lr_boots.get(configs[k].label), lr_adj.get(configs[k].label), margins,
               lr_missing=lr_missing)
        for k in promoted]
    sel_promoted = [d["key"] for d in out["selection_screen"] if d.get("promoted")]
    sel_boots = {configs[k].label: paired_bootstrap(configs[k], ref, "report", n_boot, seed, items)
                 for k in sel_promoted}
    sel_adj = holm({name: b["p"] for name, b in sel_boots.items()})
    out["confirmatory_selection"] = [
        decide(configs[k], ref, sel_boots[configs[k].label], sel_adj[configs[k].label], SIGMA_NOISE,
               margins=margins)
        for k in sel_promoted]
    out["claims"] = claims_table(configs, "report", lr_ctrl, n_boot, seed, items)
    out["claims_elsewhere"] = D.CLAIMS_ELSEWHERE
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
    planted_pairs = {"×".join(sorted(k.split("+"))): v for k, v in planted["retention_interactions"].items()}
    for fit_key in ("effects_retention_selection", "alias_chains_retention_selection"):
        for name, e in report[fit_key]["effects"].items():
            if "×" not in name:
                continue
            # an aliased estimate is the sum of every effect in its chain
            value = sum(planted_pairs.get(n, 0.0) for n in [name, *e.get("aliased_with", [])])
            checks.append({"fit": fit_key, "effect": name, "planted": value,
                           "estimate": e["estimate"], "lo": e["lo"], "hi": e["hi"],
                           "inside": e["lo"] <= value <= e["hi"]})
    return checks


def check_planted_selection(rows: list[ResultRow], planted: dict[str, Any], n_boot: int = 4000,
                            seed: int = 0) -> list[dict[str, Any]]:
    """Fake rows only: each replicated selection variant against the reference arm, on the
    selection half, compared with its planted effect."""
    configs = group(rows)
    ref = find(configs, REFERENCE)
    checks = []
    for cfg in configs.values():
        if cfg.selection is None or len(set(cfg.seeds) & set(ref.seeds)) < 2:
            continue
        b = paired_bootstrap(cfg, ref, "selection", n_boot, seed, items=None)
        value = planted["retention_selection_effects"][f"{cfg.selection[0]}/{cfg.selection[1]}"]
        checks.append({"fit": "selection_vs_reference", "effect": cfg.label, "planted": value,
                       "estimate": b["delta"], "lo": b["lo"], "hi": b["hi"], "inside": b["lo"] <= value <= b["hi"]})
    return checks


def to_markdown(report: dict[str, Any]) -> str:
    """A short human summary of `analyze`'s output."""
    pct = lambda v: "—" if v is None else f"{100 * v:+.2f}"
    pts = lambda v: "—" if v is None else f"{100 * v:.2f}"
    lines = ["# Grid analysis\n", f"Halves read: {', '.join(report['halves_read'])}\n",
             f"Rows: {report['rows']}\n"]
    nf = report["noise_floor"]
    lines.append(f"Noise floor: `{nf['config']}` over {nf['n_seeds']} seeds, "
                 f"σ_noise = {pts(nf['sigma_noise'])} points (frozen), "
                 f"measured σ(selection) = {pts(nf['sigma_selection_measured'])} points"
                 + (f", measured σ(report) = {pts(nf.get('sigma_report_measured'))} points"
                    if "sigma_report_measured" in nf else "") + "\n")
    lines.append("## Retention effects (selection half, points)\n")
    lines.append("| Effect | Estimate | 95% CI |\n|---|---|---|")
    for name, e in report["effects_retention_selection"]["effects"].items():
        alias = f" (= {' = '.join(e['aliased_with'])})" if e.get("aliased_with") else ""
        lines.append(f"| {name}{alias} | {pct(e['estimate'])} | {pct(e['lo'])} to {pct(e['hi'])} |")
    if report["effects_retention_selection"]["dropped"]:
        lines.append(f"\nNot estimable from these rows: {', '.join(report['effects_retention_selection']['dropped'])}\n")
    chains = {k: v for k, v in report["alias_chains_retention_selection"]["effects"].items() if "×" in k}
    if chains:
        lines.append("\n### Interaction chains (exploratory; each estimate is the sum of its chain)\n")
        lines.append("| Chain | Estimate | 95% CI |\n|---|---|---|")
        for name, e in chains.items():
            lines.append(f"| {' = '.join([name, *e['aliased_with']])} | {pct(e['estimate'])} | {pct(e['lo'])} to {pct(e['hi'])} |")
    if report.get("off_design_rows"):
        lines.append(f"\nOff-design rows, not in the fits or promotion: {', '.join(report['off_design_rows'])}\n")
    lines.append("\n## Promotion (selection half)\n")
    lines.append("| Rank | Config | Selection primary | Acq. ratio | Gate | Promoted |\n|---|---|---|---|---|---|")
    for d in report["promotion"]:
        lines.append(f"| {d['rank']} | {d['config']} | {pct(d['selection_primary'])} | {d['acquisition']:.3f} | "
                     f"{'pass' if d['passes_acquisition_gate'] else 'fail'} | {'yes' if d['promoted'] else ''} |")
    if report.get("selection_screen"):
        lines.append("\n## Selection on replay (selection half, against the reference arm)\n")
        lines.append("| Config | Δ vs reference | Seeds | Gate | Promoted |\n|---|---|---|---|---|")
        for d in report["selection_screen"]:
            lines.append(f"| {d['config']} | {pct(d['vs_reference'])} | {d['n_seeds']} | "
                         f"{'pass' if d['passes_acquisition_gate'] else 'fail'} | {'yes' if d.get('promoted') else ''} |")
    if "confirmatory" in report:
        lines.append("\n## Confirmatory tests vs the reference arm (report half)\n")
        lc = report.get("lr_control", {})
        lines.append(f"Lower-LR control (A13): `{lc.get('chosen_label') or 'none passes the acquisition gate'}`\n")
        lines.append("| Config | Seeds | Δ | 95% CI | p (Holm) | > 2σ | Acq. gate | Δ vs LR control | Beats LR control | Wins |\n"
                     "|---|---|---|---|---|---|---|---|---|---|")
        for d in report["confirmatory"]:
            seeds = f"{d['n_pairs']}" + ("" if d["enough_seeds"] else " (too few)")
            vs = d.get("vs_lr_control")
            lr_cell = f"{pct(vs['delta'])} ({pct(vs['ci'][0])} to {pct(vs['ci'][1])})" if vs else "n/a"
            lines.append(f"| {d['config']} | {seeds} | {pct(d['delta'])} | {pct(d['ci'][0])} to {pct(d['ci'][1])} | "
                         f"{d['p_holm']:.3f} | {'yes' if d['rule2_large_enough'] else 'no'} | "
                         f"{'pass' if d['rule3_learns_enough'] else 'fail'} | {lr_cell} | "
                         f"{'yes' if d['rule4_beats_lr_control'] else 'no'} | **{'yes' if d['beats_reference'] else 'no'}** |")
        if report.get("confirmatory_selection"):
            lines.append("\n### Selection family (Holm across signals)\n")
            lines.append("| Config | Seeds | Δ | 95% CI | p (Holm) | > 2σ | Beats reference |\n|---|---|---|---|---|---|---|")
            for d in report["confirmatory_selection"]:
                seeds = f"{d['n_pairs']}" + ("" if d["enough_seeds"] else " (too few)")
                lines.append(f"| {d['config']} | {seeds} | {pct(d['delta'])} | {pct(d['ci'][0])} to {pct(d['ci'][1])} | "
                             f"{d['p_holm']:.3f} | {'yes' if d['rule2_large_enough'] else 'no'} | "
                             f"**{'yes' if d['beats_reference'] else 'no'}** |")
        s = report["sanity_reference_vs_no_mitigation"]
        lines.append(f"\nSanity, reference vs no-mitigation: Δ = {pct(s['delta'])} "
                     f"({pct(s['lo'])} to {pct(s['hi'])}), {s['mode']} bootstrap over {s['n_pairs']} seeds.")
    if report.get("claims"):
        lines.append("\n## Published claims (exploratory, report half)\n")
        lines.append("| # | Claim | Arm vs comparator | Outcome | Δ (95% CI) | Seeds | Verdict | vs lower-LR control |\n"
                     "|---|---|---|---|---|---|---|---|")
        for c in report["claims"]:
            if "delta" not in c:
                lines.append(f"| {c['id']} | {c['claim']} | {c['arm']} vs {c['comparator']} | {c['outcome']} | — | — | {c['verdict']} | |")
                continue
            fmt = pct if c["outcome"] == "retention" else (lambda v: f"{v:+.3f}")
            vs = c.get("vs_lr_control")
            vs_cell = f"{pct(vs['delta'])} ({pct(vs['lo'])} to {pct(vs['hi'])})" if vs else ""
            lines.append(f"| {c['id']} | {c['claim']} | {c['arm']} vs {c['comparator']} | {c['outcome']} | "
                         f"{fmt(c['delta'])} ({fmt(c['lo'])} to {fmt(c['hi'])}) | {c['n_pairs']} | {c['verdict']} | {vs_cell} |")
        for k, v in report.get("claims_elsewhere", {}).items():
            lines.append(f"\nClaim {k}: {v}")
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
