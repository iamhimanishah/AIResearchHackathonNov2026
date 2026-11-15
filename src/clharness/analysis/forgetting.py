"""Why questions are forgotten: structure in WHICH questions a run loses, WHEN and HOW.

Exploratory (added on the Sunday of the hackathon, after the pre-registration). The fragile
core (manifests/eval-protected-v1-core.json, analysis/item_profile.py) shows that the plain run
forgets a small set of questions in almost every seed. This module asks the follow-up questions
from the per-item files every run already has, selection half only:

    repeatability   Within an arm, does the number of OTHER seeds that lost a question predict
                    whether a held-out seed loses it? AUC 0.5 = losses are random over the
                    questions the base model knows; 1.0 = every seed loses the same ones. No
                    threshold and no core definition, so it is not circular.
    dose vs target  Does an arm protect the fragile questions specifically, or just forget less
                    of everything? Each question's fragility is its loss rate in all OTHER arms
                    (leave-arm-out, logit), and the arm's losses are fitted on it. The slope is
                    compared with the slope the same estimator gives on data simulated from an
                    additive model in which every arm only sets a dose. Below that null band =
                    targeted, above = concentrating. Raw "core forgotten" mixes dose and target.
    timing          Of the questions lost by the end, the share already lost after the first
                    domain (code), and the share lost after code that came back during legal.
    answer drift    When two seeds of an arm lose the same question, do they move to the same
                    wrong option? Compared with the 1/(k-1) chance of agreeing at random. High
                    agreement means the question is pulled toward a specific wrong answer, not
                    just knocked off the right one.
    margins         (new runs only, from loglik_<task>.npz) the base model's margin, gold score
                    minus best wrong score, of questions later lost vs kept.
    state drift     (bench fast-mech only, from drift_<task>.npz) per-layer cosine between the
                    question's final state and the base model's, lost vs kept.

The report half is never indexed: every array is cut to `half_indices(task, n)[0]` on load.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from clharness.analysis.item_profile import _checkpoint_dirs
from clharness.config import REPO_ROOT
from clharness.executor.rows import half_indices

TASKS = ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal")
FROZEN_STREAM = "code:10000000+legal:10000000"


@dataclass
class RunItems:
    """One run's per-item outcomes on the selection half, all tasks concatenated in TASKS order."""

    arm: str
    seed: int
    correct: np.ndarray  # [n_ckpt, n] bool, checkpoint 0 = base
    chosen: np.ndarray | None  # [n_ckpt, n] int, -1 unknown
    gold: np.ndarray | None  # [n] int
    n_options: np.ndarray  # [n] int, 0 unknown
    margin_base: np.ndarray | None = None  # [n] gold minus best wrong score at base (loglik files)
    drift: dict[int, np.ndarray] = field(default_factory=dict)  # layer -> [n] cosine at the last ckpt

    @property
    def base_ok(self) -> np.ndarray:
        return self.correct[0]

    @property
    def lost(self) -> np.ndarray:
        return self.correct[0] & ~self.correct[-1]


def _arm_name(spec: Any) -> str:
    lr = spec.optim.get("lr")
    name = "+".join(spec.technique_names()) or "no-mitigation"
    return name + (f"@lr{lr:g}" if lr is not None else "")


def _arm_key(spec: Any) -> str:
    """Spec minus seed and name: the same grouping as item_profile.arm_profiles."""
    return json.dumps([spec.bench, spec.optim, [t.model_dump(mode="json") for t in spec.techniques]], sort_keys=True)


def _stream(spec: Any) -> str:
    return "+".join(f"{d.name}:{d.tokens}" for d in spec.stream)


def _margin(score: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Gold score minus the best other score, per item; NaN where unknown."""
    out = np.full(len(score), np.nan, dtype=np.float64)
    for i, (s, g) in enumerate(zip(score, gold, strict=True)):
        if g < 0 or g >= len(s) or np.isnan(s[g]):
            continue
        others = np.delete(s, g)
        others = others[~np.isnan(others)]
        if others.size:
            out[i] = float(s[g] - others.max())
    return out


def load_run(row: Any, repo_root: Path = REPO_ROOT, tasks: Sequence[str] = TASKS) -> RunItems | None:
    """A row's selection-half outcomes, or None when its item files are not on this machine."""
    ckpts = _checkpoint_dirs(row, repo_root)
    if not ckpts:
        return None
    correct, chosen, gold, n_opt, margin = [], [], [], [], []
    drift: dict[int, list[np.ndarray]] = {}
    have_choices = have_margin = True
    for task in tasks:
        paths = [c / f"items_{task}.npy" for c in ckpts]
        if not all(p.exists() for p in paths):
            return None
        full = [np.load(p) for p in paths]
        sel = half_indices(task, len(full[0]))[0]  # the report half is never indexed
        correct.append(np.stack([x[sel] > 0.5 for x in full]))
        ch = [c / f"choices_{task}.npy" for c in ckpts]
        gd = ckpts[0] / f"gold_{task}.npy"
        if have_choices and all(p.exists() for p in ch) and gd.exists():
            chosen.append(np.stack([np.load(p)[sel].astype(int) for p in ch]))
            gold.append(np.load(gd)[sel].astype(int))
        else:
            have_choices = False
        ll = ckpts[0] / f"loglik_{task}.npz"
        if have_margin and ll.exists() and gd.exists():
            z = np.load(ll)
            margin.append(_margin(z["score"][sel], np.load(gd)[sel].astype(int)))
            n_opt.append((~np.isnan(z["score"][sel])).sum(1))
        else:
            have_margin = False
            n_opt.append(np.zeros(len(sel), dtype=int))
        dr = ckpts[-1] / f"drift_{task}.npz"
        if dr.exists():
            z = np.load(dr)
            for j, layer in enumerate(z["layers"].tolist()):
                drift.setdefault(int(layer), []).append(z["cos"][sel, j])
    corr = np.concatenate(correct, axis=1)
    run = RunItems(
        arm=_arm_name(row.spec), seed=row.spec.seed, correct=corr,
        chosen=np.concatenate(chosen, axis=1) if have_choices else None,
        gold=np.concatenate(gold) if have_choices else None,
        n_options=np.concatenate(n_opt),
        margin_base=np.concatenate(margin) if have_margin else None,
        drift={L: np.concatenate(v) for L, v in drift.items() if len(v) == len(tasks)},
    )
    if not run.n_options.any():  # no loglik files: the task's option count from the frozen manifest
        run.n_options = np.concatenate([np.full(len(half_indices(t, n)[0]), k, dtype=int)
                                        for t, n, k in _option_counts(tasks, [len(np.load(ckpts[0] / f"items_{t}.npy")) for t in tasks])])
    return run


def _option_counts(tasks: Sequence[str], sizes: Sequence[int]) -> list[tuple[str, int, int]]:
    """(task, n_items, options) with options = round(1 / chance_uniform) from the eval manifest."""
    m = _eval_manifest()
    return [(t, n, max(2, round(1 / m[t]["chance_uniform"])) if m.get(t, {}).get("chance_uniform") else 0)
            for t, n in zip(tasks, sizes, strict=True)]


@lru_cache(maxsize=1)
def _eval_manifest() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "manifests" / "eval-protected-v1.json").read_text())["tasks"]


def load_arms(rows: Sequence[Any], repo_root: Path = REPO_ROOT, stream: str | None = FROZEN_STREAM
              ) -> dict[str, list[RunItems]]:
    """Finished rows on `stream` (None = any) with local item files, grouped into arms. Arms that
    share a display name but differ in parameters are named by the parameters that differ, e.g.
    `lora[lora.rank=256]`, so a reader can tell which configuration a line is."""
    by_key: dict[str, list[RunItems]] = {}
    specs: dict[str, Any] = {}
    seen: dict[str, set[int]] = {}
    for row in rows:
        if row.status != "finished" or (stream is not None and _stream(row.spec) != stream):
            continue
        key = _arm_key(row.spec)
        if row.spec.seed in seen.setdefault(key, set()):
            continue  # the same run listed in two row files
        run = load_run(row, repo_root)
        if run is None:
            continue
        seen[key].add(row.spec.seed)
        specs.setdefault(key, row.spec)
        by_key.setdefault(key, []).append(run)
    names = _arm_labels(specs)
    groups: dict[str, list[RunItems]] = {}
    for key, runs in by_key.items():
        for run in runs:
            run.arm = names[key]
        groups[names[key]] = runs
    return {k: sorted(v, key=lambda r: r.seed) for k, v in sorted(groups.items())}


def _arm_labels(specs: dict[str, Any]) -> dict[str, str]:
    """Display name per arm key: `_arm_name`, plus the parameters that differ between arms sharing
    that name (bench, optimizer settings other than lr, technique parameters)."""
    def params(spec: Any) -> dict[str, Any]:
        p: dict[str, Any] = {"bench": spec.bench}
        p.update({f"optim.{k}": v for k, v in spec.optim.items() if k != "lr"})
        p.update({f"{t.name}.{k}": v for t in spec.techniques for k, v in t.params.items()})
        return p

    by_name: dict[str, list[str]] = {}
    for key, spec in specs.items():
        by_name.setdefault(_arm_name(spec), []).append(key)
    out: dict[str, str] = {}
    for name, keys in by_name.items():
        if len(keys) == 1:
            out[keys[0]] = name
            continue
        ps = {k: params(specs[k]) for k in keys}
        differing = sorted({p for d in ps.values() for p in d
                            if len({json.dumps(q.get(p), sort_keys=True) for q in ps.values()}) > 1})
        for k in keys:
            # only the parameters this arm sets: the arm with all defaults keeps the plain name
            suffix = ",".join(f"{p}={ps[k][p]}" for p in differing if p in ps[k])
            out[k] = f"{name}[{suffix}]" if suffix else name
        if len({out[k] for k in keys}) < len(keys):  # cannot happen with distinct keys; stay unique
            for k in keys:
                out[k] = f"{out[k]}@{hashlib.sha256(k.encode()).hexdigest()[:6]}"
    return out


# --- repeatability -------------------------------------------------------------------------------

def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability a positive scores above a negative (ties count half). NaN without both."""
    pos, neg = score[label], score[~label]
    if not pos.size or not neg.size:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def repeatability(runs: Sequence[RunItems]) -> dict[str, Any] | None:
    """Leave-one-seed-out AUC: does 'how many other seeds lost it' predict a held-out seed's
    losses, over questions every seed's base model knows? Needs at least three seeds."""
    if len(runs) < 3:
        return None
    known = np.logical_and.reduce([r.base_ok for r in runs])
    lost = np.stack([r.lost for r in runs])[:, known]
    aucs = []
    for s in range(len(runs)):
        others = np.delete(lost, s, axis=0).sum(0)
        aucs.append(_auc(others.astype(float), lost[s]))
    counts = Counter(lost.sum(0).tolist())
    rate = float(lost.mean())
    n = len(runs)
    expected_all = known.sum() * rate ** n  # questions lost by every seed if losses were random
    return {"seeds": [r.seed for r in runs], "n_known": int(known.sum()), "loss_rate": rate,
            "auc": float(np.nanmean(aucs)), "auc_by_seed": aucs,
            "lost_by_k_seeds": {k: int(counts.get(k, 0)) for k in range(n + 1)},
            "lost_by_all": int(counts.get(n, 0)), "lost_by_all_if_random": float(expected_all)}


# --- dose vs target ------------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def _logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """(intercept, slope) of P(y) = sigmoid(a + b x), by Newton's method with a tiny ridge."""
    beta = np.zeros(2)
    X = np.stack([np.ones_like(x), x], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = p * (1 - p)
        hess = X.T @ (X * w[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def _leave_arm_out_slopes(lost: dict[str, np.ndarray], known: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Per arm: (intercept, slope) of its losses on each question's logit loss rate in the other arms."""
    tot_lost = sum(x.sum(0) for x in lost.values())
    tot_known = sum(x.sum(0) for x in known.values())
    out = {}
    for a, arm_lost in lost.items():
        frag = _logit((tot_lost - arm_lost.sum(0) + 0.5) / (tot_known - known[a].sum(0) + 1.0))
        mask = known[a]
        x = np.broadcast_to(frag, mask.shape)[mask]
        out[a] = _logistic_fit(x - x.mean(), arm_lost[mask].astype(float))
    return out


def _additive_fit(lost: dict[str, np.ndarray], known: dict[str, np.ndarray], iters: int = 200,
                  ridge: float = 0.1) -> tuple[np.ndarray, dict[str, float]]:
    """logit P(lost) = fragility_i + dose_a: one fragility per question, one dose per arm, no
    interaction. A small ridge pulls the fragility of never-lost questions toward the mean."""
    arms = list(lost)
    y = np.stack([lost[a].sum(0) for a in arms]).astype(float)  # [arms, n]
    n = np.stack([known[a].sum(0) for a in arms]).astype(float)
    f = np.zeros(y.shape[1])
    d = np.zeros(len(arms))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(d[:, None] + f[None, :])))
        f += ((y - n * p).sum(0) - ridge * (f - f.mean())) / ((n * p * (1 - p)).sum(0) + ridge)
        p = 1 / (1 + np.exp(-(d[:, None] + f[None, :])))
        d += (y - n * p).sum(1) / np.maximum((n * p * (1 - p)).sum(1), 1e-9)
    return f, dict(zip(arms, d.tolist(), strict=True))


def dose_and_target(arms: dict[str, list[RunItems]], n_sim: int = 100, seed: int = 0) -> list[dict[str, Any]]:
    """Per arm: its slope on leave-arm-out fragility, against the slope the same estimator gives
    when the data come from the additive model (each arm only sets a dose; no targeting). The null
    is simulated on the same seeds and known questions, so pooling over arms with different doses
    and the noise in the fragility estimate affect the null exactly as they affect the arm.
    Below the null's 95% band = targeted (protects the fragile questions beyond its dose); above
    it = concentrating (spares robust questions, loses the fragile ones). Needs three arms."""
    if len(arms) < 3:
        return []
    lost = {a: np.stack([r.lost for r in runs]) for a, runs in arms.items()}
    known = {a: np.stack([r.base_ok for r in runs]) for a, runs in arms.items()}
    observed = _leave_arm_out_slopes(lost, known)
    frag, dose = _additive_fit(lost, known)
    rng = np.random.default_rng(seed)
    null: dict[str, list[float]] = {a: [] for a in arms}
    for _ in range(n_sim):
        sim = {a: known[a] & (rng.random(known[a].shape) < 1 / (1 + np.exp(-(dose[a] + frag)))) for a in arms}
        for a, (_i, b) in _leave_arm_out_slopes(sim, known).items():
            null[a].append(b)
    out = []
    for a, runs in arms.items():
        b = observed[a][1]
        lo, mid, hi = np.percentile(null[a], [2.5, 50, 97.5])
        out.append({"arm": a, "seeds": [r.seed for r in runs], "loss_rate": float(lost[a][known[a]].mean()),
                    "slope": b, "null_slope": float(mid), "null_band": [float(lo), float(hi)],
                    "relative_slope": b / mid if mid else float("nan"),
                    "reading": "targeted" if b < lo else ("concentrating" if b > hi else "dose only")})
    return out


# --- timing and answer drift ---------------------------------------------------------------------

def timing(runs: Sequence[RunItems]) -> dict[str, float] | None:
    """Share of end-of-stream losses already lost after the first domain, and share of questions
    lost after the first domain that were right again at the end. Needs three checkpoints."""
    if runs[0].correct.shape[0] < 3:
        return None
    lost_end = lost_mid = early = back = 0
    for r in runs:
        b, mid, end = r.correct[0], r.correct[1], r.correct[-1]
        lost_end += int((b & ~end).sum())
        early += int((b & ~mid & ~end).sum())
        lost_mid += int((b & ~mid).sum())
        back += int((b & ~mid & end).sum())
    return {"lost_by_first_domain": early / lost_end if lost_end else float("nan"),
            "recovered_after_first_domain": back / lost_mid if lost_mid else float("nan"),
            "n_lost": lost_end}


def answer_drift(runs: Sequence[RunItems]) -> dict[str, float] | None:
    """When two seeds lose the same question, how often they pick the same wrong option, against
    the mean chance 1/(k-1) of agreeing at random over those pairs."""
    if len(runs) < 2 or any(r.chosen is None for r in runs):
        return None
    same = pairs = 0
    chance = 0.0
    for r1, r2 in itertools.combinations(runs, 2):
        both = r1.lost & r2.lost & (r1.chosen[-1] >= 0) & (r2.chosen[-1] >= 0)
        k = np.maximum(r1.n_options[both], 2)
        same += int((r1.chosen[-1][both] == r2.chosen[-1][both]).sum())
        pairs += int(both.sum())
        chance += float((1.0 / (k - 1)).sum())
    if not pairs:
        return None
    return {"same_wrong_option": same / pairs, "chance": chance / pairs, "n_pairs": pairs}


def _lost_vs_kept(runs: Sequence[RunItems], values: list[np.ndarray]) -> dict[str, float] | None:
    lost = np.concatenate([r.lost for r in runs])
    kept = np.concatenate([r.base_ok & r.correct[-1] for r in runs])
    v = np.concatenate(values)
    lv, kv = v[lost & ~np.isnan(v)], v[kept & ~np.isnan(v)]
    if not lv.size or not kv.size:
        return None
    return {"lost": float(np.median(lv)), "kept": float(np.median(kv)), "n_lost": int(lv.size),
            "auc_lower_predicts_loss": _auc(-np.concatenate([lv, kv]), np.r_[np.ones(lv.size, bool), np.zeros(kv.size, bool)])}


def margins(runs: Sequence[RunItems]) -> dict[str, float] | None:
    """Median base margin of questions later lost vs kept, and the AUC of a low margin for loss."""
    if any(r.margin_base is None for r in runs):
        return None
    return _lost_vs_kept(runs, [r.margin_base for r in runs])


def state_drift(runs: Sequence[RunItems]) -> dict[int, dict[str, float]]:
    """Per layer: median end-of-stream cosine to the base state, lost vs kept questions."""
    layers = set.intersection(*(set(r.drift) for r in runs)) if runs else set()
    out = {}
    for layer in sorted(layers):
        res = _lost_vs_kept(runs, [r.drift[layer] for r in runs])
        if res is not None:
            out[layer] = res
    return out


# --- report --------------------------------------------------------------------------------------

def analyze(rows: Sequence[Any], repo_root: Path = REPO_ROOT, stream: str | None = FROZEN_STREAM,
            inputs: Sequence[str] = ()) -> dict[str, Any]:
    """`inputs`: the row files the rows came from, recorded so the summary can be rebuilt. The
    dose-or-target reading of an arm depends on the other arms in the pool (each question's
    fragility is its loss rate in the OTHER arms), so the pool is recorded too."""
    arms = load_arms(rows, repo_root, stream)
    per_arm = {}
    for a, runs in arms.items():
        per_arm[a] = {"seeds": [r.seed for r in runs], "repeatability": repeatability(runs),
                      "timing": timing(runs), "answer_drift": answer_drift(runs),
                      "margins": margins(runs), "state_drift": state_drift(runs)}
    return {"half": "selection", "stream": stream, "status": "exploratory",
            "inputs": sorted(inputs), "pool": sorted(arms),
            "arms": per_arm, "dose_and_target": dose_and_target(arms)}


def _pct(x: float | None) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def format_report(res: dict[str, Any]) -> str:
    lines = [f"# Forgetting structure (exploratory, {res['half']} half, stream {res['stream']})", ""]
    if res.get("inputs"):
        lines += ["Built from: " + ", ".join(f"`{p}`" for p in res["inputs"])
                  + f" ({len(res.get('pool', []))} arms with item files). Rebuild: "
                  + "`clh forget-sets " + " ".join(res["inputs"]) + " --out results/analysis/forgetting`", ""]
    rep =[(a, v["repeatability"]) for a, v in res["arms"].items() if v["repeatability"]]
    if rep:
        lines += ["## Repeatability: do seeds lose the same questions?",
                  "AUC of 'lost by k other seeds' for a held-out seed's losses; 0.5 = random.", "",
                  "arm | seeds | known | loss rate | AUC | lost by every seed (if random)", "---|---|---|---|---|---"]
        for a, r in rep:
            lines.append(f"{a} | {len(r['seeds'])} | {r['n_known']} | {_pct(r['loss_rate'])} | {r['auc']:.2f} | "
                         f"{r['lost_by_all']} ({r['lost_by_all_if_random']:.1f})")
        lines.append("")
    if res["dose_and_target"]:
        lines += ["## Dose or target: does an arm protect the fragile questions specifically?",
                  ("Slope of the arm's losses on each question's fragility in the other arms (logit), against the "
                   "slope expected if every arm only set a dose (simulated null, 95% band). Below the band = protects "
                   "fragile questions beyond its dose; above = concentrates the damage on them. If the relative slope "
                   "tracks the loss rate across the plain learning-rate arms, the additive model does not hold; "
                   "then compare arms only at similar loss rates."),
                  ("**The reading depends on the pool.** A question's fragility is its loss rate in the OTHER "
                   f"arms listed here ({len(res.get('pool', []))} arms), so adding or removing arms can move a slope "
                   "across the band (rebuilt with more arms, no-mitigation and the reference arm moved from "
                   "targeted to dose only). Quote a reading together with its pool, never as a property of "
                   "the arm alone."), "",
                  "arm | seeds | loss rate | slope | if dose only | relative | reading", "---|---|---|---|---|---|---"]
        for d in sorted(res["dose_and_target"], key=lambda d: d["relative_slope"]):
            lines.append(f"{d['arm']} | {len(d['seeds'])} | {_pct(d['loss_rate'])} | {d['slope']:.2f} | "
                         f"{d['null_slope']:.2f} ({d['null_band'][0]:.2f} to {d['null_band'][1]:.2f}) | "
                         f"{d['relative_slope']:.2f} | {d['reading']}")
        lines.append("")
    lines += ["## When and how questions are lost", "",
              ("arm | lost | already lost after code | lost after code, back at end | same wrong option (chance) | "
               "base margin lost / kept"), "---|---|---|---|---|---"]
    for a, v in res["arms"].items():
        t, d, m = v["timing"], v["answer_drift"], v["margins"]
        lines.append(f"{a} | {t['n_lost'] if t else '-'} | {_pct(t and t['lost_by_first_domain'])} | "
                     f"{_pct(t and t['recovered_after_first_domain'])} | "
                     f"{_pct(d['same_wrong_option']) + ' (' + _pct(d['chance']) + ')' if d else '-'} | "
                     f"{f'{m['lost']:.2f} / {m['kept']:.2f}' if m else '-'}")
    drift_arms = {a: v["state_drift"] for a, v in res["arms"].items() if v["state_drift"]}
    if drift_arms:
        lines += ["", "## State drift (bench fast-mech): cosine to the base state at the end, lost / kept", ""]
        for a, sd in drift_arms.items():
            lines.append(f"{a}: " + ", ".join(f"L{L} {x['lost']:.3f} / {x['kept']:.3f}" for L, x in sd.items()))
    return "\n".join(lines) + "\n"


def agent_summary(res: dict[str, Any], max_arms: int = 8) -> str:
    """A few lines for an agent prompt: selection half only, no question text."""
    lines = ["Forgetting structure (selection half, exploratory):"]
    reps = [v["repeatability"] for v in res["arms"].values() if v["repeatability"]]
    if reps:
        lines.append("- seeds lose the same questions: held-out-seed AUC "
                     + ", ".join(f"{r['auc']:.2f}" for r in reps) + " (0.5 = random)")
    if res["dose_and_target"]:
        lines.append(f"- dose-or-target readings below are relative to this pool of {len(res.get('pool', []))} "
                     "arms; adding arms can change them")
    for d in sorted(res["dose_and_target"], key=lambda d: d["relative_slope"])[:max_arms]:
        lines.append(f"- {d['arm']}: loses {_pct(d['loss_rate'])} of known questions; fragility slope "
                     f"{d['slope']:.2f} vs {d['null_slope']:.2f} if it only set a dose -> {d['reading']}")
    for a, v in list(res["arms"].items())[:max_arms]:
        t, d = v["timing"], v["answer_drift"]
        if t and d:
            lines.append(f"- {a}: {_pct(t['lost_by_first_domain'])} of losses happen during code; "
                         f"two seeds that lose a question pick the same wrong option {_pct(d['same_wrong_option'])} "
                         f"of the time (chance {_pct(d['chance'])})")
    return "\n".join(lines)
