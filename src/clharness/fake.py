"""Fake result rows, so analysis, the agent and the page can be built before any real run exists.

The numbers are invented. Every row has backend "fake" and git_sha "fake", and must never be
mixed into a reported result.

Techniques have planted effects on retention and acquisition, plus two planted pairwise
interactions and seed noise. `PLANTED` is written next to the rows, so the attribution analysis
can be tested: it should recover these values inside its confidence intervals.
"""

from __future__ import annotations

import math
import random
from typing import Any

from clharness.analysis.design import (
    HALVES,
    LR_CONTROLS,
    NO_MITIGATION,
    REFERENCE,
    SELECTION,
    factorial,
    lr_of,
    selection_of,
)
from clharness.analysis.design import PARAMS as DESIGN_PARAMS
from clharness.analysis.design import STREAM as DESIGN_STREAM
from clharness.analysis.design import spec as design_spec
from clharness.spec import (
    Acquisition,
    BoundaryRetention,
    Cost,
    HalfRetention,
    ResultRow,
    Retention,
    RunSpec,
    StreamDomain,
    TaskRetention,
    TechniqueSpec,
)

TASKS = ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal")
BASE_SCORE = {"arc_easy": 0.706, "hellaswag": 0.634, "piqa": 0.738, "pubmedqa_pqal": 0.642}
STREAM = DESIGN_STREAM  # the frozen bench: code then legal
HALF_ITEMS = 250  # each task's 500 protected items split into a selection and a report half

# What the generator plants. Retention values are changes to the mean delta, as fractions
# (0.025 = 2.5 points less forgetting). Acquisition values are added to log(perplexity ratio):
# positive means the technique learns the new domains less well.
PLANTED: dict[str, Any] = {
    "baseline_retention_mean_delta": -0.050,
    "baseline_log_ppl_ratio": -0.150,
    "retention_main_effects": {
        "replay": 0.025,
        "lr_rewarm": 0.008,
        "lora": 0.030,
        "freeze": 0.010,
        "distill": 0.020,
        "wavg": 0.012,
    },
    "retention_interactions": {
        "distill+replay": -0.012,  # they overlap: together they give less than the sum
        "lora+replay": 0.008,
    },
    # Sized so the 5% acquisition gate (PREREG 5) passes some factorial runs and fails others.
    "acquisition_main_effects": {
        "replay": 0.010,
        "lr_rewarm": -0.010,
        "lora": 0.090,
        "freeze": 0.020,
        "distill": 0.010,
        "wavg": 0.030,
    },
    # replay's `select` signal, against random selection (plain replay); (signal, keep) keys
    # no-mitigation at a lower learning rate (PREREG A13): forgets less, learns code a little less.
    # 3e-5 learns too little to pass the 5% acquisition gate; 5e-5 passes and is the chosen control.
    "lr_control_effects": {
        "5e-05": {"retention": 0.025, "log_acquisition": 0.020},
        "3e-05": {"retention": 0.045, "log_acquisition": 0.060},
    },
    "retention_selection_effects": {
        "loss/high": 0.004,
        "loss/low": -0.002,
        "act_nov/high": 0.012,
        "act_nov/low": -0.006,
    },
    "seed_noise_sd": 0.006,
    "item_noise_sd": 0.012,
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {k: dict(v) for k, v in DESIGN_PARAMS.items()}

# Extra wall-clock per technique, as a fraction of the baseline run.
_TIME_FACTOR = {"distill": 0.30, "replay": 0.02, "lora": -0.10, "freeze": -0.15}
_SELECTION_TIME = 0.15  # scoring a pool of 4 candidates per replayed row
_FULL_PARAMS = 1_000_000_000
_TRAINABLE = {"lora": 6_500_000, "freeze": 480_000_000}


def _spec(techniques: tuple[str, ...], seed: int, **param_overrides: dict[str, Any]) -> RunSpec:
    label = "+".join(techniques) if techniques else "no-mitigation"
    for params in param_overrides.values():
        label += "".join(f"-{v}" for v in params.values())
    return RunSpec(
        name=f"fake-{label}-s{seed}",
        seed=seed,
        stream=[StreamDomain(name=n, tokens=t) for n, t in STREAM],
        techniques=[
            TechniqueSpec(name=t, params={**DEFAULT_PARAMS[t], **param_overrides.get(t, {})})
            for t in techniques
        ],
    )


def _effects(techniques: tuple[str, ...], selection: tuple[str, str] | None = None,
             lr: float | None = None) -> tuple[float, float]:
    """Planted (retention mean delta, log perplexity ratio) for a set of techniques, no noise."""
    ret = PLANTED["baseline_retention_mean_delta"]
    if lr is not None:
        e = PLANTED["lr_control_effects"][f"{lr:g}"]
        return ret + e["retention"], PLANTED["baseline_log_ppl_ratio"] + e["log_acquisition"]
    if selection:
        ret += PLANTED["retention_selection_effects"][f"{selection[0]}/{selection[1]}"]
    acq = PLANTED["baseline_log_ppl_ratio"]
    for t in techniques:
        ret += PLANTED["retention_main_effects"][t]
        acq += PLANTED["acquisition_main_effects"][t]
    for pair, effect in PLANTED["retention_interactions"].items():
        if set(pair.split("+")) <= set(techniques):
            ret += effect
    return ret, acq


def _half(rng: random.Random, mean_delta: float) -> HalfRetention:
    per_task = {}
    for task in TASKS:
        delta = round(mean_delta + rng.gauss(0, PLANTED["item_noise_sd"]), 4)
        base = BASE_SCORE[task]
        per_task[task] = TaskRetention(
            base=base, final=round(base + delta, 4), delta=delta, n_items=HALF_ITEMS
        )
    mean = round(sum(t.delta for t in per_task.values()) / len(per_task), 4)
    return HalfRetention(per_task=per_task, mean_delta=mean)


def _finished(spec: RunSpec, rng: random.Random) -> ResultRow:
    techs = tuple(spec.technique_names())
    selection = selection_of(spec)
    ret, acq = _effects(techs, selection, lr_of(spec))
    ret += rng.gauss(0, PLANTED["seed_noise_sd"])  # shared by both halves: it is run-level noise
    selection, report = _half(rng, ret), _half(rng, ret)
    # In the pilot runs most of the forgetting happened in the first domain.
    boundaries = [
        BoundaryRetention(
            domain=STREAM[0][0],
            gradient_tokens=STREAM[0][1],
            selection_mean_delta=round(selection.mean_delta * 0.8, 4),
            report_mean_delta=round(report.mean_delta * 0.8, 4),
        ),
        BoundaryRetention(
            domain=STREAM[1][0],
            gradient_tokens=spec.gradient_tokens(),
            selection_mean_delta=selection.mean_delta,
            report_mean_delta=report.mean_delta,
        ),
    ]
    ratios = {name: round(math.exp(acq + rng.gauss(0, 0.01)), 4) for name, _ in STREAM}
    mean_ratio = round(math.exp(sum(math.log(v) for v in ratios.values()) / len(ratios)), 4)
    wall = 1700.0 * (1 + sum(_TIME_FACTOR.get(t, 0.0) for t in techs) + (_SELECTION_TIME if selection else 0.0)) \
        + rng.gauss(0, 30)
    trainable = min([_TRAINABLE[t] for t in techs if t in _TRAINABLE] or [_FULL_PARAMS])
    return ResultRow(
        spec=spec,
        spec_hash=spec.spec_hash(),
        git_sha="fake",
        status="finished",
        backend="fake",
        retention=Retention(selection=selection, report=report, boundaries=boundaries),
        acquisition=Acquisition(
            ppl_ratio=ratios,
            mean_ppl_ratio=mean_ratio,
            general_ppl_ratio=round(math.exp(-4.0 * ret + rng.gauss(0, 0.01)), 4),
            # The planted acquisition effects are what each domain learned, so they show at its
            # own boundary. Same values as the final ratio here; no extra random draws, so the
            # other fake numbers stay unchanged.
            own_boundary_ppl_ratio=ratios,
            mean_own_boundary_ppl_ratio=mean_ratio,
        ),
        cost=Cost(
            gradient_tokens=spec.gradient_tokens(),
            wall_clock_s=round(wall, 1),
            gpu_type="none",
            update_bytes=trainable * 4,  # fp32 delta
            trainable_params=trainable,
            extra_forward_tokens=(spec.gradient_tokens() if "distill" in techs else 0)
            + (int(0.05 * 4 * spec.gradient_tokens()) if selection else 0),
            est_usd=round(wall / 3600 * 4.0, 2),
        ),
        answer_share={t: round(rng.uniform(0.28, 0.45), 3) for t in TASKS},
        items_path=f"fake/items/{spec.spec_hash()[:12]}.json",
    )


def _failed(spec: RunSpec, reason: str) -> ResultRow:
    return ResultRow(
        spec=spec, spec_hash=spec.spec_hash(), git_sha="fake", status="failed", reason=reason,
        backend="fake", attempt=3,
    )


def _collapsed(spec: RunSpec, rng: random.Random) -> ResultRow:
    """A run that finished but answers one option almost every time on PubMedQA. It stays
    `finished`: the analysis flags collapse from `answer_share` (analysis/PREREG.md, A9)."""
    row = _finished(spec, rng)
    return row.model_copy(update={"answer_share": {**row.answer_share, "pubmedqa_pqal": 0.97}})


def _planted_promotion(k: int = 3) -> list[frozenset[str]]:
    """The k screened configurations PREREG 6.3 would promote if there were no noise: best planted
    retention among those passing the acquisition gate against the reference arm."""
    ref_acq = _effects(tuple(REFERENCE))[1]
    limit = ref_acq + math.log(1.05)
    candidates = [t for t in factorial() if t not in (NO_MITIGATION, REFERENCE)]
    passing = [t for t in candidates if _effects(tuple(t))[1] <= limit]
    return sorted(passing, key=lambda t: -_effects(tuple(t))[0])[:k]


def fake_rows(seed: int = 0) -> list[ResultRow]:
    """About 40 rows shaped like the pre-registered study (analysis/PREREG.md, section 6): the
    noise floor, the screen (factorial, the reference halves, selection on replay), replication,
    and a few failures."""
    rng = random.Random(seed)
    rows: list[ResultRow] = []
    failed_screen = frozenset({"freeze", "wavg"})  # one screening run is lost to preemption

    def add(techniques: frozenset[str], seeds: range, selection: tuple[str, str] | None = None) -> None:
        rows.extend(_finished(design_spec(techniques, s, selection), rng) for s in seeds)

    add(NO_MITIGATION, range(5))  # the noise floor; seed 0 is also the factorial's first run
    add(REFERENCE, range(3))  # the reference arm, replicated
    for t in factorial():
        if t not in (NO_MITIGATION, REFERENCE, failed_screen):
            add(t, range(1))
    for t in HALVES:
        add(t, range(1))
    for sel in SELECTION:
        add(REFERENCE, range(1), sel)
    for t in _planted_promotion():
        add(t, range(1, 3))  # replication of the promoted configurations
    for sel in (("loss", "high"), ("act_nov", "high")):  # the selection family, replicated
        add(REFERENCE, range(1, 3), sel)
    for lr in LR_CONTROLS:  # the lower-LR no-mitigation controls (PREREG A13)
        rows.extend(_finished(design_spec(NO_MITIGATION, s, lr=lr), rng) for s in range(3))

    rows.append(_failed(design_spec(failed_screen, 0), "infrastructure: GPU preempted 3 times"))
    rows.append(_failed(_spec(("distill", "freeze"), 0), "out of memory in the teacher forward"))
    rows.append(_collapsed(_spec(("distill",), 1, distill={"weight": 10.0}), rng))
    rows.append(_collapsed(_spec(("freeze", "lora"), 0), rng))
    nan = _spec(("lora",), 0, lora={"rank": 512})
    rows.append(ResultRow(spec=nan, spec_hash=nan.spec_hash(), git_sha="fake", status="invalid",
                          reason="non-finite score or perplexity", backend="fake"))
    return rows
