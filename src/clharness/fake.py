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
STREAM = (("code", 10_000_000), ("biomedical", 10_000_000))
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
    "acquisition_main_effects": {
        "replay": 0.010,
        "lr_rewarm": -0.010,
        "lora": 0.090,
        "freeze": 0.040,
        "distill": 0.030,
        "wavg": 0.050,
    },
    "seed_noise_sd": 0.006,
    "item_noise_sd": 0.012,
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "replay": {"share": 0.05, "source": "general"},
    "lr_rewarm": {"rewarm_steps": 8, "min_lr_ratio": 0.1},
    "lora": {"rank": 8},
    "freeze": {"bottom_layers": 13},
    "distill": {"weight": 1.0},
    "wavg": {"alpha": 0.25},
}

# Extra wall-clock per technique, as a fraction of the baseline run.
_TIME_FACTOR = {"distill": 0.30, "replay": 0.02, "lora": -0.10, "freeze": -0.15}
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


def _effects(techniques: tuple[str, ...]) -> tuple[float, float]:
    """Planted (retention mean delta, log perplexity ratio) for a set of techniques, no noise."""
    ret = PLANTED["baseline_retention_mean_delta"]
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
    ret, acq = _effects(techs)
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
    wall = 1700.0 * (1 + sum(_TIME_FACTOR.get(t, 0.0) for t in techs)) + rng.gauss(0, 30)
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
        ),
        cost=Cost(
            gradient_tokens=spec.gradient_tokens(),
            wall_clock_s=round(wall, 1),
            gpu_type="none",
            update_bytes=trainable * 4,  # fp32 delta
            trainable_params=trainable,
            extra_forward_tokens=spec.gradient_tokens() if "distill" in techs else 0,
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
    """A run that finished but answers one option almost every time on PubMedQA."""
    row = _finished(spec, rng)
    share = {**row.answer_share, "pubmedqa_pqal": 0.97}
    return row.model_copy(update={
        "status": "invalid",
        "reason": "answer collapse on pubmedqa_pqal: most common answer share 0.97 > 0.90",
        "answer_share": share,
    })


def fake_rows(seed: int = 0) -> list[ResultRow]:
    """About 40 rows shaped like a real study: screen, replication, controls, and failures."""
    rng = random.Random(seed)
    rows: list[ResultRow] = []

    def add(techniques: tuple[str, ...], seeds: range, **overrides: dict[str, Any]) -> None:
        rows.extend(_finished(_spec(techniques, s, **overrides), rng) for s in seeds)

    add((), range(5))  # no mitigation: the noise floor
    add(("lr_rewarm", "replay"), range(3))  # the reference arm
    for tech in ("replay", "lr_rewarm", "lora", "freeze", "distill", "wavg"):
        add((tech,), range(1))  # screen: each technique alone, one seed
    # Selection is a parameter of replay, not a technique of its own. No planted effect: the
    # analysis should find loss-based and activation-based selection no different from random.
    for signal in ("loss", "act_nov"):
        add(("replay",), range(1), replay={"select": signal, "pool": 4})
    core = ("distill", "lora", "replay", "wavg")
    pairs = [(a, b) for i, a in enumerate(core) for b in core[i + 1:]]
    for pair in pairs:
        add(pair, range(1))  # screen: pairs of the strongest techniques
    add(("lora", "replay"), range(1, 3))  # replication of the configurations that matter
    add(("distill", "replay"), range(1, 3))
    add(("lora",), range(1, 3))
    add(("replay",), range(1, 3))
    add(("distill", "lora", "replay"), range(3))

    rows.append(_failed(_spec(("freeze", "wavg"), 0), "infrastructure: GPU preempted 3 times"))
    rows.append(_failed(_spec(("distill", "freeze"), 0), "out of memory in the teacher forward"))
    rows.append(_collapsed(_spec(("distill",), 1, distill={"weight": 10.0}), rng))
    rows.append(_collapsed(_spec(("freeze", "lora"), 0), rng))
    return rows
