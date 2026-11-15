"""The judge: a simulated bench with planted technique effects.

Nothing here trains a model. `World(seed)` is a deterministic function from a run spec to a
result row. Every agent is scored against it, and prompt optimization climbs the same score.

What a world plants (all of it differs from seed to seed, so a prompt cannot memorise it):

  * a main effect on retention for each technique, in points (`select_random` is exactly 0);
  * pairwise interactions: the reference arm and replay are redundant, the two selection signals
    are redundant, and three more pairs are drawn at random;
  * an acquisition penalty per technique (how much it raises the perplexity ratio);
  * a TRAP technique: the largest retention effect of all, bought by not learning. It fails the
    acquisition floor, alone and in every combination;
  * a COLLAPSE configuration: distillation above a weight threshold answers one option on
    PubMedQA (answer share above 0.9), so the run is invalid while its score looks fine;
  * noise with a known SD on the selection-half MC mean. A third of it is shared by runs with the
    same seed (as data order is), the rest is per run and per half;
  * an infrastructure failure rate, and a cost multiplier per technique.

The report half has the same true effect as the selection half and independent noise. With
`report_sentinel` set, every report-half number is that sentinel instead: the firewall canary.

`SimExecutor` implements `roles.Executor` on a fake clock and writes `rows.jsonl`. It can be told
to be unreachable, to hang, to fail a run, or to cut a row off mid-line. `oracle(world)` returns
the truly best valid configuration and every true effect.

The simulator ignores technique parameters (except the collapse threshold), `optim` overrides
and the domain order. Numbers are plausible, taken from `docs/baselines.md`, not measured.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .context import REPORT_SENTINEL
from .contracts import (
    MC_TASKS,
    PRIMARY_METRIC,
    Catalog,
    CatalogEntry,
    FailureKind,
    Frozen,
    ParamSpec,
    Prereg,
    ResultRow,
    RunSpec,
    StreamDomain,
    a1,
    a1_cost,
    a1_failed_row,
    append_jsonl,
    arm_label,
    atomic_write_json,
    build_spec,
    from_a1_row,
    load_rows,
    passes_acquisition_floor,
    retryable,
    spec_hash,
    technique_map,
    technique_names,
)
from .roles import Clock, ExecutorUnreachable

__all__ = [
    "REPORT_SENTINEL", "CollapseRule", "Oracle", "SimExecutor", "World", "default_catalog",
    "default_prereg", "default_stream", "make_spec", "make_worlds", "oracle",
]

SIM_GIT_SHA = "simulated"
BASELINE_MC_MEAN = -11.85  # no mitigation at LR 1e-4, from the pilot
# Per-task share of the MC-mean change (mean 1.0), from the pilot's per-task deltas.
TASK_WEIGHTS: dict[str, float] = dict(zip(MC_TASKS, (1.35, 1.40, 0.69, 0.56)))
ANSWER_SHARE_BASE: dict[str, float] = dict(zip(MC_TASKS, (0.28, 0.27, 0.52, 0.62)))
COLLAPSE_TASK = "pubmedqa_pqal"
SIM_BASE_SCORE: dict[str, float] = dict(zip(MC_TASKS, (0.70, 0.55, 0.75, 0.68)))
SIM_ITEMS_PER_HALF = 500
ACQ_BASE: dict[str, float] = {"code": 0.78, "biomedical": 0.85, "legal": 0.80, "math": 0.86}
BASE_RUN_USD = 2.50  # one no-mitigation bench run
BASE_RUN_SECONDS = 1800.0
BENCH_TOKENS = 40_000_000
ACQ_NOISE_SD = 0.004
INVALID_PENALTY = 5.0  # added to the regret of an invalid choice
MAX_COMBO = 3  # largest combination the oracle considers
SPLIT_OFFSETS: dict[str, int] = {"train": 1000, "val": 2000, "test": 3000}

TECHNIQUES: tuple[str, ...] = (
    "reference_arm", "replay", "lora", "layer_freeze", "distill", "weight_avg",
    "select_random", "select_loss", "select_activation",
)
TRAP_CANDIDATES: tuple[str, ...] = ("layer_freeze", "lora", "distill")
_EFFECT_CENTER: dict[str, float] = {
    "reference_arm": 4.0, "replay": 3.5, "lora": 2.5, "layer_freeze": 2.5, "distill": 2.5,
    "weight_avg": 1.5, "select_random": 0.0, "select_loss": 0.8, "select_activation": 2.2,
}
_ACQ_PENALTY_RANGE: dict[str, tuple[float, float]] = {
    "reference_arm": (0.010, 0.025), "replay": (0.010, 0.025), "lora": (0.015, 0.035),
    "layer_freeze": (0.015, 0.035), "distill": (0.015, 0.035), "weight_avg": (0.005, 0.020),
    "select_random": (0.0, 0.0), "select_loss": (-0.010, 0.005),
    "select_activation": (-0.010, 0.005),
}
_COST_MULTIPLIER: dict[str, float] = {
    "reference_arm": 1.15, "replay": 1.10, "lora": 0.70, "layer_freeze": 0.60, "distill": 1.60,
    "weight_avg": 1.05, "select_random": 1.00, "select_loss": 1.30, "select_activation": 1.35,
}
_FREE_PAIR_POOL: tuple[str, ...] = (
    "replay", "lora", "layer_freeze", "distill", "weight_avg", "select_activation",
)


def _rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _names(techniques: Iterable[str] | Mapping[str, Any] | RunSpec) -> tuple[str, ...]:
    if isinstance(techniques, RunSpec):
        return tuple(sorted(technique_names(techniques)))
    return tuple(sorted(set(techniques) - {"no_mitigation"}))


class CollapseRule(Frozen):
    technique: str
    param: str
    threshold: float  # the run collapses when the parameter is at or above this
    task: str


class World:
    """One simulated bench. Deterministic: the same seed and spec always give the same row."""

    def __init__(
        self,
        seed: int,
        *,
        noise_sd: float = 0.8,
        infra_fail_rate: float = 0.1,
        report_sentinel: float | None = None,
    ) -> None:
        self.seed = seed
        self.noise_sd = noise_sd
        self.infra_fail_rate = infra_fail_rate
        self.report_sentinel = report_sentinel
        self.baseline_retention = BASELINE_MC_MEAN
        rng = _rng("world", seed)
        self.trap = rng.choice(TRAP_CANDIDATES)
        self.main_effects = self._draw_main_effects(rng)
        self.acq_penalty = self._draw_acq_penalties(rng)
        self.interactions = self._draw_interactions(rng)
        self.collapse = CollapseRule(
            technique="distill", param="weight", threshold=round(rng.uniform(0.8, 1.5), 2),
            task=COLLAPSE_TASK,
        )

    # ---- what is planted -------------------------------------------------------------------

    def _draw_main_effects(self, rng: random.Random) -> dict[str, float]:
        effects = {n: round(c * rng.uniform(0.6, 1.4), 2) for n, c in _EFFECT_CENTER.items()}
        effects[self.trap] = round(max(effects.values()) + rng.uniform(2.0, 4.0), 2)
        return effects

    def _draw_acq_penalties(self, rng: random.Random) -> dict[str, float]:
        penalties = {n: round(rng.uniform(*r), 4) for n, r in _ACQ_PENALTY_RANGE.items()}
        penalties[self.trap] = round(rng.uniform(0.20, 0.35), 4)
        return penalties

    def _draw_interactions(self, rng: random.Random) -> dict[tuple[str, str], float]:
        main = self.main_effects
        pairs = {
            _pair("reference_arm", "replay"): round(-0.8 * main["replay"], 2),
            _pair("select_loss", "select_activation"): round(
                -min(main["select_loss"], main["select_activation"]), 2
            ),
        }
        free = [p for p in itertools.combinations(_FREE_PAIR_POOL, 2) if self.trap not in p]
        for pair in rng.sample(free, 3):
            pairs[_pair(*pair)] = round(rng.choice((-1, 1)) * rng.uniform(1.0, 2.5), 2)
        return pairs

    # ---- truth, without noise --------------------------------------------------------------

    def true_retention(self, techniques: Iterable[str] | Mapping[str, Any] | RunSpec) -> float:
        """True selection-half and report-half MC-mean change, in points."""
        names = _names(techniques)
        total = self.baseline_retention + sum(self.main_effects[n] for n in names)
        total += sum(self.interactions.get(p, 0.0) for p in itertools.combinations(names, 2))
        return round(total, 4)

    def true_acquisition(
        self, techniques: Iterable[str] | Mapping[str, Any] | RunSpec,
        domains: Sequence[str] = ("code", "biomedical"),
    ) -> dict[str, float]:
        """True perplexity ratio per trained domain."""
        factor = 1.0
        for name in _names(techniques):
            factor *= 1.0 + self.acq_penalty[name]
        return {d: round(ACQ_BASE.get(d, 0.9) * factor, 5) for d in domains}

    def passes_floor(self, techniques: Iterable[str] | Mapping[str, Any] | RunSpec,
                     delta: float = 0.05) -> bool:
        mine = self.true_acquisition(techniques).values()
        base = self.true_acquisition(()).values()
        return passes_acquisition_floor(sum(mine) / len(mine), sum(base) / len(base), delta)

    def collapses(self, techniques: Mapping[str, Mapping[str, Any]]) -> bool:
        rule = self.collapse
        if rule.technique not in techniques:
            return False
        default = default_catalog().get(rule.technique).params[rule.param].default
        return float(techniques[rule.technique].get(rule.param, default)) >= rule.threshold

    def is_valid(self, techniques: Mapping[str, Mapping[str, Any]] | Iterable[str],
                 delta: float = 0.05) -> bool:
        """Truly valid: passes the acquisition floor and does not collapse."""
        collapsed = isinstance(techniques, Mapping) and self.collapses(techniques)
        return self.passes_floor(techniques, delta) and not collapsed

    def regret(self, techniques: Mapping[str, Mapping[str, Any]] | Iterable[str]) -> float:
        """Points of true retention lost against the oracle's best. 0 is perfect.

        An invalid choice (trap or collapse) scores as if nothing had been done, plus a penalty.
        """
        best = oracle(self).best_retention
        if not self.is_valid(techniques):
            return round(best - self.baseline_retention + INVALID_PENALTY, 4)
        return round(best - self.true_retention(techniques), 4)

    # ---- cost ------------------------------------------------------------------------------

    def cost_usd(self, spec: RunSpec) -> float:
        scale = spec.gradient_tokens() / BENCH_TOKENS
        return round(BASE_RUN_USD * scale * self._cost_factor(spec), 4)

    def wall_seconds(self, spec: RunSpec) -> float:
        scale = spec.gradient_tokens() / BENCH_TOKENS
        return round(BASE_RUN_SECONDS * scale * self._cost_factor(spec), 1)

    def _cost_factor(self, spec: RunSpec) -> float:
        factor = 1.0
        for name in technique_names(spec):
            factor *= _COST_MULTIPLIER.get(name, 1.0)
        return factor

    def _cost(self, spec: RunSpec, fraction: float = 1.0) -> a1.Cost:
        lora = "lora" in technique_names(spec)
        frozen = "layer_freeze" in technique_names(spec)
        params = 6_500_000 if lora else (500_000_000 if frozen else 1_000_000_000)
        return a1_cost(
            int(spec.gradient_tokens() * fraction), round(self.wall_seconds(spec) * fraction, 1),
            "sim-H100", round(self.cost_usd(spec) * fraction, 4),
            update_bytes=params * 2, trainable_params=params,
        )

    # ---- one run ---------------------------------------------------------------------------

    def infra_fails(self, key: str, attempt: int) -> bool:
        return _rng("infra", self.seed, key, attempt).random() < self.infra_fail_rate

    def failed_row(self, spec: RunSpec, kind: FailureKind, reason: str,
                   attempt: int = 1) -> ResultRow:
        """The agent's view of `failed_row_a1`."""
        return from_a1_row(self.failed_row_a1(spec, kind, reason, attempt))

    def failed_row_a1(self, spec: RunSpec, kind: FailureKind, reason: str,
                      attempt: int = 1) -> a1.ResultRow:
        """A `failed` A1 row. An infrastructure failure still costs a quarter of the run."""
        fraction = 0.25 if kind == "infra" else 1.0
        return a1_failed_row(
            spec, kind, reason, git_sha=SIM_GIT_SHA, backend="local", attempt=attempt,
            cost=self._cost(spec, fraction),
        )

    def _shared_noise(self, spec: RunSpec, key: str) -> float:
        by_seed = _rng("seed", self.seed, spec.seed).gauss(0.0, self.noise_sd * 0.4**0.5)
        by_run = _rng("run", self.seed, key).gauss(0.0, self.noise_sd * 0.3**0.5)
        return by_seed + by_run

    def _half(self, true_mean: float, shared: float, key: str, half: str) -> a1.HalfRetention:
        """One half in A1's units: the world works in points, rows hold fractions."""
        rng = _rng("half", self.seed, key, half)
        task_sd = 2.0 * self.noise_sd * 0.3**0.5  # the mean of four tasks has half this SD
        deltas = {
            t: round((TASK_WEIGHTS[t] * true_mean + shared + rng.gauss(0.0, task_sd)) / 100, 5)
            for t in MC_TASKS
        }
        return _a1_half(deltas)

    def _sentinel_half(self) -> a1.HalfRetention:
        return _a1_half(dict.fromkeys(MC_TASKS, float(self.report_sentinel)))

    def _answer_share(self, key: str, collapsed: bool) -> dict[str, float]:
        rng = _rng("share", self.seed, key)
        share = {t: round(ANSWER_SHARE_BASE[t] + rng.uniform(0.0, 0.08), 3) for t in MC_TASKS}
        if collapsed:
            share[self.collapse.task] = round(rng.uniform(0.93, 0.99), 3)
        return share

    def simulate(self, spec: RunSpec, attempt: int = 1, label_collapse: bool = True) -> ResultRow:
        """The agent's view of the row this world gives for `spec`."""
        return from_a1_row(self.simulate_a1(spec, attempt, label_collapse))

    def simulate_a1(self, spec: RunSpec, attempt: int = 1,
                    label_collapse: bool = True) -> a1.ResultRow:
        """The A1 row this world gives for `spec`. Attempt 2 is the re-queue of a failed attempt 1."""
        key = spec_hash(spec)
        unknown = [n for n in technique_names(spec) if n not in self.main_effects]
        if unknown:
            return self.failed_row_a1(spec, "science",
                                      f"unknown technique: {', '.join(unknown)}", attempt)
        if self.infra_fails(key, attempt):
            return self.failed_row_a1(spec, "infra", "simulated failure: preempted", attempt)
        collapsed = self.collapses(technique_map(spec))
        shared = self._shared_noise(spec, key)
        truth = self.true_retention(spec)
        selection = self._half(truth, shared, key, "selection")
        report = self._half(truth, shared, key, "report")
        if self.report_sentinel is not None:
            report = self._sentinel_half()
        acq_rng = _rng("acq", self.seed, key)
        domains = [d.name for d in spec.stream]
        acquisition = {
            d: round(v * (1.0 + acq_rng.gauss(0.0, ACQ_NOISE_SD)), 4)
            for d, v in self.true_acquisition(spec, domains).items()
        }
        geo = math.exp(sum(math.log(v) for v in acquisition.values()) / len(acquisition))
        share = self._answer_share(key, collapsed)
        invalid = collapsed and label_collapse
        reason = f"answer collapse: {self.collapse.task} share {share[self.collapse.task]}"
        return a1.ResultRow(
            spec=spec, spec_hash=key, git_sha=SIM_GIT_SHA,
            status="invalid" if invalid else "finished", reason=reason if invalid else None,
            backend="local", attempt=attempt,
            retention=a1.Retention(selection=selection, report=report),
            acquisition=a1.Acquisition(ppl_ratio=acquisition, mean_ppl_ratio=round(geo, 4)),
            cost=self._cost(spec), answer_share=share,
        )


def _a1_half(deltas: Mapping[str, float]) -> a1.HalfRetention:
    per_task = {
        t: a1.TaskRetention(base=SIM_BASE_SCORE[t], final=round(SIM_BASE_SCORE[t] + d, 5),
                            delta=d, n_items=SIM_ITEMS_PER_HALF)
        for t, d in deltas.items()
    }
    return a1.HalfRetention(per_task=per_task,
                            mean_delta=round(sum(deltas.values()) / len(deltas), 6))


# --------------------------------------------------------------------------------------------
# Oracle
# --------------------------------------------------------------------------------------------


class Oracle(Frozen):
    """The truth about a world. Never shown to an agent; used to score one."""

    best: tuple[str, ...]  # technique names of the truly best valid configuration
    best_arm: str
    best_retention: float
    baseline_retention: float
    reference_retention: float
    main_effects: dict[str, float]
    interactions: dict[str, float]  # "a+b" -> points
    trap: str
    collapse: CollapseRule
    retention: dict[str, float]  # arm label -> true MC-mean change, every arm up to MAX_COMBO
    valid: dict[str, bool]  # arm label -> passes the acquisition floor at default parameters


def oracle(world: World, max_combo: int = MAX_COMBO, delta: float = 0.05) -> Oracle:
    """Enumerate every combination of up to `max_combo` techniques at default parameters."""
    retention, valid = {}, {}
    for size in range(max_combo + 1):
        for combo in itertools.combinations(sorted(TECHNIQUES), size):
            retention[arm_label(combo)] = world.true_retention(combo)
            valid[arm_label(combo)] = world.passes_floor(combo, delta)
    ranked = sorted((a for a in retention if valid[a]), key=lambda a: (-retention[a], len(a), a))
    best_arm = ranked[0]
    best = () if best_arm == "no_mitigation" else tuple(best_arm.split("+"))
    return Oracle(
        best=best, best_arm=best_arm, best_retention=retention[best_arm],
        baseline_retention=world.baseline_retention,
        reference_retention=world.true_retention(("reference_arm",)),
        main_effects=dict(world.main_effects),
        interactions={"+".join(p): v for p, v in world.interactions.items()},
        trap=world.trap, collapse=world.collapse, retention=retention, valid=valid,
    )


def make_worlds(n: int, split: Literal["train", "val", "test"], **world_kwargs: Any) -> list[World]:
    """`n` worlds from the seed range of a split. The three ranges never overlap."""
    if not 0 < n <= 1000:
        raise ValueError("n must be between 1 and 1000")
    return [World(SPLIT_OFFSETS[split] + i, **world_kwargs) for i in range(n)]


# --------------------------------------------------------------------------------------------
# Catalog, PREREG and specs that match the worlds
# --------------------------------------------------------------------------------------------


def _entry(name: str, description: str, role: str, control: bool = False,
           **params: ParamSpec) -> CatalogEntry:
    return CatalogEntry(
        name=name, description=description, params=params, role=role, is_control=control,
        cost_multiplier=_COST_MULTIPLIER.get(name, 1.0),
    )


@functools.lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    """The technique catalog every world understands. Parameters are placeholders for B's."""
    num = lambda default, low, high: ParamSpec(default=default, min=low, max=high)
    keep = {"keep_fraction": num(0.5, 0.1, 0.9)}
    return Catalog(entries=(
        _entry("no_mitigation", "Plain fine-tuning on the stream. The baseline.", "none", True),
        _entry("reference_arm", "Replay plus per-domain learning-rate re-warm. The arm to beat.",
               "reference", True, replay_rate=num(0.1, 0.01, 0.5)),
        _entry("replay", "Mix a share of general text into every batch.", "other",
               rate=num(0.1, 0.01, 0.5)),
        _entry("lora", "Train low-rank adapters instead of the full weights.", "other",
               rank=num(16, 4, 64)),
        _entry("layer_freeze", "Freeze the lowest layers.", "other", n_frozen=num(12, 1, 24)),
        _entry("distill", "Add a distillation loss against the base model.", "other",
               weight=num(0.5, 0.05, 2.0)),
        _entry("weight_avg", "Average the trained weights with the base weights.", "other",
               alpha=num(0.5, 0.1, 0.9)),
        _entry("select_random", "Train on a random share of each batch. Control for selection.",
               "random_selection", True, **keep),
        _entry("select_loss", "Keep the examples with the highest loss.", "selection_signal",
               **keep),
        _entry("select_activation", "Keep the examples an activation-based signal ranks highest.",
               "selection_signal", **keep),
    ))


def default_prereg(world: World | None = None) -> Prereg:
    """PREREG thresholds that match a world's noise (0.8 points when no world is given).

    The world draws in points; the PREREG, like the rows, is in fractions."""
    return Prereg(
        primary_metric=PRIMARY_METRIC, noise_sd=(world.noise_sd if world else 0.8) / 100,
        decision_multiplier=2.0, acquisition_floor_delta=0.05, collapse_share_limit=0.9,
        screen_seeds=(0,), replicate_seeds=(0, 1, 2),
    )


def default_stream() -> tuple[StreamDomain, ...]:
    """The fast bench stream proposed in DIRECTION.md: code then biomedical, 20M tokens each."""
    half = BENCH_TOKENS // 2
    return (StreamDomain(name="code", tokens=half), StreamDomain(name="biomedical", tokens=half))


def make_spec(techniques: Mapping[str, Mapping[str, Any]] | Iterable[str] = (), seed: int = 0,
              **fields: Any) -> RunSpec:
    """A spec on the default stream.

    `techniques` is a mapping of name to parameters (used as given), or just names, which get
    the catalog's default parameters so that one configuration always has one hash.
    """
    if not isinstance(techniques, Mapping):
        catalog = default_catalog()
        techniques = {
            n: {k: p.default for k, p in catalog.get(n).params.items()} if catalog.has(n) else {}
            for n in techniques
        }
    name = fields.pop("name", f"sim-{arm_label(techniques)}-s{seed}")
    return build_spec(name, seed, default_stream(), techniques, **fields)


# --------------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------------


class SimExecutor:
    """`roles.Executor` over a `World`, on an injected clock, writing `rows.jsonl`.

    * Runs take `world.wall_seconds(spec)` of clock time, `concurrency` at once. `poll` writes
      and returns the rows whose finish time has passed.
    * Idempotent by `spec_hash`: a spec in flight, or with a finished, invalid or scientifically
      failed row, is not launched again (`duplicate_submits` counts those). A spec whose last
      row is an infrastructure or timeout failure may be submitted again.
    * State survives a restart: the queue and the launch counts are kept next to the rows file,
      so a new `SimExecutor` on the same path continues where the dead one stopped. `launches`
      maps spec hash to times launched; a kill-and-resume test asserts on it.

    Fault injection: `set_unreachable`, `hang`, `fail_spec`, `truncate_next_row`.
    """

    def __init__(self, world: World, rows_path: str | Path, clock: Clock, *, concurrency: int = 8,
                 label_collapse: bool = True) -> None:
        self.world = world
        self.rows_path = Path(rows_path)
        self.clock = clock
        self.concurrency = concurrency
        self.label_collapse = label_collapse
        self.duplicate_submits = 0
        self._unreachable = 0
        self._hang_all = False
        self._truncate_next = False
        self._state_path = self.rows_path.with_name(self.rows_path.name + ".simstate.json")
        state = self._load_state()
        self._jobs: list[dict[str, Any]] = state.get("jobs", [])
        self.launches: dict[str, int] = state.get("launches", {})
        self._forced: dict[str, list[Any]] = state.get("forced", {})
        self._hung: set[str] = set(state.get("hung", []))

    # ---- fault injection -------------------------------------------------------------------

    def set_unreachable(self, calls: int) -> None:
        """The next `calls` calls to `submit` or `poll` raise `ExecutorUnreachable`."""
        self._unreachable = calls

    def hang(self, spec_hash_value: str | None = None) -> None:
        """Never finish one run (by spec hash), or every run when called with no argument."""
        if spec_hash_value is None:
            self._hang_all = True
        else:
            self._hung.add(spec_hash_value)
        self._save_state()

    def fail_spec(self, spec_hash_value: str, kind: FailureKind = "infra", times: int = 1,
                  reason: str = "injected failure") -> None:
        """Make the next `times` attempts of one spec fail with the given kind."""
        self._forced[spec_hash_value] = [kind, times, reason]
        self._save_state()

    def truncate_next_row(self) -> None:
        """Cut the next row off mid-line, as a crash during the write would. It is not returned
        by that poll; the following poll repairs the file and delivers the row."""
        self._truncate_next = True

    # ---- Executor protocol -----------------------------------------------------------------

    def healthy(self) -> bool:
        return self._unreachable == 0

    def estimate_cost(self, spec: RunSpec) -> float:
        return self.world.cost_usd(spec)

    def submit(self, specs: Sequence[RunSpec]) -> None:
        self._maybe_unreachable()
        latest = {r.spec_hash: r for r in load_rows(self.rows_path).rows}
        for spec in specs:
            key = spec_hash(spec)
            if not self._may_launch(key, latest.get(key)):
                self.duplicate_submits += 1
                continue
            self.launches[key] = self.launches.get(key, 0) + 1
            self._jobs.append({
                "spec_hash": key, "spec": spec.model_dump(mode="json"),
                "attempt": self.launches[key], "finish_at": self._finish_time(spec).isoformat(),
            })
        self._save_state()

    def poll(self) -> list[ResultRow]:
        self._maybe_unreachable()
        now = self.clock.now()
        due = [j for j in self._jobs if self._is_due(j, now)]
        rows = []
        for job in sorted(due, key=lambda j: (j["finish_at"], j["spec_hash"])):
            row = self._row_for(job)
            if self._truncate_next:
                self._write_truncated(row)
                break
            append_jsonl(self.rows_path, row)
            self._jobs.remove(job)
            rows.append(from_a1_row(row))
        self._save_state()
        return rows

    def in_flight(self) -> list[str]:
        return [j["spec_hash"] for j in self._jobs]

    # ---- internals -------------------------------------------------------------------------

    def _maybe_unreachable(self) -> None:
        if self._unreachable > 0:
            self._unreachable -= 1
            raise ExecutorUnreachable("simulated: the executor did not answer")

    def _may_launch(self, key: str, last: ResultRow | None) -> bool:
        if key in self.in_flight():
            return False
        return last is None or retryable(last)

    def _finish_time(self, spec: RunSpec) -> datetime:
        """Start when one of the `concurrency` slots is free, finish one run length later."""
        now = self.clock.now()
        busy = sorted(datetime.fromisoformat(j["finish_at"]) for j in self._jobs)
        busy = [t for t in busy if t > now]
        start = now if len(busy) < self.concurrency else busy[len(busy) - self.concurrency]
        return start + timedelta(seconds=self.world.wall_seconds(spec))

    def _is_due(self, job: Mapping[str, Any], now: datetime) -> bool:
        if self._hang_all or job["spec_hash"] in self._hung:
            return False
        return datetime.fromisoformat(job["finish_at"]) <= now

    def _row_for(self, job: Mapping[str, Any]) -> a1.ResultRow:
        spec = RunSpec.model_validate(job["spec"])
        forced = self._forced.get(job["spec_hash"])
        if forced and forced[1] > 0:
            if not self._truncate_next:
                forced[1] -= 1
            return self.world.failed_row_a1(spec, forced[0], forced[2], job["attempt"])
        return self.world.simulate_a1(spec, job["attempt"], self.label_collapse)

    def _write_truncated(self, row: a1.ResultRow) -> None:
        line = json.dumps(row.model_dump(mode="json"), sort_keys=True)
        self.rows_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rows_path, "ab") as handle:
            handle.write(line[: len(line) // 2].encode("utf-8"))
        self._truncate_next = False

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        return json.loads(self._state_path.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        atomic_write_json(self._state_path, {
            "jobs": self._jobs, "launches": self.launches, "forced": self._forced,
            "hung": sorted(self._hung),
        })
