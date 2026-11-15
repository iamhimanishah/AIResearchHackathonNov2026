"""Shared data contracts for the agent package. Every other agent module codes against this file.

The run spec and the result row are task A1's (`src/clharness/spec.py`). This module is the
only place in the agent package that knows their layout:

  * `RunSpec`, `StreamDomain`, `TechniqueSpec` are A1's classes, re-exported. The agent builds
    specs with `build_spec` (techniques sorted by name, so one configuration has one hash) and
    reads them with `technique_map`, `technique_names` and `arm_of`.
  * `spec_hash` is A1's `RunSpec.spec_hash`. `config_hash` is the same with the seed left out.
  * Rows are stored as A1 rows. `load_rows` turns each into the agent's `ResultRow` view
    (`from_a1_row`); `row_hash` is the sha256 of the A1 row, computed on read.
  * A1 has no failure kind. A failed row whose reason starts with `infrastructure:` (or
    `infra:`) or `timeout:` is retried once; any other failed or invalid row is final.

Units are A1's: retention is a fraction against base (`-0.05` is five points of forgetting,
negative means forgetting). Acquisition is held-out perplexity end / base per trained domain
(below 1 means the domain was learned); `acquisition_mean` is A1's geometric mean. `answer_share`
is the share of items given the single most common answer, per multiple-choice task.

The firewall. A `ResultRow` holds both halves of the protected set. Model agents before the
freeze get a `SelectionRow`, a separate class with no report-half field at all.
`to_selection_view` is free. `to_report_view` needs a `FreezeToken`, and a token can only be
minted from a decision log that already holds the Conductor's FREEZE decision.

Import cost: standard library, pydantic v2, pyyaml and `clharness.spec` (itself pydantic only).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from clharness import spec as a1

# --------------------------------------------------------------------------------------------
# Names shared with the bench
# --------------------------------------------------------------------------------------------

MC_TASKS: tuple[str, ...] = ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal")
PRIMARY_METRIC = "mean_delta"
NO_MITIGATION = "no_mitigation"
MODEL_AGENTS: tuple[str, ...] = ("planner", "analyst", "scout", "reporter")
CONDUCTOR = "conductor"
FREEZE_STATE_ID = "freeze"  # must equal `states.FREEZE.id`; a kernel test checks it
GENESIS_HASH = "0" * 64
WHOLE_PROPOSAL = "*"  # `Dropped.spec_hash` value for a rule that rejects the whole proposal


class ContractError(ValueError):
    """A value breaks a contract in this module."""


class FirewallError(RuntimeError):
    """Report-half data was requested without a freeze, or handed to a pre-freeze agent."""


class PreregError(ValueError):
    """PREREG thresholds are missing or do not parse. PREFLIGHT halts on this."""


class Frozen(BaseModel):
    """Base for every contract: unknown keys are errors and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Phase(StrEnum):
    """The three phases of the loop, in order."""

    SCREEN = "screen"
    REPLICATE = "replicate"
    COMBINE = "combine"


# --------------------------------------------------------------------------------------------
# Canonical JSON and hashes of the agent's own records (specs use A1's hash)
# --------------------------------------------------------------------------------------------


def _canon(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canon(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _canon(value.value)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("NaN and infinity are not allowed in hashed data; use null")
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        return {str(k): _canon(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_canon(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    raise ContractError(f"cannot canonicalise {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Stable JSON text: sorted keys, no spaces, ASCII, `1.0` written as `1`, no NaN."""
    return json.dumps(
        _canon(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(value: Any) -> str:
    """sha256 of `canonical_json(value)`."""
    return sha256_hex(canonical_json(value))


def utc_iso(moment: datetime | None = None) -> str:
    """ISO 8601 UTC timestamp to the second. Pass the injected clock's `now()` in agent code."""
    moment = moment or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def arm_label(technique_names: Iterable[str]) -> str:
    """Readable name of a configuration: `no_mitigation`, or sorted technique names joined by +."""
    names = sorted(set(technique_names) - {NO_MITIGATION})
    return "+".join(names) if names else NO_MITIGATION


# --------------------------------------------------------------------------------------------
# Run spec: A1's class, with the helpers the agent needs
# --------------------------------------------------------------------------------------------

RunSpec = a1.RunSpec
StreamDomain = a1.StreamDomain
TechniqueSpec = a1.TechniqueSpec

_SPEC_HASH_EXCLUDE = {"name", "notes", "parent_spec_hash"}


def technique_map(spec: RunSpec) -> dict[str, dict[str, Any]]:
    """`{name: params}` of a spec's techniques. No mitigation is the empty mapping."""
    return {t.name: dict(t.params) for t in spec.techniques}


def technique_names(spec: RunSpec) -> tuple[str, ...]:
    return tuple(sorted(t.name for t in spec.techniques))


def arm_of(spec: RunSpec) -> str:
    return arm_label(technique_names(spec))


def build_spec(
    name: str, seed: int, stream: Iterable[StreamDomain],
    techniques: Mapping[str, Mapping[str, Any]] | None = None, **fields: Any,
) -> RunSpec:
    """The one way the agent makes a spec. Techniques are sorted by name, so one configuration
    has one hash; `{"no_mitigation": {}}` means no technique and cannot be combined."""
    techniques = dict(techniques or {})
    if NO_MITIGATION in techniques:
        if len(techniques) > 1:
            raise ContractError(f"{NO_MITIGATION} cannot be combined with another technique")
        techniques = {}
    listed = [TechniqueSpec(name=k, params=dict(techniques[k])) for k in sorted(techniques)]
    return RunSpec(name=name, seed=seed, stream=list(stream), techniques=listed, **fields)


def spec_hash(spec: RunSpec) -> str:
    """A1's idempotency key (`RunSpec.spec_hash`). Leaves out name, notes and lineage."""
    return spec.spec_hash()


def config_hash(spec: RunSpec) -> str:
    """Key of a configuration across seeds: `spec_hash` with the seed left out as well."""
    payload = spec.model_dump(mode="json", exclude=_SPEC_HASH_EXCLUDE | {"seed"})
    return sha256_hex(a1.canonical_json(payload))


# --------------------------------------------------------------------------------------------
# Result row: A1's row on disk, the agent's views in memory
# --------------------------------------------------------------------------------------------

Status = Literal["finished", "failed", "invalid"]
FailureKind = Literal["infra", "science", "timeout"]

# A1 has no failure-kind field. A failed row whose reason starts with one of these prefixes is
# an infrastructure failure or a timeout (re-run once); any other failed or invalid row is final.
FAILURE_PREFIXES: dict[str, FailureKind] = {
    "infrastructure:": "infra", "infra:": "infra", "timeout:": "timeout",
}


def failure_kind_of(status: str, reason: str | None) -> FailureKind | None:
    if status == "finished":
        return None
    text = (reason or "").strip().lower()
    if status == "failed":
        for prefix, kind in FAILURE_PREFIXES.items():
            if text.startswith(prefix):
                return kind
    return "science"


class Boundary(Frozen):
    """Mean retention of one half after one domain of the stream."""

    domain: str
    gradient_tokens: int
    mean_delta: float


class HalfRetention(Frozen):
    """Retention on one half of the protected set, as fractions against base (A1's units)."""

    per_task_delta: dict[str, float]
    mean_delta: float  # unweighted mean over the tasks: the primary metric
    boundaries: tuple[Boundary, ...] = ()


class Cost(Frozen):
    gradient_tokens: int = 0
    wall_seconds: float = 0.0
    gpu_type: str = ""
    update_bytes: int | None = None
    trainable_params: int | None = None
    extra_forward_tokens: int = 0
    usd: float = 0.0


class _RowCommon(BaseModel):
    """Fields every view of a row carries. None of them holds report-half data."""

    spec: RunSpec
    spec_hash: str
    git_sha: str
    status: Status
    reason: str = ""  # free text from the executor: untrusted data in prompts
    failure_kind: FailureKind | None = None
    backend: str
    attempt: int = 1
    retention_selection: HalfRetention | None = None
    acquisition: dict[str, float] = Field(default_factory=dict)  # domain -> ppl end / base
    acquisition_mean: float | None = None  # A1's geometric mean over the trained domains
    general_ppl_ratio: float | None = None
    cost: Cost = Field(default_factory=Cost)
    answer_share: dict[str, float] = Field(default_factory=dict)  # MC task -> top answer share
    row_hash: str  # sha256 of the A1 row as written

    @property
    def answer_share_max(self) -> float | None:
        return max(self.answer_share.values()) if self.answer_share else None


class ResultRow(_RowCommon):
    """Both halves of one A1 row. Only the Operator, the Conductor and the page hold this class.

    Never pass it to a prompt builder: use `to_selection_view`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    retention_report: HalfRetention | None = None


class SelectionRow(_RowCommon):
    """A row with the report half removed by type. The only row class a Planner or Analyst sees."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReportRow(_RowCommon):
    """A row with both halves, stamped with the frozen set it was opened for. Reporter only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retention_report: HalfRetention | None = None
    frozen_set_hash: str


def a1_row_hash(row: a1.ResultRow) -> str:
    return sha256_hex(a1.canonical_json(row.model_dump(mode="json")))


def _half(half: a1.HalfRetention, boundaries: Iterable[Boundary]) -> HalfRetention:
    return HalfRetention(
        per_task_delta={k: v.delta for k, v in half.per_task.items()},
        mean_delta=half.mean_delta, boundaries=tuple(boundaries),
    )


def from_a1_row(row: a1.ResultRow) -> ResultRow:
    """The agent's view of one A1 row. The only place that knows A1's row layout."""
    selection = report = None
    if row.retention is not None:
        bounds = row.retention.boundaries
        selection = _half(row.retention.selection, (
            Boundary(domain=b.domain, gradient_tokens=b.gradient_tokens,
                     mean_delta=b.selection_mean_delta) for b in bounds))
        report = _half(row.retention.report, (
            Boundary(domain=b.domain, gradient_tokens=b.gradient_tokens,
                     mean_delta=b.report_mean_delta) for b in bounds))
    acq = row.acquisition
    cost = Cost()
    if row.cost is not None:
        cost = Cost(
            gradient_tokens=row.cost.gradient_tokens, wall_seconds=row.cost.wall_clock_s,
            gpu_type=row.cost.gpu_type, update_bytes=row.cost.update_bytes,
            trainable_params=row.cost.trainable_params,
            extra_forward_tokens=row.cost.extra_forward_tokens, usd=row.cost.est_usd or 0.0,
        )
    return ResultRow(
        spec=row.spec, spec_hash=row.spec_hash, git_sha=row.git_sha, status=row.status,
        reason=row.reason or "", failure_kind=failure_kind_of(row.status, row.reason),
        backend=row.backend, attempt=row.attempt, retention_selection=selection,
        retention_report=report, acquisition=dict(acq.ppl_ratio) if acq else {},
        acquisition_mean=acq.mean_ppl_ratio if acq else None,
        general_ppl_ratio=acq.general_ppl_ratio if acq else None, cost=cost,
        answer_share=dict(row.answer_share), row_hash=a1_row_hash(row),
    )


def a1_cost(gradient_tokens: int, wall_seconds: float, gpu_type: str, usd: float,
            update_bytes: int | None = None, trainable_params: int | None = None) -> a1.Cost:
    return a1.Cost(
        gradient_tokens=gradient_tokens, wall_clock_s=wall_seconds, gpu_type=gpu_type,
        update_bytes=update_bytes, trainable_params=trainable_params, est_usd=usd,
    )


def a1_failed_row(
    spec: RunSpec, kind: FailureKind, reason: str, *, git_sha: str = "", backend: str = "local",
    attempt: int = 1, cost: a1.Cost | None = None,
) -> a1.ResultRow:
    """An A1 `failed` row whose reason carries the failure kind (see `FAILURE_PREFIXES`)."""
    prefix = {"infra": "infrastructure: ", "timeout": "timeout: ", "science": ""}[kind]
    text = reason if failure_kind_of("failed", reason) == kind else prefix + reason
    return a1.ResultRow(
        spec=spec, spec_hash=spec_hash(spec), git_sha=git_sha, status="failed", reason=text,
        backend=backend, attempt=attempt, cost=cost,
    )


def to_selection_view(row: ResultRow) -> SelectionRow:
    """Drop the report half. Free: anyone may call it."""
    if type(row) is not ResultRow:
        raise FirewallError(f"to_selection_view needs a ResultRow, got {type(row).__name__}")
    return SelectionRow(**{name: getattr(row, name) for name in SelectionRow.model_fields})


def selection_views(rows: Iterable[ResultRow]) -> tuple[SelectionRow, ...]:
    return tuple(to_selection_view(r) for r in rows)


def frozen_set_hash(spec_hashes: Iterable[str]) -> str:
    """Hash of the frozen configuration set: the sorted, de-duplicated spec hashes in it."""
    return hash_obj(sorted(set(spec_hashes)))


_MINT_KEY = object()


class FreezeToken:
    """Proof that the Conductor has frozen the configuration set. Opens the report half.

    It cannot be built directly, parsed from JSON, copied or pickled. The only way to get one is
    `mint_freeze_token(decision_log, frozen_set_hash)`, which succeeds only when the hash chain
    verifies and the log holds a FREEZE decision by the Conductor that records that hash. Only the
    Conductor writes the log, so only the Conductor's freeze can open the report half. Tests of
    the Reporter get a token the same way: append a FREEZE decision to a temporary log first.
    """

    __slots__ = ("_frozen_set_hash",)

    def __init__(self, frozen_set_hash_value: str, _key: object = None) -> None:
        if _key is not _MINT_KEY:
            raise FirewallError(
                "FreezeToken cannot be constructed directly. Use mint_freeze_token(log, hash) "
                "after the Conductor has logged the FREEZE decision."
            )
        object.__setattr__(self, "_frozen_set_hash", frozen_set_hash_value)

    @property
    def frozen_set_hash(self) -> str:
        return self._frozen_set_hash

    def __setattr__(self, name: str, value: Any) -> None:
        raise FirewallError("FreezeToken is immutable")

    def __reduce__(self) -> Any:
        raise FirewallError("FreezeToken cannot be pickled or copied")

    def __repr__(self) -> str:
        return f"FreezeToken({self._frozen_set_hash[:12]})"


def to_report_view(row: ResultRow, token: FreezeToken) -> ReportRow:
    """Open both halves of a row. Needs a real `FreezeToken`; anything else raises."""
    if type(token) is not FreezeToken:
        raise FirewallError("to_report_view needs a FreezeToken minted after the freeze")
    if type(row) is not ResultRow:
        raise FirewallError(f"to_report_view needs a ResultRow, got {type(row).__name__}")
    names = [n for n in ReportRow.model_fields if n != "frozen_set_hash"]
    values = {name: getattr(row, name) for name in names}
    return ReportRow(**values, frozen_set_hash=token.frozen_set_hash)


# --------------------------------------------------------------------------------------------
# Pre-registration and the technique catalog
# --------------------------------------------------------------------------------------------


class Prereg(Frozen):
    """Thresholds the agent reads from `analysis/PREREG.md` (task C1). No defaults on purpose."""

    primary_metric: str  # field of HalfRetention, normally "mean_delta"
    noise_sd: float = Field(gt=0)  # SD of the primary metric across seeds, a fraction (task C3)
    decision_multiplier: float = Field(gt=0)  # a difference must exceed multiplier * noise_sd
    acquisition_floor_delta: float = Field(ge=0)  # allowed relative rise of the perplexity ratio
    collapse_share_limit: float = Field(gt=0, le=1)  # top-answer share above this is invalid
    screen_seeds: tuple[int, ...] = Field(min_length=1)
    replicate_seeds: tuple[int, ...] = Field(min_length=1)

    @property
    def decision_threshold(self) -> float:
        return self.decision_multiplier * self.noise_sd


_YAML_BLOCK = re.compile(r"```ya?ml[^\n]*\n(.*?)```", re.DOTALL)
PREREG_BLOCK_KEY = "agent_thresholds"


def _prereg_candidates(text: str, is_markdown: bool) -> list[Any]:
    blocks = _YAML_BLOCK.findall(text) if is_markdown else [text]
    parsed = []
    for block in blocks:
        try:
            parsed.append(yaml.safe_load(block))
        except yaml.YAMLError:
            continue
    return parsed


def parse_prereg(text: str, is_markdown: bool = True) -> Prereg:
    """Read thresholds from PREREG text.

    Markdown: the first fenced yaml block that is a mapping with the key `agent_thresholds`
    (preferred) or `noise_sd`. Plain YAML: the same mapping at the top level.
    """
    for data in _prereg_candidates(text, is_markdown):
        if not isinstance(data, dict):
            continue
        data = data.get(PREREG_BLOCK_KEY, data)
        if isinstance(data, dict) and "noise_sd" in data:
            try:
                return Prereg.model_validate(data)
            except ValidationError as exc:
                raise PreregError(f"PREREG thresholds do not validate: {exc}") from exc
    raise PreregError(f"no yaml block with `{PREREG_BLOCK_KEY}` or `noise_sd` found in PREREG")


def load_prereg(path: str | Path) -> Prereg:
    """Load thresholds from `PREREG.md` or a `.yaml` file. Raises `PreregError` when absent."""
    path = Path(path)
    if not path.exists():
        raise PreregError(f"PREREG file not found: {path.name}")
    return parse_prereg(path.read_text(encoding="utf-8"), is_markdown=path.suffix.lower() == ".md")


def passes_acquisition_floor(
    acquisition_mean: float, baseline_acquisition_mean: float, delta: float
) -> bool:
    """True when acquisition is no worse than no-mitigation by more than `delta` (relative).

    Both arguments are mean perplexity ratios (lower is better). One definition, used by the
    Gatekeeper, the Analyst, the Conductor's ADVANCE rule and the simulator's oracle.
    """
    return acquisition_mean <= baseline_acquisition_mean * (1.0 + delta)


def is_collapsed(answer_share: Mapping[str, float], limit: float) -> bool:
    """True when any multiple-choice task gives one answer to more than `limit` of its items."""
    return any(share > limit for share in answer_share.values())


def usable(row: Any, prereg: Any) -> bool:
    """A finished row whose numbers may be ranked: finite and not collapsed onto one answer.

    A run can finish and still report NaN (a diverged loss) or answer one letter to everything;
    such a row must never win FREEZE or seed the next phase.
    """
    if row.status != "finished":
        return False
    values = (row.retention_selection.mean_delta, row.acquisition_mean)
    if any(v is not None and not math.isfinite(v) for v in values):
        return False
    if row.retention_selection.mean_delta is None:
        return False
    return not is_collapsed(row.answer_share, prereg.collapse_share_limit)


ParamValue = bool | int | float | str
CatalogRole = Literal["none", "reference", "random_selection", "selection_signal", "other"]


class ParamSpec(Frozen):
    """One technique parameter: its default and what the Gatekeeper allows."""

    default: ParamValue
    min: float | None = None
    max: float | None = None
    choices: tuple[ParamValue, ...] | None = None

    def allows(self, value: Any) -> bool:
        if self.choices is not None:
            return value in self.choices
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return type(value) is type(self.default)
        low_ok = self.min is None or value >= self.min
        return low_ok and (self.max is None or value <= self.max)


class CatalogEntry(Frozen):
    """One technique plugin as the agent sees it.

    `role`: `none` is the no-mitigation control, `reference` the reference arm, `random_selection`
    the control for any `selection_signal` technique, `other` an ordinary technique.
    """

    name: str
    description: str  # one line; shown to the Planner and the Scout
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    cost_multiplier: float = Field(gt=0)  # run cost relative to no mitigation at equal tokens
    is_control: bool
    role: CatalogRole


class Catalog(Frozen):
    entries: tuple[CatalogEntry, ...]

    @model_validator(mode="after")
    def _names_unique(self) -> Catalog:
        names = [e.name for e in self.entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate technique name in catalog")
        return self

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entries)

    def has(self, name: str) -> bool:
        return name in self.names()

    def get(self, name: str) -> CatalogEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def by_role(self, role: CatalogRole) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.entries if e.role == role)

    def controls(self) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.entries if e.is_control)

    def cost_multiplier(self, techniques: Iterable[str]) -> float:
        """Product of the cost multipliers of the named techniques (1.0 for none)."""
        product = 1.0
        for name in techniques:
            product *= self.get(name).cost_multiplier
        return product

    def problems(self, techniques: Mapping[str, Mapping[str, Any]]) -> list[str]:
        """Unknown techniques, unknown parameters and out-of-range values. Empty means valid."""
        found = []
        for name, params in techniques.items():
            if not self.has(name):
                found.append(f"unknown technique: {name}")
                continue
            allowed = self.get(name).params
            for key, value in params.items():
                if key not in allowed:
                    found.append(f"{name}: unknown parameter {key}")
                elif not allowed[key].allows(value):
                    found.append(f"{name}.{key}={value!r} is outside the allowed range")
        return found


# --------------------------------------------------------------------------------------------
# Planner output and Gatekeeper verdict
# --------------------------------------------------------------------------------------------

SpecRole = Literal["control", "treatment", "replicate", "combo"]


class Predicted(Frozen):
    """What the Planner expects, stated before the run. Scored afterwards into a `Lesson`."""

    retention_mean_delta: float  # fraction, selection half
    acquisition_ratio: float  # mean perplexity ratio over the trained domains


class ProposedSpec(Frozen):
    spec: RunSpec
    role: SpecRole
    reason: str
    predicted: Predicted


class Proposal(Frozen):
    """The Planner's only output. It launches nothing: the Gatekeeper decides."""

    phase: Phase
    specs: tuple[ProposedSpec, ...]
    rationale: str
    cites_findings: tuple[str, ...] = ()  # Finding ids the plan relies on

    def spec_hashes(self) -> tuple[str, ...]:
        return tuple(spec_hash(p.spec) for p in self.specs)


class Dropped(Frozen):
    spec_hash: str  # `WHOLE_PROPOSAL` when the rule rejects the proposal as a whole
    rule: str  # stable rule id, for example "budget" or "missing_control"
    reason: str


class Verdict(Frozen):
    """The Gatekeeper's answer. `token` is first on purpose: read it before anything else."""

    token: Literal["ACCEPT", "REJECT", "TRIM"]
    kept: tuple[str, ...] = ()  # spec hashes that may launch
    dropped: tuple[Dropped, ...] = ()

    @model_validator(mode="after")
    def _token_matches_content(self) -> Verdict:
        if self.token == "ACCEPT" and self.dropped:
            raise ValueError("ACCEPT cannot drop a spec; use TRIM")
        if self.token == "REJECT" and (self.kept or not self.dropped):
            raise ValueError("REJECT keeps nothing and states at least one reason")
        if self.token == "TRIM" and not (self.kept and self.dropped):
            raise ValueError("TRIM keeps at least one spec and drops at least one")
        return self


def accepted_specs(proposal: Proposal, verdict: Verdict) -> tuple[RunSpec, ...]:
    """The specs of `proposal` that `verdict` lets through, in proposal order, without repeats."""
    if verdict.token == "REJECT":
        return ()
    kept, seen, out = set(verdict.kept), set(), []
    for item in proposal.specs:
        key = spec_hash(item.spec)
        if key in kept and key not in seen:
            seen.add(key)
            out.append(item.spec)
    return tuple(out)


# --------------------------------------------------------------------------------------------
# Findings, lessons, scout output, memory entries, budget view, vetoes
# --------------------------------------------------------------------------------------------


class CitedNumber(Frozen):
    """A number in a claim, with the row and field it was read from."""

    label: str
    value: float
    row_hash: str
    field_path: str  # for example "retention_selection.per_task_delta.piqa"


class Finding(Frozen):
    id: str
    kind: Literal["effect", "interaction", "failure", "anomaly", "null_result"]
    claim: str
    evidence_row_hashes: tuple[str, ...]
    numbers: tuple[CitedNumber, ...] = ()
    confidence: Literal["low", "medium", "high"]
    author: str  # "analyst", "analyst:deterministic", ...
    created_at: str


def finding_id(kind: str, claim: str, evidence_row_hashes: Iterable[str]) -> str:
    """Deterministic id, so a resumed READ step stores the same finding under the same id."""
    return "F-" + hash_obj([kind, claim, sorted(evidence_row_hashes)])[:10]


class Lesson(Frozen):
    """Something the loop learned about its own judgement, for example a calibration error."""

    id: str
    kind: Literal["calibration", "failure", "process"]
    text: str
    technique: str | None = None
    evidence_row_hashes: tuple[str, ...] = ()
    predicted: float | None = None
    observed: float | None = None
    author: str
    created_at: str


class Source(Frozen):
    """One search result the Scout looked at. `title` and `snippet` are untrusted text."""

    id: str
    query: str
    url: str
    title: str
    snippet: str = ""
    published: str | None = None
    retrieved_at: str
    used: bool = False


class Candidate(Frozen):
    """A technique the Scout found that is not in the catalog. A human reviews it before use."""

    name: str
    description: str
    mechanism: str
    why_not_in_catalog: str
    expected_effect: str
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    est_cost_multiplier: float = Field(gt=0)
    source_ids: tuple[str, ...] = Field(min_length=1)
    review: Literal["pending", "approved", "rejected"] = "pending"
    author: str = "scout"
    created_at: str


class ClaimPrediction(Frozen):
    """What a claim card expects, stated before any run: test arm minus control arm, retention."""

    direction: Literal[-1, 0, 1]  # +1: the test arm forgets less; 0: no difference
    size_points: float = Field(ge=0, le=20)  # rough size of the retention difference, in points


class ClaimCard(Frozen):
    """A published claim the plugins we already have can test (docs/agent/SCOUT-HANDOFF.md 3.2).

    The Scout drafts it from search results; code checks every field (`agents.check_claim`).
    `test_arm` and `control_arm` use catalog techniques only (`{}` is no mitigation); `*_lr` is
    the run learning rate when the claim is about it (None = the bench rate). Every run made from
    a card is exploratory: never part of the win rule or a Holm family (analysis/PREREG.md).
    """

    id: str
    claim: str  # one sentence, as the paper states it (untrusted text)
    source_ids: tuple[str, ...] = Field(min_length=1)
    test_arm: dict[str, dict[str, Any]]
    control_arm: dict[str, dict[str, Any]] = Field(default_factory=dict)
    test_lr: float | None = None
    control_lr: float | None = None
    prediction: ClaimPrediction
    duplicate_of: int | None = None  # id of the `design.CLAIMS` entry it repeats
    status: Literal["proposed", "accepted", "rejected", "tested"] = "proposed"
    author: str = "scout"
    created_at: str

    @property
    def plannable(self) -> bool:
        return self.status == "accepted" and self.duplicate_of is None


def claim_card_id(claim: str, source_ids: Iterable[str]) -> str:
    """`claim-` + hash of the claim text and the sorted source ids."""
    return "claim-" + hash_obj([claim.strip(), sorted(source_ids)])[:10]


class TriedEntry(Frozen):
    """Memory `tried`: one line per spec hash. Built from a SelectionRow, so no report half."""

    spec_hash: str
    arm: str
    seed: int
    status: Status
    mean_delta: float | None = None
    acquisition_mean: float | None = None
    row_hash: str


class FailureEntry(Frozen):
    """Memory `failures`: a run that failed or was invalid, with the executor's reason."""

    spec_hash: str
    arm: str
    seed: int
    failure_kind: FailureKind
    reason: str
    row_hash: str


class BudgetView(Frozen):
    """What the Planner and the Gatekeeper may know about money and time."""

    compute_usd_left: float  # spendable now: cap minus reserve minus spent and committed
    compute_usd_reserve: float  # held back for re-queues and the final replicates
    model_usd_left: float
    minutes_left: float | None  # to the deadline; None when no deadline is set
    concurrency: int
    minutes_per_run: float
    baseline_run_usd: float  # estimated cost of one no-mitigation run


class Veto(Frozen):
    """One line a human appends to `veto.jsonl`. The Conductor reads it at every transition."""

    id: str
    ts: str
    kind: Literal["halt", "freeze_now", "block_technique", "block_spec", "skip_scout",
                  "approve_candidate", "reject_candidate"]
    target: str = ""  # technique name, spec hash or candidate name, when the kind needs one
    reason: str
    author: str


# --------------------------------------------------------------------------------------------
# Citation checker (shared by the Analyst and the Reporter)
# --------------------------------------------------------------------------------------------


def resolve_field_path(obj: Any, path: str) -> Any:
    """Follow a dotted path through models and mappings. Raises `KeyError` when it is absent.

    A path into the report half cannot resolve on a `SelectionRow`, because the field is not
    there. Task names contain `/` but never `.`, so they are safe path segments.
    """
    current = obj
    for part in path.split("."):
        if isinstance(current, BaseModel) and part in type(current).model_fields:
            current = getattr(current, part)
        elif isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise KeyError(path)
    return current


def expand_hash_prefix(prefix: str, known: Iterable[str]) -> str | None:
    """The one hash in `known` that starts with `prefix`, or None if none or several do."""
    matches = [h for h in known if h.startswith(prefix)] if prefix else []
    return matches[0] if len(matches) == 1 else None


def check_numbers(
    numbers: Iterable[CitedNumber], rows_by_hash: Mapping[str, Any], abs_tol: float = 0.00051
) -> list[str]:
    """Confirm each cited number against its row. Returns one problem per number that fails.

    `rows_by_hash` maps full row hashes to `SelectionRow` or `ReportRow`. A cited `row_hash` may
    be a unique prefix. The default tolerance accepts a fraction rounded to three decimals.
    """
    problems = []
    for number in numbers:
        full = expand_hash_prefix(number.row_hash, rows_by_hash)
        if full is None:
            problems.append(f"{number.label}: row {number.row_hash[:12]} not found")
            continue
        try:
            actual = resolve_field_path(rows_by_hash[full], number.field_path)
        except KeyError:
            problems.append(f"{number.label}: no field {number.field_path} in the row")
            continue
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            problems.append(f"{number.label}: {number.field_path} is not a number")
        elif not math.isclose(number.value, actual, abs_tol=abs_tol, rel_tol=0.0):
            problems.append(f"{number.label}: cited {number.value}, row has {actual}")
    return problems


# --------------------------------------------------------------------------------------------
# Files: atomic state writes, tolerant JSONL
# --------------------------------------------------------------------------------------------


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write a whole file so a reader sees the old content or the new, never a mixture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomic write of a model or plain data as indented JSON (for `state.json`)."""
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


@dataclass
class JsonlRead:
    """Result of a tolerant read. `partial` is an unfinished last line (a writer died mid-line)."""

    records: list[dict[str, Any]] = field(default_factory=list)
    partial: str | None = None
    bad_lines: list[int] = field(default_factory=list)  # 1-based numbers of unreadable lines


def _parse_object(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_jsonl(path: str | Path) -> JsonlRead:
    """Read a JSONL file without failing on damage.

    A last line with no newline that does not parse is skipped and reported in `partial`. Any
    other unreadable line is skipped and its number is reported in `bad_lines`. A missing file
    reads as empty.
    """
    out = JsonlRead()
    path = Path(path)
    if not path.exists():
        return out
    segments = path.read_bytes().split(b"\n")
    tail = segments.pop()
    for number, raw in enumerate(segments, start=1):
        if not raw.strip():
            continue
        parsed = _parse_object(raw)
        if parsed is None:
            out.bad_lines.append(number)
        else:
            out.records.append(parsed)
    if tail.strip():
        parsed = _parse_object(tail)
        if parsed is None:
            out.partial = tail.decode("utf-8", errors="replace")
        else:
            out.records.append(parsed)
    return out


def _repair_tail(path: Path) -> None:
    """Make the file end with a newline: finish a complete last line, drop a partial one."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    if _parse_object(data[cut:]) is not None:
        with open(path, "ab") as handle:
            handle.write(b"\n")
    else:
        os.truncate(path, cut)


def append_jsonl(path: str | Path, record: Mapping[str, Any] | BaseModel) -> None:
    """Append one record as one line and fsync. Only the file's single writer may call this.

    A partial last line left by a crash is removed first, so the new line is never glued to it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    data = record.model_dump(mode="json") if isinstance(record, BaseModel) else dict(record)
    line = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with open(path, "ab") as handle:
        handle.write(line.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class RowsRead:
    """Rows that parsed and verified, plus what was skipped and why."""

    rows: list[ResultRow] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    partial: bool = False


def load_rows(path: str | Path) -> RowsRead:
    """Read a JSONL file of A1 rows into agent views. Rows that do not validate are skipped.

    A1's validator checks `spec_hash` against the spec, so a tampered spec is skipped too.
    """
    read = read_jsonl(path)
    out = RowsRead(partial=read.partial is not None)
    out.problems.extend(f"line {n}: not valid JSON" for n in read.bad_lines)
    for index, record in enumerate(read.records):
        try:
            row = a1.ResultRow.model_validate(record)
        except ValidationError as exc:
            out.problems.append(f"record {index}: {exc.errors()[0]['msg']}")
            continue
        out.rows.append(from_a1_row(row))
    return out


def latest_by_spec(rows: Iterable[ResultRow]) -> dict[str, ResultRow]:
    """The latest row for each spec hash: highest `attempt`, then the last one written."""
    latest: dict[str, ResultRow] = {}
    for row in rows:
        old = latest.get(row.spec_hash)
        if old is None or row.attempt >= old.attempt:
            latest[row.spec_hash] = row
    return latest


# --------------------------------------------------------------------------------------------
# Decision log
# --------------------------------------------------------------------------------------------


class Decision(Frozen):
    """One logged decision, with the versions that made it. Hash-chained to the one before."""

    seq: int
    ts: str
    key: str  # idempotency key: one per logical step, for example "screen/gate#2"
    state: str  # `State.id`, for example "screen/plan" or "freeze"
    actor: str  # "conductor", "planner", "gatekeeper", "operator", "analyst", ...
    input_row_hashes: tuple[str, ...] = ()
    prompt_hash: str | None = None  # hash of the full prompt; joins to llm_calls.jsonl
    card_hash: str | None = None  # hash of the agent card (the prompt version) that decided
    skill_hashes: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    proposal: Proposal | None = None
    verdict: Verdict | None = None
    reason: str
    cites_findings: tuple[str, ...] = ()
    data: dict[str, Any] = Field(default_factory=dict)  # step-specific facts, JSON only
    prev_hash: str
    hash: str


class ChainReport(Frozen):
    ok: bool
    count: int
    head: str  # hash of the last valid entry, or GENESIS_HASH for an empty log
    problems: tuple[str, ...] = ()
    truncated_tail: bool = False  # an unfinished last line was found and ignored


def _decision_hash(data: Mapping[str, Any]) -> str:
    body = dict(data)
    body.pop("hash", None)
    return hash_obj(body)


def _check_link(index: int, record: Mapping[str, Any], prev: str, keys: set[str]) -> list[str]:
    problems = []
    try:
        decision = Decision.model_validate(record)
    except ValidationError as exc:
        return [f"entry {index}: does not validate ({exc.errors()[0]['msg']})"]
    if decision.seq != index:
        problems.append(f"entry {index}: seq is {decision.seq}")
    if decision.prev_hash != prev:
        problems.append(f"entry {index}: prev_hash does not match the entry before")
    if decision.hash != _decision_hash(record):
        problems.append(f"entry {index}: hash does not match the content")
    if decision.key in keys:
        problems.append(f"entry {index}: duplicate key {decision.key}")
    return problems


class DecisionLog:
    """Append-only, hash-chained JSONL decision log. One writer: the Conductor.

    * `append` is idempotent by `key`: a resumed step that logs again gets the first entry back
      and nothing is written.
    * `verify` recomputes every hash and link. Editing, removing or reordering an entry breaks it.
      Removing entries from the END is only visible against a head hash stored elsewhere, so the
      Conductor keeps the head in `state.json` and passes it as `expected_head`.
    * A last line cut short by a crash is tolerated on read, reported, and repaired on append.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[Decision]:
        """Every entry that parses, in order. Damage is skipped here; `verify` reports it."""
        out = []
        for record in read_jsonl(self.path).records:
            try:
                out.append(Decision.model_validate(record))
            except ValidationError:
                continue
        return out

    def get(self, key: str) -> Decision | None:
        for decision in self.entries():
            if decision.key == key:
                return decision
        return None

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def last(self) -> Decision | None:
        entries = self.entries()
        return entries[-1] if entries else None

    def head(self) -> str:
        last = self.last()
        return last.hash if last else GENESIS_HASH

    def verify(self, expected_head: str | None = None) -> ChainReport:
        read = read_jsonl(self.path)
        problems = [f"line {n}: not valid JSON" for n in read.bad_lines]
        prev, keys = GENESIS_HASH, set()
        for index, record in enumerate(read.records):
            problems.extend(_check_link(index, record, prev, keys))
            prev = str(record.get("hash", ""))
            keys.add(str(record.get("key", "")))
        if expected_head is not None and prev != expected_head:
            problems.append("head hash differs from the one recorded in state.json")
        return ChainReport(
            ok=not problems, count=len(read.records), head=prev, problems=tuple(problems),
            truncated_tail=read.partial is not None,
        )

    def append(
        self,
        *,
        key: str,
        state: str,
        actor: str,
        reason: str,
        ts: str | datetime,
        input_row_hashes: Sequence[str] = (),
        prompt_hash: str | None = None,
        card_hash: str | None = None,
        skill_hashes: Mapping[str, str] | None = None,
        model: str | None = None,
        proposal: Proposal | None = None,
        verdict: Verdict | None = None,
        cites_findings: Sequence[str] = (),
        data: Mapping[str, Any] | None = None,
    ) -> Decision:
        """Append one decision, or return the existing one with the same `key`."""
        entries = self.entries()
        for existing in entries:
            if existing.key == key:
                return existing
        body = {
            "seq": len(entries),
            "ts": utc_iso(ts) if isinstance(ts, datetime) else ts,
            "key": key, "state": str(state), "actor": actor,
            "input_row_hashes": tuple(input_row_hashes),
            "prompt_hash": prompt_hash, "card_hash": card_hash,
            "skill_hashes": dict(skill_hashes or {}), "model": model,
            "proposal": proposal, "verdict": verdict, "reason": reason,
            "cites_findings": tuple(cites_findings), "data": json.loads(canonical_json(data or {})),
            "prev_hash": entries[-1].hash if entries else GENESIS_HASH,
        }
        draft = Decision(**body, hash="")
        dumped = draft.model_dump(mode="json")
        dumped["hash"] = _decision_hash(dumped)
        append_jsonl(self.path, dumped)
        return Decision.model_validate(dumped)


def mint_freeze_token(log: DecisionLog, frozen_set_hash_value: str) -> FreezeToken:
    """Mint the token that opens the report half. See `FreezeToken` for the guard.

    The log must verify and must contain a decision with `state == "freeze"`,
    `actor == "conductor"` and `data["frozen_set_hash"]` equal to the hash asked for.
    """
    report = log.verify()
    if not report.ok:
        raise FirewallError(f"decision log does not verify: {report.problems[0]}")
    for decision in log.entries():
        frozen = decision.data.get("frozen_set_hash")
        is_freeze = decision.state == FREEZE_STATE_ID and decision.actor == CONDUCTOR
        if is_freeze and frozen == frozen_set_hash_value:
            return FreezeToken(frozen_set_hash_value, _MINT_KEY)
    raise FirewallError("no FREEZE decision for this configuration set: the report half stays shut")
