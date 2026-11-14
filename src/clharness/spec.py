"""The contract between workstreams: the run spec (input to one run) and the result row (output).

Everything outside the bench core depends on the harness only through these two formats
(docs/DIRECTION.md, "The contract between workstreams"). The executor turns a `RunSpec` into a
`ResultRow`; analysis, the agent and the results page read rows from a JSONL file.

Units, so nobody has to guess:
  * scores and deltas are fractions in [0, 1] (accuracy 0.68, a drop of 5 points is -0.05)
  * perplexity ratios are trained / base; below 1 means the model learned the domain
  * tokens are gradient tokens; time is seconds; sizes are bytes

This module imports only pydantic and the standard library, so it is safe to import from the
agent, the page and the analysis code without pulling in torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

# Optimizer fields a spec may override. Mirrors config.OptimCfg plus the per-domain re-warm
# fields that task A4 adds. Kept here as a literal so this module does not import config.
OPTIM_OVERRIDE_KEYS = frozenset({
    "lr", "weight_decay", "betas", "warmup_steps", "schedule", "grad_clip", "batch_size",
    "micro_batch_size", "schedule_scope", "rewarm_steps", "min_lr_ratio",
})

# Fields that label a spec but do not change what is run. They are left out of the hash, so a
# renamed or annotated copy of a spec is still recognised as the same experiment.
_UNHASHED_SPEC_FIELDS = ("name", "notes", "parent_spec_hash")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------------
# Run spec
# --------------------------------------------------------------------------------------------


class StreamDomain(_Strict):
    name: str  # key into the data manifest's domains, e.g. "code"
    tokens: int = Field(gt=0)  # gradient tokens to spend on this domain


class TechniqueSpec(_Strict):
    name: str  # plugin name under clharness.techniques, e.g. "replay"
    params: dict[str, Any] = Field(default_factory=dict)


class RunSpec(_Strict):
    """One run. Everything not listed here comes from the frozen bench config."""

    name: str  # human-readable label; not part of the hash
    seed: int
    bench: str = "fast"  # which frozen bench config: configs/bench/<bench>.yaml
    stream: list[StreamDomain] = Field(min_length=1)  # domains in presentation order
    optim: dict[str, Any] = Field(default_factory=dict)  # overrides of the bench optimizer
    # Stored sorted by name: the order techniques are listed in carries no meaning, so two
    # orderings of one set are the same experiment and get the same hash.
    techniques: list[TechniqueSpec] = Field(default_factory=list)
    notes: str = ""  # free text; not part of the hash
    parent_spec_hash: str | None = None  # the spec this one follows up on; not part of the hash

    @field_validator("techniques")
    @classmethod
    def _sorted(cls, v: list[TechniqueSpec]) -> list[TechniqueSpec]:
        return sorted(v, key=lambda t: t.name)

    @model_validator(mode="after")
    def _check(self) -> RunSpec:
        unknown = set(self.optim) - OPTIM_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"unknown optim override(s): {sorted(unknown)}")
        names = [t.name for t in self.techniques]
        if len(names) != len(set(names)):
            raise ValueError(f"a technique is listed twice: {names}")
        return self

    def technique_names(self) -> list[str]:
        return [t.name for t in self.techniques]

    def gradient_tokens(self) -> int:
        return sum(d.tokens for d in self.stream)

    def spec_hash(self) -> str:
        """Idempotency key: sha256 of the canonical JSON of the fields that affect the run."""
        payload = self.model_dump(mode="json", exclude=set(_UNHASHED_SPEC_FIELDS))
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace. Lists keep their order (it is meaningful)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# --------------------------------------------------------------------------------------------
# Result row
# --------------------------------------------------------------------------------------------


class TaskRetention(_Strict):
    base: float  # score at checkpoint 0 (the base model, under this run's precision)
    final: float  # score at the end of the stream
    delta: float  # final - base; negative is forgetting
    n_items: int = Field(gt=0)


class HalfRetention(_Strict):
    """Retention on one half of the protected set."""

    per_task: dict[str, TaskRetention]
    mean_delta: float  # unweighted mean of the per-task deltas


class BoundaryRetention(_Strict):
    """Mean retention after one domain. Per-checkpoint detail lives in the file at `items_path`."""

    domain: str  # the domain that just finished
    gradient_tokens: int  # cumulative
    selection_mean_delta: float
    report_mean_delta: float


class Retention(_Strict):
    # The protected set is split in two (docs/DIRECTION.md, rules for experiments). Decisions
    # about which configurations to pursue use `selection`; reported numbers use `report`.
    selection: HalfRetention
    report: HalfRetention
    boundaries: list[BoundaryRetention] = Field(default_factory=list)


class Acquisition(_Strict):
    ppl_ratio: dict[str, float]  # per trained domain: held-out perplexity, final / base
    mean_ppl_ratio: float  # geometric mean over the trained domains
    general_ppl_ratio: float | None = None  # untrained general-web slice; above 1 is forgetting
    # Per trained domain: held-out perplexity at the boundary that ends that domain's own
    # training, / base. It measures what was learned before later domains could erase it, which
    # the final ratio mixes with forgetting (analysis/PREREG.md, section 4). Empty in rows
    # written before this field existed.
    own_boundary_ppl_ratio: dict[str, float] = Field(default_factory=dict)
    mean_own_boundary_ppl_ratio: float | None = None  # geometric mean of the above


class Cost(_Strict):
    gradient_tokens: int
    wall_clock_s: float
    gpu_type: str  # e.g. "H100"; "none" for fake rows
    update_bytes: int | None = None  # size of the weight delta that would ship
    trainable_params: int | None = None
    extra_forward_tokens: int = 0  # tokens run forward-only (teacher passes, scoring passes)
    est_usd: float | None = None


class ResultRow(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    spec: RunSpec
    spec_hash: str
    git_sha: str  # commit of the code that produced the row
    status: Literal["finished", "failed", "invalid"]
    reason: str | None = None  # required unless finished
    backend: Literal["modal", "tinker", "local", "fake"]
    attempt: int = Field(default=1, ge=1)
    started_at: str | None = None  # ISO 8601, UTC
    finished_at: str | None = None
    retention: Retention | None = None
    acquisition: Acquisition | None = None
    cost: Cost | None = None
    # Per multiple-choice task: share of the most common answer at the end of the stream.
    # Near 1.0 means the model collapsed to one answer (see docs/baselines.md).
    answer_share: dict[str, float] = Field(default_factory=dict)
    items_path: str | None = None  # pointer to per-item outcomes; not stored in the row

    @model_validator(mode="after")
    def _check(self) -> ResultRow:
        if self.spec_hash != self.spec.spec_hash():
            raise ValueError("spec_hash does not match the spec")
        if self.status == "finished":
            missing = [f for f in ("retention", "acquisition", "cost") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"finished row is missing {missing}")
        elif not self.reason:
            raise ValueError(f"a {self.status} row needs a reason")
        return self


# --------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------


def load_spec(path: str | Path) -> RunSpec:
    return RunSpec.model_validate_json(Path(path).read_text())


def read_rows(path: str | Path) -> list[ResultRow]:
    """Read a JSONL file of result rows. Raises ValueError naming the first bad line."""
    rows: list[ResultRow] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(ResultRow.model_validate_json(line))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
    return rows


def write_rows(path: str | Path, rows: Iterable[ResultRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row_to_line(r) for r in rows))


def append_row(path: str | Path, row: ResultRow) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(row_to_line(row))


def row_to_line(row: ResultRow) -> str:
    return json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"


def json_schemas() -> dict[str, dict[str, Any]]:
    """File name -> JSON Schema, as committed under configs/schema/."""
    return {
        "run_spec.schema.json": RunSpec.model_json_schema(),
        "result_row.schema.json": ResultRow.model_json_schema(),
    }
