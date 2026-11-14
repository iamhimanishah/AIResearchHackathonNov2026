"""Agent configuration: one strict YAML file, loaded once by the Conductor.

Same rules as `clharness.config`: unknown keys are errors, and anything that changes what the
loop spends or decides has no default. Paths default to the stores in ARCHITECTURE.md section 5.

Model names have no default. A slot left at `None` means "must be set before a real run":
`require_models()` raises and PREFLIGHT halts. Tests use the fake model and never set them.

Each part keeps its own extra settings under `parts.<name>` and validates them itself with
`cfg.part("<name>", MySettings)`, so no part needs to edit this file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import MODEL_AGENTS, Phase

T = TypeVar("T", bound=BaseModel)
PACKAGE_DIR = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """The agent configuration is missing something a real run needs."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(Strict):
    """Every store of ARCHITECTURE.md section 5. Relative paths resolve against `root`."""

    rows: str = "results/rows.jsonl"  # written by the executor (workstream 2)
    state: str = "results/agent/state.json"
    decisions: str = "results/agent/decisions.jsonl"
    llm_calls: str = "results/agent/llm_calls.jsonl"
    memory_dir: str = "results/agent/memory"  # tried, failures, findings, lessons .jsonl
    sources: str = "results/agent/sources.jsonl"
    candidates_dir: str = "results/agent/candidates"  # Scout candidate cards
    report_md: str = "results/agent/report.md"
    report_json: str = "results/agent/report.json"
    veto: str = "results/agent/veto.jsonl"  # appended by a human
    prereg: str = "analysis/PREREG.md"  # task C1
    attribution: str | None = None  # workstream 4 output on the selection half, when it exists
    catalog: str | None = None  # technique catalog file; None = the part that owns it decides
    cards_dir: str | None = None  # None = the package's own cards/
    skills_dir: str | None = None  # None = the package's own skills/


class ModelPrice(Strict):
    """Dollars per million tokens. Used to check the model budget before each call."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class ExecutorCfg(Strict):
    concurrency: int = Field(gt=0)  # runs in flight at once
    minutes_per_run: float = Field(gt=0)  # planning estimate for one bench run
    poll_seconds: float = Field(default=60.0, gt=0)
    max_unreachable: int = Field(default=3, gt=0)  # LAUNCH halts after this many failures
    max_requeues: int = Field(default=1, ge=0)  # per spec, infrastructure failures only


def _default_models() -> dict[str, str | None]:
    return {agent: None for agent in MODEL_AGENTS}


class AgentConfig(Strict):
    root: str = "."  # repository root; relative paths resolve against it
    paths: Paths = Paths()
    compute_budget_usd: float = Field(gt=0)  # hard cap, checked before every launch
    compute_reserve_fraction: float = Field(ge=0, lt=1)  # held back for re-queues and replicates
    model_budget_usd: float = Field(gt=0)  # hard cap, checked before every model call
    deadline: datetime | None  # wall clock; reaching it moves the loop to FREEZE
    autonomy: Literal["full", "veto"]  # "veto": read veto.jsonl at every transition
    max_gate_rejections: int = Field(default=2, ge=0)
    phase_timeout_minutes: dict[Phase, float]
    executor: ExecutorCfg
    scout_enabled: bool
    models: dict[str, str | None] = Field(default_factory=_default_models)
    prices: dict[str, ModelPrice] = Field(default_factory=dict)  # model name -> price
    llm_max_output_tokens: int = Field(default=4000, gt=0)
    parts: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def _known_agents(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        unknown = sorted(set(value) - set(MODEL_AGENTS))
        if unknown:
            raise ValueError(f"unknown model agent(s): {unknown}")
        return {**_default_models(), **value}

    @model_validator(mode="after")
    def _every_phase_has_a_timeout(self) -> AgentConfig:
        missing = [p.value for p in Phase if p not in self.phase_timeout_minutes]
        if missing:
            raise ValueError(f"phase_timeout_minutes is missing {missing}")
        return self

    @property
    def spendable_compute_usd(self) -> float:
        """The compute cap minus the reserve. Proposals are budgeted against this."""
        return self.compute_budget_usd * (1.0 - self.compute_reserve_fraction)

    def path(self, name: str) -> Path:
        """Absolute path of a store, for example `cfg.path("decisions")`."""
        value = getattr(self.paths, name)
        if value is None:
            defaults = {"cards_dir": PACKAGE_DIR / "cards", "skills_dir": PACKAGE_DIR / "skills"}
            if name not in defaults:
                raise ConfigError(f"paths.{name} is not set")
            return defaults[name]
        path = Path(value)
        return path if path.is_absolute() else Path(self.root).resolve() / path

    def memory_path(self, store: str) -> Path:
        """Path of one memory file: `tried`, `failures`, `findings` or `lessons`."""
        return self.path("memory_dir") / f"{store}.jsonl"

    def require_models(self, agents: tuple[str, ...] = MODEL_AGENTS) -> dict[str, str]:
        """Model name per agent. Raises if a slot is unset or a model has no price."""
        missing = [a for a in agents if not self.models.get(a)]
        if missing:
            raise ConfigError(f"no model set for: {', '.join(missing)} (set `models`)")
        chosen = {a: str(self.models[a]) for a in agents}
        unpriced = sorted({m for m in chosen.values() if m not in self.prices})
        if unpriced:
            raise ConfigError(f"no price set for model(s): {', '.join(unpriced)}")
        return chosen

    def part(self, name: str, schema: type[T]) -> T:
        """Validate and return `parts.<name>` with the part's own settings model."""
        try:
            return schema.model_validate(self.parts.get(name, {}))
        except ValidationError as exc:
            raise ConfigError(f"parts.{name} is invalid: {exc}") from exc


def load_agent_config(path: str | Path, root: str | Path | None = None) -> AgentConfig:
    """Load the YAML config. `root` overrides the file's `root` (useful for a temp directory)."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if root is not None:
        raw["root"] = str(root)
    return AgentConfig.model_validate(raw)


def config_for_tests(root: str | Path, **overrides: Any) -> AgentConfig:
    """A complete config rooted in a temp directory, for tests of any part.

    Budgets are small, there is no deadline, models are unset and the fake model is priced.
    PREREG is expected as `prereg.yaml` under `root`. Pass keyword overrides for anything else.
    """
    base: dict[str, Any] = {
        "root": str(root),
        "paths": {"prereg": "prereg.yaml"},
        "compute_budget_usd": 200.0,
        "compute_reserve_fraction": 0.15,
        "model_budget_usd": 5.0,
        "deadline": None,
        "autonomy": "veto",
        "phase_timeout_minutes": {"screen": 240.0, "replicate": 240.0, "combine": 180.0},
        "executor": {"concurrency": 8, "minutes_per_run": 30.0},
        "scout_enabled": False,
        "prices": {"fake": {"input_per_mtok": 1.0, "output_per_mtok": 4.0}},
    }
    base.update(overrides)
    return AgentConfig.model_validate(base)
