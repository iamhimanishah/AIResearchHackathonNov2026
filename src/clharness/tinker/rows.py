"""The C5 row: one JSONL line per Tinker run, in `results/tinker/rows.jsonl`.

This is deliberately NOT the shared `ResultRow`: the C5 tasks are not the protected evals
(analysis/PREREG.md section 10 marks the track exploratory), so a parallel format keeps the
confirmatory contract untouched. Rows carry the accuracy matrix R[stage, task] and point at
the per-item correctness file under `results/tinker/items/`, which the paired bootstrap
resamples.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TINKER_SCHEMA_VERSION = 1
ROWS_PATH = Path("results/tinker/rows.jsonl")
ITEMS_DIR = Path("results/tinker/items")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraCfg(_Strict):
    rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True


class Hyperparameters(_Strict):
    lr: float = 3e-4
    batch_size: int = 32
    max_len: int = 256
    lr_schedule: str = "constant"
    replay_per_task: int | None = None  # set on the replay-200 arm


class Tokens(_Strict):
    train: int = 0  # tokens sent through forward_backward (input + target)
    prefill: int = 0  # tokens prefilled by compute_logprobs calls (uncached upper bound)
    sample: int = 0  # generated tokens; C5 never samples


class TinkerRow(_Strict):
    schema_version: Literal[1] = TINKER_SCHEMA_VERSION
    arm: str  # naive | replay-200 | joint (| a technique arm, later)
    seed: int
    model: str
    lora: LoraCfg = LoraCfg()
    epochs: int = 2
    hyperparameters: Hyperparameters = Hyperparameters()
    template_sha256: str = ""
    manifest_sha256: str = ""
    manifest: str = "manifests/tinker-c5.json"
    smoke: bool = False
    n_train: int = 0
    n_eval: int = 0
    # R[s][t] = accuracy on task t after stage s; null for tasks not yet trained.
    r_matrix: list[list[float | None]] = Field(default_factory=list)
    # Realized replay share per stage, in examples and in tokens (replay-200 only).
    replay_share_examples: list[float] = Field(default_factory=list)
    replay_share_tokens: list[float] = Field(default_factory=list)
    items_path: str | None = None  # .npy of shape (n_stages, n_tasks, n_eval), NaN = unscored
    tokens: Tokens | None = None
    est_usd: float | None = None
    status: Literal["finished", "failed", "invalid"]
    reason: str | None = None
    git_sha: str = "unknown"
    started_at: str | None = None
    finished_at: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TinkerRow:
        if self.status == "finished":
            missing = [f for f in ("tokens", "items_path") if getattr(self, f) is None]
            if missing or not self.r_matrix:
                raise ValueError(f"finished row is missing {missing or ['r_matrix']}")
        elif not self.reason:
            raise ValueError(f"a {self.status} row needs a reason")
        return self


def read_tinker_rows(path: str | Path = ROWS_PATH) -> list[TinkerRow]:
    rows: list[TinkerRow] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(TinkerRow.model_validate_json(line))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
    return rows


def append_tinker_row(path: str | Path, row: TinkerRow) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")


def write_tinker_rows(path: str | Path, rows: Iterable[TinkerRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r.model_dump(mode="json"), sort_keys=True) + "\n"
                            for r in rows))


def spent_usd(path: str | Path = ROWS_PATH) -> float:
    """USD already committed to C5: sum of est_usd over every recorded run (failed included —
    a failed run can still have spent tokens)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return sum(r.est_usd or 0.0 for r in read_tinker_rows(p))
