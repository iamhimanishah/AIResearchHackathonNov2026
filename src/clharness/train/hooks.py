"""Pluggable hooks for the training loop.

The loop itself only knows how to (a) present the domain stream, (b) take optimizer steps,
(c) count gradient tokens, (d) call hooks. Everything that *measures* (evals, activation
capture, SAE statistics, CKA) and everything that will later *intervene* (Phase 3 selection /
replay / protection policies) is a hook. Keep the loop unchanged; add hooks.

Lifecycle:
  on_run_start(ctx)                once, after model/data are loaded, before step 0
  on_domain_start(ctx, domain)     when the stream switches to a new domain
  select_batch(ctx, batch) -> batch|None   (policy hook) may filter/reweight/replace a
                                   micro-batch; return None to skip it. Default: identity.
  wants_activations() -> [layers]  layers whose residuals the loop should capture during the
                                   training forward (detached) for on_micro_batch
  on_micro_batch(ctx, step, domain_index, chunk_ids, batch, logits, seq_loss, acts)
                                   after each training forward, before backward: detached logits
                                   [B,T,V], per-sequence token-sum losses [B], captured residuals
  on_step_end(ctx, step_stats)     after every optimizer step
  on_checkpoint(ctx, ckpt)         at every checkpoint (every N steps + domain boundaries + end)
  on_domain_end(ctx, domain)
  on_run_end(ctx)
Hooks return dicts of scalars to be logged (or None).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class CheckpointInfo:
    index: int  # 0 = base model before any training
    step: int
    gradient_tokens: int
    domain: str  # domain currently being trained (or "base")
    domain_index: int
    is_domain_boundary: bool
    path: str | None  # weights path if saved
    is_tier2: bool


@dataclass
class StepStats:
    step: int
    domain: str
    loss: float
    lr: float
    grad_norm: float
    gradient_tokens: int  # cumulative
    tokens_in_domain: int
    seqs: int


@dataclass
class RunContext:
    """Everything a hook may need. Populated by the loop."""

    cfg: Any  # RunConfig
    model: torch.nn.Module
    tokenizer: Any
    device: str
    data_manifest: dict[str, Any]
    eval_manifest: dict[str, Any] | None
    output_dir: str
    autocast: Any = None  # zero-arg factory returning a context manager (bf16 autocast or nullcontext)
    state: dict[str, Any] = field(default_factory=dict)  # scratch space shared across hooks

    def ac(self):
        import contextlib

        return self.autocast() if self.autocast is not None else contextlib.nullcontext()


@runtime_checkable
class Hook(Protocol):
    name: str

    def on_run_start(self, ctx: RunContext) -> dict[str, float] | None: ...
    def on_domain_start(self, ctx: RunContext, domain: str) -> dict[str, float] | None: ...
    def select_batch(self, ctx: RunContext, batch: torch.Tensor) -> torch.Tensor | None: ...
    def on_step_end(self, ctx: RunContext, stats: StepStats) -> dict[str, float] | None: ...
    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo) -> dict[str, float] | None: ...
    def on_domain_end(self, ctx: RunContext, domain: str) -> dict[str, float] | None: ...
    def on_run_end(self, ctx: RunContext) -> dict[str, float] | None: ...


class BaseHook:
    """No-op defaults; subclass and override what you need."""

    name = "base"

    def on_run_start(self, ctx: RunContext):
        return None

    def on_domain_start(self, ctx: RunContext, domain: str):
        return None

    def select_batch(self, ctx: RunContext, batch: torch.Tensor):
        return batch

    def on_step_end(self, ctx: RunContext, stats: StepStats):
        return None

    def wants_activations(self) -> list[int]:
        return []

    def on_micro_batch(self, ctx: RunContext, step: int, domain_index: int, chunk_ids, batch, logits, seq_loss, acts):
        return None

    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo):
        return None

    def on_domain_end(self, ctx: RunContext, domain: str):
        return None

    def on_run_end(self, ctx: RunContext):
        return None
