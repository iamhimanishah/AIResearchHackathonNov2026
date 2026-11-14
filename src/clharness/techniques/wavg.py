"""Weight averaging: pull the weights part of the way back toward the base model.

After training on a domain, every trainable weight is moved toward its value before any
training:  theta <- alpha * theta_base + (1 - alpha) * theta.  alpha = 0 changes nothing,
alpha = 1 puts the base model back. The idea is that the averaged model keeps much of what
the domain taught while undoing part of the drift that causes forgetting.

Parameters:
  alpha       weight on the base model, 0..1 (modes boundary and final; default 0.25)
  mode        boundary  average at the end of every domain; training continues from the
                        averaged weights
              final     average once, at the end of the last domain
              ema       keep an exponential moving average (EMA) of the weights during training
                        and put it into the model at the end of every domain
  ema_decay   EMA decay per optimizer step, 0..1 (mode ema only; default 0.99)

Averaging happens in on_domain_end, which the loop calls right before the boundary checkpoint,
so the boundary evals and saved weights measure the averaged model. (on_run_end runs after the
last checkpoint, so anything done there would never be measured.)

Diagnostics, per averaging, in `self.history` and in the next boundary checkpoint's logs
(`wavg/dist_before`, `wavg/dist_after`):
  boundary, final   L2 distance from the base weights, over the averaged parameters, before and
                    after; after == (1 - alpha) * before
  ema               L2 distance between the raw weights and the EMA before the copy; after is 0

Cost: no extra forward passes and no change to gradient tokens. Memory: one CPU copy of the
trainable weights (boundary, final) or one extra float32 copy on the model's device (ema).
Frozen parameters never move, so they are skipped.

Known limits:
  - The optimizer's state (AdamW moments) is not reset after averaging, so the next steps use
    moments gathered on the un-averaged weights.
  - In mode ema, mid-domain checkpoints measure the raw weights; only boundaries see the EMA.
  - The EMA's memory is about 1 / (1 - ema_decay) steps. With ~150 steps per domain an EMA at
    0.999 would still be ~86% base weights at the domain end (0.999^150 ~ 0.86), which is why
    the default is 0.99.
  - With LoRA the averaged parameters are the adapter factors A and B. Averaging the factors is
    not the same as averaging the weight delta B @ A (the product is not linear in the factors).
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field

from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook, CheckpointInfo, RunContext, StepStats


@register
class WeightAveraging(BaseHook):
    name = "wavg"

    class Params(TechniqueParams):
        alpha: float = Field(0.25, ge=0, le=1)  # weight on the base model (modes boundary, final)
        mode: Literal["boundary", "final", "ema"] = "boundary"
        ema_decay: float = Field(0.99, ge=0, le=1)  # mode ema only

    def __init__(self, params: Params):
        self.p = params
        self.params: dict[str, torch.nn.Parameter] = {}  # the parameters this hook averages
        self.base: dict[str, torch.Tensor] = {}  # name -> base value on CPU (boundary, final)
        self.ema: dict[str, torch.Tensor] = {}  # name -> float32 EMA on the model's device (ema)
        self.history: list[dict[str, float]] = []
        self.n_domains = 0
        self.domain_count = 0  # domains started so far; names may repeat, so count, don't compare
        self._pending: dict[str, float] | None = None  # diagnostics for the next boundary ckpt

    def on_run_start(self, ctx: RunContext):
        self.n_domains = len(ctx.cfg.data.domains)
        self.domain_count = 0
        self.history, self._pending = [], None
        # frozen parameters never move, and integer tensors cannot be averaged
        self.params = {
            n: p for n, p in ctx.model.named_parameters() if p.requires_grad and p.is_floating_point()
        }
        with torch.no_grad():
            if self.p.mode == "ema":
                self.base = {}
                self.ema = {n: p.detach().float().clone() for n, p in self.params.items()}
            else:
                self.ema = {}
                self.base = {n: p.detach().to("cpu", copy=True) for n, p in self.params.items()}

    def on_domain_start(self, ctx: RunContext, domain: str):
        self.domain_count += 1

    def on_step_end(self, ctx: RunContext, stats: StepStats):
        if self.p.mode == "ema":
            d = self.p.ema_decay
            with torch.no_grad():
                for n, p in self.params.items():
                    self.ema[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def on_domain_end(self, ctx: RunContext, domain: str):
        last = self.domain_count == self.n_domains
        if self.p.mode == "final" and not last:
            return None
        with torch.no_grad():
            if self.p.mode == "ema":
                before, after = self._copy_ema()
            else:
                before, after = self._average()
        entry = {"domain_index": self.domain_count - 1, "dist_before": before, "dist_after": after}
        self.history.append(entry)
        self._pending = {"dist_before": before, "dist_after": after}
        return dict(self._pending)

    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo):
        # the boundary checkpoint right after on_domain_end measures the averaged model
        if ckpt.is_domain_boundary and self._pending is not None:
            out, self._pending = self._pending, None
            return out
        return None

    def _average(self) -> tuple[float, float]:
        """theta <- alpha * base + (1 - alpha) * theta in place; returns L2 distance from base
        before and after."""
        a = self.p.alpha
        sq_before = sq_after = 0.0
        for n, p in self.params.items():
            base = self.base[n].to(p.device, torch.float32)
            # fp32 arithmetic even for bf16 weights. For fp32 weights .float() is the parameter
            # itself, not a copy, so measure the distance before writing into it.
            cur = p.detach().float()
            sq_before += float((cur - base).pow(2).sum())
            new = base * a + cur * (1.0 - a)
            p.copy_(new.to(p.dtype))
            sq_after += float((p.detach().float() - base).pow(2).sum())
        return sq_before**0.5, sq_after**0.5

    def _copy_ema(self) -> tuple[float, float]:
        """Put the EMA into the model; returns L2 distance between raw weights and EMA before
        the copy, and 0 after."""
        sq = 0.0
        for n, p in self.params.items():
            sq += float((p.detach().float() - self.ema[n]).pow(2).sum())
            p.copy_(self.ema[n].to(p.dtype))
        return sq**0.5, 0.0
