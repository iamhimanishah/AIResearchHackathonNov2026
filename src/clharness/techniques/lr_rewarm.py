"""Per-domain learning-rate re-warm: restart the warmup and the decay in every domain.

Half of the reference arm (Ibrahim et al. 2024, replay plus re-warming). With the global
schedule the learning rate is warmed up once, at the start of the stream. With re-warm, every
domain starts with `rewarm_steps` of linear warmup and then decays to `lr * min_lr_ratio` by the
end of that domain.

This technique has no hook behaviour. The schedule lives in the training loop (`optim` settings,
task A4); the technique only switches it on through `config_overrides`, so a run spec turns
re-warm on and off by name like any other technique.

Cost: none. Gradient tokens are unchanged. Note that a decaying schedule takes smaller steps in
total than the fast bench's constant one, so compare acquisition as well as retention.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook


@register
class LRRewarm(BaseHook):
    name = "lr_rewarm"

    class Params(TechniqueParams):
        rewarm_steps: int = Field(8, ge=1)  # linear warmup at the start of every domain (~5% of a fast-bench domain)
        min_lr_ratio: float = Field(0.1, ge=0, le=1)  # the decay ends at lr * min_lr_ratio
        schedule: Literal["cosine", "linear", "constant"] = "cosine"  # shape of the decay within a domain

    def __init__(self, params: Params):
        self.p = params

    @classmethod
    def config_overrides(cls, params: Params) -> dict[str, Any]:
        return {"optim": {
            "schedule_scope": "per_domain", "rewarm_steps": params.rewarm_steps,
            "min_lr_ratio": params.min_lr_ratio, "schedule": params.schedule,
        }}
