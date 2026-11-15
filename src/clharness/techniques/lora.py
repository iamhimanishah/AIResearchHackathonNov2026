"""LoRA: train small low-rank adapters instead of the model's weights.

Every targeted linear layer W gets two small matrices A (rank x in) and B (out x rank) and
computes W x + (alpha / rank) * B A x. Only A and B train; every original weight stays exactly
as it was. B starts at zero, so before the first step the model's output is the base model's.

The adapters are added in place with `peft.inject_adapter_in_model`, not `get_peft_model`. A
`PeftModel` wrapper moves every module under `base_model.model...`; in-place injection keeps
`model.model.layers[i]` where it was and only replaces the targeted `Linear` leaves, so the code
that finds decoder layers (activation capture) works unchanged.

Parameters
  rank     adapter rank r
  alpha    LoRA scaling numerator (the update is scaled by alpha / rank); None means 2 * rank
  targets  "attn": q_proj, k_proj, v_proj, o_proj. "attn_mlp": those plus gate_proj, up_proj,
           down_proj.
  lr       learning rate for the whole run, set through `config_overrides`; None keeps the
           bench's. The default, 1e-3, suits rank 8. With alpha = 2 * rank the size of a step
           grows roughly with lr * sqrt(rank), so keep lr <= 1e-3 * sqrt(8 / rank): rank 256
           diverges at 1e-3 and trains normally at the bench's 1e-4. Rank 8 at 1e-4 learns
           little of the new domain, so there it forgets little partly because it learns
           little. The runs are in docs/TECHNIQUES.md, "Exploratory runs".

No dropout and no bias training. If `freeze` ran first (it does: techniques are built in name
order), the layers it froze get no adapters, since an adapter there would un-freeze the layer in
effect.

Cost: far fewer trainable parameters, less optimizer memory and a much smaller update to ship.
Gradient tokens are unchanged. The forward cost is about the same (the adapters are small), and
backward still runs through the whole model.

Known limits
  - The checkpoint's state dict has the adapter keys (`...q_proj.base_layer.weight`,
    `...q_proj.lora_A.default.weight`), so loading a saved checkpoint into a plain model needs
    the adapter merged first. The fast bench keeps no weights, so it is not affected.
  - The learning-rate override applies to the whole run, including any other technique's
    parameters that train.
  - The plugin does not enforce the rank and learning-rate limit above, and the default lr does
    not depend on rank. The explore gate enforces the limit (`lora_lr_limit` in
    `agent/catalog_real.py`).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field

from clharness.model import decoder_layers
from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook, RunContext

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def _is_adapter(name: str) -> bool:
    return "lora_" in name


@register
class LoRA(BaseHook):
    name = "lora"

    class Params(TechniqueParams):
        rank: int = Field(8, ge=1)
        alpha: float | None = Field(None, gt=0)  # LoRA scaling; None means 2 * rank
        targets: Literal["attn", "attn_mlp"] = "attn_mlp"  # attention projections, or attention + MLP projections
        lr: float | None = Field(1e-3, gt=0)  # learning rate for the run; None keeps the bench's

    def __init__(self, params: Params):
        self.p = params
        self.trainable_params = 0
        self.adapted_modules = 0
        self.target_names: list[str] = []

    @classmethod
    def config_overrides(cls, params: Params) -> dict[str, Any]:
        return {"optim": {"lr": params.lr}} if params.lr is not None else {}

    def _targets(self, model: torch.nn.Module, frozen: set[int]) -> list[str]:
        """Full names of the Linear leaves to adapt: target projections in non-frozen layers."""
        wanted = set(ATTN if self.p.targets == "attn" else ATTN + MLP)
        layer_of = {id(m): i for i, layer in enumerate(decoder_layers(model)) for m in layer.modules()}
        names = [
            n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in wanted
            and id(m) in layer_of and layer_of[id(m)] not in frozen
        ]
        if not names:
            raise ValueError(
                f"lora: no Linear module named {sorted(wanted)} in the model's trainable decoder layers "
                f"(frozen layers: {sorted(frozen)})"
            )
        return names

    def configure_model(self, ctx: RunContext):
        from peft import LoraConfig, inject_adapter_in_model

        model = ctx.model
        frozen = set(ctx.state.get("frozen_layers") or [])
        self.target_names = self._targets(model, frozen)
        config = LoraConfig(
            r=self.p.rank,
            lora_alpha=self.p.alpha if self.p.alpha is not None else 2 * self.p.rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=self.target_names,
        )
        injected = inject_adapter_in_model(config, model)
        if injected is not None and injected is not model:
            raise RuntimeError("lora: peft returned a new model; expected in-place injection")
        # only the adapters train; freeze the rest in case peft left anything trainable
        for n, p in model.named_parameters():
            if not _is_adapter(n):
                p.requires_grad_(False)
        adapters = [(n, p) for n, p in model.named_parameters() if _is_adapter(n)]
        if not adapters or not all(p.requires_grad for _, p in adapters):
            raise RuntimeError("lora: adapter parameters are missing or not trainable after injection")
        self.trainable_params = sum(p.numel() for _, p in adapters)
        self.adapted_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0)
        if self.adapted_modules != len(self.target_names):
            raise RuntimeError(f"lora: adapted {self.adapted_modules} modules, expected {len(self.target_names)}")
        if getattr(getattr(ctx.cfg, "train", None), "gradient_checkpointing", False):
            # reentrant checkpointing passes no gradient through a block whose inputs do not
            # require grad; with frozen embeddings the adapters would silently get none
            model.enable_input_require_grads()

    def on_run_start(self, ctx: RunContext):
        return {"trainable_params": float(self.trainable_params), "adapted_modules": float(self.adapted_modules)}
