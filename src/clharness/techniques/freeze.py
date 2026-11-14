"""Freeze: stop some decoder layers (and the embeddings) from training.

The chosen layers keep their base weights for the whole run: their parameters get
`requires_grad=False` before the optimizer is built, so the optimizer never sees them (no
update, no weight decay). The idea is that the bottom layers hold general features that the new
domains do not need to change.

Parameters
  bottom_layers  freeze the first n decoder layers (Gemma 3 1B has 26; the default is half)
  top_layers     freeze the last n decoder layers
  embeddings     freeze the token embeddings. In Gemma they are tied to the output head, so this
                 freezes the head as well.

At least one decoder layer must stay trainable. The frozen layer indices are kept on the hook
(`frozen_layers`) and in `ctx.state["frozen_layers"]`, so `lora` puts no adapters in them.

Cost: fewer trainable parameters and less optimizer memory. Gradient tokens are unchanged.
Backward still runs through every layer above the lowest trainable one, so the compute saving
is only the frozen layers' weight gradients (plus the backward below the lowest trainable
layer when the bottom layers are frozen).
"""

from __future__ import annotations

from pydantic import Field

from clharness.model import decoder_layers
from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook, RunContext


@register
class Freeze(BaseHook):
    name = "freeze"

    class Params(TechniqueParams):
        bottom_layers: int = Field(13, ge=0)  # freeze the first n decoder layers (Gemma 3 1B has 26)
        top_layers: int = Field(0, ge=0)  # freeze the last n decoder layers
        embeddings: bool = True  # freeze the token embeddings (tied to the output head in Gemma)

    def __init__(self, params: Params):
        self.p = params
        self.frozen_layers: list[int] = []
        self.frozen_params = 0

    def configure_model(self, ctx: RunContext):
        model = ctx.model
        layers = decoder_layers(model)
        n = len(layers)
        if self.p.bottom_layers + self.p.top_layers >= n:
            raise ValueError(
                f"freeze: bottom_layers + top_layers = {self.p.bottom_layers + self.p.top_layers} "
                f"must be smaller than the model's {n} decoder layers"
            )
        frozen = sorted(set(range(self.p.bottom_layers)) | set(range(n - self.p.top_layers, n)))
        params = [p for i in frozen for p in layers[i].parameters()]
        if self.p.embeddings:
            params += list(model.get_input_embeddings().parameters())
        for p in params:
            p.requires_grad_(False)
        self.frozen_layers = frozen
        self.frozen_params = sum(p.numel() for p in {id(p): p for p in params}.values())
        ctx.state["frozen_layers"] = list(frozen)
        if getattr(getattr(ctx.cfg, "train", None), "gradient_checkpointing", False) and self.p.embeddings:
            # reentrant checkpointing passes no gradient through a block whose inputs do not
            # require grad; frozen embeddings would otherwise silently stop all training
            model.enable_input_require_grads()

    def on_run_start(self, ctx: RunContext):
        return {"frozen_layers": float(len(self.frozen_layers)), "frozen_params": float(self.frozen_params)}
