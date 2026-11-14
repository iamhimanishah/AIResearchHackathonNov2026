"""Residual-stream activation capture with plain HF forward hooks.

Hook site: the *output* of decoder block L (`model.layers.L.output` in Gemma Scope 2's own
config.json), which equals `output_hidden_states[L+1]`. This is what Gemma Scope 2
`resid_post` SAEs were trained on. `tests/test_shapes.py` asserts the equality.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from transformers import PreTrainedModel

from clharness.model import decoder_layers


class ResidualCapture:
    """Context manager that stores block outputs for the requested layers.

    Captured tensors are detached and (optionally) moved to `store_device`. Padding positions
    are the caller's responsibility (pass the attention mask to the statistics code).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layers: list[int],
        store_device: str | None = None,
        store_dtype: torch.dtype | None = None,
    ):
        self.model = model
        self.layers = list(layers)
        self.store_device = store_device
        self.store_dtype = store_dtype
        self.acts: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.detach()
            if self.store_dtype is not None:
                h = h.to(self.store_dtype)
            if self.store_device is not None:
                h = h.to(self.store_device)
            self.acts[layer] = h

        return hook

    def __enter__(self) -> "ResidualCapture":
        blocks = decoder_layers(self.model)
        for layer in self.layers:
            if not 0 <= layer < len(blocks):
                raise IndexError(f"layer {layer} out of range for {len(blocks)} blocks")
            self._handles.append(blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.acts.clear()


@contextmanager
def capture_residuals(
    model: PreTrainedModel, layers: list[int], **kw
) -> Iterator[ResidualCapture]:
    with ResidualCapture(model, layers, **kw) as cap:
        yield cap


@torch.no_grad()
def forward_with_capture(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: list[int],
    **kw,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """One forward pass; returns (logits, {layer: resid_post [B, T, d]})."""
    with ResidualCapture(model, layers, **kw) as cap:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits, dict(cap.acts)
