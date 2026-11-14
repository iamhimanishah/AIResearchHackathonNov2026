"""Model + tokenizer loading for Gemma 3 (text-only causal LM)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from clharness.env import hf_token

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True)
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_id: str
    device: str
    dtype: torch.dtype

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        # bf16 on cuda/mps, fp32 on cpu (bf16 matmul on cpu is slow and less precise)
        return torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    return _DTYPES[name]


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    # Gemma's tokenizer prepends <bos> by default; the PT model expects it. Assert rather than assume.
    ids = tok("x", add_special_tokens=True)["input_ids"]
    if ids[0] != tok.bos_token_id:
        raise RuntimeError(f"tokenizer for {model_id} does not prepend BOS; got {ids[:3]}")
    return tok


def load_model(
    model_id: str = "google/gemma-3-1b-pt",
    device: str = "cpu",
    dtype: str = "auto",
    attn_implementation: str | None = None,
) -> LoadedModel:
    """Load a Gemma 3 text model in eval mode on `device`.

    attn_implementation: None lets transformers pick ("sdpa" normally). Use "eager" when you
    need attention weights or when the SDPA path misbehaves on a backend (e.g. some MPS ops).
    """
    torch_dtype = resolve_dtype(dtype, device)
    kwargs = dict(dtype=torch_dtype, token=hf_token())
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    tok = load_tokenizer(model_id)
    return LoadedModel(model=model, tokenizer=tok, model_id=model_id, device=device, dtype=torch_dtype)


def decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """The list of decoder blocks, independent of the wrapping class.

    Gemma3ForCausalLM -> .model (Gemma3TextModel) -> .layers
    """
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")
    return layers
