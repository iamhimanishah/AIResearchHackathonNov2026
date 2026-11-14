"""Gemma Scope 2 SAEs via SAELens, plus reconstruction / drift metrics.

We never train an SAE. Loading goes through SAELens' registry so the release/id strings in
configs are the only source of truth (see configs/model/gemma3-1b-pt.yaml).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from sae_lens import SAE


@dataclass(frozen=True)
class SAESpec:
    release: str
    sae_id: str
    layer: int


@dataclass
class ReconMetrics:
    """Reconstruction quality of a fixed SAE on a (possibly updated) model's activations.

    All values are token-averaged over non-padding positions.
      mse:               mean squared error per dimension, mean over tokens
      fvu:               fraction of variance unexplained = sum((x-x̂)^2) / sum((x-mean(x))^2)
      variance_explained: 1 - fvu
      l0:                mean number of active features per token
      cos:               mean cosine(x, x̂)
      x_norm:            mean ||x||  (tracks activation-scale drift, which also moves mse)
    """

    layer: int
    n_tokens: int
    mse: float
    fvu: float
    variance_explained: float
    l0: float
    cos: float
    x_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_sae(spec: SAESpec, device: str) -> SAE:
    """Load one SAE. Float32 always: the SAEs are published in fp32 and mixed-precision
    encode would change thresholds' effective behaviour."""
    sae = SAE.from_pretrained(release=spec.release, sae_id=spec.sae_id, device=device)
    sae.eval()
    hook = sae.cfg.metadata.hook_name
    expected = f"blocks.{spec.layer}.hook_resid_post"
    if hook != expected:
        raise ValueError(f"{spec.sae_id}: hook {hook!r} != expected {expected!r}")
    return sae


def load_saes(specs: list[SAESpec], device: str) -> dict[int, SAE]:
    return {s.layer: load_sae(s, device) for s in specs}


@torch.no_grad()
def reconstruction_metrics(
    sae: SAE, acts: torch.Tensor, mask: torch.Tensor | None, layer: int
) -> ReconMetrics:
    """acts: [B, T, d] residual activations; mask: [B, T] (1 = real token) or None."""
    x = acts.reshape(-1, acts.shape[-1])
    if mask is not None:
        x = x[mask.reshape(-1).bool()]
    x = x.to(sae.device, torch.float32)
    feats = sae.encode(x)
    x_hat = sae.decode(feats)
    err = x - x_hat
    ss_res = err.pow(2).sum()
    ss_tot = (x - x.mean(0, keepdim=True)).pow(2).sum()
    fvu = (ss_res / ss_tot).item()
    return ReconMetrics(
        layer=layer,
        n_tokens=int(x.shape[0]),
        mse=err.pow(2).mean().item(),
        fvu=fvu,
        variance_explained=1.0 - fvu,
        l0=(feats > 0).float().sum(-1).mean().item(),
        cos=torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item(),
        x_norm=x.norm(dim=-1).mean().item(),
    )
