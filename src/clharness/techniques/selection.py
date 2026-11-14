"""Scores for choosing which replay candidates to keep (used by the `replay` technique).

Two signals, each computed on candidate sequences with a forward-only pass:

  loss     mean token loss of the candidate under the current model (the same quantity as
           `per_sequence_token_loss` in the training loop): high = the model has started to
           forget it, or never knew it.
  act_nov  activation novelty at one layer: distance of the candidate's mean-pooled residual
           (BOS excluded) from running diagonal mean/variance statistics of the sequences the
           model has been training on. The same definition as `act_nov_L` in
           `clharness.train.signals`: score = sqrt(mean((x - mean)^2 / var)), var clamped at
           1e-6, statistics in float64 on CPU, Welford updates, one update per sequence.

This module registers no technique.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch

from clharness.activations import ResidualCapture
from clharness.train.loop import per_sequence_token_loss


def pool_residual(acts: torch.Tensor, ids: torch.Tensor, bos: int | None) -> torch.Tensor:
    """Mean over positions of residuals `acts` [B, T, d], excluding positions where `ids` [B, T]
    is the BOS token. Returns float64 on CPU [B, d] (copied first, then cast: no fp64 on MPS)."""
    a = acts.float()
    mask = torch.ones(ids.shape, dtype=torch.bool) if bos is None else ids != bos
    m3 = mask.to(a.device).unsqueeze(-1).to(a.dtype)
    return ((a * m3).sum(1) / m3.sum(1).clamp(min=1)).cpu().double()


class ActNovStats:
    """Running diagonal mean/variance of pooled residuals; `score` is the activation novelty."""

    def __init__(self):
        self.n = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        """False while fewer than 2 sequences have been seen: there is no variance yet."""
        return self.n > 1

    def update(self, pooled: torch.Tensor) -> None:
        """Add the rows of `pooled` [B, d], one Welford step per row."""
        pooled = pooled.detach().cpu().double()
        if self.mean is None:
            self.mean = torch.zeros(pooled.shape[1], dtype=torch.float64)
            self.m2 = torch.zeros(pooled.shape[1], dtype=torch.float64)
        for row in pooled:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    def score(self, pooled: torch.Tensor) -> torch.Tensor | None:
        """Novelty of each row of `pooled` [B, d] as float64 [B]; None if not `ready`."""
        if not self.ready:
            return None
        var = (self.m2 / (self.n - 1)).clamp(min=1e-6)
        x = pooled.detach().cpu().double()
        return (((x - self.mean) ** 2) / var).mean(1).sqrt()


@torch.no_grad()
def candidate_forward(
    model: torch.nn.Module,
    rows: torch.Tensor,
    chunk: int,
    device: str | torch.device = "cpu",
    autocast: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
    layer: int | None = None,
    bos: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One forward-only pass over candidate `rows` [n, seq_len+1], `chunk` rows at a time.

    Returns (mean token loss [n] float64 CPU, pooled residual at `layer` [n, d] float64 CPU, or
    None when `layer` is None). Does not change the model's train/eval mode."""
    losses, pooled = [], []
    for i in range(0, rows.shape[0], chunk):
        part = rows[i : i + chunk]
        ids, labels = part[:, :-1].to(device), part[:, 1:].to(device)
        layers = [] if layer is None else [layer]
        with ResidualCapture(model, layers) as cap, autocast():
            logits = model(input_ids=ids, use_cache=False).logits
        losses.append(per_sequence_token_loss(logits, labels).cpu().double())
        if layer is not None:
            pooled.append(pool_residual(cap.acts[layer], ids, bos))
        del logits
    return torch.cat(losses), (torch.cat(pooled) if layer is not None else None)


def loss_scores(model, rows, chunk, device="cpu", autocast=contextlib.nullcontext) -> torch.Tensor:
    """Mean token loss of each candidate row under `model`: float64 [n] on CPU."""
    return candidate_forward(model, rows, chunk, device, autocast)[0]


def pick(scores: torch.Tensor, k: int, keep: str) -> torch.Tensor:
    """Indices of the k highest (`keep="high"`) or lowest (`"low"`) scores. Ties go to the
    candidate earlier in the pool, so the result is deterministic."""
    if keep not in ("high", "low"):
        raise ValueError(f"keep must be 'high' or 'low', not {keep!r}")
    key = -scores if keep == "high" else scores
    order = torch.sort(key, stable=True).indices  # stable: equal scores keep pool order
    return order[:k]
