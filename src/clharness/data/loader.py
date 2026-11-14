"""Packed-sequence access over the token streams written by build.py.

Training stream: contiguous chunks of seq_len+1 tokens (inputs = [:-1], labels = [1:]);
the chunk order is a seeded permutation so a domain is not presented in source order.
Every training token counts as a gradient token (no padding in packed chunks), which is what
makes the gradient-token counter exact.

Held-out and probe sets: contiguous chunks of exactly seq_len tokens, in fixed order, no
shuffling; the count is fixed by the manifest so perplexity is comparable across checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from clharness.config import REPO_ROOT


def load_tokens(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    arr = np.load(p)
    if arr.dtype != np.int32 or arr.ndim != 1:
        raise ValueError(f"{p}: expected 1-D int32, got {arr.dtype} {arr.shape}")
    return arr


def n_chunks(tokens: np.ndarray, seq_len: int, stride: int) -> int:
    return max(0, (len(tokens) - stride) // stride) if stride > 0 else 0


def fixed_chunks(tokens: np.ndarray, seq_len: int, max_seqs: int | None = None) -> torch.Tensor:
    """[n, seq_len] int64 tensor of contiguous, non-overlapping chunks in stream order."""
    n = len(tokens) // seq_len
    if max_seqs is not None:
        n = min(n, max_seqs)
    if n == 0:
        raise ValueError("token stream shorter than one sequence")
    return torch.from_numpy(tokens[: n * seq_len].astype(np.int64)).view(n, seq_len)


class PackedStream:
    """Seeded iterator of [micro_batch, seq_len+1] chunks over one domain's token stream."""

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.chunk = seq_len + 1
        self.n = len(tokens) // self.chunk
        if self.n == 0:
            raise ValueError("token stream shorter than one training chunk")
        rng = np.random.default_rng(seed)
        self.order = rng.permutation(self.n)
        self.pos = 0

    @property
    def tokens_available(self) -> int:
        return self.n * self.seq_len

    def batches(self, micro_batch: int) -> Iterator[torch.Tensor]:
        """Yield until the stream is exhausted (one epoch). Caller stops early by token budget.
        Each yielded tensor carries `.chunk_ids` (the stream chunk indices) for per-example logging."""
        while self.pos < self.n:
            idx = self.order[self.pos : self.pos + micro_batch]
            self.pos += len(idx)
            rows = np.stack([self.tokens[i * self.chunk : (i + 1) * self.chunk] for i in idx])
            t = torch.from_numpy(rows.astype(np.int64))
            t.chunk_ids = torch.from_numpy(idx.astype(np.int64))
            yield t

    def state_dict(self) -> dict:
        return {"pos": self.pos}

    def load_state_dict(self, d: dict) -> None:
        self.pos = int(d["pos"])
