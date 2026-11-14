"""Replay: mix a fixed share of old-domain sequences into every new domain's training.

While the model trains on a domain, a share of the training sequences is swapped for sequences
from a replay source (by default `general`, the pretraining-like domain). Rows are replaced,
not added: the gradient tokens per step and per run are unchanged, and the new domain simply
gets (1 - share) of them.

Parameters
  share        fraction of training sequences that are replayed (0 < share < 1)
  source       data-manifest domain whose `stream` split is replayed. Replay is off while the
               run trains on the source itself, so `source: code` means "replay code while
               training the domains after it".
  buffer_seqs  None: walk the source stream, fresh sequences each time (new permutation when it
               runs out). N: a fixed buffer of N sequences, sampled with replacement.

Cost: no extra forward passes and no extra gradient tokens; the source stream is held in memory.

The replayed rows are the last k rows of a micro-batch. The returned batch carries `.chunk_ids`
with -1 at replayed rows (so per-example signals do not log them as new-domain chunks) and
`.replay_mask` (bool [B], True at replayed rows).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pydantic import Field

from clharness.data.loader import PackedStream, load_tokens
from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook, RunContext, StepStats

SEED_OFFSET = 10_007  # training streams use seed + domain_index; keep the replay order distinct


@register
class Replay(BaseHook):
    name = "replay"

    class Params(TechniqueParams):
        share: float = Field(0.05, gt=0, lt=1)
        source: str = "general"
        buffer_seqs: int | None = Field(None, gt=0)

    def __init__(self, params: Params):
        self.p = params
        self.active = False
        self.rows_seen = 0
        self.rows_replayed = 0
        self.source_epochs = 0

    # -- source stream ----------------------------------------------------------------------
    def on_run_start(self, ctx: RunContext):
        domains = ctx.data_manifest.get("domains", {})
        try:
            path = domains[self.p.source]["splits"]["stream"]["tokens"]["path"]
        except KeyError:
            raise ValueError(
                f"replay source {self.p.source!r} has no stream split in the data manifest; "
                f"available domains: {sorted(domains)}"
            ) from None
        self.seed = ctx.cfg.train.seed + SEED_OFFSET
        self.stream = PackedStream(load_tokens(path), ctx.cfg.data.seq_len, seed=self.seed)
        self.order, self.pos = self.stream.order, 0
        self.buffer = None
        if self.p.buffer_seqs is not None:
            self.buffer = self.order[: min(self.p.buffer_seqs, self.stream.n)]
            self.rng = np.random.default_rng(self.seed)
        self.active = False
        self.rows_seen = self.rows_replayed = 0
        self.source_epochs = 1  # passes over the source stream started

    def _next_chunks(self, k: int) -> np.ndarray:
        if self.buffer is not None:
            return self.buffer[self.rng.integers(0, len(self.buffer), size=k)]
        out = []
        for _ in range(k):
            if self.pos == self.stream.n:  # out of fresh sequences: start another pass
                self.order = np.random.default_rng(self.seed + self.source_epochs).permutation(
                    self.stream.n
                )
                self.pos = 0
                self.source_epochs += 1
            out.append(self.order[self.pos])
            self.pos += 1
        return np.asarray(out, dtype=np.int64)

    def _rows(self, chunk_ids: np.ndarray) -> torch.Tensor:
        c, toks = self.stream.chunk, self.stream.tokens
        rows = np.stack([toks[i * c : (i + 1) * c] for i in chunk_ids]).astype(np.int64)
        return torch.from_numpy(rows)

    # -- mixing -----------------------------------------------------------------------------
    def on_domain_start(self, ctx: RunContext, domain: str):
        self.active = domain != self.p.source  # never replay a domain into itself

    def select_batch(self, ctx: RunContext, batch: torch.Tensor):
        if not self.active:
            return batch
        b = batch.shape[0]
        self.rows_seen += b
        # running total, not per batch: round(0.05 * 2) is 0 on every 2-row micro-batch
        k = math.floor(self.p.share * self.rows_seen + 1e-9) - self.rows_replayed
        k = max(0, min(b, k))
        chunk_ids = getattr(batch, "chunk_ids", None)
        chunk_ids = torch.arange(b) if chunk_ids is None else chunk_ids.clone()
        mask = torch.zeros(b, dtype=torch.bool)
        out = batch.clone()
        if k:
            out[b - k :] = self._rows(self._next_chunks(k)).to(out.device, out.dtype)
            chunk_ids[b - k :] = -1
            mask[b - k :] = True
            self.rows_replayed += k
        # tensor clone/slicing drops Python attributes, so set them on every returned batch
        out.chunk_ids = chunk_ids
        out.replay_mask = mask
        return out

    # -- logging ----------------------------------------------------------------------------
    def _stats(self) -> dict[str, float]:
        share = self.rows_replayed / self.rows_seen if self.rows_seen else 0.0
        return {"share": share, "rows": float(self.rows_replayed)}

    def on_step_end(self, ctx: RunContext, stats: StepStats):
        return self._stats()

    def on_run_end(self, ctx: RunContext):
        return {**self._stats(), "source_epochs": float(self.source_epochs)}
