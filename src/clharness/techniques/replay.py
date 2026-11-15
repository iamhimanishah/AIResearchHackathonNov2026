"""Replay: mix a fixed share of old-domain sequences into every new domain's training.

While the model trains on a domain, a share of the training sequences is swapped for sequences
from a replay source (by default `general`, the pretraining-like domain). Rows are replaced,
not added: the gradient tokens per step and per run are unchanged, and the new domain simply
gets (1 - share) of them.

Parameters
  share        fraction of training sequences that are replayed (0 < share < 1)
  source       data-manifest domain whose `stream` split is replayed. A source outside the
               run's stream (such as `general`) is replayed from the first domain on. A source
               that is one of the stream's domains is replayed only after the run has trained
               on it, and never into itself, so `source: code` means "replay code while
               training the domains after it".
  buffer_seqs  None: walk the source stream, fresh sequences each time (new permutation when it
               runs out). N: a fixed buffer of N sequences, sampled with replacement.
  select       how the replayed rows are chosen. "random" (the default, and the control): take
               the next rows of the source. "loss" or "act_nov": draw a pool of `pool` candidates
               per replayed row, score each with a forward-only pass of the current model, and
               keep the best-scoring ones (see `selection.py` for the scores). "loss" is the
               candidate's mean token loss; "act_nov" is how far its mean-pooled residual at
               `layer` lies from running statistics of the sequences the model has trained on.
  pool         candidates drawn per replayed row (ignored for random). From a buffer, the pool is
               sampled without replacement unless the buffer is smaller than the pool.
  keep         "high": keep the highest-scoring candidates; "low": the lowest.
  layer        residual layer for act_nov. Its statistics are updated from the non-replayed
               rows of every training forward (captured by the loop); until 2 sequences have
               been seen there are none, and that call falls back to random (`fallback_rows`).

Cost: no extra gradient tokens, and the source stream is held in memory. random: no extra
forward passes. loss/act_nov: forward-only passes over share x pool of the training tokens
(e.g. 20% at share 0.05 and pool 4), counted in `ctx.state["extra_forward_tokens"]`; without a
buffer the source is walked `pool` times faster.

Logged: `share`, `rows`; with a scored selection also `scored_rows`, `fallback_rows` (cumulative)
and the running means `score_pool` (all scored candidates) and `score_kept` (the kept ones).

The replayed rows are the last k rows of a micro-batch. The returned batch carries `.chunk_ids`
with -1 at replayed rows (so per-example signals do not log them as new-domain chunks) and
`.replay_mask` (bool [B], True at replayed rows).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from clharness.data.loader import PackedStream, load_tokens
from clharness.techniques import TechniqueParams, register
from clharness.techniques.selection import ActNovStats, candidate_forward, pick, pool_residual
from clharness.train.hooks import BaseHook, RunContext, StepStats

SEED_OFFSET = 10_007  # training streams use seed + domain_index; keep the replay order distinct


@register
class Replay(BaseHook):
    name = "replay"

    class Params(TechniqueParams):
        share: float = Field(0.05, gt=0, lt=1)
        source: str = "general"
        buffer_seqs: int | None = Field(None, gt=0)
        select: Literal["random", "loss", "act_nov"] = "random"  # how replay rows are chosen from the candidate pool
        pool: int = Field(4, ge=1)  # candidates drawn per replayed row (ignored for random)
        keep: Literal["high", "low"] = "high"  # keep the highest- or the lowest-scoring candidates
        layer: int = Field(13, ge=0)  # residual layer for act_nov

    def __new__(cls, params: Params):
        # Activation-novelty selection needs the training forward's residuals, so it lives in a
        # subclass. The loop computes a per-sequence loss for every micro-batch as soon as any
        # hook overrides on_micro_batch; plain replay must not make every run pay for that.
        return super().__new__(_ReplayActNov if cls is Replay and params.select == "act_nov" else cls)

    def __init__(self, params: Params):
        self.p = params
        self.active = False
        self._trained: set[str] = set()
        self.rows_seen = 0
        self.rows_replayed = 0
        self.source_epochs = 0
        self._reset_selection()

    def _reset_selection(self):
        self.stats = ActNovStats() if self.p.select == "act_nov" else None
        self.last_pool: torch.Tensor | None = None  # candidate rows of the latest selection
        self.last_scores: torch.Tensor | None = None  # their scores [pool size]; None on a fallback
        self.last_chosen: torch.Tensor | None = None  # indices into last_pool that were kept
        self.scored_rows = self.fallback_rows = 0
        self._sum_pool = self._sum_kept = 0.0
        self._n_kept = 0

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
        self._trained: set[str] = set()  # stream domains this run has finished
        self.rows_seen = self.rows_replayed = 0
        self.source_epochs = 1  # passes over the source stream started
        self._reset_selection()
        self.bos = getattr(ctx.tokenizer, "bos_token_id", None)
        if self.p.select != "random":
            ctx.state.setdefault("extra_forward_tokens", 0)

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

    def _pool_chunks(self, n: int) -> np.ndarray:
        """Chunk ids of n candidates: the next n of the stream, or n buffer entries (without
        replacement unless the buffer is smaller than the pool)."""
        if self.buffer is not None and n <= len(self.buffer):
            return self.buffer[self.rng.choice(len(self.buffer), size=n, replace=False)]
        return self._next_chunks(n)

    def _rows(self, chunk_ids: np.ndarray) -> torch.Tensor:
        c, toks = self.stream.chunk, self.stream.tokens
        rows = np.stack([toks[i * c : (i + 1) * c] for i in chunk_ids]).astype(np.int64)
        return torch.from_numpy(rows)

    # -- mixing -----------------------------------------------------------------------------
    def on_domain_start(self, ctx: RunContext, domain: str):
        # Never replay a domain into itself. A source that is part of this run's stream is old
        # text only once the run has trained on it: replaying a later domain early would feed
        # the model that domain before its turn.
        in_stream = self.p.source in {d.name for d in ctx.cfg.data.domains}
        self.active = domain != self.p.source and (not in_stream or self.p.source in self._trained)

    def on_domain_end(self, ctx: RunContext, domain: str):
        self._trained.add(domain)

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
            rows = self._rows(self._next_chunks(k)) if self.p.select == "random" else self._select(ctx, k, b)
            out[b - k :] = rows.to(out.device, out.dtype)
            chunk_ids[b - k :] = -1
            mask[b - k :] = True
            self.rows_replayed += k
        # tensor clone/slicing drops Python attributes, so set them on every returned batch
        out.chunk_ids = chunk_ids
        out.replay_mask = mask
        return out

    # -- selection (loss / act_nov) ---------------------------------------------------------
    def _select(self, ctx: RunContext, k: int, chunk: int) -> torch.Tensor:
        """Draw k * pool candidates, score them with a forward-only pass (`chunk` rows at a
        time, no more than the training micro-batch), return the k rows `keep` asks for."""
        cand = self._rows(self._pool_chunks(k * self.p.pool))
        self.last_pool = cand
        if self.stats is not None and not self.stats.ready:  # act_nov before any statistics
            self.last_scores, self.last_chosen = None, torch.arange(k)
            self.fallback_rows += k
            return cand[:k]  # the pool is a random draw, so its first k rows are a random pick
        layer = self.p.layer if self.stats is not None else None
        loss, pooled = candidate_forward(ctx.model, cand, chunk, ctx.device, ctx.ac, layer, self.bos)
        ctx.state["extra_forward_tokens"] = ctx.state.get("extra_forward_tokens", 0) + cand[:, :-1].numel()
        scores = loss if self.stats is None else self.stats.score(pooled)
        chosen = pick(scores, k, self.p.keep)
        self.last_scores, self.last_chosen = scores, chosen
        self.scored_rows += cand.shape[0]
        self._sum_pool += float(scores.sum())
        self._sum_kept += float(scores[chosen].sum())
        self._n_kept += k
        return cand[chosen]

    # -- logging ----------------------------------------------------------------------------
    def _stats(self) -> dict[str, float]:
        share = self.rows_replayed / self.rows_seen if self.rows_seen else 0.0
        out = {"share": share, "rows": float(self.rows_replayed)}
        if self.p.select != "random":
            out["scored_rows"] = float(self.scored_rows)
            out["fallback_rows"] = float(self.fallback_rows)
            if self.scored_rows:
                out["score_pool"] = self._sum_pool / self.scored_rows
                out["score_kept"] = self._sum_kept / self._n_kept
        return out

    def on_step_end(self, ctx: RunContext, stats: StepStats):
        return self._stats()

    def on_run_end(self, ctx: RunContext):
        return {**self._stats(), "source_epochs": float(self.source_epochs)}


class _ReplayActNov(Replay):
    """Replay with `select: act_nov`: also keeps running statistics of the trained rows'
    residuals at `layer`. Built by `Replay(params)`; not registered under a name of its own."""

    def wants_activations(self) -> list[int]:
        return [self.p.layer] if self.p.select == "act_nov" else []

    @torch.no_grad()
    def on_micro_batch(self, ctx: RunContext, step: int, domain_index: int, chunk_ids, batch, logits, seq_loss, acts):
        """act_nov: add this forward's trained, non-replayed rows to the running statistics
        (after the rows were used, so no candidate is ever scored against itself)."""
        if self.stats is None:
            return
        replayed = getattr(batch, "replay_mask", None)
        trained = torch.ones(batch.shape[0], dtype=torch.bool) if replayed is None else ~replayed.cpu()
        if trained.any():
            a = acts[self.p.layer]
            self.stats.update(pool_residual(a[trained.to(a.device)], batch[trained, :-1], self.bos))
