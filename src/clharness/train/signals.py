"""Per-example importance signals logged during training (Phase 2's signal side, for free).

For every training sequence, at the step it is trained on, record:
  loss         mean token NLL under the current model (the "surprise" scalar: SuRe/Rho-1 family)
  entropy      mean per-token entropy of the output distribution (uncertainty channel)
  act_nov_L    raw-activation novelty at layer L: Mahalanobis distance of the sequence's
               mean-pooled residual from a running (diagonal) mean/variance of what the model
               has been trained on — the hippocampal-mismatch analogue (plan §3 Phase 2, primary)
  sae_rare_L   SAE-feature novelty at layer L: mean over active features of -log(running
               firing frequency); high = the sequence fires features the stream rarely fires
  sae_fvu_L    SAE weakness at layer L: fraction of variance unexplained on this sequence
               (features the model represents poorly under the fixed SAE)
Running statistics are updated *after* scoring each micro-batch, so a score never sees itself.
Output: <run>/signals.npz appended per step (compact: ~10 floats per sequence), plus sequence
identity (domain index, chunk index) so Phase 2 can join labels to signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from clharness.sae import SAESpec, load_sae
from clharness.train.hooks import BaseHook, RunContext


class SignalLogHook(BaseHook):
    name = "signals"

    def on_run_start(self, ctx: RunContext):
        cfg = ctx.cfg
        self.enabled = cfg.train.signal_log
        if not self.enabled:
            return {"enabled": 0.0}
        self.layers = list(cfg.model.capture_layers)
        seen: set[int] = set()
        self.saes = {}
        for sc in cfg.model.saes:  # primary SAE per layer only
            if sc.layer not in seen:
                self.saes[sc.layer] = load_sae(SAESpec(sc.release, sc.sae_id, sc.layer), ctx.device)
                seen.add(sc.layer)
        d = next(iter(self.saes.values())).cfg.d_in
        self.n = 0
        self.mean = {L: torch.zeros(d, dtype=torch.float64) for L in self.layers}
        self.m2 = {L: torch.zeros(d, dtype=torch.float64) for L in self.layers}
        self.feat_count = {L: torch.zeros(self.saes[L].cfg.d_sae, dtype=torch.float64) for L in self.layers}
        self.feat_tokens = 0
        self.bos = ctx.tokenizer.bos_token_id
        self.rows: list[np.ndarray] = []
        self.cols = ["step", "domain", "chunk", "loss", "entropy"] + [f"act_nov_L{L}" for L in self.layers] \
            + [f"sae_rare_L{L}" for L in self.layers] + [f"sae_fvu_L{L}" for L in self.layers]
        self.out = Path(ctx.output_dir) / "signals.npz"
        return {"enabled": 1.0}

    def wants_activations(self) -> list[int]:
        return self.layers if getattr(self, "enabled", False) else []

    @torch.no_grad()
    def on_micro_batch(self, ctx: RunContext, step: int, domain_index: int, chunk_ids: torch.Tensor,
                       batch: torch.Tensor, logits: torch.Tensor, seq_loss: torch.Tensor, acts: dict[int, torch.Tensor]):
        if not self.enabled:
            return
        ids = batch[:, :-1].to(logits.device)
        mask = (ids != self.bos)  # exclude BOS from representation statistics, as everywhere else
        B = ids.shape[0]
        # entropy of the output distribution, chunked over the 262k vocab
        ent = torch.zeros(B, dtype=torch.float64)
        flat = logits.reshape(-1, logits.shape[-1])
        T = ids.shape[1]
        for j in range(0, flat.shape[0], 1024):
            lp = torch.log_softmax(flat[j : j + 1024].float(), dim=-1)
            e = -(lp.exp() * lp).sum(-1)  # [n]
            idx = torch.arange(j, j + e.shape[0], device=e.device) // T
            ent.index_add_(0, idx.cpu(), e.cpu().double())
        ent /= T
        cols: dict[str, torch.Tensor] = {"loss": seq_loss.cpu().double(), "entropy": ent}
        for L in self.layers:
            a = acts[L].float()  # [B, T, d]
            m3 = mask.unsqueeze(-1).to(a.dtype)
            pooled = ((a * m3).sum(1) / m3.sum(1).clamp(min=1)).cpu().double()  # [B, d]  (copy, then cast: no fp64 on MPS)
            if self.n > 1:
                var = (self.m2[L] / (self.n - 1)).clamp(min=1e-6)
                cols[f"act_nov_L{L}"] = (((pooled - self.mean[L]) ** 2) / var).mean(1).sqrt()
            else:
                cols[f"act_nov_L{L}"] = torch.zeros(B, dtype=torch.float64)
            # SAE rarity + FVU per sequence
            sae = self.saes[L]
            x = a[mask]  # [n_tok, d] (BOS excluded)
            seq_of_tok = torch.arange(B, device=a.device).unsqueeze(1).expand(B, T)[mask]
            feats = sae.encode(x)
            active = feats > 0
            if self.feat_tokens > 0:
                freq = (self.feat_count[L] / self.feat_tokens).clamp(min=1.0 / max(self.feat_tokens, 1)).float().to(a.device)  # cast before moving (no fp64 on MPS)
                rar_tok = -(active.float() * torch.log(freq)).sum(1) / active.float().sum(1).clamp(min=1)
            else:
                rar_tok = torch.zeros(x.shape[0], device=a.device)
            err = (x - sae.decode(feats)).pow(2).sum(1)
            tot = (x - x.mean(0, keepdim=True)).pow(2).sum(1)
            rare = torch.zeros(B, dtype=torch.float64).index_add_(0, seq_of_tok.cpu(), rar_tok.cpu().double())
            fe = torch.zeros(B, dtype=torch.float64).index_add_(0, seq_of_tok.cpu(), err.cpu().double())
            ft = torch.zeros(B, dtype=torch.float64).index_add_(0, seq_of_tok.cpu(), tot.cpu().double())
            cnt = torch.zeros(B, dtype=torch.float64).index_add_(0, seq_of_tok.cpu(), torch.ones(rar_tok.shape[0], dtype=torch.float64))
            cols[f"sae_rare_L{L}"] = rare / cnt.clamp(min=1)
            cols[f"sae_fvu_L{L}"] = fe / ft.clamp(min=1e-12)
            # update running stats AFTER scoring (Welford, diagonal; n counts sequences)
            n = self.n
            for row in pooled:
                n += 1
                delta = row - self.mean[L]
                self.mean[L] += delta / n
                self.m2[L] += delta * (row - self.mean[L])
            self.feat_count[L] += active.sum(0).cpu().double()
        self.n += B
        self.feat_tokens += int(mask.sum())
        rows = np.stack(
            [np.full(B, step, dtype=np.float64), np.full(B, domain_index, dtype=np.float64), chunk_ids.double().numpy()]
            + [cols[c].numpy() for c in self.cols[3:]], axis=1,
        )
        self.rows.append(rows)

    def on_checkpoint(self, ctx: RunContext, ckpt):
        if not self.enabled or not self.rows:
            return None
        data = np.concatenate(self.rows)
        prev = np.load(self.out)["data"] if self.out.exists() else np.zeros((0, data.shape[1]))
        np.savez_compressed(self.out, data=np.concatenate([prev, data]), columns=np.array(self.cols))
        self.rows = []
        return {"rows_logged": float(data.shape[0])}

    def on_run_end(self, ctx: RunContext):
        return self.on_checkpoint(ctx, None)
