"""Compact per-SAE-feature statistics over a probe set.

Stored per checkpoint, per layer: for each of d_sae features
  freq[f]      = fraction of probe tokens on which feature f fires (act > 0)
  mean_mag[f]  = mean activation of f over tokens where it fires (0 if never)
  mean_act[f]  = mean activation of f over all tokens (= freq * mean_mag)
  mean_pre[f]  = mean JumpReLU *pre*-activation (x·W_enc + b_enc) over all tokens
  near_thr[f]  = fraction of tokens whose pre-activation lies within ±margin of the threshold
plus the reconstruction metrics from clharness.sae. Raw activations are never stored.

mean_pre / near_thr exist to separate genuine fading (pre-activation drifts down) from
threshold flicker (pre-activation hovers at the threshold, so tiny shifts flip the feature on
and off and *look* like decay). See docs/research-notes.md T1.

At 16k features x 3 layers x float32 this is ~200KB per checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sae_lens import SAE

from clharness.sae import ReconMetrics


@dataclass
class FeatureStats:
    layer: int
    n_tokens: int
    freq: torch.Tensor  # [d_sae] float32
    mean_mag: torch.Tensor  # [d_sae] float32
    mean_act: torch.Tensor  # [d_sae] float32
    mean_pre: torch.Tensor  # [d_sae] float32
    near_thr: torch.Tensor  # [d_sae] float32
    recon: ReconMetrics

    def summary(self) -> dict[str, float]:
        alive = (self.freq > 0).float()
        return {
            "n_tokens": float(self.n_tokens),
            "n_alive": float(alive.sum()),
            "frac_alive": float(alive.mean()),
            "mean_freq_alive": float(self.freq[self.freq > 0].mean()) if alive.sum() > 0 else 0.0,
            "near_thr_mean": float(self.near_thr.mean()),  # avg fraction of tokens within ±margin of threshold
            "mean_l0": self.recon.l0,
            "fvu": self.recon.fvu,
            "variance_explained": self.recon.variance_explained,
            "mse": self.recon.mse,
            "cos": self.recon.cos,
            "x_norm": self.recon.x_norm,
        }


class FeatureStatsAccumulator:
    """Streams probe batches through the SAE, accumulating sums on the SAE's device."""

    def __init__(self, sae: SAE, layer: int, thr_margin_frac: float = 0.1):
        self.sae = sae
        self.layer = layer
        self.thr_margin_frac = thr_margin_frac  # margin = frac * threshold (per feature)
        self.threshold = getattr(sae, "threshold", None)
        d = sae.cfg.d_sae
        # per-batch sums are computed on the SAE's device in fp32 and accumulated on the CPU in
        # fp64 (MPS has no fp64; the d_sae-sized transfer per batch is negligible)
        self.n = 0
        self.count = torch.zeros(d, dtype=torch.float64)
        self.sum_act = torch.zeros(d, dtype=torch.float64)
        self.sum_pre = torch.zeros(d, dtype=torch.float64)
        self.near_thr = torch.zeros(d, dtype=torch.float64)
        # reconstruction sums
        self.ss_res = 0.0
        self.sum_x = torch.zeros(sae.cfg.d_in, dtype=torch.float64)
        self.sum_xx = 0.0
        self.sum_l0 = 0.0
        self.sum_cos = 0.0
        self.sum_norm = 0.0

    @torch.no_grad()
    def update(self, acts: torch.Tensor, mask: torch.Tensor | None) -> None:
        x = acts.reshape(-1, acts.shape[-1])
        if mask is not None:
            x = x[mask.reshape(-1).bool()]
        if x.shape[0] == 0:
            return
        x = x.to(self.sae.device, torch.float32)
        feats = self.sae.encode(x)
        x_hat = self.sae.decode(feats)
        active = feats > 0
        self.n += x.shape[0]
        self.count += active.sum(0).cpu().double()
        self.sum_act += feats.sum(0).cpu().double()
        if self.threshold is not None:
            pre = self.sae.process_sae_in(x) @ self.sae.W_enc + self.sae.b_enc  # JumpReLU pre-activation
            self.sum_pre += pre.sum(0).cpu().double()
            margin = self.thr_margin_frac * self.threshold.abs()
            self.near_thr += ((pre - self.threshold).abs() <= margin).sum(0).cpu().double()
        self.ss_res += float((x - x_hat).pow(2).sum())
        self.sum_x += x.sum(0).cpu().double()
        self.sum_xx += float(x.pow(2).sum())
        self.sum_l0 += float(active.sum())
        self.sum_cos += float(torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).sum())
        self.sum_norm += float(x.norm(dim=-1).sum())

    def finalize(self) -> FeatureStats:
        n = self.n
        if n == 0:
            raise ValueError("no tokens accumulated")
        freq = (self.count / n).float()
        mean_mag = torch.where(self.count > 0, self.sum_act / self.count.clamp(min=1), 0.0).float()
        mean_act = (self.sum_act / n).float()
        mean_x = self.sum_x / n
        ss_tot = self.sum_xx - n * float(mean_x.pow(2).sum())
        fvu = self.ss_res / ss_tot if ss_tot > 0 else float("nan")
        recon = ReconMetrics(
            layer=self.layer,
            n_tokens=n,
            mse=self.ss_res / (n * self.sae.cfg.d_in),
            fvu=fvu,
            variance_explained=1.0 - fvu,
            l0=self.sum_l0 / n,
            cos=self.sum_cos / n,
            x_norm=self.sum_norm / n,
        )
        return FeatureStats(
            layer=self.layer,
            n_tokens=n,
            freq=freq.cpu(),
            mean_mag=mean_mag.cpu(),
            mean_act=mean_act.cpu(),
            mean_pre=(self.sum_pre / n).float(),
            near_thr=(self.near_thr / n).float(),
            recon=recon,
        )
