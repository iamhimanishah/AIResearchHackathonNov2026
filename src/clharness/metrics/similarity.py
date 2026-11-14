"""Representation-similarity measures that are robust to Gemma's massive-activation dimensions.

Why: linear CKA weights dimensions by variance (Kornblith 2019) and is driven by outliers
(Davari et al. 2022). Gemma residuals have a few dimensions ~10^2-10^5x the median magnitude
(Sun et al. 2024), so raw CKA mostly reports what those dimensions did. We therefore report,
next to raw CKA:

  cka_std      linear CKA after per-dimension standardisation with the *base model's* mean/std
               on the probe set (so every dimension counts equally; the standardiser is fixed
               across checkpoints)
  cka_nomassive linear CKA with the top-k massive dimensions (by base-model mean |x|) removed
  mknn         mutual k-nearest-neighbour overlap on a fixed token subsample, computed on the
               standardised, massive-dimension-free subspace: for each token, the fraction of
               its k nearest neighbours (cosine, current model) that are also among its k nearest
               neighbours under the base model. Basis-free, 1.0 iff local neighbourhood structure
               is unchanged.

All accumulate on the CPU in float64 (MPS has no float64; copy first, then cast).
"""

from __future__ import annotations

import torch

from clharness.metrics.cka import LinearCKAAccumulator


class Standardizer:
    """Per-dimension mean/std from base-model activations; frozen after `finalize`."""

    def __init__(self, d: int):
        self.n = 0
        self.s = torch.zeros(d, dtype=torch.float64)
        self.ss = torch.zeros(d, dtype=torch.float64)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.mean_abs = torch.zeros(d, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().cpu().to(torch.float64)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)
        self.mean_abs += x.abs().sum(0)

    def finalize(self) -> None:
        self.mean = self.s / self.n
        var = self.ss / self.n - self.mean**2
        self.std = var.clamp(min=1e-12).sqrt()
        self.mean_abs = self.mean_abs / self.n

    def massive_dims(self, k: int) -> torch.Tensor:
        """Indices of the k dimensions with the largest base-model mean |x|."""
        return torch.topk(self.mean_abs, k).indices

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().cpu().to(torch.float64)
        return (x - self.mean) / self.std


class RobustSimilarity:
    """Accumulates cka_std, cka_nomassive and (on a fixed subsample) mutual-kNN overlap."""

    def __init__(self, std: Standardizer, k_massive: int, knn_tokens: int, knn_k: int):
        assert std.mean is not None, "finalize the Standardizer first"
        d = std.mean.shape[0]
        self.std = std
        self.keep = torch.ones(d, dtype=torch.bool)
        self.keep[std.massive_dims(k_massive)] = False
        self.cka_std = LinearCKAAccumulator(d, d)
        self.cka_nm = LinearCKAAccumulator(int(self.keep.sum()), int(self.keep.sum()))
        self.knn_tokens = knn_tokens
        self.knn_k = knn_k
        self._cur: list[torch.Tensor] = []
        self._ref: list[torch.Tensor] = []
        self._n_knn = 0

    def update(self, cur: torch.Tensor, ref: torch.Tensor) -> None:
        """cur, ref: [n, d] activations of the current and base model on the same tokens."""
        c = self.std.apply(cur)
        r = self.std.apply(ref)
        self.cka_std.update(c, r)
        self.cka_nm.update(c[:, self.keep], r[:, self.keep])
        room = self.knn_tokens - self._n_knn
        if room > 0:
            self._cur.append(c[:room, self.keep])
            self._ref.append(r[:room, self.keep])
            self._n_knn += min(room, c.shape[0])

    @staticmethod
    def _knn_sets(x: torch.Tensor, k: int) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=1)
        sim = x @ x.T
        sim.fill_diagonal_(-2.0)
        return torch.topk(sim, k, dim=1).indices  # [n, k]

    def compute(self) -> dict[str, float]:
        out = {"cka_std": self.cka_std.compute(), "cka_nomassive": self.cka_nm.compute()}
        if self._n_knn >= self.knn_k + 1:
            c = torch.cat(self._cur)
            r = torch.cat(self._ref)
            kc = self._knn_sets(c, self.knn_k)
            kr = self._knn_sets(r, self.knn_k)
            n = kc.shape[0]
            # overlap per token = |kc ∩ kr| / k
            mask_c = torch.zeros(n, n, dtype=torch.bool).scatter_(1, kc, True)
            mask_r = torch.zeros(n, n, dtype=torch.bool).scatter_(1, kr, True)
            out["mknn"] = float((mask_c & mask_r).sum(1).double().mean() / self.knn_k)
            out["mknn_tokens"] = float(n)
        return out
