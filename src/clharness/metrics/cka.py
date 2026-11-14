"""Linear CKA (Kornblith et al. 2019) between two activation matrices.

Drift-immune in the sense that it needs no fixed basis: it compares the *geometry* of the
probe-set representations under the current checkpoint vs. the base model, and is invariant to
orthogonal transforms and isotropic scaling. Computed in float64 on the CPU for stability.

Memory note: we use the feature-space form (d x d Gram matrices, d=1152), so cost is
O(n d^2) regardless of the number of probe tokens n. Streaming accumulation lets us avoid
holding all probe activations at once.
"""

from __future__ import annotations

import torch


class LinearCKAAccumulator:
    """Accumulate cross/self second-moment matrices for X (current) and Y (base) token by token.

    linear_CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F  ||Yc^T Yc||_F)   (columns centred)
    Centering is applied at finalisation from accumulated sums, so batches can arrive in any
    order. Both inputs must be [n, d] with matching token order.
    """

    def __init__(self, d_x: int, d_y: int):
        self.n = 0
        self.sx = torch.zeros(d_x, dtype=torch.float64)
        self.sy = torch.zeros(d_y, dtype=torch.float64)
        self.xx = torch.zeros(d_x, d_x, dtype=torch.float64)
        self.yy = torch.zeros(d_y, d_y, dtype=torch.float64)
        self.xy = torch.zeros(d_x, d_y, dtype=torch.float64)

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"token count mismatch {x.shape[0]} vs {y.shape[0]}")
        # Copy to CPU *then* cast. A fused `.to("cpu", torch.float64)` from an MPS tensor
        # returned wrong values (self-CKA 1e-5) on torch 2.14 / macOS 15; two steps are correct.
        x = x.detach().float().cpu().to(torch.float64)
        y = y.detach().float().cpu().to(torch.float64)
        self.n += x.shape[0]
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.xx += x.T @ x
        self.yy += y.T @ y
        self.xy += x.T @ y

    def compute(self) -> float:
        if self.n < 2:
            raise ValueError("need at least 2 tokens")
        n = self.n
        mx, my = self.sx / n, self.sy / n
        cxx = self.xx - n * torch.outer(mx, mx)
        cyy = self.yy - n * torch.outer(my, my)
        cxy = self.xy - n * torch.outer(mx, my)
        num = cxy.pow(2).sum()
        den = cxx.norm() * cyy.norm()
        return float(num / den)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """One-shot convenience for small inputs. x, y: [n, d]."""
    acc = LinearCKAAccumulator(x.shape[1], y.shape[1])
    acc.update(x, y)
    return acc.compute()
