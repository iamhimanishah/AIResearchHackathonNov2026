"""SignalLogHook on a stand-in SAE: shapes, finiteness, novelty behaviour, and the MPS/CPU
device-dtype path (fp64 must be cast after copying to CPU)."""

import types

import numpy as np
import pytest
import torch

from clharness.train.signals import SignalLogHook

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


class _FakeSAE(torch.nn.Module):
    """Linear ReLU 'SAE' with the attributes the hook touches."""

    def __init__(self, d, m, device):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W = (torch.randn(d, m, generator=g) / d**0.5).to(device)
        self.cfg = types.SimpleNamespace(d_in=d, d_sae=m)

    def encode(self, x):
        return torch.relu(x @ self.W)

    def decode(self, f):
        return f @ self.W.T


@pytest.mark.parametrize("device", DEVICES)
def test_signal_hook_scores_and_running_stats(tmp_path, device):
    d, m, B, T, V = 16, 64, 3, 8, 50
    hook = SignalLogHook()
    hook.enabled, hook.layers, hook.saes = True, [7], {7: _FakeSAE(d, m, device)}
    hook.n, hook.mean, hook.m2 = 0, {7: torch.zeros(d, dtype=torch.float64)}, {7: torch.zeros(d, dtype=torch.float64)}
    hook.feat_count, hook.feat_tokens, hook.bos, hook.rows = {7: torch.zeros(m, dtype=torch.float64)}, 0, 2, []
    hook.cols = ["step", "domain", "chunk", "loss", "entropy", "act_nov_L7", "sae_rare_L7", "sae_fvu_L7"]
    hook.out = tmp_path / "signals.npz"
    g = torch.Generator().manual_seed(1)
    ctx = types.SimpleNamespace(output_dir=str(tmp_path))
    for step in range(4):
        batch = torch.randint(3, V, (B, T + 1), generator=g)
        batch[:, 0] = 2  # BOS
        logits = torch.randn(B, T, V, generator=g).to(device)
        acts = {7: (torch.randn(B, T, d, generator=g) + (5.0 if step == 3 else 0.0)).to(device)}
        seq_loss = torch.rand(B, generator=g).to(device)
        hook.on_micro_batch(ctx, step + 1, 0, torch.arange(B), batch, logits, seq_loss, acts)
    hook.on_checkpoint(ctx, None)
    z = np.load(hook.out)
    data, cols = z["data"], list(z["columns"])
    assert data.shape == (4 * B, len(cols)) and np.isfinite(data).all()
    nov = data[:, cols.index("act_nov_L7")]
    assert nov[:B].max() == 0.0  # nothing to compare against on the first micro-batch
    assert nov[-B:].mean() > 3 * nov[B:-B].mean()  # the shifted batch is novel
    assert hook.n == 4 * B and hook.feat_tokens == 4 * B * (T - 1)  # BOS (position 0) excluded from feature stats
    fvu = data[:, cols.index("sae_fvu_L7")]
    assert (fvu >= 0).all() and np.isfinite(fvu).all()  # a random stand-in SAE reconstructs badly; only sign/finiteness are meaningful here
