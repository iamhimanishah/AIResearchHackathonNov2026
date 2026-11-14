"""Scores used by replay's candidate selection (src/clharness/techniques/selection.py)."""

import math

import numpy as np
import torch
from test_loop_tiny import VOCAB, _tiny_model

from clharness.activations import ResidualCapture
from clharness.techniques.selection import (
    ActNovStats,
    candidate_forward,
    loss_scores,
    pick,
    pool_residual,
)


def test_act_nov_needs_two_updates():
    s = ActNovStats()
    assert not s.ready and s.score(torch.zeros(1, 4)) is None
    s.update(torch.ones(1, 4))
    assert s.n == 1 and not s.ready and s.score(torch.zeros(1, 4)) is None
    s.update(torch.zeros(1, 4))
    assert s.ready and s.score(torch.zeros(1, 4)) is not None


def test_act_nov_far_scores_higher_than_near():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 16, generator=g)
    s = ActNovStats()
    s.update(x[:250])
    s.update(x[250:])  # two calls: Welford carries over
    assert s.n == 500
    near, far = torch.zeros(1, 16), torch.full((1, 16), 6.0)
    sc = s.score(torch.cat([near, far]))
    assert sc.dtype == torch.float64 and torch.isfinite(sc).all()
    assert sc[1] > sc[0]
    # matches the closed form with the sample variance (ddof=1), as in signals.py
    xn = x.double().numpy()
    want = np.sqrt((((far.double().numpy() - xn.mean(0)) ** 2) / xn.var(0, ddof=1)).mean())
    assert math.isclose(float(sc[1]), float(want), rel_tol=1e-9)


def test_act_nov_constant_input_is_finite():
    s = ActNovStats()
    s.update(torch.ones(5, 3))  # zero variance: clamped, not a division by zero
    assert torch.isfinite(s.score(torch.zeros(2, 3))).all()


def test_pool_residual_excludes_bos():
    acts = torch.tensor([[[1.0], [3.0], [5.0]]])  # [1, 3, 1]
    ids = torch.tensor([[2, 7, 8]])
    assert pool_residual(acts, ids, bos=2).item() == 4.0
    assert pool_residual(acts, ids, bos=None).item() == 3.0
    assert pool_residual(acts, ids, bos=2).dtype == torch.float64


def test_pick_high_low_and_ties():
    s = torch.tensor([1.0, 3.0, 3.0, 0.0, 3.0], dtype=torch.float64)
    assert pick(s, 2, "high").tolist() == [1, 2]  # ties go to the earlier candidate
    assert pick(s, 1, "low").tolist() == [3]
    assert pick(torch.zeros(4), 3, "low").tolist() == [0, 1, 2]


def test_loss_scores_match_mean_token_loss_and_chunking():
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    model = lm.model.train()
    rows = torch.from_numpy(np.random.default_rng(0).integers(3, VOCAB, size=(5, 17))).long()
    got = loss_scores(model, rows, chunk=2)
    with torch.no_grad():
        logits = model(input_ids=rows[:, :-1], use_cache=False).logits
        want = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), rows[:, 1:].reshape(-1), reduction="none").reshape(5, -1).mean(1)
    assert got.shape == (5,) and got.dtype == torch.float64
    torch.testing.assert_close(got, want.double(), atol=1e-5, rtol=1e-5)
    assert model.training  # train/eval mode left as it was
    assert all(p.grad is None for p in model.parameters())


def test_candidate_forward_pools_the_residual():
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    model = lm.model.eval()
    rows = torch.from_numpy(np.random.default_rng(1).integers(3, VOCAB, size=(3, 9))).long()
    rows[:, 0] = 2  # BOS first, as in packed data
    loss, pooled = candidate_forward(model, rows, chunk=1, layer=1, bos=2)
    with torch.no_grad(), ResidualCapture(model, [1]) as cap:
        model(input_ids=rows[:, :-1], use_cache=False)
    want = cap.acts[1][:, 1:].double().mean(1)  # BOS position excluded
    assert loss.shape == (3,) and pooled.shape == (3, 64)
    torch.testing.assert_close(pooled, want, atol=1e-5, rtol=1e-5)
    assert not model.training
