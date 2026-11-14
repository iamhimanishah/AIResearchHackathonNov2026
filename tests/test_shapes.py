"""Cheap unit tests: metric properties, stream packing / token counting, config strictness.
No model download."""

import numpy as np
import pytest
import torch

from clharness.config import REPO_ROOT, RunConfig, load_run_config, load_yaml_with_includes
from clharness.data.loader import PackedStream, fixed_chunks
from clharness.metrics.cka import LinearCKAAccumulator, linear_cka
from clharness.train.loop import lr_at

# ---- CKA ---------------------------------------------------------------------------------------


def test_cka_invariances():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 16, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g, dtype=torch.float64))
    assert abs(linear_cka(x, x) - 1.0) < 1e-9
    assert abs(linear_cka(x, x @ q * 3.0 + 1.0) - 1.0) < 1e-9  # rotation, scale, shift
    y = torch.randn(500, 16, generator=g, dtype=torch.float64)
    assert linear_cka(x, y) < 0.2  # independent


def test_cka_streaming_equals_batch():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(300, 8, generator=g)
    y = x @ torch.randn(8, 8, generator=g) + 0.1 * torch.randn(300, 8, generator=g)
    acc = LinearCKAAccumulator(8, 8)
    for i in range(0, 300, 70):
        acc.update(x[i : i + 70], y[i : i + 70])
    assert abs(acc.compute() - linear_cka(x, y)) < 1e-9


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_cka_accumulator_from_mps_tensors():
    """Regression: a fused MPS->CPU float64 cast returned garbage (self-CKA ~1e-5)."""
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(2000, 64, generator=g) * 3e4).bfloat16()  # Gemma-scale residual magnitudes
    acc = LinearCKAAccumulator(64, 64)
    acc.update(x.to("mps").float(), x.float())
    assert abs(acc.compute() - 1.0) < 1e-6


# ---- packing / gradient tokens ----------------------------------------------------------------


def test_packed_stream_counts_and_seed():
    toks = np.arange(10_000, dtype=np.int32)
    s1 = PackedStream(toks, seq_len=64, seed=3)
    s2 = PackedStream(toks, seq_len=64, seed=3)
    assert s1.n == 10_000 // 65
    assert s1.tokens_available == s1.n * 64
    b1 = list(s1.batches(4))
    b2 = list(s2.batches(4))
    assert all(torch.equal(a, b) for a, b in zip(b1, b2))
    assert sum(b.shape[0] for b in b1) == s1.n
    assert all(b.shape[1] == 65 for b in b1)
    # gradient tokens per micro-batch = rows * seq_len (labels = [:,1:])
    assert sum(b[:, 1:].numel() for b in b1) == s1.tokens_available
    # a chunk is contiguous in the stream
    row = b1[0][0]
    assert torch.equal(row, torch.arange(int(row[0]), int(row[0]) + 65))
    s3 = PackedStream(toks, seq_len=64, seed=4)
    assert not torch.equal(next(s3.batches(4)), b1[0])


def test_fixed_chunks():
    toks = np.arange(1000, dtype=np.int32)
    c = fixed_chunks(toks, 128)
    assert c.shape == (7, 128) and c.dtype == torch.int64
    assert fixed_chunks(toks, 128, max_seqs=2).shape == (2, 128)


def test_lr_schedule():
    class O:
        lr, warmup_steps, schedule = 1.0, 2, "cosine"

    assert lr_at(0, 10, O) == 0.5 and lr_at(1, 10, O) == 1.0
    assert abs(lr_at(10, 10, O)) < 1e-12
    O.schedule = "constant"
    assert lr_at(9, 10, O) == 1.0


# ---- config strictness ------------------------------------------------------------------------


def test_smoke_config_loads_and_is_frozen():
    cfg = load_run_config(REPO_ROOT / "configs" / "smoke.yaml")
    assert cfg.model.capture_layers == [7, 13, 22]
    assert [d.name for d in cfg.data.domains] == ["code", "biomedical", "legal", "math"]
    with pytest.raises(Exception):
        cfg.optim.lr = 1.0  # frozen


def test_unknown_key_and_missing_key_are_errors():
    raw = load_yaml_with_includes(REPO_ROOT / "configs" / "smoke.yaml")
    bad = {**raw, "optim": {**raw["optim"], "learning_rate": 1.0}}
    with pytest.raises(Exception, match="learning_rate"):
        RunConfig.model_validate(bad)
    missing = {**raw, "optim": {k: v for k, v in raw["optim"].items() if k != "lr"}}
    with pytest.raises(Exception, match="lr"):
        RunConfig.model_validate(missing)


# ---- robust similarity ---------------------------------------------------------------------------


def test_robust_similarity_identity_and_outlier_insensitivity():
    from clharness.metrics.similarity import RobustSimilarity, Standardizer

    g = torch.Generator().manual_seed(5)
    n, d = 1500, 32
    base = torch.randn(n, d, generator=g)
    base[:, 0] *= 1000.0  # one massive dimension, Gemma-style
    std = Standardizer(d)
    std.update(base)
    std.finalize()
    assert std.massive_dims(1).item() == 0
    # identical activations -> all measures 1
    rs = RobustSimilarity(std, k_massive=1, knn_tokens=1000, knn_k=5)
    rs.update(base, base)
    r = rs.compute()
    assert abs(r["cka_std"] - 1) < 1e-9 and abs(r["cka_nomassive"] - 1) < 1e-9 and abs(r["mknn"] - 1) < 1e-9
    # perturb ONLY the massive dimension: raw CKA moves a lot, robust measures barely
    cur = base.clone()
    cur[:, 0] = cur[:, 0] * 0.3 + 500.0 * torch.randn(n, generator=g)
    raw = linear_cka(cur, base)
    rs2 = RobustSimilarity(std, k_massive=1, knn_tokens=1000, knn_k=5)
    rs2.update(cur, base)
    r2 = rs2.compute()
    assert raw < 0.7, raw
    assert r2["cka_nomassive"] > 0.999 and r2["mknn"] > 0.9, r2
