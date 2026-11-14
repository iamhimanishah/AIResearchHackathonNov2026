"""Replay technique on the tiny random model (CPU, seconds, no downloads).

The tiny manifest gets a third domain, `general`, whose tokens come from a narrow range [3, 50)
so a replayed row can be told apart from a code or math row (uniform in [3, 512))."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from test_loop_hooks import SEQ, _cfg
from test_loop_tiny import VOCAB, _abs_entry, _synthetic_manifest, _tiny_model

from clharness.config import RunConfig
from clharness.data.manifest import read_manifest, write_manifest
from clharness.spec import TechniqueSpec
from clharness.techniques import build_hooks
from clharness.train.hooks import BaseHook
from clharness.train.loop import Trainer

CHUNK = SEQ + 1
GENERAL_HI = 50


def _add_general(tmp_path: Path, n_chunks: int = 40) -> np.ndarray:
    """Add a `general` domain (stream split only) to the tiny manifest; return its tokens."""
    path = tmp_path / "manifests" / "tiny.json"
    if not path.exists():
        _synthetic_manifest(tmp_path, SEQ, seed=7)
    man = read_manifest(path)
    d = tmp_path / "data" / "tiny" / "general"
    d.mkdir(parents=True)
    toks = np.random.default_rng(11).integers(3, GENERAL_HI, size=n_chunks * CHUNK, dtype=np.int32)
    np.save(d / "stream.tokens.npy", toks)
    (d / "stream.jsonl").write_text("{}\n")
    split = {"n_docs": 1, "n_tokens": len(toks), "jsonl": _abs_entry(d / "stream.jsonl"),
             "tokens": _abs_entry(d / "stream.tokens.npy")}
    man["domains"]["general"] = {"source": {"name": "synthetic"}, "splits": {"stream": split}}
    write_manifest(man, path)
    return toks


def _chunks(tokens: np.ndarray) -> set[tuple[int, ...]]:
    """Every training chunk of a token file, as a set of row tuples."""
    n = len(tokens) // CHUNK
    return {tuple(tokens[i * CHUNK : (i + 1) * CHUNK].tolist()) for i in range(n)}


def _domain_tokens(tmp_path: Path, dom: str) -> np.ndarray:
    return np.load(tmp_path / "data" / "tiny" / dom / "stream.tokens.npy")


class Spy(BaseHook):
    """Listed after replay: records what the model is actually trained on."""

    name = "spy"

    def __init__(self):
        self.domain = None
        self.batches: list[dict] = []
        self.steps = []

    def on_domain_start(self, ctx, domain):
        self.domain = domain

    def select_batch(self, ctx, batch):
        self.batches.append({
            "domain": self.domain, "rows": batch.clone(),
            "chunk_ids": getattr(batch, "chunk_ids", None),
            "mask": getattr(batch, "replay_mask", None),
        })
        return batch

    def on_step_end(self, ctx, stats):
        self.steps.append(stats)


def _train(tmp_path, monkeypatch, name, techniques, seed=None, **optim):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    raw = _cfg(tmp_path, name, **optim).dump()  # code then math, 16 rows each
    raw["techniques"] = techniques
    if seed is not None:
        raw["train"]["seed"] = seed
    cfg = RunConfig.model_validate(raw)
    hooks = build_hooks(cfg.techniques)
    spy = Spy()
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    Trainer(cfg, [*hooks, spy], eval_manifest=None, model_override=lm).run()
    metrics = [json.loads(line) for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
    return spy, metrics


def _replayed_rows(spy: Spy) -> list[tuple[int, ...]]:
    return [tuple(r.tolist()) for b in spy.batches if b["mask"] is not None
            for r, m in zip(b["rows"], b["mask"], strict=True) if m]


def _last(metrics: list[dict], key: str) -> float:
    return [m[key] for m in metrics if key in m][-1]


def test_replay_swaps_rows_without_changing_gradient_tokens(tmp_path, monkeypatch):
    general = _add_general(tmp_path)
    base, _ = _train(tmp_path, monkeypatch, "base", [])
    spy, metrics = _train(tmp_path, monkeypatch, "replay", [{"name": "replay", "params": {"share": 0.25}}])

    assert spy.steps[-1].gradient_tokens == base.steps[-1].gradient_tokens == 8 * 4 * SEQ
    assert [s.gradient_tokens for s in spy.steps] == [s.gradient_tokens for s in base.steps]

    general_chunks = _chunks(general)
    n_rows = n_replayed = 0
    for b in spy.batches:
        assert b["mask"] is not None and b["chunk_ids"] is not None  # set on every active batch
        for row, cid, m in zip(b["rows"], b["chunk_ids"], b["mask"], strict=True):
            n_rows += 1
            from_general = tuple(row.tolist()) in general_chunks
            assert bool(m) == from_general == (int(cid) == -1)
            if from_general:
                assert int(row.max()) < GENERAL_HI
            n_replayed += int(m)
    assert n_rows == 32 and n_replayed == 8  # exactly 25% of the rows
    assert _last(metrics, "replay/share") == 0.25
    assert _last(metrics, "replay/rows") == 8.0


def test_small_share_uses_a_running_total(tmp_path, monkeypatch):
    _add_general(tmp_path)
    spy, metrics = _train(tmp_path, monkeypatch, "small", [{"name": "replay", "params": {"share": 0.05}}])
    seen = replayed = 0
    for b in spy.batches:
        assert b["rows"].shape[0] == 2  # micro-batch 2: round(0.05 * 2) would be 0 every time
        seen += 2
        replayed += int(b["mask"].sum())
        assert replayed == math.floor(0.05 * seen + 1e-9)
    assert seen == 32 and replayed == math.floor(0.05 * 32) == 1
    assert _last(metrics, "replay/rows") == 1.0


def test_source_code_replays_code_only_after_code(tmp_path, monkeypatch):
    spy, metrics = _train(tmp_path, monkeypatch, "code-src",
                          [{"name": "replay", "params": {"share": 0.25, "source": "code"}}])
    code_chunks = _chunks(_domain_tokens(tmp_path, "code"))
    during_code = [b for b in spy.batches if b["domain"] == "code"]
    during_math = [b for b in spy.batches if b["domain"] == "math"]
    assert during_code and all(b["mask"] is None for b in during_code)  # untouched: no self-replay
    rows = _replayed_rows(spy)
    assert len(rows) == 4  # 25% of the 16 math rows
    assert all(r in code_chunks for r in rows)
    assert sum(int(b["mask"].sum()) for b in during_math) == 4
    assert _last(metrics, "replay/share") == 0.25  # over rows seen while active


def test_replay_rows_follow_the_seed(tmp_path, monkeypatch):
    _add_general(tmp_path)
    tech = [{"name": "replay", "params": {"share": 0.25}}]
    a, _ = _train(tmp_path, monkeypatch, "a", tech)
    b, _ = _train(tmp_path, monkeypatch, "b", tech)
    c, _ = _train(tmp_path, monkeypatch, "c", tech, seed=_cfg(tmp_path, "probe").train.seed + 1)
    assert _replayed_rows(a) == _replayed_rows(b)
    assert len(_replayed_rows(c)) == len(_replayed_rows(a))
    assert _replayed_rows(c) != _replayed_rows(a)


def test_fixed_buffer(tmp_path, monkeypatch):
    general = _add_general(tmp_path)
    spy, _ = _train(tmp_path, monkeypatch, "buf",
                    [{"name": "replay", "params": {"share": 0.25, "buffer_seqs": 2}}])
    rows = _replayed_rows(spy)
    assert len(rows) == 8
    assert len(set(rows)) <= 2
    assert set(rows) <= _chunks(general)


def test_short_source_wraps_around(tmp_path, monkeypatch):
    general = _add_general(tmp_path, n_chunks=3)  # 8 replayed rows needed, 3 available
    spy, metrics = _train(tmp_path, monkeypatch, "wrap", [{"name": "replay", "params": {"share": 0.25}}])
    rows = _replayed_rows(spy)
    assert len(rows) == 8 and set(rows) <= _chunks(general)
    assert len(set(rows)) == 3
    assert _last(metrics, "replay/source_epochs") > 1


def test_bad_source_and_params_raise(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"'nope'.*\['code', 'math'\]"):
        _train(tmp_path, monkeypatch, "nope", [{"name": "replay", "params": {"source": "nope"}}])
    for params in ({"share": 0}, {"share": 1.5}, {"shar": 0.1}, {"buffer_seqs": 0}):
        with pytest.raises(ValueError, match="replay"):
            build_hooks([TechniqueSpec(name="replay", params=params)])
