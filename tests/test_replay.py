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


# -- selection: choosing which replay candidates to keep ----------------------------------------


class SelectionSpy(BaseHook):
    """Listed after replay. At every micro-batch where replay just selected rows, snapshots the
    replay hook's last selection, recomputes each candidate's mean token loss independently
    under the model as it is at that moment, and records the act_nov statistics' count."""

    name = "selspy"

    def __init__(self, replay):
        self.replay = replay
        self.records: list[dict] = []

    def select_batch(self, ctx, batch):
        mask = getattr(batch, "replay_mask", None)
        r = self.replay
        if mask is None or not bool(mask.any()) or r.last_pool is None:  # no selection (or random)
            return batch
        pool = r.last_pool.clone()
        with torch.no_grad():
            logits = ctx.model(input_ids=pool[:, :-1], use_cache=False).logits.float()
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), pool[:, 1:].reshape(-1), reduction="none"
            ).reshape(pool.shape[0], -1)
        self.records.append({
            "k": int(mask.sum()), "pool": pool,
            "scores": None if r.last_scores is None else r.last_scores.clone(),
            "chosen": r.last_chosen.clone(), "recomputed": nll.mean(1).double(),
            "replayed": batch[mask].clone(), "stats_n": None if r.stats is None else r.stats.n,
        })
        return batch


def _train_sel(tmp_path, monkeypatch, name, params, techniques=(), seed=None):
    """Train with replay(params) plus optional other techniques; return (spy, selspy, replay
    hook, trainer state, metrics)."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    raw = _cfg(tmp_path, name).dump()
    raw["techniques"] = [{"name": "replay", "params": params}, *techniques]
    if seed is not None:
        raw["train"]["seed"] = seed
    cfg = RunConfig.model_validate(raw)
    hooks = build_hooks(cfg.techniques)
    replay = next(h for h in hooks if h.name == "replay")
    spy, selspy = Spy(), SelectionSpy(replay)
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    trainer = Trainer(cfg, [*hooks, selspy, spy], eval_manifest=None, model_override=lm)
    trainer.run()
    metrics = [json.loads(line) for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
    return spy, selspy, replay, trainer.ctx.state, metrics


def test_random_select_is_unchanged(tmp_path, monkeypatch):
    _add_general(tmp_path)
    plain, _ = _train(tmp_path, monkeypatch, "plain", [{"name": "replay", "params": {"share": 0.25}}])
    spy, _, replay, state, metrics = _train_sel(
        tmp_path, monkeypatch, "rand", {"share": 0.25, "select": "random", "pool": 7, "keep": "low", "layer": 1})
    assert _replayed_rows(spy) == _replayed_rows(plain)
    assert state.get("extra_forward_tokens", 0) == 0
    assert replay.wants_activations() == [] and replay.last_pool is None
    assert not any("replay/scored_rows" in m for m in metrics)


def test_loss_select_keeps_the_top_scores(tmp_path, monkeypatch):
    general = _add_general(tmp_path)
    base, _ = _train(tmp_path, monkeypatch, "base", [])
    pool = 4
    spy, selspy, _, state, metrics = _train_sel(
        tmp_path, monkeypatch, "loss-high", {"share": 0.25, "select": "loss", "pool": pool})
    assert selspy.records
    for rec in selspy.records:
        k, scores = rec["k"], rec["scores"]
        assert scores.shape == (k * pool,) and rec["pool"].shape[0] == k * pool
        torch.testing.assert_close(scores, rec["recomputed"], atol=1e-5, rtol=1e-5)
        assert sorted(rec["chosen"].tolist()) == sorted(torch.topk(scores, k).indices.tolist())
        # the kept rows are what is trained on
        assert torch.equal(rec["replayed"], rec["pool"][rec["chosen"]])
    # share, gradient tokens and the extra-forward count
    rows = _replayed_rows(spy)
    assert len(rows) == 8 and set(rows) <= _chunks(general)
    assert [s.gradient_tokens for s in spy.steps] == [s.gradient_tokens for s in base.steps]
    assert _last(metrics, "replay/share") == 0.25
    assert _last(metrics, "replay/scored_rows") == pool * 8
    assert _last(metrics, "replay/fallback_rows") == 0
    assert state["extra_forward_tokens"] == pool * 8 * SEQ
    assert _last(metrics, "replay/score_kept") >= _last(metrics, "replay/score_pool")


def test_loss_select_keep_low(tmp_path, monkeypatch):
    _add_general(tmp_path)
    _, high, _, _, _ = _train_sel(tmp_path, monkeypatch, "high", {"share": 0.25, "select": "loss"})
    _, low, _, _, metrics = _train_sel(tmp_path, monkeypatch, "low", {"share": 0.25, "select": "loss", "keep": "low"})
    for rec in low.records:
        k, scores = rec["k"], rec["scores"]
        torch.testing.assert_close(scores, rec["recomputed"], atol=1e-5, rtol=1e-5)
        assert sorted(rec["chosen"].tolist()) == sorted(torch.topk(scores, k, largest=False).indices.tolist())
        assert torch.equal(rec["replayed"], rec["pool"][rec["chosen"]])
    assert len(high.records) == len(low.records)
    assert all(torch.equal(h["pool"], lo["pool"]) for h, lo in zip(high.records, low.records, strict=True))
    assert [r["replayed"].tolist() for r in high.records] != [r["replayed"].tolist() for r in low.records]
    assert _last(metrics, "replay/score_kept") <= _last(metrics, "replay/score_pool")


def test_loss_select_follows_the_seed(tmp_path, monkeypatch):
    _add_general(tmp_path)
    params = {"share": 0.25, "select": "loss", "pool": 3}
    a, sa, _, _, _ = _train_sel(tmp_path, monkeypatch, "a", params)
    b, sb, _, _, _ = _train_sel(tmp_path, monkeypatch, "b", params)
    assert _replayed_rows(a) == _replayed_rows(b) and len(_replayed_rows(a)) == 8
    assert [r["chosen"].tolist() for r in sa.records] == [r["chosen"].tolist() for r in sb.records]


def test_loss_select_from_a_buffer(tmp_path, monkeypatch):
    general = _add_general(tmp_path)
    for n_buf, pool in ((3, 4), (10, 2)):  # buffer smaller than the pool, then larger
        spy, selspy, replay, _, _ = _train_sel(
            tmp_path, monkeypatch, f"buf{n_buf}", {"share": 0.25, "buffer_seqs": n_buf, "select": "loss", "pool": pool})
        buffer_rows = {tuple(r.tolist()) for r in replay._rows(replay.buffer)}
        rows = _replayed_rows(spy)
        assert len(rows) == 8 and set(rows) <= buffer_rows <= _chunks(general)
        assert len(buffer_rows) == n_buf
        for rec in selspy.records:
            cands = [tuple(r.tolist()) for r in rec["pool"]]
            assert set(cands) <= buffer_rows
            if len(cands) <= n_buf:
                assert len(set(cands)) == len(cands)  # without replacement when the buffer allows


def test_act_nov_select_end_to_end(tmp_path, monkeypatch):
    _add_general(tmp_path)
    pool = 3
    spy, selspy, replay, state, metrics = _train_sel(
        tmp_path, monkeypatch, "actnov", {"share": 0.5, "select": "act_nov", "layer": 1, "pool": pool})
    assert replay.wants_activations() == [1]
    first = selspy.records[0]
    assert first["stats_n"] == 0 and first["scores"] is None  # before any training forward
    fallbacks = 0
    for rec in selspy.records:
        if rec["stats_n"] < 2:
            assert rec["scores"] is None
            fallbacks += rec["k"]
        else:
            k, scores = rec["k"], rec["scores"]
            assert scores.shape == (k * pool,) and torch.isfinite(scores).all()
            assert sorted(rec["chosen"].tolist()) == sorted(torch.topk(scores, k).indices.tolist())
        assert torch.equal(rec["replayed"], rec["pool"][rec["chosen"]])
    n_replayed = len(_replayed_rows(spy))
    assert n_replayed == 16  # half of the 32 rows
    assert fallbacks == 2  # one replayed and one trained row per micro-batch: stats after 2 forwards
    assert _last(metrics, "replay/fallback_rows") == fallbacks
    assert _last(metrics, "replay/scored_rows") == pool * (n_replayed - fallbacks)
    assert state["extra_forward_tokens"] == pool * (n_replayed - fallbacks) * SEQ
    trained = sum(int((~b["mask"]).sum()) for b in spy.batches)
    assert trained == 16 and replay.stats.n == trained  # replayed rows never enter the statistics


def test_act_nov_counts_rows_of_the_source_domain(tmp_path, monkeypatch):
    """With `source: code` replay is off during code: all code rows feed the statistics, so the
    first selection (during math) already has them and nothing falls back."""
    spy, selspy, replay, _, metrics = _train_sel(
        tmp_path, monkeypatch, "actnov-code", {"share": 0.25, "source": "code", "select": "act_nov", "layer": 1})
    # first selection: second math micro-batch (floor(0.25 * 4) = 1), after 16 code + 2 math rows
    assert selspy.records and selspy.records[0]["stats_n"] == 16 + 2
    assert _last(metrics, "replay/fallback_rows") == 0
    assert replay.stats.n == 32 - len(_replayed_rows(spy))


def test_bad_selection_params_raise():
    for params in ({"select": "loss", "pool": 0}, {"select": "nope"}, {"keep": "middle"}, {"layer": -1}):
        with pytest.raises(ValueError, match="replay"):
            build_hooks([TechniqueSpec(name="replay", params=params)])
