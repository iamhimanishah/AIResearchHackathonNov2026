"""Mechanism probes: per-option log-likelihoods and per-question representation drift.

CPU, seconds, no downloads: a tiny random Gemma 3 stands in for the real model.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from clharness.evals.lm_eval_runner import _mc_choices
from clharness.executor.run import config_from_spec
from clharness.spec import load_spec
from clharness.train.hooks import CheckpointInfo, RunContext
from clharness.train.measure import EvalHook, ItemDriftHook, default_hooks, loglik_arrays

REPO = Path(__file__).resolve().parents[1]
SPECS = REPO / "configs" / "bench" / "specs"


# --- per-option log-likelihoods ----------------------------------------------------------------

class _FakeTask:
    def doc_to_choice(self, doc):
        return doc["choices"]

    def doc_to_target(self, doc):
        return doc["gold"]


def _sample(doc_id, choices, gold, lls):
    return {"doc_id": doc_id, "doc": {"choices": choices, "gold": gold},
            "filtered_resps": [(ll, False) for ll in lls]}


def test_mc_choices_keeps_the_scores_the_pick_is_made_from():
    samples = [_sample(0, ["a", "bbbb"], 1, [-2.0, -4.0]),  # raw picks 0; per-char picks 1 (-2 vs -1)
               _sample(1, ["x", "y", "z"], "2", [-3.0, -2.0, -1.0])]
    chosen, gold, scores = _mc_choices(_FakeTask(), samples, "acc_norm,none")
    assert chosen == {0: 1, 1: 2} and gold == {0: 1, 1: 2}
    assert scores[0] == ([-2.0, -4.0], [-2.0, -1.0])
    _, _, raw_scores = _mc_choices(_FakeTask(), samples, "acc,none")
    assert raw_scores[0] == ([-2.0, -4.0], [-2.0, -4.0])


def test_loglik_arrays_pad_ragged_items_with_nan_and_reproduce_the_choices():
    raw, score = loglik_arrays([([-1.0, -2.0], [-0.5, -2.0]), None, ([-3.0, -1.0, -2.0], [-3.0, -1.0, -2.0])])
    assert raw.shape == score.shape == (3, 3) and raw.dtype == np.float32
    assert np.isnan(raw[0, 2]) and np.isnan(raw[1]).all()
    assert np.nanargmax(score[[0, 2]], axis=1).tolist() == [0, 1]


def test_eval_hook_saves_loglik_and_does_not_log_it(tmp_path, monkeypatch):
    """The `_loglik` list becomes loglik_<task>.npz and never leaks into the logged scalars."""
    from clharness.train import measure

    fake = {"arc_easy": {"acc": 0.5, "_items": [1.0, 0.0], "_item_metric": "acc", "_choices": [1, 0],
                         "_gold": [1, 1], "_loglik": [([-2.0, -1.0], [-2.0, -1.0]), ([-1.0, -3.0], [-1.0, -3.0])]}}
    monkeypatch.setattr(measure, "run_lm_eval", lambda *a, **k: {t: dict(v) for t, v in fake.items()})
    hook = EvalHook()
    hook.slices, hook.doc_slices, hook.tier2, hook.base_items = {}, {}, [], {}
    hook.tier1 = [SimpleNamespace(task="arc_easy")]
    hook.n_choices = {"arc_easy": 4.0}
    cfg = SimpleNamespace(eval=SimpleNamespace(batch_size=2, include_path=None, max_length=64),
                          train=SimpleNamespace(seed=0))
    ctx = RunContext(cfg=cfg, model=None, tokenizer=None, device="cpu", data_manifest={}, eval_manifest={},
                     output_dir=str(tmp_path))
    out = hook.on_checkpoint(ctx, CheckpointInfo(0, 0, 0, "base", 0, True, str(tmp_path), False))
    saved = np.load(tmp_path / "loglik_arc_easy.npz")
    assert saved["score"].argmax(1).tolist() == np.load(tmp_path / "choices_arc_easy.npy").tolist()
    assert not any("loglik" in k for k in out)


# --- per-question representation drift ---------------------------------------------------------

def _tiny_model():
    cfg = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                           max_position_embeddings=256, sliding_window=64)
    torch.manual_seed(0)
    return Gemma3ForCausalLM(cfg).float().eval()


def _ctx(model, tmp_path, batch_size=4):
    cfg = SimpleNamespace(eval=SimpleNamespace(item_drift_layers=[0, 2], batch_size=batch_size, max_length=64))
    return RunContext(cfg=cfg, model=model, tokenizer=SimpleNamespace(pad_token_id=0), device="cpu",
                      data_manifest={}, eval_manifest={}, output_dir=str(tmp_path))


def _ckpt(tmp_path, index):
    d = tmp_path / f"ckpt_{index:04d}"
    d.mkdir()
    return CheckpointInfo(index, index, 0, "base" if index == 0 else "code", 0, True, str(d), False)


ITEMS = {"arc_easy": [[2, 5, 9], [2, 7, 7, 7, 7, 3], None, [2, 11]], "piqa": [[2, 4, 4, 8, 1]]}


def test_item_drift_is_one_at_base_and_moves_with_the_weights(tmp_path):
    model = _tiny_model()
    ctx = _ctx(model, tmp_path)
    hook = ItemDriftHook(item_tokens=ITEMS)
    assert hook.on_run_start(ctx) == {"n_items": 5.0, "n_layers": 2.0}
    hook.on_checkpoint(ctx, _ckpt(tmp_path, 0))
    base = np.load(tmp_path / "ckpt_0000" / "drift_arc_easy.npz")
    assert base["layers"].tolist() == [0, 2]
    assert base["cos"].shape == base["norm_ratio"].shape == (4, 2)
    assert np.isnan(base["cos"][2]).all()  # the item with no prompt
    ok = [0, 1, 3]
    assert np.allclose(base["cos"][ok], 1.0, atol=1e-5) and np.allclose(base["norm_ratio"][ok], 1.0, atol=1e-5)
    assert (tmp_path / "item_drift_base.pt").exists()

    with torch.no_grad():  # change the last block only: layer 0 must not move, layer 2 must
        for p in model.model.layers[2].parameters():
            p.add_(torch.randn_like(p) * 0.5)
    out = hook.on_checkpoint(ctx, _ckpt(tmp_path, 1))
    after = np.load(tmp_path / "ckpt_0001" / "drift_arc_easy.npz")
    assert np.allclose(after["cos"][ok, 0], 1.0, atol=1e-5)
    assert (after["cos"][ok, 1] < 0.999).all()
    assert out["arc_easy/L0/cos_mean"] == pytest.approx(1.0, abs=1e-5) and out["arc_easy/L2/cos_mean"] < 0.999
    assert (tmp_path / "ckpt_0001" / "drift_piqa.npz").exists()


def test_item_drift_state_does_not_depend_on_batching(tmp_path):
    """Right padding must not change a question's last-token state: alone or in a batch, the same."""
    model = _tiny_model()
    batched = ItemDriftHook(item_tokens=ITEMS)
    batched.on_run_start(_ctx(model, tmp_path, batch_size=4))
    alone = ItemDriftHook(item_tokens=ITEMS)
    alone.on_run_start(_ctx(model, tmp_path, batch_size=1))
    a = batched._states(_ctx(model, tmp_path, batch_size=4), ITEMS["arc_easy"])
    b = alone._states(_ctx(model, tmp_path, batch_size=1), ITEMS["arc_easy"])
    ok = [0, 1, 3]
    assert torch.allclose(a[ok], b[ok], atol=1e-4)


def test_fast_mech_trains_and_scores_exactly_like_the_frozen_bench():
    """fast-mech may differ from fast only by turning the drift probe on."""
    spec = load_spec(SPECS / "no-mitigation-s0.json")
    fast = config_from_spec(spec)
    mech = config_from_spec(spec.model_copy(update={"bench": "fast-mech"}))
    for section in ("model", "data", "optim", "train", "techniques"):
        assert getattr(fast, section) == getattr(mech, section), section
    assert fast.eval.item_drift_layers == [] and mech.eval.item_drift_layers == [7, 13, 22]
    assert fast.eval.model_copy(update={"item_drift_layers": [7, 13, 22]}) == mech.eval
    assert not any(isinstance(h, ItemDriftHook) for h in default_hooks(fast))
    assert any(isinstance(h, ItemDriftHook) for h in default_hooks(mech))
