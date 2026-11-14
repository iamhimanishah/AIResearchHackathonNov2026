"""Weight averaging (`wavg`) on the tiny random model (CPU, seconds, no downloads)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_loop_hooks import _cfg, _state
from test_loop_tiny import VOCAB, _tiny_model

from clharness.spec import TechniqueSpec
from clharness.techniques import build_hooks
from clharness.train.hooks import BaseHook, CheckpointInfo
from clharness.train.loop import Trainer


class PreSpy(BaseHook):
    """Placed before wavg: its on_domain_end sees the weights before averaging."""

    name = "pre"

    def __init__(self):
        self.pre: list[dict[str, torch.Tensor]] = []

    def on_domain_end(self, ctx, domain):
        self.pre.append(_state(ctx.model))


class CkptSpy(BaseHook):
    """Records the weights and checkpoint info at every checkpoint (after any averaging)."""

    name = "ckpt"

    def __init__(self):
        self.infos: list[CheckpointInfo] = []
        self.states: list[dict[str, torch.Tensor]] = []

    def on_checkpoint(self, ctx, ckpt):
        self.infos.append(ckpt)
        self.states.append(_state(ctx.model))


def _wavg(**params):
    (hook,) = build_hooks([TechniqueSpec(name="wavg", params=params)])
    return hook


def _train(tmp_path, monkeypatch, name, hooks, lm=None, every=1000):
    """Two domains of four steps; `every` sets the mid-domain checkpoint interval."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = lm or _tiny_model(vocab=VOCAB, dtype=torch.float32)
    cfg = _cfg(tmp_path, name)
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"checkpoint_every_steps": every})})
    spy = CkptSpy()
    trainer = Trainer(cfg, [*hooks, spy], eval_manifest=None, model_override=lm)
    trainer.run()
    return trainer, spy, lm


def _equal(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _boundaries(spy):
    """(info, state) of the boundary checkpoints after training (not the base checkpoint)."""
    return [(i, s) for i, s in zip(spy.infos, spy.states) if i.is_domain_boundary and i.index > 0]


def test_alpha_one_restores_the_base_at_every_boundary(tmp_path, monkeypatch):
    _, spy, _ = _train(tmp_path, monkeypatch, "wavg-a1", [PreSpy(), _wavg(alpha=1.0)])
    base = spy.states[0]
    bounds = _boundaries(spy)
    assert len(bounds) == 2
    for _, state in bounds:
        assert _equal(state, base)


def test_alpha_zero_matches_no_technique(tmp_path, monkeypatch):
    _, _, plain = _train(tmp_path, monkeypatch, "wavg-plain", [])
    _, _, avg = _train(tmp_path, monkeypatch, "wavg-a0", [_wavg(alpha=0.0)])
    a, b = plain.model.state_dict(), avg.model.state_dict()
    assert _equal(a, b)
    assert not _equal(a, _state(_tiny_model(vocab=VOCAB, dtype=torch.float32).model))  # it trained


def test_final_mode_averages_only_after_the_last_domain(tmp_path, monkeypatch):
    pre = PreSpy()
    hook = _wavg(alpha=0.5, mode="final")
    _, spy, _ = _train(tmp_path, monkeypatch, "wavg-final", [pre, hook])
    base = spy.states[0]
    (_, first), (_, last) = _boundaries(spy)
    assert len(pre.pre) == 2
    assert _equal(first, pre.pre[0])  # untouched after the first domain
    assert not _equal(last, pre.pre[1])
    for k in last:
        assert torch.allclose(last[k], 0.5 * base[k] + 0.5 * pre.pre[1][k], atol=1e-6), k
    assert [h["domain_index"] for h in hook.history] == [1]


def test_boundary_mode_diagnostics_reach_the_boundary_checkpoints(tmp_path, monkeypatch):
    hook = _wavg(alpha=0.5)
    trainer, spy, _ = _train(tmp_path, monkeypatch, "wavg-diag", [hook], every=3)
    assert [h["domain_index"] for h in hook.history] == [0, 1]
    for h in hook.history:
        assert h["dist_before"] > 0 and h["dist_after"] > 0
        assert h["dist_after"] == pytest.approx(0.5 * h["dist_before"], rel=1e-4)

    # checkpoints: base, step 3, boundary 4, step 6, boundary 8 (see test_loop_tiny)
    assert [i.step for i in spy.infos] == [0, 3, 4, 6, 8]
    metrics = [json.loads(p.read_text())
               for p in sorted((Path(trainer.out_dir) / "checkpoints").glob("*/metrics.json"))]
    assert len(metrics) == 5
    with_diag = [i for i, m in enumerate(metrics) if "wavg/dist_before" in m]
    assert with_diag == [2, 4]  # the two post-training boundaries only
    for m, h in zip((metrics[2], metrics[4]), hook.history):
        assert m["wavg/dist_before"] == pytest.approx(h["dist_before"])
        assert m["wavg/dist_after"] == pytest.approx(h["dist_after"])


def test_boundary_mode_on_domain_end_returns_the_diagnostics():
    hook = _wavg(alpha=0.5)
    lin = torch.nn.Linear(4, 3)
    ctx = SimpleNamespace(model=lin, cfg=SimpleNamespace(data=SimpleNamespace(domains=[1, 2])))
    hook.on_run_start(ctx)
    hook.on_domain_start(ctx, "a")
    with torch.no_grad():
        lin.weight.add_(1.0)
    out = hook.on_domain_end(ctx, "a")
    assert out == {"dist_before": pytest.approx(12**0.5), "dist_after": pytest.approx(0.5 * 12**0.5)}


def test_ema_decay_zero_matches_no_technique(tmp_path, monkeypatch):
    _, _, plain = _train(tmp_path, monkeypatch, "wavg-plain", [])
    _, _, ema = _train(tmp_path, monkeypatch, "wavg-ema0", [_wavg(mode="ema", ema_decay=0.0)])
    assert _equal(plain.model.state_dict(), ema.model.state_dict())


def test_ema_decay_one_restores_the_base_at_every_boundary(tmp_path, monkeypatch):
    hook = _wavg(mode="ema", ema_decay=1.0)
    _, spy, _ = _train(tmp_path, monkeypatch, "wavg-ema1", [hook])
    base = spy.states[0]
    bounds = _boundaries(spy)
    assert len(bounds) == 2
    for _, state in bounds:
        assert _equal(state, base)
    assert len(hook.history) == 2
    assert all(h["dist_before"] > 0 and h["dist_after"] == 0.0 for h in hook.history)


def test_frozen_parameters_are_skipped(tmp_path, monkeypatch):
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    frozen = "model.layers.0.mlp.up_proj.weight"
    dict(lm.model.named_parameters())[frozen].requires_grad = False
    before = lm.model.state_dict()[frozen].clone()
    hook = _wavg(alpha=0.5)
    _train(tmp_path, monkeypatch, "wavg-frozen", [hook], lm=lm)
    assert frozen not in hook.base and frozen not in hook.params
    assert hook.base  # the rest were snapshotted
    assert torch.equal(lm.model.state_dict()[frozen], before)
    assert len(hook.history) == 2


def test_bf16_weights_stay_bf16():
    hook = _wavg(alpha=0.5)
    lin = torch.nn.Linear(4, 3).to(torch.bfloat16)
    base = lin.weight.detach().float().clone()
    ctx = SimpleNamespace(model=lin, cfg=SimpleNamespace(data=SimpleNamespace(domains=[1])))
    hook.on_run_start(ctx)
    hook.on_domain_start(ctx, "a")
    with torch.no_grad():
        lin.weight.add_(2.0)
    hook.on_domain_end(ctx, "a")
    assert lin.weight.dtype == torch.bfloat16
    assert torch.allclose(lin.weight.float(), base + 1.0, atol=1e-2)


@pytest.mark.parametrize("params, match", [
    ({"alpha": 1.5}, "alpha"),
    ({"mode": "swa"}, "mode"),
    ({"alhpa": 0.5}, "alhpa"),
    ({"ema_decay": -0.1}, "ema_decay"),
])
def test_bad_params_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        build_hooks([TechniqueSpec(name="wavg", params=params)])
