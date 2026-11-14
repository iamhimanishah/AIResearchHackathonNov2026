"""The freeze technique on the tiny random Gemma 3 model (2 decoder layers, CPU, seconds)."""

import pytest
import torch
from test_loop_hooks import _cfg, _run, _state
from test_loop_tiny import VOCAB, _tiny_model

from clharness.spec import TechniqueSpec
from clharness.techniques import build_hooks
from clharness.train.loop import Trainer


def _freeze(**params):
    (hook,) = build_hooks([TechniqueSpec(name="freeze", params=params)])
    return hook


def _is_frozen(name: str, embeddings: bool = True) -> bool:
    # lm_head.weight is tied to the token embeddings in Gemma
    return ".layers.0." in name or (embeddings and ("embed_tokens" in name or name.startswith("lm_head")))


def _trainer(tmp_path, monkeypatch, name, hooks) -> Trainer:
    """Build the trainer (which calls configure_model) without running it."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    return Trainer(_cfg(tmp_path, name), hooks, eval_manifest=None, model_override=lm)


def test_frozen_layers_do_not_move_and_the_rest_trains(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    hook = _freeze(bottom_layers=1)
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    before = _state(lm.model)
    trainer = Trainer(_cfg(tmp_path, "freeze"), [hook], eval_manifest=None, model_override=lm)
    trainer.run()
    after = lm.model.state_dict()

    frozen = [k for k in before if _is_frozen(k)]
    assert any(".layers.0." in k for k in frozen) and any("embed_tokens" in k for k in frozen)
    for k in frozen:
        assert torch.equal(before[k], after[k]), k
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and not any(_is_frozen(k) for k in moved)
    assert any(".layers.1." in k for k in moved)

    assert hook.frozen_layers == [0] and trainer.ctx.state["frozen_layers"] == [0]


def test_trainable_params_shrink_to_what_is_left(tmp_path, monkeypatch):
    plain, _, _ = _run(tmp_path, monkeypatch, "plain", [])
    frozen, _, lm = _run(tmp_path, monkeypatch, "freeze", [_freeze(bottom_layers=1)])
    expected = sum(p.numel() for n, p in lm.model.named_parameters() if not _is_frozen(n))
    assert frozen.ctx.state["trainable_params"] == expected < plain.ctx.state["trainable_params"]
    assert plain.ctx.state["trainable_params"] == plain.ctx.state["total_params"]


def test_embeddings_false_leaves_the_embeddings_trainable(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, "freeze-noemb", [_freeze(bottom_layers=1, embeddings=False)])
    model = trainer.model
    assert model.get_input_embeddings().weight.requires_grad
    assert not any(p.requires_grad for p in model.model.layers[0].parameters())
    assert all(p.requires_grad for p in model.model.layers[1].parameters())
    expected = sum(p.numel() for n, p in model.named_parameters() if not _is_frozen(n, embeddings=False))
    assert trainer.ctx.state["trainable_params"] == expected


def test_top_layers_freezes_from_the_top(tmp_path, monkeypatch):
    hook = _freeze(bottom_layers=0, top_layers=1, embeddings=False)
    trainer = _trainer(tmp_path, monkeypatch, "freeze-top", [hook])
    assert hook.frozen_layers == [1] == trainer.ctx.state["frozen_layers"]
    assert all(p.requires_grad for p in trainer.model.model.layers[0].parameters())
    assert not any(p.requires_grad for p in trainer.model.model.layers[1].parameters())


def test_freezing_every_layer_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="2 decoder layers"):
        _trainer(tmp_path, monkeypatch, "freeze-all", [_freeze(bottom_layers=2)])
    with pytest.raises(ValueError, match="2 decoder layers"):
        _trainer(tmp_path, monkeypatch, "freeze-all2", [_freeze(bottom_layers=1, top_layers=1)])


@pytest.mark.parametrize("params", [{"bottom_layers": -1}, {"top_layers": -2}, {"bottom": 1}, {"embeddings": "maybe"}])
def test_bad_params_are_rejected(params):
    with pytest.raises(ValueError, match="freeze"):
        build_hooks([TechniqueSpec(name="freeze", params=params)])
