"""The lora technique on the tiny random Gemma 3 model (2 decoder layers, CPU, seconds)."""

import pytest
import torch
from test_loop_hooks import _cfg, _state
from test_loop_tiny import VOCAB, _tiny_model

from clharness.activations import ResidualCapture
from clharness.config import RunConfig
from clharness.model import decoder_layers
from clharness.spec import TechniqueSpec
from clharness.techniques import apply_config_overrides, build_hooks
from clharness.train.hooks import RunContext
from clharness.train.loop import Trainer

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
# tiny model: hidden 64, 2 heads x 32 (q, o: 64x64), 1 kv head x 32 (k, v: 64->32), MLP 64<->128
PER_LAYER_ATTN = (64 + 64) + (64 + 32) + (64 + 32) + (64 + 64)  # = 448 per unit of rank
PER_LAYER_MLP = 3 * (64 + 128)  # = 576 per unit of rank


def _lora(**params):
    (hook,) = build_hooks([TechniqueSpec(name="lora", params=params)])
    return hook


def _trainer(tmp_path, monkeypatch, name, hooks, lm=None) -> Trainer:
    """Build the trainer (which calls configure_model) without running it."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = lm or _tiny_model(vocab=VOCAB, dtype=torch.float32)
    return Trainer(_cfg(tmp_path, name), hooks, eval_manifest=None, model_override=lm)


def _expected_adapter_params(model, rank: int, names=ATTN + MLP) -> int:
    return sum(rank * (m.in_features + m.out_features) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in names)


def _logits(model, x):
    model.eval()
    with torch.no_grad():
        return model(input_ids=x, use_cache=False).logits


def test_only_adapters_train_and_output_starts_at_the_base_model(tmp_path, monkeypatch):
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    before = _state(lm.model)
    x = torch.randint(3, VOCAB, (2, 16), generator=torch.Generator().manual_seed(1))
    base_out = _logits(lm.model, x)

    trainer = _trainer(tmp_path, monkeypatch, "lora", [_lora(rank=4)], lm=lm)
    assert trainer.model is lm.model  # injected in place, not wrapped
    assert torch.equal(_logits(lm.model, x), base_out)  # B = 0: the adapted model is the base model

    trainer.run()
    after = lm.model.state_dict()
    base = {k.replace(".base_layer", ""): v for k, v in after.items() if "lora_" not in k}
    assert set(base) == set(before)
    for k, v in base.items():
        assert torch.equal(v, before[k]), k
    lora_b = [v for k, v in after.items() if "lora_B" in k]
    assert lora_b and all(v.abs().sum() > 0 for v in lora_b)  # B starts at zero; training moved every one
    assert not torch.equal(_logits(lm.model, x), base_out)


def test_trainable_params_are_the_adapters_and_scale_with_rank(tmp_path, monkeypatch):
    counts = {}
    for rank in (4, 8):
        hook = _lora(rank=rank)
        lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
        expected = _expected_adapter_params(lm.model, rank)
        trainer = _trainer(tmp_path, monkeypatch, f"lora-r{rank}", [hook], lm=lm)
        assert expected == 2 * (PER_LAYER_ATTN + PER_LAYER_MLP) * rank
        assert trainer.ctx.state["trainable_params"] == hook.trainable_params == expected
        assert hook.on_run_start(trainer.ctx) == {"trainable_params": float(expected), "adapted_modules": 14.0}
        counts[rank] = expected
    assert counts[8] == 2 * counts[4]


def test_attn_targets_adapt_fewer_modules(tmp_path, monkeypatch):
    attn, both = _lora(targets="attn"), _lora(targets="attn_mlp")
    t_attn = _trainer(tmp_path, monkeypatch, "lora-attn", [attn])
    _trainer(tmp_path, monkeypatch, "lora-both", [both])
    assert attn.adapted_modules == 2 * len(ATTN) < both.adapted_modules == 2 * len(ATTN + MLP)
    assert attn.trainable_params == 2 * PER_LAYER_ATTN * 8
    adapted = {n.rsplit(".", 1)[-1] for n, m in t_attn.model.named_modules() if hasattr(m, "lora_A")}
    assert adapted == set(ATTN)


def _with(cfg: RunConfig, *techniques: dict) -> RunConfig:
    return RunConfig.model_validate({**cfg.dump(), "techniques": list(techniques)})


def test_lr_override(tmp_path):
    base = _cfg(tmp_path, "lora-lr", lr=1e-4)
    assert apply_config_overrides(_with(base, {"name": "lora"})).optim.lr == 1e-3
    assert apply_config_overrides(_with(base, {"name": "lora", "params": {"lr": 5e-3}})).optim.lr == 5e-3
    kept = apply_config_overrides(_with(base, {"name": "lora", "params": {"lr": None}}))
    assert kept.optim.lr == 1e-4 and kept.optim == base.optim


def test_lora_respects_an_earlier_freeze(tmp_path, monkeypatch):
    # listed lora-first: build_hooks orders by name, so freeze still runs first
    hooks = build_hooks([TechniqueSpec(name="lora", params={"rank": 4}),
                         TechniqueSpec(name="freeze", params={"bottom_layers": 1})])
    assert [h.name for h in hooks] == ["freeze", "lora"]
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    before = _state(lm.model)
    trainer = _trainer(tmp_path, monkeypatch, "freeze-lora", hooks, lm=lm)
    names = [n for n, _ in lm.model.named_parameters()]
    assert not any(".layers.0." in n and "lora_" in n for n in names)
    assert any(".layers.1." in n and "lora_" in n for n in names)
    assert hooks[1].adapted_modules == len(ATTN + MLP)
    assert trainer.ctx.state["trainable_params"] == (PER_LAYER_ATTN + PER_LAYER_MLP) * 4

    trainer.run()
    after = lm.model.state_dict()
    layer0 = [k for k in before if ".layers.0." in k]
    assert layer0
    for k in layer0:
        assert torch.equal(before[k], after[k]), k


def test_decoder_layers_are_still_found_after_injection(tmp_path, monkeypatch):
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    layers = lm.model.model.layers
    trainer = _trainer(tmp_path, monkeypatch, "lora-capture", [_lora(rank=4)], lm=lm)
    assert decoder_layers(trainer.model) is layers and len(layers) == 2
    x = torch.randint(3, VOCAB, (2, 16), generator=torch.Generator().manual_seed(2))
    with torch.no_grad(), ResidualCapture(trainer.model, [0, 1]) as cap:
        trainer.model(input_ids=x, use_cache=False)
    assert sorted(cap.acts) == [0, 1] and all(a.shape == (2, 16, 64) for a in cap.acts.values())


def test_no_matching_module_is_an_error(tmp_path):
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    ctx = RunContext(cfg=_cfg(tmp_path, "lora-none"), model=lm.model, tokenizer=None, device="cpu",
                     data_manifest={}, eval_manifest=None, output_dir=str(tmp_path))
    ctx.state["frozen_layers"] = [0, 1]  # every layer frozen: nothing left to adapt
    with pytest.raises(ValueError, match="no Linear module"):
        _lora().configure_model(ctx)


@pytest.mark.parametrize("params", [{"rank": 0}, {"alpha": 0}, {"targets": "mlp"}, {"lr": -1.0}, {"ranks": 4}])
def test_bad_params_are_rejected(params):
    with pytest.raises(ValueError, match="lora"):
        build_hooks([TechniqueSpec(name="lora", params=params)])
