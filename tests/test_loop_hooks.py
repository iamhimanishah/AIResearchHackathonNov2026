"""The three loop extensions techniques need, on the tiny random model (CPU, seconds):
per-domain LR re-warm (A4), the pre-optimizer model hook (A2) and the loss-term hook (A3)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_loop_tiny import VOCAB, RecordingHook, _synthetic_manifest, _tiny_model

from clharness.config import REPO_ROOT, RunConfig, load_run_config
from clharness.train.hooks import BaseHook
from clharness.train.loop import Trainer, lr_at

SEQ = 32


def _cfg(tmp_path: Path, name: str, **optim) -> RunConfig:
    """Two domains, four optimizer steps each."""
    manifest = tmp_path / "manifests" / "tiny.json"
    if not manifest.exists():
        _synthetic_manifest(tmp_path, SEQ, seed=7)
    raw = load_run_config(REPO_ROOT / "configs" / "smoke.yaml").dump()
    raw["name"] = name
    raw["eval"]["lm_eval_tasks"] = []
    raw["eval"]["perplexity_slices"] = []
    raw["data"].update({"manifest": str(manifest), "seq_len": SEQ, "probe_max_seqs_per_domain": None,
                        "domains": [{"name": "code", "tokens": 4 * 4 * SEQ}, {"name": "math", "tokens": 4 * 4 * SEQ}]})
    raw["optim"].update({"batch_size": 4, "micro_batch_size": 2, "warmup_steps": 1, "lr": 1e-3, **optim})
    raw["train"].update({"precision": "fp32", "device": "cpu", "checkpoint_every_steps": 1000,
                         "output_dir": os.path.relpath(tmp_path, REPO_ROOT)})
    return RunConfig.model_validate(raw)


def _run(tmp_path, monkeypatch, name, hooks, **optim):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    rec = RecordingHook()
    trainer = Trainer(_cfg(tmp_path, name, **optim), [*hooks, rec], eval_manifest=None, model_override=lm)
    trainer.run()
    return trainer, rec, lm


def _state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# -- A4: per-domain re-warm -------------------------------------------------------------------


def test_lr_at_global_scope_is_unchanged():
    cfg = SimpleNamespace(lr=1.0, warmup_steps=2, schedule="cosine")  # a config from before A4
    assert [lr_at(s, 10, cfg) for s in (0, 1, 2)] == [0.5, 1.0, 1.0]
    assert lr_at(10, 10, cfg) == pytest.approx(0.0)
    const = SimpleNamespace(lr=1.0, warmup_steps=2, schedule="constant", schedule_scope="global",
                            rewarm_steps=5, min_lr_ratio=0.5)
    assert [lr_at(s, 10, const) for s in (0, 1, 5, 9)] == [0.5, 1.0, 1.0, 1.0]


def test_lr_at_per_domain_rewarms_and_decays_to_the_floor():
    cfg = SimpleNamespace(lr=1.0, warmup_steps=99, schedule="cosine", schedule_scope="per_domain",
                          rewarm_steps=2, min_lr_ratio=0.1)
    curve = [lr_at(100 + s, 1000, cfg, domain_step=s, domain_steps=6) for s in range(6)]
    assert curve[:3] == [0.5, 1.0, 1.0]  # two warmup steps, then the peak
    assert curve[3] > curve[4] > curve[5] > 0.1
    assert lr_at(0, 0, cfg, domain_step=6, domain_steps=6) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="per_domain"):
        lr_at(0, 10, cfg)


def test_trainer_rewarms_at_each_domain_boundary(tmp_path, monkeypatch):
    _, rec, _ = _run(tmp_path, monkeypatch, "rewarm", [], schedule="linear",
                     schedule_scope="per_domain", rewarm_steps=2, min_lr_ratio=0.1)
    lrs = [s.lr for s in rec.steps]
    assert lrs[:4] == pytest.approx([0.5e-3, 1e-3, 1e-3, 0.55e-3])
    assert lrs[4:] == pytest.approx(lrs[:4])  # the second domain repeats the first's curve


# -- A2: model hook -------------------------------------------------------------------------


class FreezeAllButLastLayer(BaseHook):
    name = "freeze"

    def configure_model(self, ctx):
        for n, p in ctx.model.named_parameters():
            p.requires_grad = ".layers.1." in n


class Lora(BaseHook):
    name = "lora"

    def configure_model(self, ctx):
        from peft import LoraConfig, get_peft_model

        return get_peft_model(ctx.model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))


def test_frozen_weights_do_not_move(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    before = _state(lm.model)
    trainer = Trainer(_cfg(tmp_path, "freeze"), [FreezeAllButLastLayer()], eval_manifest=None, model_override=lm)
    trainer.run()
    after = lm.model.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and all(".layers.1." in k for k in moved)
    expected = sum(p.numel() for n, p in lm.model.named_parameters() if ".layers.1." in n)
    assert trainer.ctx.state["trainable_params"] == expected < trainer.ctx.state["total_params"]
    first = json.loads((Path(trainer.out_dir) / "metrics.jsonl").read_text().splitlines()[0])
    assert first["train/trainable_params"] == expected


def test_lora_wrapped_model_trains_only_the_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    before = _state(lm.model)
    trainer = Trainer(_cfg(tmp_path, "lora"), [Lora()], eval_manifest=None, model_override=lm)
    assert trainer.model is trainer.ctx.model and trainer.model is not lm.model  # the wrapper replaced it
    trainer.run()
    after = trainer.model.state_dict()
    lora_b = [v for k, v in after.items() if "lora_B" in k]
    assert lora_b and any(v.abs().sum() > 0 for v in lora_b)  # lora_B starts at zero
    base = {k: v for k, v in after.items() if "lora_" not in k}
    assert len(base) == len(before)
    for k, v in base.items():
        original = before[k.removeprefix("base_model.model.").replace(".base_layer", "")]
        assert torch.equal(v, original), k
    assert trainer.ctx.state["trainable_params"] == sum(v.numel() for k, v in after.items() if "lora_" in k)


# -- A3: loss hook --------------------------------------------------------------------------


class ZeroTerm(BaseHook):
    name = "zero"

    def extra_loss(self, ctx, batch, logits, labels):
        return logits.sum() * 0.0


class PullLogitsDown(BaseHook):
    name = "pull"

    def extra_loss(self, ctx, batch, logits, labels):
        assert logits.requires_grad
        return 0.5 * logits.float().mean()


def test_zero_extra_loss_changes_nothing_and_a_real_one_does(tmp_path, monkeypatch):
    plain, rec_plain, lm_plain = _run(tmp_path, monkeypatch, "plain", [])
    zero, rec_zero, lm_zero = _run(tmp_path, monkeypatch, "zero", [ZeroTerm()])
    pull, rec_pull, lm_pull = _run(tmp_path, monkeypatch, "pull", [PullLogitsDown()])
    a, b, c = lm_plain.model.state_dict(), lm_zero.model.state_dict(), lm_pull.model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)  # bit-identical training
    assert any(not torch.equal(a[k], c[k]) for k in a)
    assert [s.loss for s in rec_plain.steps] == [s.loss for s in rec_zero.steps]

    def step_logs(trainer):
        lines = (Path(trainer.out_dir) / "metrics.jsonl").read_text().splitlines()
        return [r for r in map(json.loads, lines) if "train/loss" in r]

    assert all("train/extra_loss/pull" in r for r in step_logs(pull))
    assert all(r["train/extra_loss/zero"] == 0.0 for r in step_logs(zero))
    assert not any(k.startswith("train/extra_loss") for r in step_logs(plain) for k in r)
    # train/loss stays the language-model loss: the first step sees the same weights and batch
    assert step_logs(pull)[0]["train/loss"] == pytest.approx(step_logs(plain)[0]["train/loss"])
