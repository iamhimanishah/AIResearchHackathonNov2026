"""Distillation from the base model (B5), on fixed random logits and the tiny random model (CPU)."""

import json
import math
from pathlib import Path

import pytest
import torch
from test_loop_hooks import _cfg, _state
from test_loop_tiny import VOCAB, RecordingHook, _tiny_model

from clharness.config import RunConfig
from clharness.spec import TechniqueSpec
from clharness.techniques import build_hooks
from clharness.techniques.distill import Distill, distill_kl
from clharness.train.hooks import RunContext
from clharness.train.loop import Trainer


def _distill(**params) -> Distill:
    (hook,) = build_hooks([TechniqueSpec(name="distill", params=params)])
    return hook


def _train(tmp_path, monkeypatch, name, params, cfg: RunConfig | None = None, **optim):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    base = _state(lm.model)
    hook, rec = _distill(**params), RecordingHook()
    trainer = Trainer(cfg or _cfg(tmp_path, name, **optim), [hook, rec], eval_manifest=None, model_override=lm)
    trainer.run()
    return trainer, hook, lm, base


def _step_logs(trainer) -> list[dict]:
    lines = (Path(trainer.out_dir) / "metrics.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if "train/loss" in r]


def _logits(seed: int, shape=(3, 5, VOCAB), scale=3.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def _plain_kl(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per position, averaged, the textbook way."""
    s, t = s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1])
    return torch.nn.functional.kl_div(torch.log_softmax(s, -1), torch.softmax(t, -1), reduction="batchmean")


# -- the term itself --------------------------------------------------------------------------


def test_kl_matches_the_plain_computation():
    s, t = _logits(0), _logits(1)
    exact = _plain_kl(s, t)
    assert distill_kl(s, t, top_k=None) == pytest.approx(float(exact), rel=1e-5)
    assert distill_kl(s, t, top_k=None, chunk=4) == pytest.approx(float(exact), rel=1e-5)  # chunking is exact
    assert distill_kl(s, t, top_k=VOCAB) == pytest.approx(float(exact), rel=1e-5)
    # one token in the rest bucket: the bucketed KL is the exact KL, through the bucket code path
    assert distill_kl(s, t, top_k=VOCAB - 1) == pytest.approx(float(exact), rel=1e-4)
    previous = 0.0
    for k in (1, 4, 16, 64, 256):
        v = float(distill_kl(s, t, top_k=k))
        assert -1e-6 <= v <= float(exact) + 1e-6
        assert v >= previous - 1e-5  # a finer bucketing never loses information
        previous = v
    assert previous > 0


def test_kl_is_zero_when_the_models_agree_and_respects_the_mask():
    s = _logits(2, scale=20.0)  # peaked: the top-k mass is near 1, the hard case for the rest bucket
    for k in (None, 1, 8):
        v = distill_kl(s, s.clone(), top_k=k)
        assert math.isfinite(float(v)) and abs(float(v)) < 1e-6
    t = _logits(3)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[1, 2] = mask[2, 4] = True
    expected = _plain_kl(s[mask], t[mask])
    assert distill_kl(s, t, mask, top_k=None) == pytest.approx(float(expected), rel=1e-5)
    assert float(distill_kl(s, t, torch.zeros_like(mask))) == 0.0


@pytest.mark.parametrize("top_k", [None, 8])
def test_temperature_is_logits_over_t_times_t_squared(top_k):
    s, t = _logits(4), _logits(5)
    assert distill_kl(s, t, temperature=2.0, top_k=top_k) == pytest.approx(
        4 * float(distill_kl(s / 2, t / 2, top_k=top_k)), rel=1e-5
    )


@pytest.mark.parametrize("top_k", [None, 8])
def test_gradient_reaches_the_student_logits_only(top_k):
    s = _logits(6).requires_grad_(True)
    t = _logits(7).requires_grad_(True)
    distill_kl(s, t, top_k=top_k).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0 and torch.isfinite(s.grad).all()
    assert t.grad is None


def test_bf16_logits_give_a_float32_term():
    s, t = _logits(8).bfloat16(), _logits(9).bfloat16()
    v = distill_kl(s, t, top_k=8)
    assert v.dtype == torch.float32 and math.isfinite(float(v)) and float(v) > 0


# -- the hook ----------------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [None, 8])
def test_first_step_kl_is_zero(tmp_path, monkeypatch, top_k):
    _, hook, _, _ = _train(tmp_path, monkeypatch, f"distill-first-{top_k}", {"top_k": top_k})
    assert len(hook.kl_history) == 8  # 2 domains x 4 steps
    assert abs(hook.kl_history[0]) < 1e-6  # both micro-batches of step 1 see the base weights
    assert hook.kl_history[-1] > 1e-4  # and the student then moves away


def test_a_heavier_weight_stays_closer_to_base(tmp_path, monkeypatch):
    _, light, lm_light, base = _train(tmp_path, monkeypatch, "distill-light", {"weight": 0.1})
    _, heavy, lm_heavy, _ = _train(tmp_path, monkeypatch, "distill-heavy", {"weight": 10.0})
    assert heavy.kl_history[-1] < light.kl_history[-1]

    def distance(model):
        after = model.state_dict()
        return sum(float((after[k] - base[k]).float().pow(2).sum()) for k in base)

    assert distance(lm_heavy.model) < distance(lm_light.model)


def test_teacher_is_frozen_and_gets_no_gradient(tmp_path, monkeypatch):
    _, hook, lm, base = _train(tmp_path, monkeypatch, "distill-frozen", {})
    teacher = hook.teacher
    assert teacher is not lm.model and not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    after = teacher.state_dict()
    assert after.keys() == base.keys() and all(torch.equal(after[k], base[k]) for k in base)
    assert any(not torch.equal(lm.model.state_dict()[k], base[k]) for k in base)  # the student did move


def test_extra_loss_has_gradient_into_the_student(tmp_path):
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    ctx = RunContext(cfg=None, model=lm.model, tokenizer=lm.tokenizer, device="cpu",
                     data_manifest={}, eval_manifest=None, output_dir=str(tmp_path))
    hook = _distill(top_k=8)
    hook.on_run_start(ctx)
    with torch.no_grad():  # move the student off the teacher so the gradient is not ~0
        for p in lm.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    batch = torch.randint(3, VOCAB, (2, 17))
    logits = lm.model(input_ids=batch[:, :-1], use_cache=False).logits
    logits.retain_grad()
    term = hook.extra_loss(ctx, batch, logits, batch[:, 1:])
    assert term.requires_grad and term.ndim == 0 and float(term) > 0
    term.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert all(p.grad is None for p in hook.teacher.parameters())
    assert ctx.state["extra_forward_tokens"] == 2 * 16


def test_cost_counter_and_logged_term(tmp_path, monkeypatch):
    # one micro-batch per step, so train/extra_loss/distill is exactly one extra_loss return
    trainer, hook, _, _ = _train(tmp_path, monkeypatch, "distill-log", {"weight": 2.0}, micro_batch_size=4)
    assert trainer.ctx.state["extra_forward_tokens"] == trainer.gradient_tokens == 8 * 4 * 32
    rows = _step_logs(trainer)
    assert len(rows) == 8
    for r in rows:
        assert r["train/extra_loss/distill"] == pytest.approx(2.0 * r["distill/kl"], rel=1e-5, abs=1e-9)
    assert [r["distill/kl"] for r in rows] == pytest.approx(hook.kl_history)


def test_bf16_mixed_runs_without_nan(tmp_path, monkeypatch):
    raw = _cfg(tmp_path, "distill-bf16").dump()
    raw["train"]["precision"] = "bf16-mixed"
    _, hook, _, _ = _train(tmp_path, monkeypatch, "distill-bf16", {"top_k": 8}, cfg=RunConfig.model_validate(raw))
    assert len(hook.kl_history) == 8 and all(math.isfinite(v) and v > -1e-3 for v in hook.kl_history)


@pytest.mark.parametrize("params", [{"weight": 0}, {"top_k": 0}, {"temperature": 0}, {"wieght": 1.0}])
def test_bad_params_are_rejected(params):
    with pytest.raises(ValueError, match="distill"):
        build_hooks([TechniqueSpec(name="distill", params=params)])
