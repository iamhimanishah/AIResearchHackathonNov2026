"""Distillation from the base model (B5), on fixed random logits and the tiny random model (CPU)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_loop_hooks import _cfg, _state
from test_loop_tiny import VOCAB, RecordingHook, _tiny_model
from test_replay import _add_general

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


# -- rows: which rows of the batch are distilled ------------------------------------------------


def _ctx_and_moved_student(tmp_path, techniques=()):
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    cfg = SimpleNamespace(techniques=[SimpleNamespace(name=n) for n in techniques])
    ctx = RunContext(cfg=cfg, model=lm.model, tokenizer=lm.tokenizer, device="cpu",
                     data_manifest={}, eval_manifest=None, output_dir=str(tmp_path))
    return lm, ctx


@pytest.mark.parametrize(("rows", "kept"), [("replay", [1, 3]), ("new", [0, 2])])
def test_rows_distils_only_the_chosen_rows(tmp_path, rows, kept):
    lm, ctx = _ctx_and_moved_student(tmp_path, techniques=["replay"])
    hook = _distill(top_k=8, rows=rows)
    hook.on_run_start(ctx)
    with torch.no_grad():  # move the student off the teacher so the term is not ~0
        for p in lm.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    batch = torch.randint(3, VOCAB, (4, 17))
    batch.replay_mask = torch.tensor([False, True, False, True])
    logits = lm.model(input_ids=batch[:, :-1], use_cache=False).logits
    term = hook.extra_loss(ctx, batch, logits, batch[:, 1:])
    # the mean KL of the chosen rows alone, times the share of the batch's positions they hold
    # (2 of 4 rows), so a token is weighted as it would be under rows: all
    teacher = hook.teacher(input_ids=batch[kept, :-1], use_cache=False).logits
    assert float(term) == pytest.approx(0.5 * float(distill_kl(logits[kept], teacher, top_k=8)), rel=1e-5)
    assert float(term) > 0
    assert ctx.state["extra_forward_tokens"] == len(kept) * 16  # the teacher saw only those rows


def test_rows_replay_skips_a_batch_with_no_replayed_rows(tmp_path):
    lm, ctx = _ctx_and_moved_student(tmp_path, techniques=["replay"])
    hook = _distill(rows="replay")
    hook.on_run_start(ctx)
    batch = torch.randint(3, VOCAB, (2, 17))
    batch.replay_mask = torch.tensor([False, False])
    logits = lm.model(input_ids=batch[:, :-1], use_cache=False).logits
    assert hook.extra_loss(ctx, batch, logits, batch[:, 1:]) is None
    assert ctx.state["extra_forward_tokens"] == 0  # no teacher pass was paid for
    assert hook.on_step_end(ctx, None) is None


def test_rows_replay_needs_the_replay_technique(tmp_path):
    _, ctx = _ctx_and_moved_student(tmp_path, techniques=[])
    with pytest.raises(ValueError, match="needs the replay technique"):
        _distill(rows="replay").on_run_start(ctx)


@pytest.mark.parametrize(("rows", "expected_rows"), [("replay", 8), ("new", 24), ("all", 32)])
def test_rows_in_a_real_run_with_replay(tmp_path, monkeypatch, rows, expected_rows):
    # code then math, 32 training rows of 32 tokens; replay swaps in exactly 25% of them
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _add_general(tmp_path)
    techniques = [{"name": "replay", "params": {"share": 0.25, "source": "general"}},
                  {"name": "distill", "params": {"rows": rows, "top_k": 8}}]
    cfg = RunConfig.model_validate({**_cfg(tmp_path, f"distill-rows-{rows}").dump(), "techniques": techniques})
    hooks = build_hooks(cfg.techniques)
    trainer = Trainer(cfg, hooks, eval_manifest=None, model_override=_tiny_model(vocab=VOCAB, dtype=torch.float32))
    trainer.run()
    assert trainer.gradient_tokens == 32 * 32
    assert trainer.ctx.state["extra_forward_tokens"] == expected_rows * 32
    distill = next(h for h in hooks if h.name == "distill")
    assert all(math.isfinite(k) and k >= 0 for k in distill.kl_history)


@pytest.mark.parametrize("mask", [[False, True, False, False], [False, True, False, True], [True, True, True, False]])
def test_a_distilled_token_is_pulled_equally_hard_in_every_rows_mode(tmp_path, mask):
    # the gradient on a replayed row's logits must not depend on how many replayed rows share
    # the micro-batch, nor on whether the other rows are distilled too
    lm, ctx = _ctx_and_moved_student(tmp_path, techniques=["replay"])
    with torch.no_grad():
        for p in lm.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    batch = torch.randint(3, VOCAB, (4, 17))
    batch.replay_mask = torch.tensor(mask)
    grads = {}
    for rows in ("all", "replay"):
        hook = _distill(top_k=8, rows=rows)
        hook.teacher = None
        hook.on_run_start(ctx)
        # the teacher is a copy of the moved student, so move the student again to get a non-zero term
        logits = (lm.model(input_ids=batch[:, :-1], use_cache=False).logits + 0.3 * torch.sin(torch.arange(VOCAB))).detach().requires_grad_()
        hook.extra_loss(ctx, batch, logits, batch[:, 1:]).backward()
        grads[rows] = logits.grad
    replayed = batch.replay_mask
    assert grads["all"][replayed].abs().sum() > 0
    assert torch.allclose(grads["replay"][replayed], grads["all"][replayed], rtol=1e-4, atol=1e-9)
    assert grads["replay"][~replayed].abs().sum() == 0  # the other rows are left alone


def test_teacher_keeps_its_own_weights_when_the_student_forward_was_replaced(tmp_path):
    # train.compile stores a wrapper bound to the student as the instance attribute `forward`;
    # a deep copy would carry that wrapper along and the teacher would compute the student
    lm, ctx = _ctx_and_moved_student(tmp_path)
    original = lm.model.forward
    lm.model.forward = lambda *a, **k: original(*a, **k)  # stands in for torch.compile's wrapper
    hook = _distill(top_k=8)
    hook.on_run_start(ctx)
    assert "forward" not in hook.teacher.__dict__
    with torch.no_grad():  # move the student; a real teacher must not follow
        for p in lm.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    batch = torch.randint(3, VOCAB, (2, 17))
    student = lm.model(input_ids=batch[:, :-1], use_cache=False).logits
    teacher = hook.teacher(input_ids=batch[:, :-1], use_cache=False).logits
    assert not torch.allclose(student, teacher)
    assert float(hook.extra_loss(ctx, batch, student, batch[:, 1:])) > 1e-4
