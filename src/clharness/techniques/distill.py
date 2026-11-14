"""Distillation from the base model: penalise the trained model for drifting away from it.

At the start of the run the hook keeps a frozen copy of the model (the teacher). On every
training micro-batch it runs the teacher on the same input ids and adds

    weight * T^2 * mean over valid positions of KL(teacher || student)

to the language-model loss, where both next-token distributions are softened by the
temperature T. This is the Learning-without-Forgetting recipe (Li and Hoiem 2016): the new
domain is learned, but the model is pulled back toward the base model's predictions.

Parameters
  weight       multiplier on the KL term (> 0)
  temperature  divides both models' logits before the softmax (> 0); the term is scaled by
               T^2 so its gradient keeps the same size across temperatures
  top_k        None: exact KL over the whole vocabulary. k: KL over the teacher's k most likely
               tokens plus one "everything else" bucket. That is a lower bound on the exact KL,
               exactly 0 when the two models agree, and it ignores how the student spreads the
               rest of its mass over tokens the teacher thinks unlikely.

Logged: `distill/kl`, the mean of the term before `weight` (so including T^2) over the step's
micro-batches; `train/extra_loss/distill` is `weight` times it.

Cost: one extra forward-only pass per micro-batch (about 1.3x the compute of a plain step,
since a training step is one forward plus a backward of about twice that), counted in
`ctx.state["extra_forward_tokens"]`, plus one frozen copy of the model in memory and the
teacher's [B, T, V] logits for the current micro-batch. Gradient tokens are unchanged.

Known limits
  - It distils on whatever the batch holds, including rows a replay hook swapped in.
  - On CUDA the teacher is held in bfloat16 to halve its memory, so even before the first
    update the term is not exactly 0 there. Elsewhere it keeps the model's dtype.
  - The teacher is copied in `on_run_start`. LoRA adapters injected before that start as a
    zero change, so the copy still computes the base model; a technique that changes the
    weights themselves before the run starts would change the teacher too.
"""

from __future__ import annotations

import copy

import torch
from pydantic import Field

from clharness.techniques import TechniqueParams, register
from clharness.train.hooks import BaseHook, RunContext, StepStats

_FILL = -1e9  # stands in for -inf when masking logits out: finite, so no NaN in logsumexp or its gradient


def _bucket_kl(s: torch.Tensor, t: torch.Tensor, top_k: int) -> torch.Tensor:
    """Per-row KL(teacher || student) over the teacher's top-k tokens plus a "rest" bucket.

    `s` (with gradient) and `t` (no gradient, overwritten in place) are float32 [n, V] logits
    already divided by the temperature. The rest bucket's log-probability is a logsumexp over
    the tokens outside the top k (not log(1 - sum p_topk), which loses all precision when the
    top-k mass is near 1), so student top-k and rest probabilities come from one log-softmax
    and add up to 1."""
    lse_s = torch.logsumexp(s, dim=-1, keepdim=True)
    lse_t = torch.logsumexp(t, dim=-1, keepdim=True)
    top_t, idx = t.topk(top_k, dim=-1)
    log_pt = top_t - lse_t  # [n, k]
    log_ps = s.gather(-1, idx) - lse_s
    t.scatter_(-1, idx, _FILL)  # the teacher's copy is ours to overwrite
    log_pt_rest = torch.logsumexp(t, dim=-1, keepdim=True) - lse_t  # [n, 1]
    log_ps_rest = torch.logsumexp(s.scatter(-1, idx, _FILL), dim=-1, keepdim=True) - lse_s
    log_pt, log_ps = torch.cat([log_pt, log_pt_rest], -1), torch.cat([log_ps, log_ps_rest], -1)
    return (log_pt.exp() * (log_pt - log_ps)).sum(-1)


def distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    chunk: int = 512,
) -> torch.Tensor:
    """T^2 * mean over positions where `mask` is True of KL(teacher || student) at temperature T.

    Logits are [..., V] in any dtype; the probability math is float32, `chunk` positions at a
    time, so no float32 copy of the whole [B, T, V] tensor is ever alive (at a 262k vocabulary
    one chunk of 512 positions is 0.5 GB per float32 temporary). Gradient flows into
    `student_logits` only. Returns a float32 scalar; 0 if no position is valid."""
    V = student_logits.shape[-1]
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"student logits {tuple(student_logits.shape)} != teacher {tuple(teacher_logits.shape)}")
    s_flat, t_flat = student_logits.reshape(-1, V), teacher_logits.detach().reshape(-1, V)
    m_flat = (torch.ones(s_flat.shape[0], dtype=torch.bool, device=s_flat.device) if mask is None
              else mask.reshape(-1).to(s_flat.device))
    exact = top_k is None or top_k >= V
    total = s_flat.new_zeros((), dtype=torch.float32)
    for i in range(0, s_flat.shape[0], chunk):
        m = m_flat[i : i + chunk]
        if not bool(m.any()):
            continue
        s = s_flat[i : i + chunk][m].float() / temperature
        t = t_flat[i : i + chunk][m].float() / temperature
        if exact:
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(s, -1), torch.log_softmax(t, -1), log_target=True, reduction="sum"
            )
        else:
            kl = _bucket_kl(s, t, top_k).sum()
        total = total + kl
    n = int(m_flat.sum())
    return total * (temperature**2) / max(n, 1)


@register
class Distill(BaseHook):
    name = "distill"

    class Params(TechniqueParams):
        weight: float = Field(1.0, gt=0)  # multiplier on the KL term
        temperature: float = Field(1.0, gt=0)  # softens both distributions; the term is scaled by T^2
        top_k: int | None = Field(64, ge=1)  # teacher's top-k tokens plus one "rest" bucket; None = full vocabulary

    def __init__(self, params: Params):
        self.p = params
        self.teacher: torch.nn.Module | None = None
        self.kl_history: list[float] = []
        self._kl_sum, self._kl_n = 0.0, 0

    def on_run_start(self, ctx: RunContext):
        teacher = copy.deepcopy(ctx.model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if str(ctx.device).startswith("cuda"):
            teacher = teacher.to(torch.bfloat16)
        self.teacher = teacher
        self.kl_history = []
        self._kl_sum, self._kl_n = 0.0, 0
        ctx.state.setdefault("extra_forward_tokens", 0)

    def extra_loss(self, ctx: RunContext, batch: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
        if self.teacher is None:
            raise RuntimeError("distill: extra_loss called before on_run_start")
        ids = batch[:, :-1].to(logits.device)  # the input ids the loop fed the student
        with torch.no_grad(), ctx.ac():
            teacher_logits = self.teacher(input_ids=ids, use_cache=False).logits
        ctx.state["extra_forward_tokens"] = ctx.state.get("extra_forward_tokens", 0) + ids.numel()
        kl = distill_kl(logits, teacher_logits, labels != -100, self.p.temperature, self.p.top_k)
        del teacher_logits
        self._kl_sum += float(kl.detach())
        self._kl_n += 1
        return self.p.weight * kl

    def on_step_end(self, ctx: RunContext, stats: StepStats):
        if self._kl_n == 0:
            return None
        kl = self._kl_sum / self._kl_n
        self._kl_sum, self._kl_n = 0.0, 0
        self.kl_history.append(kl)
        return {"kl": kl}
