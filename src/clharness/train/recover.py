"""Recoverability probe (Phase 1 Q1): how much of a protected-set drop is spurious?

At every domain-boundary checkpoint: snapshot the weights, re-warm the model on a small fixed
budget of *general* text with a fresh optimizer, re-run the tier-1 measurements, log the
recovered share, and restore the weights so the main run is untouched. Following Zheng et al.
(ICLR 2025): a drop that comes back after a brief re-warm on unrelated data is lost task
alignment, not lost knowledge.

Cost: `recover_tokens` of training + one extra tier-1 eval per boundary. Memory: one CPU copy of
the weights (4 GB for 1B in fp32).

Logged, per tier-1 metric m: recover/<m> (value after re-warm), recover/<m>_delta (after − before),
e.g. recover/ppl/general and recover/lm_eval/arc_easy/acc_delta.
For perplexities lower is better; for accuracies higher is better; consumers compute the
"recovered share" as (after − before) / (base − before) with the sign handled per metric.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import torch

from clharness.data.loader import PackedStream, load_tokens
from clharness.train.loop import _empty_cache, chunked_cross_entropy
from clharness.train.hooks import BaseHook, CheckpointInfo, RunContext


class RecoverabilityHook(BaseHook):
    name = "recover"

    def __init__(self, eval_hook):
        self.eval_hook = eval_hook  # reuse the run's EvalHook so the measurements are identical

    def on_run_start(self, ctx: RunContext):
        cfg = ctx.cfg
        self.tokens = cfg.train.recover_tokens
        if self.tokens <= 0:
            return {"enabled": 0.0}
        entry = ctx.data_manifest["domains"][cfg.train.recover_domain]["splits"]["stream"]
        self.stream_tokens = load_tokens(entry["tokens"]["path"])
        return {"enabled": 1.0, "tokens": float(self.tokens)}

    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo):
        cfg = ctx.cfg
        if self.tokens <= 0 or not ckpt.is_domain_boundary or ckpt.index == 0:
            return None
        model = ctx.model
        t0 = time.time()
        # "before" = the EvalHook's measurements from this same checkpoint (published in ctx.state);
        # recomputing them would double the probe's cost for no information
        last = ctx.state.get("eval_last")
        before = last[1] if last and last[0] == ckpt.index else self.eval_hook.on_checkpoint(ctx, ckpt)
        snapshot = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        was_training = model.training

        # brief re-warm on general text, fresh optimizer, same precision path as the main loop
        o = cfg.optim
        stream = PackedStream(self.stream_tokens, cfg.data.seq_len, seed=cfg.train.seed + 1000 + ckpt.index)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.recover_lr, betas=o.betas, weight_decay=0.0)
        model.train()
        # Same batch size as the main run (grad accumulation) and a linear warmup over the first
        # 20% of steps: a fresh Adam on tiny batches at full LR *hurt* general perplexity in the
        # first CUDA check (+0.39 ppl after 122 micro-steps), which would masquerade as "genuine
        # forgetting". The probe must be a gentle re-warm, not a second training run.
        accum = max(1, o.batch_size // o.micro_batch_size)
        n_steps = max(1, self.tokens // (o.batch_size * cfg.data.seq_len))
        warm = max(1, n_steps // 5)
        seen, step, micro = 0, 0, 0
        opt.zero_grad(set_to_none=True)
        for batch in stream.batches(o.micro_batch_size):
            ids = batch[:, :-1].to(ctx.device)
            labels = batch[:, 1:].to(ctx.device)
            with ctx.ac():
                logits = model(input_ids=ids, use_cache=False).logits
            loss = chunked_cross_entropy(logits, labels, labels.numel()) / accum
            loss.backward()
            seen += labels.numel()
            micro += 1
            if micro % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = cfg.train.recover_lr * min(1.0, (step + 1) / warm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if seen >= self.tokens:
                break
        opt.zero_grad(set_to_none=True)
        del opt
        model.zero_grad(set_to_none=True)
        _empty_cache(ctx.device)
        model.eval()
        # per-item files from the "after" eval go to <ckpt>/recover/, not over the checkpoint's own
        # (before 2026-09-09 they overwrote items_<task>.npy at every boundary checkpoint)
        after_ckpt = dataclasses.replace(ckpt, path=str(Path(ckpt.path) / "recover") if ckpt.path else None)
        if after_ckpt.path:
            Path(after_ckpt.path).mkdir(parents=True, exist_ok=True)
        after = self.eval_hook.on_checkpoint(ctx, after_ckpt)

        # restore
        model.load_state_dict(snapshot, strict=True)
        del snapshot
        _empty_cache(ctx.device)
        if was_training:
            model.train()

        out = {"tokens_used": float(seen), "wall_s": time.time() - t0}
        for k, v in after.items():
            if "wall" in k or not isinstance(v, (int, float)):
                continue
            out[k] = v
            if k in before and isinstance(before[k], (int, float)):
                out[f"{k}_delta"] = v - before[k]
        return out
