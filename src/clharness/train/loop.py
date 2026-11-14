"""Config-driven continual fine-tuning driver.

Responsibilities (and nothing else): present domains in order with a per-domain gradient-token
budget, take optimizer steps, count gradient tokens, checkpoint on schedule + at domain
boundaries, and dispatch to hooks. Measurement lives in clharness.train.measure; policies (Phase 3)
will live next to it. See hooks.py for the contract.

Gradient tokens: every token position whose label contributes to the loss. With packed
sequences and no padding that is micro_batch * seq_len per micro-step; we count from the label
mask so a future policy that masks tokens (Rho-1 style) is counted correctly and automatically.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from clharness.config import REPO_ROOT, RunConfig
from clharness.data.loader import PackedStream, load_tokens
from clharness.data.manifest import read_manifest, verify_manifest
from clharness.env import pick_device, seed_everything
from clharness.model import LoadedModel, load_model
from clharness.train.checkpoint import save_checkpoint
from clharness.train.hooks import BaseHook, CheckpointInfo, Hook, RunContext, StepStats
from clharness.wandb_log import RunLogger


def lr_at(step: int, total_steps: int, cfg, domain_step: int | None = None, domain_steps: int | None = None) -> float:
    """Learning rate for the optimizer step about to be taken (0-based `step`).

    With `schedule_scope: per_domain` the warmup and decay restart in every domain: pass the
    step index within the domain and the domain's step count. The decay ends at
    lr * min_lr_ratio."""
    warmup = cfg.warmup_steps
    if getattr(cfg, "schedule_scope", "global") == "per_domain":
        if domain_step is None or domain_steps is None:
            raise ValueError("per_domain schedule needs domain_step and domain_steps")
        step, total_steps, warmup = domain_step, domain_steps, cfg.rewarm_steps
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    if cfg.schedule == "constant":
        return cfg.lr
    frac = (step - warmup) / max(1, total_steps - warmup)
    frac = min(1.0, frac)
    floor = cfg.lr * getattr(cfg, "min_lr_ratio", 0.0)
    if cfg.schedule == "linear":
        return floor + (cfg.lr - floor) * (1.0 - frac)
    if cfg.schedule == "cosine":
        return floor + (cfg.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(cfg.schedule)


def per_sequence_token_loss(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """[B] token-sum NLL per sequence (no grad), chunked fp32 softmax; ignore_index=-100."""
    B, T, V = logits.shape
    with torch.no_grad():
        flat, tgt = logits.reshape(-1, V), labels.reshape(-1)
        out = torch.zeros(B, dtype=torch.float32, device=logits.device)
        for i in range(0, flat.shape[0], chunk):
            l = torch.nn.functional.cross_entropy(flat[i : i + chunk].float(), tgt[i : i + chunk], ignore_index=-100, reduction="none")
            out.index_add_(0, torch.arange(i, i + l.shape[0], device=logits.device) // T, l)
        return out / (labels != -100).sum(1).clamp(min=1)


def chunked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, n_tokens: int, chunk: int = 1024) -> torch.Tensor:
    """Mean token loss over the label mask, computing the fp32 softmax one `chunk` of tokens at a
    time. With a 262k vocabulary a whole-micro-batch fp32 copy of the logits (plus its
    log-softmax and gradient) is ~3 x 8 GB at micro-batch 8 x 1024; chunking bounds the
    temporaries at ~3 x 1 GB. Gradient flows through each chunk; the sum is exact."""
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    tgt = labels.reshape(-1)
    total = flat.new_zeros((), dtype=torch.float32)
    for i in range(0, flat.shape[0], chunk):
        total = total + torch.nn.functional.cross_entropy(
            flat[i : i + chunk].float(), tgt[i : i + chunk], ignore_index=-100, reduction="sum"
        )
    return total / max(n_tokens, 1)


def autocast_ctx(device: str, precision: str):
    """Factory for the forward-pass autocast context implied by `precision` on `device`."""
    import contextlib

    if precision != "bf16-mixed":
        return contextlib.nullcontext
    dev_type = "cuda" if device.startswith("cuda") else device
    return lambda: torch.autocast(device_type=dev_type, dtype=torch.bfloat16)


def _empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _merge_logs(target: dict[str, Any], prefix: str, d: dict[str, Any] | None) -> None:
    if d:
        for k, v in d.items():
            target[f"{prefix}/{k}"] = v


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        hooks: list[Hook],
        eval_manifest: dict[str, Any] | None,
        model_override: LoadedModel | None = None,  # tests: inject a tiny model instead of loading cfg.model
    ):
        self.cfg = cfg
        self.hooks = hooks
        self.device = pick_device(cfg.train.device)
        seed_everything(cfg.train.seed)

        self.out_dir = (REPO_ROOT / cfg.train.output_dir / cfg.name).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "config.resolved.json", "w") as f:
            json.dump(cfg.dump(), f, indent=2, sort_keys=True)

        self.data_manifest = read_manifest(cfg.data.manifest)
        verify_manifest(self.data_manifest)
        if self.data_manifest["seq_len"] != cfg.data.seq_len:
            raise ValueError("cfg.data.seq_len != manifest seq_len")

        weight_dtype = {"bf16-pure": "bfloat16", "bf16-mixed": "float32", "fp32": "float32"}[cfg.train.precision]
        self.lm: LoadedModel = model_override or load_model(
            cfg.model.model_id, device=self.device, dtype=weight_dtype,
            attn_implementation=cfg.model.attn_implementation,
        )
        self.model = self.lm.model
        self.autocast = autocast_ctx(self.device, cfg.train.precision)
        self.ctx = RunContext(
            cfg=cfg, model=self.model, tokenizer=self.lm.tokenizer, device=self.device,
            data_manifest=self.data_manifest, eval_manifest=eval_manifest, output_dir=str(self.out_dir),
            autocast=self.autocast,
        )
        # Techniques that change the model (LoRA wrap, layer freezing) do it here, before
        # torch.compile and before the optimizer picks up the trainable parameters.
        for h in self.hooks:
            configure = getattr(h, "configure_model", None)
            new_model = configure(self.ctx) if configure else None
            if new_model is not None:
                self.model = self.ctx.model = new_model
        self.ctx.state["trainable_params"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.ctx.state["total_params"] = sum(p.numel() for p in self.model.parameters())
        if cfg.train.compile:
            # compile the decoder forward only; ResidualCapture hooks still attach to the eager
            # submodules because torch.compile keeps module identity
            self.model.forward = torch.compile(self.model.forward, dynamic=False)
        if cfg.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.logger = RunLogger(cfg.dump(), cfg.wandb.run_name or cfg.name, cfg.wandb.project, cfg.wandb.tags, self.out_dir)
        self.logger.attach_file(REPO_ROOT / cfg.data.manifest, "data-manifest")
        self.logger.attach_file(self.out_dir / "config.resolved.json", "config")

        self.step = 0
        self.gradient_tokens = 0
        self.ckpt_index = 0
        self.anomalous_grad_steps = 0
        self._gnorm_ref = 1.0
        self._extra_loss: dict[str, float] = {}  # per hook, summed over the current step's micro-batches
        self.total_steps = self._planned_steps()

    # -- planning ---------------------------------------------------------------------------
    def _planned_steps(self) -> int:
        toks_per_step = self.cfg.optim.batch_size * self.cfg.data.seq_len
        return sum(math.ceil(d.tokens / toks_per_step) for d in self.cfg.data.domains)

    # -- one optimizer step -------------------------------------------------------------------
    def _make_optimizer(self) -> torch.optim.Optimizer:
        o = self.cfg.optim
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "norm" in n.lower() else decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=o.lr, betas=o.betas,
        )

    def _loss_and_count(self, batch: torch.Tensor, domain_index: int) -> tuple[torch.Tensor, int, float]:
        """batch: [b, seq_len+1] int64. Returns (loss to backpropagate, #gradient tokens, LM loss).
        The first is the mean token loss plus any hook's extra_loss term; the last is the mean
        token loss alone. Dispatches on_micro_batch to hooks with detached logits / per-sequence
        losses / captured residuals."""
        ids = batch[:, :-1].to(self.device)
        labels = batch[:, 1:].to(self.device)
        layers = sorted({L for h in self.hooks for L in h.wants_activations()})
        from clharness.activations import ResidualCapture

        with ResidualCapture(self.model, layers) as cap, self.autocast():
            logits = self.model(input_ids=ids, use_cache=False).logits
        mask = labels != -100
        loss = chunked_cross_entropy(logits, labels, int(mask.sum()))
        lm_loss = float(loss)
        for h in self.hooks:
            extra = getattr(h, "extra_loss", None)
            term = extra(self.ctx, batch, logits, labels) if extra else None
            if term is not None:
                term = term.float()  # keep the added term in fp32
                loss = loss + term
                self._extra_loss[h.name] = self._extra_loss.get(h.name, 0.0) + float(term.detach())
        if layers or any(type(h).on_micro_batch is not BaseHook.on_micro_batch for h in self.hooks):
            seq_loss = per_sequence_token_loss(logits.detach(), labels)
            chunk_ids = getattr(batch, "chunk_ids", torch.arange(batch.shape[0]))
            for h in self.hooks:
                h.on_micro_batch(self.ctx, self.step + 1, domain_index, chunk_ids, batch, logits.detach(), seq_loss, cap.acts)
        return loss, int(mask.sum()), lm_loss

    # -- checkpoint + hook dispatch -----------------------------------------------------------
    def _checkpoint(self, domain: str, domain_index: int, boundary: bool, optimizer) -> None:
        cfg = self.cfg
        is_tier2 = (self.ckpt_index % cfg.eval.tier2_every_n_checkpoints == 0) or boundary
        meta = {
            "index": self.ckpt_index, "step": self.step, "gradient_tokens": self.gradient_tokens,
            "domain": domain, "domain_index": domain_index, "is_domain_boundary": boundary,
            "is_tier2": is_tier2, "time": time.time(),
        }
        path = save_checkpoint(self.out_dir, self.ckpt_index, self.model, optimizer, meta, cfg.train.keep_checkpoint_weights, cfg.train.keep_optimizer_state)
        info = CheckpointInfo(
            index=self.ckpt_index, step=self.step, gradient_tokens=self.gradient_tokens, domain=domain,
            domain_index=domain_index, is_domain_boundary=boundary, path=str(path), is_tier2=is_tier2,
        )
        logs: dict[str, Any] = {
            "ckpt/index": self.ckpt_index, "ckpt/gradient_tokens": self.gradient_tokens,
            "ckpt/domain_index": domain_index, "ckpt/is_boundary": float(boundary), "ckpt/is_tier2": float(is_tier2),
        }
        t0 = time.time()
        for h in self.hooks:
            th = time.time()
            _merge_logs(logs, h.name, h.on_checkpoint(self.ctx, info))
            logs[f"{h.name}/wall_s"] = time.time() - th
        logs["ckpt/hooks_wall_s"] = time.time() - t0
        self.logger.log(logs, step=self.step)
        with open(path / "metrics.json", "w") as f:
            json.dump(logs, f, indent=2, sort_keys=True, default=float)
        self.ckpt_index += 1
        _empty_cache(self.device)

    # -- main loop ----------------------------------------------------------------------------
    def run(self) -> None:
        cfg, o = self.cfg, self.cfg.optim
        if o.batch_size % o.micro_batch_size != 0:
            raise ValueError("batch_size must be a multiple of micro_batch_size")
        accum = o.batch_size // o.micro_batch_size
        optimizer = self._make_optimizer()

        logs: dict[str, Any] = {}
        for h in self.hooks:
            _merge_logs(logs, h.name, h.on_run_start(self.ctx))
        logs["train/trainable_params"] = self.ctx.state["trainable_params"]
        logs["train/total_params"] = self.ctx.state["total_params"]
        self.logger.log(logs, step=0)
        toks_per_step = o.batch_size * cfg.data.seq_len

        # fail on an over-budget domain before spending 10 minutes on the base checkpoint's evals
        for di, dom in enumerate(cfg.data.domains):
            entry = self.data_manifest["domains"][dom.name]["splits"]["stream"]
            avail = PackedStream(load_tokens(entry["tokens"]["path"]), cfg.data.seq_len, seed=0).tokens_available
            if avail < dom.tokens:
                raise ValueError(f"{dom.name}: budget {dom.tokens} tokens > {avail} available (single epoch)")

        # checkpoint 0 = base model, measured before any update
        self._checkpoint(domain="base", domain_index=-1, boundary=True, optimizer=None)

        self.model.train()
        pbar = tqdm(total=self.total_steps, desc="train", dynamic_ncols=True)
        for di, dom in enumerate(cfg.data.domains):
            stream_entry = self.data_manifest["domains"][dom.name]["splits"]["stream"]
            tokens = load_tokens(stream_entry["tokens"]["path"])
            stream = PackedStream(tokens, cfg.data.seq_len, seed=cfg.train.seed + di)
            if stream.tokens_available < dom.tokens:
                raise ValueError(
                    f"{dom.name}: budget {dom.tokens} tokens > {stream.tokens_available} available (single epoch)"
                )
            for h in self.hooks:
                _merge_logs(logs := {}, h.name, h.on_domain_start(self.ctx, dom.name))
            self.logger.log({**logs, "domain/index": di}, step=self.step)

            tokens_in_domain = 0
            domain_start_step = self.step
            domain_steps = math.ceil(dom.tokens / toks_per_step)
            batches = stream.batches(o.micro_batch_size)
            done = False
            while not done:
                optimizer.zero_grad(set_to_none=True)
                step_loss, step_tokens, step_seqs = 0.0, 0, 0
                self._extra_loss = {}
                for _ in range(accum):
                    try:
                        batch = next(batches)
                    except StopIteration:
                        done = True
                        break
                    for h in self.hooks:
                        batch = h.select_batch(self.ctx, batch)
                        if batch is None:
                            break
                    if batch is None or batch.shape[0] == 0:
                        continue
                    loss, n_tok, lm_loss = self._loss_and_count(batch, di)
                    (loss / accum).backward()
                    step_loss += lm_loss / accum
                    step_tokens += n_tok
                    step_seqs += batch.shape[0]
                if step_tokens == 0:
                    break
                lr = lr_at(self.step, self.total_steps, o, self.step - domain_start_step, domain_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), o.grad_clip)
                # Guard: a near-zero or non-finite grad norm at a normal loss is a backend fault
                # (seen on MPS under memory pressure), not a training signal. Count it visibly.
                gnorm_f = float(gnorm)
                if not math.isfinite(gnorm_f) or (self.step > 0 and gnorm_f < 1e-3 * self._gnorm_ref):
                    self.anomalous_grad_steps += 1
                    print(f"[warn] step {self.step + 1}: grad_norm={gnorm_f:.3g} (ref {self._gnorm_ref:.3g})")
                elif math.isfinite(gnorm_f):
                    self._gnorm_ref = 0.9 * self._gnorm_ref + 0.1 * gnorm_f if self.step > 0 else gnorm_f
                optimizer.step()
                self.step += 1
                self.gradient_tokens += step_tokens
                tokens_in_domain += step_tokens
                stats = StepStats(
                    step=self.step, domain=dom.name, loss=step_loss, lr=lr, grad_norm=float(gnorm),
                    gradient_tokens=self.gradient_tokens, tokens_in_domain=tokens_in_domain, seqs=step_seqs,
                )
                logs = {
                    "train/loss": stats.loss, "train/lr": lr, "train/grad_norm": stats.grad_norm,
                    "train/gradient_tokens": self.gradient_tokens, "train/tokens_in_domain": tokens_in_domain,
                    "train/anomalous_grad_steps": self.anomalous_grad_steps,
                }
                for hook_name, total in self._extra_loss.items():
                    logs[f"train/extra_loss/{hook_name}"] = total / accum
                for h in self.hooks:
                    _merge_logs(logs, h.name, h.on_step_end(self.ctx, stats))
                self.logger.log(logs, step=self.step)
                pbar.update(1)
                pbar.set_postfix(loss=f"{stats.loss:.3f}", dom=dom.name, gtok=self.gradient_tokens)

                if tokens_in_domain >= dom.tokens:
                    done = True
                if not done and self.step % cfg.train.checkpoint_every_steps == 0:
                    self._checkpoint(dom.name, di, boundary=False, optimizer=optimizer)
                    self.model.train()

            for h in self.hooks:
                _merge_logs(logs := {}, h.name, h.on_domain_end(self.ctx, dom.name))
            self._checkpoint(dom.name, di, boundary=True, optimizer=optimizer)
            self.model.train()
        pbar.close()

        logs = {}
        for h in self.hooks:
            _merge_logs(logs, h.name, h.on_run_end(self.ctx))
        self.logger.log({**logs, "run/final_gradient_tokens": self.gradient_tokens}, step=self.step)
        self.logger.finish()
