"""Strict, config-driven settings.

Rules:
  * Every result-affecting field is required in YAML: no silent defaults.
  * Unknown keys are errors (typos cannot silently disable something).
  * `include:` lets a run config pull in shared fragments (model, eval suite); later keys win.
  * The fully-resolved config is what gets logged to wandb and written next to checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from clharness.spec import TechniqueSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SAECfg(Strict):
    release: str
    sae_id: str
    layer: int


class ModelCfg(Strict):
    model_id: str
    dtype: Literal["auto", "bfloat16", "float16", "float32"]
    attn_implementation: str | None = None  # None = transformers default; not result-affecting
    capture_layers: list[int]
    saes: list[SAECfg]  # one or more per captured layer; extra SAEs at a layer are replication controls
    exclude_bos_from_stats: bool
    sae_fvu_invalid_delta: float  # pre-registered: FVU rise over base > this => layer's feature stats flagged invalid

    @model_validator(mode="after")
    def _layers_match(self) -> "ModelCfg":
        sae_layers = {s.layer for s in self.saes}
        if set(self.capture_layers) != sae_layers:
            raise ValueError(f"capture_layers {self.capture_layers} != SAE layers {sorted(sae_layers)}")
        ids = [s.sae_id for s in self.saes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sae_id")
        return self


class LMEvalTaskCfg(Strict):
    """One lm-eval task with a frozen item list."""

    task: str
    n_items: int  # number of items drawn (fixed by manifest)
    num_fewshot: int
    tier: Literal[1, 2]
    gen_kwargs: dict[str, Any] | None = None
    batch_size: int | None = None  # override the suite batch size (long-prompt tasks -> 1)


class PerplexitySliceCfg(Strict):
    name: str  # matches a held-out slice in the data manifest
    max_seqs: int | None = None  # None = whole slice


class EvalCfg(Strict):
    suite: str  # frozen item lists live in manifests/eval-<suite>.json, shared by every run
    lm_eval_tasks: list[LMEvalTaskCfg]
    perplexity_slices: list[PerplexitySliceCfg]
    tier2_every_n_checkpoints: int
    item_seed: int  # seed used to draw item indices when manifests are built
    batch_size: int
    max_length: int  # lm-eval context cap (left-truncates prompts); logits are [B, T, 262k] fp32
    include_path: str | None = None  # extra lm-eval task dir (custom tasks), repo-relative


class DomainCfg(Strict):
    name: str  # key into the data manifest's "domains"; its "stream" split is the training stream
    tokens: int  # gradient tokens to spend on this domain


class DataCfg(Strict):
    manifest: str  # repo-relative path to data manifest json
    domains: list[DomainCfg]  # in presentation order
    probe_set: str  # name of probe set in manifest
    probe_max_seqs_per_domain: int | None  # None = every probe chunk; cap for cheap configs
    seq_len: int


class OptimCfg(Strict):
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    warmup_steps: int
    schedule: Literal["constant", "cosine", "linear"]
    grad_clip: float
    batch_size: int  # sequences per optimizer step (after grad accumulation)
    micro_batch_size: int
    # Per-domain re-warm (Ibrahim et al. 2024). The defaults keep the single global schedule.
    #   global:     warmup_steps at the start of the stream, `schedule` over the whole stream
    #   per_domain: rewarm_steps of linear warmup at the start of every domain, then `schedule`
    #               over the rest of that domain
    schedule_scope: Literal["global", "per_domain"] = "global"
    rewarm_steps: int = 0  # per_domain only
    min_lr_ratio: float = 0.0  # floor of the linear / cosine decay, as a fraction of lr


class TrainCfg(Strict):
    seed: int
    # bf16-pure: bf16 weights + AdamW (cheap; small-lr updates below bf16's 4e-3 precision are
    #   partly rounded away -> smoke/plumbing only)
    # bf16-mixed: fp32 master weights, bf16 autocast forward (the pilot setting; ~16 GB for 1B)
    # fp32: everything fp32 (reference / debugging)
    precision: Literal["bf16-pure", "bf16-mixed", "fp32"]
    compile: bool  # torch.compile the model forward (CUDA); measured in the throughput probe before use
    checkpoint_every_steps: int
    keep_checkpoint_weights: bool  # False = metrics only (saves disk on sweeps)
    keep_optimizer_state: bool  # also save AdamW state with the weights (8 GB per ckpt at 1B; only for resume)
    output_dir: str
    device: Literal["auto", "cuda", "mps", "cpu"]
    probe_batch_size: int
    cka_max_tokens: int  # base-model probe activations kept for CKA (bf16, per layer); caps memory
    massive_dims_k: int  # top-k massive-activation dims (by base mean |x|) tracked and excluded from cka_nomassive
    mknn_tokens: int  # token subsample for mutual-kNN overlap (O(n^2) memory: 4096 -> 128 MB)
    mknn_k: int  # neighbours per token for mutual-kNN
    recover_tokens: int  # recoverability probe at domain boundaries: re-warm tokens (0 = off)
    recover_lr: float  # LR for the re-warm
    recover_domain: str  # manifest domain whose *stream* split is the re-warm text (normally "general")
    readout_probe: bool  # linear domain-id probes on probe-set activations (fading vs readout loss)
    signal_log: bool  # per-example importance signals logged during training (Phase 2 signal side)
    gradient_checkpointing: bool


class WandbCfg(Strict):
    project: str | None  # None -> from WANDB_PROJECT env, else offline
    run_name: str | None
    tags: list[str]


class RunConfig(Strict):
    name: str
    model: ModelCfg
    eval: EvalCfg
    data: DataCfg
    optim: OptimCfg
    train: TrainCfg
    wandb: WandbCfg
    # Optional, and empty means no mitigation. It must keep its default because
    # `clh reeval` re-validates the frozen pilot config.resolved.json files, which predate it.
    # Same {name, params} entries as RunSpec.techniques, so a spec's list passes straight through.
    techniques: tuple[TechniqueSpec, ...] = ()

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BaselineConfig(Strict):
    """Eval-only config (deliverable 2): model + eval suite, no training."""

    name: str
    model: ModelCfg
    eval: EvalCfg
    device: Literal["auto", "cuda", "mps", "cpu"]

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml_with_includes(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    includes = raw.pop("include", []) or []
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc) if not Path(inc).is_absolute() else Path(inc)
        if not inc_path.exists():
            inc_path = REPO_ROOT / inc
        merged = _deep_merge(merged, load_yaml_with_includes(inc_path))
    return _deep_merge(merged, raw)


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(load_yaml_with_includes(Path(path)))


def load_baseline_config(path: str | Path) -> BaselineConfig:
    return BaselineConfig.model_validate(load_yaml_with_includes(Path(path)))


def load_fragment(path: str | Path, model: type[Strict]) -> Any:
    """Validate a single fragment file (e.g. configs/model/*.yaml) against one section model."""
    raw = load_yaml_with_includes(Path(path))
    key = {ModelCfg: "model", EvalCfg: "eval", DataCfg: "data"}[model]
    return model.model_validate(raw[key])
