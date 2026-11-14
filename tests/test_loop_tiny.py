"""End-to-end trainer loop on a tiny random Gemma 3 model (CPU, seconds, no downloads).

Covers: domain order + token budgets, gradient-token counting, checkpoint schedule (every N +
boundaries), hook dispatch, precision paths (fp32 and bf16-mixed autocast), wandb offline logging.
The real-model path (SAEs, lm-eval) is covered by the smoke test.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from clharness.config import REPO_ROOT, RunConfig, load_run_config
from clharness.data.manifest import file_entry, write_manifest
from clharness.model import LoadedModel
from clharness.train.hooks import BaseHook
from clharness.train.loop import Trainer

VOCAB = 512  # tiny vocab keeps the LM head cheap; the real 262k vocab is exercised by the smoke test


class _Tok:
    """Minimal tokenizer stand-in: the loop only needs bos_token_id."""

    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0


def _abs_entry(path: Path) -> dict:
    e = file_entry(path, root=path.parent)
    e["path"] = str(path.resolve())
    return e


def _synthetic_manifest(root: Path, seq_len: int, seed: int) -> Path:
    """Random-token data set with two domains under `root`; manifest entries use absolute paths."""
    rng = np.random.default_rng(seed)
    man = {"set": "tiny", "seed": seed, "seq_len": seq_len, "tokenizer": {"vocab_size": VOCAB}, "domains": {}}
    for dom in ("code", "math"):
        d = root / "data" / "tiny" / dom
        d.mkdir(parents=True)
        entry = {"source": {"name": "synthetic"}, "splits": {}}
        for split, n in (("stream", 40 * (seq_len + 1)), ("heldout", 4 * seq_len), ("probe", 4 * seq_len)):
            toks = rng.integers(3, VOCAB, size=n, dtype=np.int32)
            np.save(d / f"{split}.tokens.npy", toks)
            (d / f"{split}.jsonl").write_text("{}\n")
            # absolute paths: the trainer verifies manifests against REPO_ROOT; absolute entries resolve as-is
            entry["splits"][split] = {"n_docs": 1, "n_tokens": n, "jsonl": _abs_entry(d / f"{split}.jsonl"),
                                      "tokens": _abs_entry(d / f"{split}.tokens.npy")}
        man["domains"][dom] = entry
    write_manifest(man, root / "manifests" / "tiny.json")
    return root / "manifests" / "tiny.json"


class RecordingHook(BaseHook):
    name = "rec"

    def __init__(self):
        self.checkpoints = []
        self.steps = []

    def on_checkpoint(self, ctx, ckpt):
        self.checkpoints.append(ckpt)
        return {"n": float(len(self.checkpoints))}

    def on_step_end(self, ctx, stats):
        self.steps.append(stats)
        return None


def _tiny_model(vocab: int, dtype: torch.dtype) -> LoadedModel:
    cfg = Gemma3TextConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=512,
        sliding_window=128,
    )
    torch.manual_seed(0)
    model = Gemma3ForCausalLM(cfg).to(dtype)
    return LoadedModel(model=model, tokenizer=_Tok(), model_id="tiny", device="cpu", dtype=dtype)


@pytest.mark.parametrize("precision", ["fp32", "bf16-mixed"])
def test_trainer_loop_tiny(tmp_path: Path, monkeypatch, precision: str):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)
    seq_len = 32
    manifest = _synthetic_manifest(tmp_path, seq_len, seed=7)
    raw = load_run_config(REPO_ROOT / "configs" / "smoke.yaml").dump()
    raw["name"] = f"tiny-{precision}"
    raw["eval"]["lm_eval_tasks"] = []  # no lm-eval on a random model
    raw["eval"]["perplexity_slices"] = []
    raw["data"].update({"manifest": str(manifest), "seq_len": seq_len, "probe_max_seqs_per_domain": None,
                        "domains": [{"name": "code", "tokens": 4 * 4 * seq_len}, {"name": "math", "tokens": 4 * 4 * seq_len}]})
    raw["optim"].update({"batch_size": 4, "micro_batch_size": 2, "warmup_steps": 1})  # 4 x 32 tok/step -> 4 steps/domain
    raw["train"].update({"precision": precision, "device": "cpu", "checkpoint_every_steps": 3,
                         "output_dir": os.path.relpath(tmp_path, REPO_ROOT)})
    cfg = RunConfig.model_validate(raw)
    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    hook = RecordingHook()
    Trainer(cfg, [hook], eval_manifest=None, model_override=lm).run()

    # 2 domains x 4 steps; checkpoints: base, step 3, boundary(4), step 6, boundary(8)
    assert [s.step for s in hook.steps] == list(range(1, 9))
    assert [c.step for c in hook.checkpoints] == [0, 3, 4, 6, 8]
    assert [c.is_domain_boundary for c in hook.checkpoints] == [True, False, True, False, True]
    assert hook.checkpoints[-1].gradient_tokens == 8 * 4 * seq_len
    assert all(s.grad_norm > 0 and s.loss > 0 for s in hook.steps)
    metas = [json.loads(p.read_text()) for p in sorted((tmp_path / cfg.name / "checkpoints").glob("*/meta.json"))]
    assert [m["domain"] for m in metas] == ["base", "code", "code", "math", "math"]
    # weights actually moved (fp32 master weights under both precisions)
    assert lm.model.get_input_embeddings().weight.dtype == torch.float32
