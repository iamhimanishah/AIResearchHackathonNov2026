"""The tiny-model check for an agent-written technique. Runs on a laptop CPU in seconds.

    python -m clharness.techniques.agent_check --spec technique.json

`technique.json` is `{"source": "...", "params": {...}}`. Prints one JSON line:
`{"ok": true, ...}` or `{"ok": false, "problems": [...]}`. Exit status is 0 either way; the
caller reads the line. Run it as a subprocess with a timeout: a technique that hangs or crashes
the interpreter then fails the check instead of the loop.

What it checks, on a tiny randomly initialised Gemma 3 model trained for two domains of four
optimizer steps each on random tokens:

  * the source passes `agent_code.validate_source` and its class can be constructed;
  * training finishes without raising;
  * every step's loss and gradient norm are finite and the gradient norm is above zero;
  * the run spends exactly the gradient tokens the plain run spends (matched compute);
  * at least one parameter is still trainable, and the weights moved.

It says nothing about whether the technique helps. That is what the GPU runs are for.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

VOCAB, SEQ, STEPS = 512, 32, 4


def _tiny_config(root: Path, name: str) -> Any:
    import numpy as np

    from clharness.config import REPO_ROOT, RunConfig, load_run_config
    from clharness.data.manifest import file_entry, write_manifest

    manifest_path = root / "manifests" / "tiny.json"
    if not manifest_path.exists():
        rng = np.random.default_rng(7)
        manifest = {"set": "tiny", "seed": 7, "seq_len": SEQ, "tokenizer": {"vocab_size": VOCAB}, "domains": {}}
        for domain in ("code", "legal", "general"):
            d = root / "data" / "tiny" / domain
            d.mkdir(parents=True)
            entry: dict[str, Any] = {"source": {"name": "synthetic"}, "splits": {}}
            for split, n in (("stream", 60 * (SEQ + 1)), ("heldout", 4 * SEQ), ("probe", 4 * SEQ)):
                np.save(d / f"{split}.tokens.npy", rng.integers(3, VOCAB, size=n, dtype=np.int32))
                (d / f"{split}.jsonl").write_text("{}\n")
                files = {}
                for key, fname in (("jsonl", f"{split}.jsonl"), ("tokens", f"{split}.tokens.npy")):
                    files[key] = file_entry(d / fname, root=d)
                    files[key]["path"] = str((d / fname).resolve())
                entry["splits"][split] = {"n_docs": 1, "n_tokens": n, **files}
            manifest["domains"][domain] = entry
        write_manifest(manifest, manifest_path)
    raw = load_run_config(REPO_ROOT / "configs" / "smoke.yaml").dump()
    raw["name"] = name
    raw["eval"]["lm_eval_tasks"] = []
    raw["eval"]["perplexity_slices"] = []
    raw["model"].update({"capture_layers": [], "saes": []})
    tokens = STEPS * 4 * SEQ
    raw["data"].update({"manifest": str(manifest_path), "seq_len": SEQ, "probe_max_seqs_per_domain": None,
                        "domains": [{"name": "code", "tokens": tokens}, {"name": "legal", "tokens": tokens}]})
    raw["optim"].update({"batch_size": 4, "micro_batch_size": 2, "warmup_steps": 1, "lr": 1e-3})
    raw["train"].update({"precision": "fp32", "device": "cpu", "checkpoint_every_steps": 1000,
                         "recover_tokens": 0, "signal_log": False, "readout_probe": False,
                         "output_dir": os.path.relpath(root, REPO_ROOT)})
    return RunConfig.model_validate(raw)


def _train(root: Path, name: str, hooks: list[Any]) -> dict[str, Any]:
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from clharness.model import LoadedModel
    from clharness.train.hooks import BaseHook
    from clharness.train.loop import Trainer

    class Tok:
        bos_token_id, eos_token_id, pad_token_id = 2, 1, 0

    class Record(BaseHook):
        name = "record"

        def __init__(self) -> None:
            self.steps: list[Any] = []

        def on_step_end(self, ctx, stats):
            self.steps.append(stats)

    torch.manual_seed(0)
    model = Gemma3ForCausalLM(Gemma3TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, max_position_embeddings=512, sliding_window=128))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    lm = LoadedModel(model=model, tokenizer=Tok(), model_id="tiny", device="cpu", dtype=torch.float32)
    record = Record()
    trainer = Trainer(_tiny_config(root, name), [*hooks, record], eval_manifest=None, model_override=lm)
    trainer.run()
    after = trainer.model.state_dict()
    moved = any(k in after and not torch.equal(v, after[k]) for k, v in before.items()) or len(after) != len(before)
    return {"gradient_tokens": trainer.gradient_tokens, "steps": record.steps, "moved": moved,
            "trainable": trainer.ctx.state["trainable_params"]}


def tiny_check(source: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the check in this process. Use `main` through a subprocess for untrusted timing."""
    from clharness.techniques.agent_code import AgentCode, validate_source

    os.environ.setdefault("WANDB_MODE", "disabled")
    t0 = time.time()
    problems = validate_source(source)
    if problems:
        return {"ok": False, "stage": "static", "problems": problems}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            plain = _train(root, "plain", [])
            hook = AgentCode(AgentCode.Params(source=source, params=params or {}))
            run = _train(root, "technique", [hook])
        except Exception as exc:  # noqa: BLE001 - whatever it raises is the finding
            return {"ok": False, "stage": "tiny-model", "problems": [f"{type(exc).__name__}: {str(exc)[:300]}"]}
    if run["gradient_tokens"] != plain["gradient_tokens"]:
        problems.append(f"spent {run['gradient_tokens']} gradient tokens; the plain run spends "
                        f"{plain['gradient_tokens']} (compute must match)")
    if len(run["steps"]) != len(plain["steps"]):
        problems.append(f"took {len(run['steps'])} optimizer steps; the plain run takes {len(plain['steps'])}")
    if any(not math.isfinite(s.loss) or not math.isfinite(s.grad_norm) for s in run["steps"]):
        problems.append("a loss or gradient norm was not finite")
    if any(s.grad_norm <= 0 for s in run["steps"]):
        problems.append("a step had zero gradient")
    if run["trainable"] <= 0:
        problems.append("no parameter is left trainable")
    if not run["moved"]:
        problems.append("the weights did not move")
    return {"ok": not problems, "stage": "tiny-model", "problems": problems,
            "gradient_tokens": run["gradient_tokens"], "trainable_params": run["trainable"],
            "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True, help='JSON file: {"source": "...", "params": {...}}')
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    result = tiny_check(spec["source"], spec.get("params") or {})
    sys.stdout.write("AGENT_CHECK " + json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
