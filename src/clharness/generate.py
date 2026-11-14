"""`clh generate RUN --ckpts 0,10,21,32,43` — greedy completions of a fixed prompt set from the
base model and saved checkpoints, for the demo's side-by-side panel. Writes
results/runs/<run>/generations.json. Mac-friendly: one checkpoint at a time (see the reeval note
in the parking lot about loading several checkpoints in one process — here each load is
followed by a fingerprint check, and the outputs are visibly different per checkpoint)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from clharness.config import REPO_ROOT, RunConfig

# Completion-style prompts (Gemma 3 1B PT is a base model). One or two per domain plus general.
DEFAULT_PROMPTS: list[dict[str, str]] = [
    {"tag": "general", "prompt": "Question: Why does the Earth have seasons?\nAnswer:"},
    {"tag": "general", "prompt": "Question: What is the capital of Australia?\nAnswer:"},
    {"tag": "reasoning", "prompt": "Question: A train travels at 60 miles per hour for 2.5 hours. How far does it travel?\nAnswer:"},
    {"tag": "code", "prompt": "def is_prime(n):\n    \"\"\"Return True if n is a prime number.\"\"\"\n"},
    {"tag": "code", "prompt": "# Python: count how many times each word appears in a list of words\ndef word_counts(words):\n"},
    {"tag": "biomedical", "prompt": "Question: What is the function of hemoglobin in the blood?\nAnswer:"},
    {"tag": "biomedical", "prompt": "Question: Does regular exercise reduce the risk of heart disease? Answer yes, no, or maybe, then explain.\nAnswer:"},
    {"tag": "legal", "prompt": "Question: In a criminal trial, who has the burden of proof and what is the standard?\nAnswer:"},
    {"tag": "math", "prompt": "Question: What is 17 multiplied by 24?\nAnswer:"},
    {"tag": "math", "prompt": "Problem: Solve for x: 3x + 7 = 22.\nSolution:"},
]


def _fingerprint(model) -> float:
    return float(model.state_dict()["model.layers.13.mlp.down_proj.weight"].float().sum())


def generate_panel(run: str, ckpts: list[int], prompts: list[dict[str, str]] | None = None, max_new_tokens: int = 64,
                   device: str | None = None) -> Path:
    from clharness.env import pick_device, seed_everything
    from clharness.model import load_model

    prompts = prompts or DEFAULT_PROMPTS
    run_dir = REPO_ROOT / "results" / "runs" / run
    cfg = RunConfig.model_validate(json.load(open(run_dir / "config.resolved.json")))
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    ck_rows = {int(r["ckpt/index"]): r for r in rows if "ckpt/index" in r}
    names = [d.name for d in cfg.data.domains]
    seed_everything(cfg.train.seed)
    dev = pick_device(device or "auto")
    lm = load_model(cfg.model.model_id, device=dev, dtype=cfg.model.dtype)
    lm.model.eval()
    base_sd = {k: v.detach().clone() for k, v in lm.model.state_dict().items()} if 0 in ckpts else None
    out: dict[str, Any] = {"run": run, "prompts": prompts, "max_new_tokens": max_new_tokens, "checkpoints": {}}
    for i in ckpts:
        ck = run_dir / "checkpoints" / f"ckpt_{i:04d}"
        if i == 0:
            lm.model.load_state_dict(base_sd)
        else:
            wf = ck / "model.bf16.pt" if (ck / "model.bf16.pt").exists() else ck / "model.pt"
            lm.model.load_state_dict(torch.load(wf, map_location=dev), strict=True)
        fp = _fingerprint(lm.model)
        r = ck_rows.get(i, {})
        di = int(r.get("ckpt/domain_index", -1))
        label = "base model" if i == 0 else f"after {names[di]}" if 0 <= di < len(names) else f"checkpoint {i}"
        t0 = time.time()
        outputs = []
        for p in prompts:
            ids = lm.tokenizer(p["prompt"], return_tensors="pt").to(dev)
            with torch.no_grad():
                gen = lm.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                        pad_token_id=lm.tokenizer.pad_token_id or lm.tokenizer.eos_token_id)
            text = lm.tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            # cut at the first blank line: completion-style prompts run on otherwise
            cut = text.split("\n\n")[0] if "\n\n" in text else text
            outputs.append({"text": text, "short": cut.strip()})
        out["checkpoints"][str(i)] = {"label": label, "domain_index": di, "gradient_tokens": r.get("ckpt/gradient_tokens"),
                                      "fingerprint": fp, "outputs": outputs}
        print(f"ckpt {i:2d} ({label}) fingerprint {fp:.4f}: {len(prompts)} prompts in {time.time() - t0:.0f}s", flush=True)
    path = run_dir / "generations.json"
    json.dump(out, open(path, "w"), indent=1)
    return path
