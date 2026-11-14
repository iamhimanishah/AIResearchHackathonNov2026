"""Deliverable 2: baseline every candidate eval on the base model.

For each task: score, uniform-chance level (mean over selected items of 1/#choices), majority-
class level (most frequent gold index among selected items), margin above both, standard error,
item count, wall-clock. Multiple-choice tasks report acc (and acc_norm when defined); generation
tasks report their primary metric with chance = 0.

Results are written incrementally to results/baselines/evals_<suite>.json so a killed run keeps
what it finished, and re-running skips tasks already recorded (delete the file to redo).
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import lm_eval

from clharness.config import REPO_ROOT, BaselineConfig, LMEvalTaskCfg
from clharness.evals.lm_eval_runner import build_eval_manifest, make_task_manager, run_lm_eval

PRIMARY = {  # metric to report per output type / task family
    "gsm8k": "exact_match/strict-match",
    "mbpp": "pass_at_1",
    "humaneval": "pass@1/create_test",
}


from clharness.evals.lm_eval_runner import chance_levels as _chance_levels  # noqa: E402


def _primary_metric(task_name: str, metrics: dict[str, Any]) -> tuple[str, float]:
    for key, m in PRIMARY.items():
        if task_name.startswith(key) and m in metrics:
            return m, metrics[m]
    for m in ("acc_norm", "acc"):
        if m in metrics:
            return m, metrics[m]
    nums = {k: v for k, v in metrics.items() if isinstance(v, float) and k not in ("wall_s",)}
    k = next(iter(nums))
    return k, nums[k]


def _dev_tag(device: str) -> str:
    return device.split(":")[0]


def run_baseline(cfg: BaselineConfig, device: str, model, tokenizer, limit_items: int | None = None) -> dict[str, Any]:
    # outputs are keyed by device: MPS and CUDA numerics differ, and a resumable file from one
    # backend must never be mistaken for the other's result
    out_path = REPO_ROOT / "results" / "baselines" / f"evals_{cfg.name}_{_dev_tag(device)}{'_dryrun' if limit_items else ''}.json"
    results: dict[str, Any] = json.load(open(out_path)) if out_path.exists() else {"suite": cfg.name, "device": _dev_tag(device), "tasks": {}}
    if results.get("device") not in (None, _dev_tag(device)):
        raise RuntimeError(f"{out_path} was produced on {results['device']}, not {device}")

    tasks = cfg.eval.lm_eval_tasks
    if limit_items:
        tasks = [t.model_copy(update={"n_items": min(t.n_items, limit_items)}) for t in tasks]
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"lm_eval_tasks": tasks})})
    manifest = build_eval_manifest(cfg.eval, cfg.eval.suite + ("-dryrun" if limit_items else ""))
    tm = make_task_manager(cfg.eval.include_path)
    task_objs = lm_eval.tasks.get_task_dict([t.task for t in tasks], tm)

    for t in tasks:
        if t.task in results["tasks"]:
            print(f"skip {t.task} (done)")
            continue
        entry = manifest["tasks"][t.task]
        levels = _chance_levels(task_objs[t.task], entry["indices"])
        t0 = time.time()
        res = run_lm_eval(model, tokenizer, [t], manifest, cfg.eval.batch_size, device, cfg.eval.include_path, seed=0,
                          max_length=cfg.eval.max_length)
        wall = time.time() - t0
        metrics = {k.replace(",none", "").replace(",", "/"): v for k, v in res[t.task].items()}
        mname, score = _primary_metric(t.task, metrics)
        n = entry["n_items"]
        se = math.sqrt(max(score * (1 - score), 1e-12) / n)
        rec = {
            "tier": t.tier, "num_fewshot": t.num_fewshot, "n_items": n, "n_total_docs": entry["n_total_docs"],
            "metric": mname, "score": score, "stderr": se, **levels,
            "margin_uniform": score - levels["chance_uniform"],
            "margin_majority": (score - levels["chance_majority"]) if levels["chance_majority"] is not None else None,
            "wall_s": wall, "s_per_item": wall / n, "all_metrics": metrics,
        }
        results["tasks"][t.task] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out_path, "w"), indent=2, sort_keys=True)
        print(f"{t.task:40s} {mname:26s} {score:.3f} ±{se:.3f}  chance {levels['chance_uniform']:.3f}"
              f"  maj {levels['chance_majority'] if levels['chance_majority'] is not None else '-'}"
              f"  n={n}  {wall:.0f}s")
    return results


def summarize(results: dict[str, Any]) -> str:
    """Markdown table; MMLU leaves are collapsed into one weighted row."""
    rows = []
    mm = {k: v for k, v in results["tasks"].items() if k.startswith("mmlu_")}
    other = {k: v for k, v in results["tasks"].items() if not k.startswith("mmlu_")}
    if mm:
        n = sum(v["n_items"] for v in mm.values())
        score = sum(v["score"] * v["n_items"] for v in mm.values()) / n
        ch = sum(v["chance_uniform"] * v["n_items"] for v in mm.values()) / n
        maj = sum(v["chance_majority"] * v["n_items"] for v in mm.values()) / n
        wall = sum(v["wall_s"] for v in mm.values())
        other["mmlu (57 subjects)"] = {
            "tier": 1, "num_fewshot": 0, "metric": "acc", "score": score,
            "stderr": math.sqrt(score * (1 - score) / n), "chance_uniform": ch, "chance_majority": maj,
            "margin_uniform": score - ch, "n_items": n, "wall_s": wall,
        }
    rows.append("| task | tier | shots | metric | score | ±se | chance | majority | margin | n | wall |")
    rows.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in other.items():
        maj = v.get("chance_majority")
        rows.append(
            f"| {k} | {v['tier']} | {v['num_fewshot']} | {v['metric']} | {v['score']:.3f} | {v['stderr']:.3f} | "
            f"{v['chance_uniform']:.3f} | {'-' if maj is None else f'{maj:.3f}'} | {v['margin_uniform']:+.3f} | "
            f"{v['n_items']} | {v['wall_s']/60:.1f} min |"
        )
    return "\n".join(rows)
