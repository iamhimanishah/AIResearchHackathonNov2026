"""lm-eval-harness wrapper with frozen item lists.

Item freezing: for each task we draw `n_items` indices into `task.eval_docs` (test split if the
task has one, else validation — lm-eval's own rule) with `item_seed`, and store them with a
sha256 of the selected docs in manifests/eval-<suite>.json. At eval time we pass those indices
via `simple_evaluate(samples=...)`, so the harness scores exactly those items. The manifest hash
check catches upstream dataset changes.

The model is passed in-process (HFLM(pretrained=<model object>)), so no reload per checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

# mbpp / humaneval score by *executing* model-generated code (HF `code_eval`): each program runs
# in a subprocess with a timeout under the metric's reliability guard (os.system, shutil.rmtree,
# ... disabled). lm-eval's mbpp module checks this flag at import time, so it must be set before
# lm_eval is imported. Still: run evals on a machine you'd be fine re-imaging.
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

import lm_eval  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from lm_eval.tasks import TaskManager, get_task_dict  # noqa: E402
import numpy as np  # noqa: E402
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from clharness.config import REPO_ROOT, EvalCfg, LMEvalTaskCfg
from clharness.data.manifest import sha256_text, write_manifest


def make_task_manager(include_path: str | None) -> TaskManager:
    if include_path:
        p = Path(include_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return TaskManager(include_path=str(p))
    return TaskManager()


def _doc_hash(docs: list[dict]) -> str:
    return sha256_text(json.dumps(docs, sort_keys=True, default=str))


def chance_levels(task, indices: list[int]) -> dict[str, float | None]:
    """Uniform-chance (mean 1/#choices) and majority-class levels over the selected items;
    generation tasks get chance 0. Used for chance-adjusted flip metrics and baseline tables."""
    from collections import Counter

    docs = task.eval_docs
    if getattr(task, "OUTPUT_TYPE", None) != "multiple_choice":
        return {"chance_uniform": 0.0, "chance_majority": None, "n_choices_mean": None}
    inv, golds = [], []
    for i in indices:
        d = docs[i]
        choices = task.doc_to_choice(d)
        inv.append(1.0 / len(choices))
        t = task.doc_to_target(d)
        if isinstance(t, str):
            if t.strip().lstrip("-").isdigit():
                t = int(t)
            elif t in choices:
                t = choices.index(t)
            else:
                # WinoGrande scores the sentence ending given each candidate context: the target
                # is the ending, and the gold index is what doc_to_text returns.
                alt = task.doc_to_text(d)
                t = alt if isinstance(alt, int) else None
        if t is not None:
            golds.append(int(t))
    maj = Counter(golds).most_common(1)[0][1] / len(golds) if golds else None
    return {"chance_uniform": sum(inv) / len(inv), "chance_majority": maj, "n_choices_mean": len(inv) / sum(inv)}


def build_eval_manifest(cfg: EvalCfg, suite_name: str) -> dict[str, Any]:
    """Draw and record item indices (+ chance levels) for every lm-eval task in the suite."""
    tm = make_task_manager(cfg.include_path)
    task_dict = lm_eval.tasks.get_task_dict([t.task for t in cfg.lm_eval_tasks], tm)
    manifest: dict[str, Any] = {"suite": suite_name, "item_seed": cfg.item_seed, "tasks": {}}
    for tcfg in cfg.lm_eval_tasks:
        task = task_dict[tcfg.task]
        docs = task.eval_docs
        n_total = len(docs)
        rng = random.Random(f"{cfg.item_seed}:{tcfg.task}")
        n = min(tcfg.n_items, n_total)
        idx = sorted(rng.sample(range(n_total), n))
        selected = [docs[i] for i in idx]
        manifest["tasks"][tcfg.task] = {
            "tier": tcfg.tier,
            "num_fewshot": tcfg.num_fewshot,
            "n_total_docs": n_total,
            "n_items": n,
            "indices": idx,
            "docs_sha256": _doc_hash(selected),
            "dataset_path": getattr(task.config, "dataset_path", None),
            "dataset_name": getattr(task.config, "dataset_name", None),
            **chance_levels(task, idx),
        }
    write_manifest(manifest, REPO_ROOT / "manifests" / f"eval-{suite_name}.json")
    return manifest


def verify_eval_manifest(cfg: EvalCfg, manifest: dict[str, Any]) -> list[str]:
    """Re-derive the doc hashes from the live datasets; return a list of problems (empty = ok)."""
    tm = make_task_manager(cfg.include_path)
    task_dict = lm_eval.tasks.get_task_dict(list(manifest["tasks"]), tm)
    problems = []
    for name, entry in manifest["tasks"].items():
        docs = task_dict[name].eval_docs
        if len(docs) != entry["n_total_docs"]:
            problems.append(f"{name}: dataset size {len(docs)} != {entry['n_total_docs']}")
            continue
        h = _doc_hash([docs[i] for i in entry["indices"]])
        if h != entry["docs_sha256"]:
            problems.append(f"{name}: selected docs changed (hash mismatch)")
    return problems


_ITEM_METRIC_PREFERENCE = ("acc_norm", "acc", "pass_at_1", "pass@1", "exact_match")


def _primary_item_metric(samples: list[dict]) -> str | None:
    if not samples:
        return None
    keys = samples[0].keys()
    for m in _ITEM_METRIC_PREFERENCE:
        for k in keys:
            if k == m or k.startswith(m + ","):
                return k
    return None


def _mc_choices(task, samples: list[dict], metric: str) -> tuple[dict[int, int], dict[int, int]]:
    """Chosen option index per doc (mirrors lm-eval's `pred` / `pred_norm`) and the gold index.

    Uses the same `doc_to_choice` and character-length normalisation as lm-eval's
    `process_results` for multiple_choice tasks, so `chosen == gold` reproduces acc / acc_norm.
    """
    norm = metric.startswith("acc_norm")
    chosen: dict[int, int] = {}
    gold: dict[int, int] = {}
    for smp in samples:
        doc = smp["doc"]
        choices = task.doc_to_choice(doc)
        lls = np.array([float(r[0]) for r in smp["filtered_resps"]][: len(choices)])
        if lls.size != len(choices):
            continue
        if norm:
            lls = lls / np.array([float(len(c)) for c in choices])
        chosen[int(smp["doc_id"])] = int(np.argmax(lls))
        g = task.doc_to_target(doc)
        if isinstance(g, str):  # same rule as chance_levels: digit string = index, else a choice string
            g = int(g) if g.strip().lstrip("-").isdigit() else (choices.index(g) if g in choices else -1)
        gold[int(smp["doc_id"])] = int(g) if isinstance(g, (int, np.integer)) else -1
    return chosen, gold


def run_lm_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tasks: list[LMEvalTaskCfg],
    manifest: dict[str, Any],
    batch_size: int,
    device: str,
    include_path: str | None,
    seed: int,
    max_length: int = 2048,
) -> dict[str, Any]:
    """Evaluate the in-memory model on the frozen items. Returns {task: {metric: value, ...}}."""
    if not tasks:
        return {}
    was_training = model.training
    model.eval()
    tm = make_task_manager(include_path)
    results: dict[str, Any] = {}
    # group by num_fewshot / gen_kwargs / batch size, since simple_evaluate takes them per call
    groups: dict[tuple, list[LMEvalTaskCfg]] = {}
    for t in tasks:
        key = (t.num_fewshot, json.dumps(t.gen_kwargs, sort_keys=True), t.batch_size or batch_size)
        groups.setdefault(key, []).append(t)
    for (nfs, gk, bs), group in groups.items():
        names = [t.task for t in group]
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs, device=device, max_length=max_length)
        t0 = time.time()
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=names,
            num_fewshot=nfs,
            samples={n: manifest["tasks"][n]["indices"] for n in names},
            gen_kwargs=json.loads(gk),
            task_manager=tm,
            log_samples=True,  # per-item correctness for chance-adjusted 1->0 / 0->1 flips
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            bootstrap_iters=0,
            confirm_run_unsafe_code=True,
            verbosity="ERROR",
        )
        wall = time.time() - t0
        task_objs = get_task_dict(names, tm)
        for n in names:
            r = {k: v for k, v in out["results"][n].items() if isinstance(v, (int, float))}
            r["wall_s"] = wall / len(names)
            r["n_items"] = manifest["tasks"][n]["n_items"]
            # per-item primary-metric vector in manifest index order (doc_id = index into eval_docs)
            samples = out.get("samples", {}).get(n, [])
            metric = _primary_item_metric(samples)
            if metric is not None:
                by_doc = {int(smp["doc_id"]): float(smp[metric]) for smp in samples}
                r["_items"] = [by_doc.get(i, float("nan")) for i in manifest["tasks"][n]["indices"]]
                r["_item_metric"] = metric
            # multiple-choice tasks: which option the model picked (for the degenerate-answer alarm)
            task = task_objs.get(n)
            if task is not None and getattr(task, "OUTPUT_TYPE", None) == "multiple_choice" and samples:
                ch, gd = _mc_choices(task, samples, metric or "acc")
                r["_choices"] = [ch.get(i, -1) for i in manifest["tasks"][n]["indices"]]
                r["_gold"] = [gd.get(i, -1) for i in manifest["tasks"][n]["indices"]]
            results[n] = r
    if was_training:
        model.train()
    return results
