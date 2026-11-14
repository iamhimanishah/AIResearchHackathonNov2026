"""Per-item view of the protected set across a run: which questions were lost, when, by which
domain, and whether they came back. The gallery in the report and Phase 1's cross-seed item
analysis read this file.

Reads items_<task>.npy (correctness per checkpoint), choices_/gold_<task>.npy when present, the
run's metrics.jsonl (checkpoint -> domain), and the eval manifest (frozen indices) plus the live
datasets through lm-eval for the question text. Writes results/runs/<run>/items.json.

Categories (per item, from the correctness vector c[0..K] over checkpoints):
  stable_correct   correct at base and at every checkpoint
  stable_wrong     wrong at base and at every checkpoint
  lost             correct at base, wrong at the end, and wrong at each of the last `persist`
                   checkpoints (not a one-checkpoint flicker)
  recovered        correct at base, wrong somewhere in between, correct at the end
  gained           wrong at base, correct at the end
  unstable         everything else (flickering)
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT

DOMAIN_NAMES = ["code", "biomedical", "legal", "math"]


def _correctness_matrix(run_dir: Path, task: str) -> np.ndarray | None:
    files = sorted(glob.glob(str(run_dir / "checkpoints" / "ckpt_*" / f"items_{task}.npy")))
    if not files:
        return None
    return np.stack([np.load(f) for f in files])  # [n_ckpt, n_items]


def _ckpt_domains(run_dir: Path) -> list[int]:
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    return [int(r["ckpt/domain_index"]) for r in rows if "ckpt/index" in r]


def classify(c: np.ndarray, persist: int = 3) -> str:
    ok = ~np.isnan(c)
    if ok.sum() < 2:
        return "unknown"
    v = c[ok] > 0.5
    if v.all():
        return "stable_correct"
    if not v.any():
        return "stable_wrong"
    base, end = v[0], v[-1]
    if base and not end and not v[-persist:].any():
        return "lost"
    if base and end:
        return "recovered"
    if not base and end:
        return "gained"
    return "unstable"


def _texts(task_name: str, indices: list[int], include_path: str | None) -> list[dict[str, Any]]:
    from clharness.evals.lm_eval_runner import get_task_dict, make_task_manager

    task = get_task_dict([task_name], make_task_manager(include_path))[task_name]
    docs = task.eval_docs
    out = []
    mc = getattr(task, "OUTPUT_TYPE", None) == "multiple_choice"
    for i in indices:
        d = docs[i]
        q = task.doc_to_text(d)
        entry: dict[str, Any] = {"question": q if isinstance(q, str) else str(q)}
        if mc:
            choices = task.doc_to_choice(d)
            g = task.doc_to_target(d)
            if isinstance(g, str):
                g = int(g) if g.strip().lstrip("-").isdigit() else (choices.index(g) if g in choices else -1)
            entry["choices"] = choices
            entry["gold"] = int(g)
        out.append(entry)
    return out


def build_items(run: str, with_text: bool = True, persist: int = 3, max_question_chars: int = 600) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / run
    cfg = json.load(open(run_dir / "config.resolved.json"))
    manifest = json.load(open(REPO_ROOT / "manifests" / f"eval-{cfg['eval']['suite']}.json"))
    domains = _ckpt_domains(run_dir)
    out: dict[str, Any] = {"run": run, "n_checkpoints": len(domains), "checkpoint_domain": domains, "tasks": {}}
    for task in manifest["tasks"]:
        C = _correctness_matrix(run_dir, task)
        if C is None:
            continue
        idx = manifest["tasks"][task]["indices"]
        texts = _texts(task, idx, cfg["eval"].get("include_path")) if with_text else [{} for _ in idx]
        ck_files = sorted(glob.glob(str(run_dir / "checkpoints" / "ckpt_*" / f"items_{task}.npy")))
        ck_index = [int(Path(f).parent.name.split("_")[1]) for f in ck_files]
        choices_by_ck = {}
        for k in ck_index:
            fc = run_dir / "checkpoints" / f"ckpt_{k:04d}" / f"choices_{task}.npy"
            if fc.exists():
                choices_by_ck[k] = np.load(fc)
        items = []
        for j in range(C.shape[1]):
            c = C[:, j]
            cat = classify(c, persist)
            first_loss = None
            if cat in ("lost", "recovered", "unstable") and c[0] > 0.5:
                wrong = np.where(c < 0.5)[0]
                first_loss = int(ck_index[wrong[0]]) if wrong.size else None
            entry = {
                "index": int(idx[j]),
                "category": cat,
                "correct": "".join("1" if x > 0.5 else ("0" if x == x else "-") for x in c),
                "n_wrong": int((c < 0.5).sum()),
                "first_loss_ckpt": first_loss,
                "first_loss_domain": DOMAIN_NAMES[domains[first_loss]] if first_loss is not None and 0 <= domains[first_loss] < 4 else None,
                **{k: (v[:max_question_chars] if isinstance(v, str) else v) for k, v in texts[j].items()},
            }
            if choices_by_ck:
                entry["chosen"] = {str(k): int(v[j]) for k, v in choices_by_ck.items()}
            items.append(entry)
        counts = Counter(e["category"] for e in items)
        by_domain = Counter(e["first_loss_domain"] for e in items if e["category"] == "lost")
        out["tasks"][task] = {
            "n_items": len(items),
            "acc_base": float(np.nanmean(C[0])),
            "acc_end": float(np.nanmean(C[-1])),
            "counts": dict(counts),
            "lost_by_domain": dict(by_domain),
            "items": items,
        }
    json.dump(out, open(run_dir / "items.json", "w"), indent=1)
    return out


def summary(t: dict[str, Any]) -> str:
    L = [f"# {t['run']}: {t['n_checkpoints']} checkpoints"]
    L.append(f"{'task':16s} {'base':>6s} {'end':>6s}  {'lost':>5s} {'recov':>5s} {'gain':>5s} {'unst':>5s}  lost by domain")
    for name, tk in t["tasks"].items():
        c = tk["counts"]
        L.append(f"{name:16s} {tk['acc_base']:6.3f} {tk['acc_end']:6.3f}  {c.get('lost', 0):5d} {c.get('recovered', 0):5d} "
                 f"{c.get('gained', 0):5d} {c.get('unstable', 0):5d}  " + ", ".join(f"{d}:{n}" for d, n in sorted(tk["lost_by_domain"].items(), key=lambda x: -x[1])))
    return "\n".join(L)
