"""C5 tasks and prompts: SST-2 -> MNLI -> AG News, each subsampled to 5k train / 1k eval.

The subsample is frozen in `manifests/tinker-c5.json`: per task it records the dataset
revision, the chosen train/eval indices and a sha256 of the selected raw examples, so a
run trains and scores exactly the items every other run did. `build_manifest` is the only
function that touches the network; runs and tests read the manifest and inject their own
loader.

Each task has one text-to-text prompt template and one fixed verbalizer per label (the
scored completion is " " + the label word). Both live here in code so their hash can be
recorded in the manifest and in every result row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT
from clharness.data.manifest import sha256_text
from clharness.spec import canonical_json

# Pinned hub revisions (checked 2026-11-14). `datasets>=4` needs the namespaced repo ids.
GLUE_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
AGNEWS_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"

MANIFEST_SEED = 20261114
N_TRAIN = 5000
N_EVAL = 1000
SMOKE_N_TRAIN = 50
SMOKE_N_EVAL = 50
TASK_ORDER = ["sst2", "mnli", "ag_news"]

MANIFEST_PATH = REPO_ROOT / "manifests" / "tinker-c5.json"


@dataclass(frozen=True)
class TaskDef:
    name: str
    repo_id: str
    config: str | None
    revision: str
    train_split: str
    eval_split: str
    fields: tuple[str, ...]  # text fields interpolated into the template, in order
    labels: tuple[str, ...]  # dataset ClassLabel names, index order; also the verbalizers
    template: str  # one {field} placeholder per entry in `fields`, ends with the answer cue


TASKS: dict[str, TaskDef] = {
    "sst2": TaskDef(
        name="sst2",
        repo_id="nyu-mll/glue",
        config="sst2",
        revision=GLUE_REVISION,
        train_split="train",
        eval_split="validation",  # 872 items; topped up from train to reach 1k
        fields=("sentence",),
        labels=("negative", "positive"),
        template="Review: {sentence}\n"
                 "Question: Is the sentiment of this review negative or positive?\n"
                 "Answer:",
    ),
    "mnli": TaskDef(
        name="mnli",
        repo_id="nyu-mll/glue",
        config="mnli",
        revision=GLUE_REVISION,
        train_split="train",
        eval_split="validation_matched",
        fields=("premise", "hypothesis"),
        labels=("entailment", "neutral", "contradiction"),
        template="Premise: {premise}\n"
                 "Hypothesis: {hypothesis}\n"
                 "Question: Does the premise entail the hypothesis? "
                 "Answer entailment, neutral, or contradiction.\n"
                 "Answer:",
    ),
    "ag_news": TaskDef(
        name="ag_news",
        repo_id="fancyzhx/ag_news",
        config=None,
        revision=AGNEWS_REVISION,
        train_split="train",
        eval_split="test",
        fields=("text",),
        labels=("World", "Sports", "Business", "Sci/Tech"),
        template="Article: {text}\n"
                 "Question: Which section does this article belong to: "
                 "World, Sports, Business, or Sci/Tech?\n"
                 "Answer:",
    ),
}


def verbalizer(task: TaskDef, label: int) -> str:
    """The scored completion for a label. The leading space is part of the tokenization."""
    return " " + task.labels[label]


def template_payload() -> dict[str, Any]:
    return {name: {"template": t.template, "verbalizers": list(t.labels)}
            for name, t in TASKS.items()}


def template_sha256() -> str:
    return sha256_text(canonical_json(template_payload()))


def render_prompt(task: TaskDef, row: dict[str, Any]) -> str:
    return task.template.format(**{f: str(row[f]).strip() for f in task.fields})


# --- manifest ---------------------------------------------------------------------------------


def _default_load_split(task: TaskDef, split: str):
    from datasets import load_dataset

    kw: dict[str, Any] = {}
    if task.config:
        kw["name"] = task.config
    return load_dataset(task.repo_id, split=split, revision=task.revision, **kw)


def _sha_examples(train_rows: list[dict], eval_rows: list[dict]) -> str:
    return sha256_text(canonical_json({"train": train_rows, "eval": eval_rows}))


def _fields_of(task: TaskDef, row: dict[str, Any]) -> dict[str, Any]:
    return {f: row[f] for f in task.fields} | {"label": int(row["label"])}


def build_manifest(
    out: str | Path = MANIFEST_PATH,
    seed: int = MANIFEST_SEED,
    n_train: int = N_TRAIN,
    n_eval: int = N_EVAL,
    load_split=None,
) -> dict[str, Any]:
    """Draw the frozen subsamples and write the manifest.

    Per task: `n_train` train indices from the train split, and `n_eval` eval items from the
    official eval split. SST-2's validation split has only 872 items, so the shortfall is
    topped up from the train split, excluding the train subsample. `load_split(task, split)`
    is injectable so tests need no dataset download.
    """
    load_split = load_split or _default_load_split
    manifest: dict[str, Any] = {
        "name": "tinker-c5",
        "seed": seed,
        "n_train": n_train,
        "n_eval": n_eval,
        "template_sha256": template_sha256(),
        "tasks": {},
    }
    for i, name in enumerate(TASK_ORDER):
        task = TASKS[name]
        train_ds = load_split(task, task.train_split)
        eval_ds = load_split(task, task.eval_split)
        for ds, split in ((train_ds, task.train_split), (eval_ds, task.eval_split)):
            names = ds.features["label"].names
            if list(names) != list(task.labels):
                raise ValueError(
                    f"{name} {split}: dataset label names {list(names)} != registry {list(task.labels)}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        train_idx = sorted(int(x) for x in rng.choice(len(train_ds), size=n_train, replace=False))
        k = min(n_eval, len(eval_ds))
        eval_idx = sorted(int(x) for x in rng.choice(len(eval_ds), size=k, replace=False))
        eval_items = [{"split": task.eval_split, "index": j} for j in eval_idx]
        if len(eval_items) < n_eval:  # sst2: 872 validation + 128 held-out train rows
            pool = sorted(set(range(len(train_ds))) - set(train_idx))
            top_up = sorted(int(x) for x in rng.choice(len(pool), size=n_eval - len(eval_items),
                                                     replace=False))
            eval_items += [{"split": task.train_split, "index": pool[j]} for j in top_up]
        train_rows = [_fields_of(task, train_ds[j]) for j in train_idx]
        eval_rows = [_fields_of(task, (eval_ds if it["split"] == task.eval_split else train_ds)
                                [it["index"]]) for it in eval_items]
        manifest["tasks"][name] = {
            "dataset_path": task.repo_id,
            "dataset_name": task.config,
            "revision": task.revision,
            "train_split": task.train_split,
            "eval_split": task.eval_split,
            "train_indices": train_idx,
            "eval_items": eval_items,
            "sha256": _sha_examples(train_rows, eval_rows),
        }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def read_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# --- loading the frozen items -------------------------------------------------------------------


def load_task_examples(
    task: TaskDef,
    manifest: dict[str, Any],
    n_train: int | None = None,
    n_eval: int | None = None,
    load_split=None,
) -> dict[str, list[dict[str, Any]]]:
    """The task's frozen rows: {"train": [...], "eval": [...]}. `n_train`/`n_eval` take a
    prefix of the frozen lists (the --smoke path); the subsample itself never changes."""
    load_split = load_split or _default_load_split
    entry = manifest["tasks"][task.name]
    splits: dict[str, Any] = {}

    def rows(split: str, index: int) -> dict[str, Any]:
        if split not in splits:
            splits[split] = load_split(task, split)
        return _fields_of(task, splits[split][index])

    train_idx = entry["train_indices"][: n_train or len(entry["train_indices"])]
    eval_items = entry["eval_items"][: n_eval or len(entry["eval_items"])]
    return {
        "train": [rows(task.train_split, i) for i in train_idx],
        "eval": [rows(it["split"], it["index"]) for it in eval_items],
    }


def verify_task_sha(task: TaskDef, manifest: dict[str, Any], load_split=None) -> None:
    """Re-hash the manifest's selected raw examples; fails loudly if the data moved."""
    entry = manifest["tasks"][task.name]
    got = load_task_examples(task, manifest, load_split=load_split)
    if _sha_examples(got["train"], got["eval"]) != entry["sha256"]:
        raise ValueError(f"{task.name}: examples no longer match manifests/tinker-c5.json")
