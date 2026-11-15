"""Where a run's forgetting falls: on the fragile core, or spread over everything else.

Exploratory (found on the Saturday night of the hackathon, after the pre-registration). On the
selection half the plain run does not forget at random: a small frozen set of questions, the
"fragile core" (manifests/eval-protected-v1-core.json, written by scripts/make_core_items.py),
is lost in almost every seed, while most questions the base model knows are never lost. Two
runs with the same retention score can therefore differ in WHAT they lost. This module reads a
run's per-item files and reports three shares, all on the selection half:

    core_forgotten     share of the core questions the run now gets wrong
    noncore_forgotten  share of the other questions the base model got right that are now wrong
    gained             share of the questions the base model got wrong that are now right

CAVEAT. The five no-mitigation runs at lr 1e-4 define the core (a question is in it because at
least 4 of those 5 forgot it), so their own `core_forgotten` is high by construction and their
`noncore_forgotten` is low by construction. Compare other arms with each other, not with that arm.

Only selection-half positions are ever read: the report half stays behind the firewall
(analysis/PREREG.md).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT
from clharness.executor.rows import half_indices

CORE_MANIFEST = REPO_ROOT / "manifests" / "eval-protected-v1-core.json"
OUTCOMES = ("core_forgotten", "noncore_forgotten")


@lru_cache(maxsize=4)
def core_manifest(path: Path = CORE_MANIFEST) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def set_sizes(path: Path = CORE_MANIFEST) -> dict[str, int]:
    """How many questions each outcome is measured on: the n in its binomial noise."""
    m = core_manifest(path)
    return {"core_forgotten": m["n_core"], "noncore_forgotten": m["n_noncore_base_correct"]}


def _checkpoint_dirs(row: Any, repo_root: Path) -> list[Path]:
    """The run's local checkpoint folders. `items_path` is repo-relative; rows written before the
    local mirror existed point into the cloud volume, so fall back to the mirror's fixed place."""
    candidates = [row.items_path] if row.items_path else []
    candidates.append(f"results/items/{row.spec_hash[:16]}/checkpoints")
    for rel in candidates:
        if Path(rel).is_absolute():
            continue
        found = sorted((Path(repo_root) / rel).glob("ckpt_*"))
        if len(found) >= 2:
            return found
    return []


def item_profile(row: Any, repo_root: Path = REPO_ROOT, manifest: Path = CORE_MANIFEST) -> dict[str, Any] | None:
    """The three shares for one result row, or None when its item files are not on this machine.

    The base model is the run's first checkpoint and the end of the stream is its last. A core
    question counts as forgotten when the last checkpoint scores it at or below 0.5. See the
    module docstring for the caveat on the no-mitigation arm at lr 1e-4."""
    ckpts = _checkpoint_dirs(row, repo_root)
    if not ckpts:
        return None
    tasks = core_manifest(manifest)["tasks"]
    core_lost = core_n = other_lost = other_n = gained = wrong_n = 0
    for task, info in tasks.items():
        first, last = ckpts[0] / f"items_{task}.npy", ckpts[-1] / f"items_{task}.npy"
        if not first.exists() or not last.exists():
            return None
        base, final = np.load(first), np.load(last)
        if len(base) != info["n_items"] or len(final) != info["n_items"]:
            return None
        selection = half_indices(task, len(base))[0]  # the report half is never indexed
        is_core = np.zeros(len(base), dtype=bool)
        is_core[info["core_positions"]] = True
        b, f, c = base[selection] > 0.5, final[selection] > 0.5, is_core[selection]
        core_lost += int((c & ~f).sum())
        core_n += int(c.sum())
        other_lost += int((b & ~c & ~f).sum())
        other_n += int((b & ~c).sum())
        gained += int((~b & f).sum())
        wrong_n += int((~b).sum())
    if not core_n or not other_n or not wrong_n:
        return None
    return {"core_forgotten": core_lost / core_n, "noncore_forgotten": other_lost / other_n,
            "gained": gained / wrong_n, "n_core": core_n, "n_noncore": other_n, "n_base_wrong": wrong_n}


def defines_core(spec: Any) -> bool:
    """True for the arm whose seeds define the core: no technique, no optimizer override."""
    return not spec.techniques and not spec.optim


def arm_profiles(rows: Sequence[Any], repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Mean of the three shares per arm, over the seeds whose item files are present."""
    arms: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.status != "finished" or row.retention is None:
            continue
        spec = row.spec
        lr = spec.optim.get("lr")
        stream = "+".join(f"{d.name}:{d.tokens}" for d in spec.stream)
        key = json.dumps([stream, spec.optim, [t.model_dump(mode="json") for t in spec.techniques]], sort_keys=True)
        arm = arms.setdefault(key, {
            "arm": ("+".join(spec.technique_names()) or "no-mitigation") + (f"@lr{lr:g}" if lr is not None else ""),
            "stream": stream, "defines_core": defines_core(spec), "by_seed": {}, "no_files": 0})
        profile = item_profile(row, repo_root)
        if profile is None:
            arm["no_files"] += 1
        else:
            arm["by_seed"][spec.seed] = profile
    out, used = [], set()
    for arm in arms.values():
        while (arm["stream"], arm["arm"]) in used:  # same techniques, different parameters
            arm["arm"] += "'"
        used.add((arm["stream"], arm["arm"]))
        seeds = sorted(arm["by_seed"])
        means = {k: statistics.mean(arm["by_seed"][s][k] for s in seeds) if seeds else None
                 for k in ("core_forgotten", "noncore_forgotten", "gained")}
        out.append({"arm": arm["arm"], "stream": arm["stream"], "defines_core": arm["defines_core"],
                    "seeds": seeds, "no_files": arm["no_files"], **means})
    return sorted(out, key=lambda a: (a["stream"], a["arm"]))


def format_table(profiles: Sequence[dict[str, Any]]) -> str:
    sizes = set_sizes()
    lines = [f"Selection half only. Core = {sizes['core_forgotten']} questions the no-mitigation run at lr 1e-4 "
             f"forgets in at least 4 of 5 seeds; other known = {sizes['noncore_forgotten']} more the base model "
             "gets right.",
             "arm | stream | seeds | core forgotten | other known forgotten | unknown gained"]
    starred = False
    for a in profiles:
        if not a["seeds"]:
            lines.append(f"{a['arm']} | {a['stream']} | none | no item files here ({a['no_files']} runs)")
            continue
        star = "*" if a["defines_core"] else ""
        starred = starred or bool(star)
        lines.append(f"{a['arm']}{star} | {a['stream']} | {a['seeds']} | {a['core_forgotten']:.1%} | "
                     f"{a['noncore_forgotten']:.1%} | {a['gained']:.1%}")
    if starred:
        lines.append("* these seeds define the core, so this arm's core rate is inflated and its other-known "
                     "rate deflated by construction; do not compare other arms against it.")
    return "\n".join(lines)
