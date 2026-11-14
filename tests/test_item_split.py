"""The pre-registered selection/report split (analysis/PREREG.md, section 3) must stay exactly
as committed, cover every protected item once, and keep both halves equally hard."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "manifests" / "eval-protected-v1-split.json"


def _builder():
    spec = importlib.util.spec_from_file_location("make_item_split", REPO / "scripts" / "make_item_split.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_split_matches_a_fresh_build():
    mod = _builder()
    assert SPLIT.read_text() == json.dumps(mod.build(), indent=1, sort_keys=True) + "\n"


def test_executor_uses_the_pinned_split_for_protected_tasks():
    from clharness.executor.rows import half_indices

    split = json.loads(SPLIT.read_text())["tasks"]
    for task, t in split.items():
        sel, rep = half_indices(task, t["n_items"])
        assert sel.tolist() == t["selection_positions"], task
        assert rep.tolist() == t["report_positions"], task


def test_halves_are_disjoint_complete_and_balanced():
    split = json.loads(SPLIT.read_text())
    protected = json.loads((REPO / "manifests" / "eval-protected-v1.json").read_text())
    for task, t in split["tasks"].items():
        sel, rep = set(t["selection_positions"]), set(t["report_positions"])
        n = protected["tasks"][task]["n_items"]
        assert not sel & rep, task
        assert sel | rep == set(range(n)), task
        assert len(sel) == len(rep) == n // 2, task
        # stratified by base correctness: the halves differ by at most one item's worth
        assert abs(t["selection_base_acc"] - t["report_base_acc"]) <= 1.0 / (n // 2) + 1e-9, task
