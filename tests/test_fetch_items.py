"""Per-item files for the item-level bootstrap (analysis/PREREG.md section 7): the committed
mirror under results/items/, `clh fetch-items` filling it from the Modal volume, and the
bootstrap switching to mode "items" when the files are there. No network in these tests."""

import subprocess
from argparse import Namespace

import numpy as np
import pytest

from clharness import cli
from clharness.analysis import attribution as A
from clharness.executor.rows import half_indices
from clharness.executor.run import items_path
from clharness.fake import fake_rows
from clharness.spec import read_rows, write_rows

TASKS = ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal")
N_ITEMS = 500


def _items_dir(tmp_path, name="run") -> str:
    """A checkpoint directory holding base and final items_<task>.npy for every task."""
    rng = np.random.default_rng(0)
    for ckpt, p_correct in (("ckpt_0000", 0.68), ("ckpt_0002", 0.62)):
        d = tmp_path / name / ckpt
        d.mkdir(parents=True)
        for t in TASKS:
            np.save(d / f"items_{t}.npy", (rng.random(N_ITEMS) < p_correct).astype(float))
    return str(tmp_path / name)


def test_checkpoint_items_reads_the_mirror_and_returns_the_right_half(tmp_path):
    row = fake_rows(seed=0)[0].model_copy(update={"items_path": _items_dir(tmp_path)})
    base, _final = A.checkpoint_items(row, "arc_easy", "report")
    sel_idx, rep_idx = half_indices("arc_easy", N_ITEMS)
    stored_base = np.load(tmp_path / "run" / "ckpt_0000" / "items_arc_easy.npy")
    assert len(base) == len(rep_idx) == 250
    assert not len(set(sel_idx) & set(rep_idx))
    np.testing.assert_array_equal(base, stored_base[rep_idx].astype(float))


def test_checkpoint_items_none_without_files(tmp_path):
    row = fake_rows(seed=0)[0].model_copy(update={"items_path": str(tmp_path / "missing")})
    assert A.checkpoint_items(row, "arc_easy", "selection") is None


def test_paired_bootstrap_uses_items_mode_when_the_files_exist(tmp_path):
    rows = fake_rows(seed=0)
    a, b = rows[0], next(r for r in rows if set(r.spec.technique_names()) == A.REFERENCE)
    a = a.model_copy(update={"items_path": _items_dir(tmp_path, "a")})
    b = b.model_copy(update={"items_path": _items_dir(tmp_path, "b")})
    x = A.Config(key="x", techniques=A.NO_MITIGATION, rows=[a])
    r = A.Config(key="r", techniques=A.REFERENCE, rows=[b])
    boot = A.paired_bootstrap(x, r, "selection", n_boot=200)
    assert boot["mode"] == "items" and boot["n_pairs"] == 1
    # a missing file on one side must fall back to the parametric mode, not crash
    (tmp_path / "b" / "ckpt_0002" / "items_piqa.npy").unlink()
    boot = A.paired_bootstrap(x, r, "selection", n_boot=200)
    assert boot["mode"] == "parametric"


def test_fetch_items_builds_the_modal_volume_get_command(tmp_path, monkeypatch):
    row = next(r for r in fake_rows(seed=0)
               if r.status == "finished" and set(r.spec.technique_names()) == A.REFERENCE)
    row = row.model_copy(update={"backend": "modal", "items_path": "/__modal/volumes/vo-x/"
                                                                 f"runs/{row.spec_hash[:16]}/checkpoints"})
    rows_file = tmp_path / "rows.jsonl"
    write_rows(rows_file, [row])
    calls = []

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        dest = tmp_path / "results" / "items" / row.spec_hash[:16]
        for ck in ("ckpt_0000", "ckpt_0002"):
            (dest / "checkpoints" / ck).mkdir(parents=True, exist_ok=True)
            np.save(dest / "checkpoints" / ck / "items_arc_easy.npy", np.ones(N_ITEMS))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = cli.cmd_fetch_items(Namespace(files=[str(rows_file)], volume="clh-work"))
    assert rc == 0
    assert calls == [["modal", "volume", "get", "--force", "clh-work",
                      f"runs/{row.spec_hash[:16]}/checkpoints",
                      str(tmp_path / "results" / "items" / row.spec_hash[:16])]]
    # the row's items_path is rewritten to the repo-relative mirror
    reread = read_rows(rows_file)[0]
    assert reread.items_path == items_path(row.spec)
    # a second run sees the files already present and calls nothing
    calls.clear()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("should not call modal"))
    assert cli.cmd_fetch_items(Namespace(files=[str(rows_file)], volume="clh-work")) == 0
