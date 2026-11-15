"""C5 analysis: gap closed and forgetting on hand-computed matrices, the unstable-
denominator warning, and the item-resampling bootstrap."""

import numpy as np

from clharness.tinker.analysis import analyze, final_avg_acc, forgetting, r_heatmap_svg
from clharness.tinker.rows import TinkerRow, Tokens, append_tinker_row, write_tinker_rows


def _row(arm, seed, r, est=1.0, items_path="x.npy", smoke=False):
    return TinkerRow(arm=arm, seed=seed, model="Qwen/Qwen3-8B", status="finished",
                     r_matrix=r, items_path=items_path, tokens=Tokens(train=10),
                     est_usd=est, smoke=smoke)


NAIVE_R = [[0.90, None, None], [0.80, 0.90, None], [0.70, 0.85, 0.90]]
REPLAY_R = [[0.90, None, None], [0.85, 0.90, None], [0.80, 0.87, 0.90]]
JOINT_R = [[0.92, 0.93, 0.95]]


def test_point_metrics():
    naive = _row("naive", 0, NAIVE_R)
    joint = _row("joint", 0, JOINT_R)
    assert abs(final_avg_acc(naive) - (0.70 + 0.85 + 0.90) / 3) < 1e-9
    assert abs(final_avg_acc(joint) - (0.92 + 0.93 + 0.95) / 3) < 1e-9
    # forgetting: best-minus-final per task = (0.9-0.7, 0.9-0.85, 0.9-0.9)
    assert abs(forgetting(naive) - (0.20 + 0.05 + 0.0) / 3) < 1e-9
    assert forgetting(joint) == 0.0  # one stage: nothing to forget


def _write_planted(tmp_path, joint_r=JOINT_R):
    """Planted rows + per-item files: each cell's correctness is `round(acc*100)` ones."""
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    rows = []
    for arm, r in [("naive", NAIVE_R), ("replay-200", REPLAY_R), ("joint", joint_r)]:
        for s in (0, 1):
            item = np.zeros((len(r), 3, 100), dtype=np.float32)
            for st, rowvals in enumerate(r):
                for t, v in enumerate(rowvals):
                    if v is None:
                        item[st, t, :] = np.nan
                    else:
                        item[st, t, : round(v * 100)] = 1.0
            p = items_dir / f"{arm}-s{s}.npy"
            np.save(p, item)
            rows.append(_row(arm, s, r, items_path=str(p)))
    rows_path = tmp_path / "rows.jsonl"
    write_tinker_rows(rows_path, rows)
    return rows_path


def test_gap_closed_and_forgetting_with_item_bootstrap(tmp_path):
    rows_path = _write_planted(tmp_path)
    out = analyze(rows_path=rows_path, out_dir=tmp_path / "out", n_boot=2000, seed=0)
    naive_m = sum(final_avg_acc(_row("naive", s, NAIVE_R)) for s in (0, 1)) / 2
    joint_m = sum(final_avg_acc(_row("joint", s, JOINT_R)) for s in (0, 1)) / 2
    replay_m = sum(final_avg_acc(_row("replay-200", s, REPLAY_R)) for s in (0, 1)) / 2
    expected_gap = (replay_m - naive_m) / (joint_m - naive_m)
    gc = out["arms"]["replay-200"]["gap_closed"]
    assert abs(gc["mean"] - expected_gap) < 0.05
    assert gc["lo"] <= gc["mean"] <= gc["hi"]
    assert "naive" in out["arms"] and "joint" in out["arms"]
    assert "gap_closed" not in out["arms"]["naive"]
    fg = out["arms"]["naive"]["forgetting"]
    assert abs(fg["mean"] - (0.20 + 0.05 + 0.0) / 3) < 1e-9
    assert "unstable denominator" not in str(out["warnings"])
    assert (tmp_path / "out" / "analysis.json").exists()
    assert (tmp_path / "out" / "analysis.md").exists()


def test_unstable_denominator_warning(tmp_path):
    # joint barely above naive: denominator < 0.03 triggers the PREREG-10 warning
    weak_joint = [[0.80, 0.83, 0.85]]  # mean 0.8267 vs naive 0.8167 -> 0.0100
    rows_path = _write_planted(tmp_path, joint_r=weak_joint)
    out = analyze(rows_path=rows_path, out_dir=tmp_path / "out", n_boot=200, seed=0)
    assert out["denominator"]["unstable"] is True
    assert any("unstable denominator" in w for w in out["warnings"])


def test_seed_only_fallback_without_item_files(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    rows = [_row("naive", s, NAIVE_R, items_path="does/not/exist.npy") for s in (0, 1)]
    rows += [_row("replay-200", s, REPLAY_R, items_path="does/not/exist.npy") for s in (0, 1)]
    rows += [_row("joint", s, JOINT_R, items_path="does/not/exist.npy") for s in (0, 1)]
    write_tinker_rows(rows_path, rows)
    out = analyze(rows_path=rows_path, out_dir=tmp_path / "out", n_boot=200, seed=0)
    gc = out["arms"]["replay-200"]["gap_closed"]
    assert abs(gc["mean"] - ((0.8567 - 0.8167) / (0.9333 - 0.8167))) < 0.06


def test_a_later_smoke_row_does_not_leak_into_the_metrics(tmp_path):
    """The review reproduction: `clh tinker-run --smoke` appends to the same rows.jsonl,
    so a smoke row written after the real naive s0 row used to replace it in _by_arm
    (0.50 instead of 0.80) and its 50-item array could break the bootstrap."""
    rows_path = _write_planted(tmp_path)
    base = analyze(rows_path=rows_path, out_dir=tmp_path / "a", n_boot=500, seed=0)
    smoke_items = np.zeros((3, 3, 50), dtype=np.float32)  # the smaller smoke shape
    smoke_items[:, :, :25] = 1.0
    items_path = tmp_path / "items" / "naive-s0-smoke.npy"
    np.save(items_path, smoke_items)
    append_tinker_row(rows_path, _row("naive", 0, [[0.50, 0.50, 0.50]], est=0.01,
                                      items_path=str(items_path), smoke=True))
    out = analyze(rows_path=rows_path, out_dir=tmp_path / "b", n_boot=500, seed=0)
    for arm in ("naive", "replay-200", "joint"):
        b, o = base["arms"][arm], out["arms"][arm]
        assert o["final_avg_acc"]["mean"] == b["final_avg_acc"]["mean"]
        assert o["forgetting"]["mean"] == b["forgetting"]["mean"]
    assert (out["arms"]["replay-200"]["gap_closed"]["mean"]
            == base["arms"]["replay-200"]["gap_closed"]["mean"])
    assert out["smoke_rows_skipped"] == 1


def test_heatmap_svg_renders(tmp_path):
    rows_path = _write_planted(tmp_path)
    svg = r_heatmap_svg(rows_path)
    assert svg.startswith("<svg") and "naive" in svg and "joint" in svg
