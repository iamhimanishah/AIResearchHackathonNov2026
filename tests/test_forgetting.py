"""analysis/forgetting.py on planted per-item files: no GPU, no network."""

from pathlib import Path

import numpy as np
import pytest

from clharness import spec as a1
from clharness.analysis import forgetting as fg
from clharness.executor.rows import half_indices

TASKS = fg.TASKS
N = 500
STREAM = [a1.StreamDomain(name="code", tokens=10_000_000), a1.StreamDomain(name="legal", tokens=10_000_000)]


def _spec(seed: int, lr: float | None = None, techniques=()) -> a1.RunSpec:
    return a1.RunSpec(name=f"x-s{seed}", seed=seed, bench="fast", stream=STREAM,
                      optim={"lr": lr} if lr else {}, techniques=list(techniques))


def _row(spec: a1.RunSpec) -> a1.ResultRow:
    half = a1.HalfRetention(per_task={"arc_easy": a1.TaskRetention(base=0.7, final=0.6, delta=-0.1, n_items=250)},
                            mean_delta=-0.1)
    return a1.ResultRow(
        spec=spec, spec_hash=spec.spec_hash(), git_sha="a" * 40, backend="modal", status="finished",
        retention=a1.Retention(selection=half, report=half),
        acquisition=a1.Acquisition(ppl_ratio={"legal": 0.8}, mean_ppl_ratio=0.8),
        cost=a1.Cost(gradient_tokens=20_000_000, wall_clock_s=1000.0, gpu_type="H100", est_usd=1.0),
        items_path=f"results/items/{spec.spec_hash()[:16]}/checkpoints")


def _write(root: Path, spec: a1.RunSpec, p_lost: dict[str, np.ndarray], rng: np.random.Generator,
           mid_share: float = 0.5, report_lost: bool = False, margin: bool = False, drift: bool = False) -> None:
    """Three checkpoints. The base model knows every question but the first 20 of each task's
    selection half. A known question is lost with probability p_lost[task][position]; half of the
    losses already happen after code. A lost question moves to option (gold + 1) mod 4."""
    for task in TASKS:
        sel, rep = half_indices(task, N)
        base = np.ones(N, dtype=np.float32)
        base[sel[:20]] = 0.0
        lost = (rng.random(N) < p_lost[task]) & (base > 0.5)
        lost[rep] = report_lost  # whatever the report half holds must not matter
        early = lost & (rng.random(N) < mid_share)
        mid, final = base.copy(), base.copy()
        mid[early], final[lost] = 0.0, 0.0
        gold = np.zeros(N, dtype=np.int16)
        chosen = [np.where(base > 0.5, 0, 1).astype(np.int16)]
        chosen += [np.where(x > 0.5, 0, 1).astype(np.int16) for x in (mid, final)]
        for i, (vals, ch) in enumerate(zip((base, mid, final), chosen, strict=True)):
            d = root / "results" / "items" / spec.spec_hash()[:16] / "checkpoints" / f"ckpt_{i:04d}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"items_{task}.npy", vals)
            np.save(d / f"choices_{task}.npy", ch)
            np.save(d / f"gold_{task}.npy", gold)
            if margin and i == 0:  # lost questions start with a small margin, kept ones with a large one
                score = np.full((N, 4), -5.0, dtype=np.float32)
                score[:, 0] = np.where(lost, -4.5, -1.0)
                np.savez(d / f"loglik_{task}.npz", raw=score, score=score)
            if drift and i == 2:
                cos = np.where(lost, 0.80, 0.97).astype(np.float32)[:, None].repeat(2, 1)
                np.savez(d / f"drift_{task}.npz", layers=np.array([7, 13]), cos=cos, norm_ratio=np.ones_like(cos))


def _fragility(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A fixed logit fragility per question, the same for every arm."""
    return {t: rng.normal(-3.0, 1.5, N) for t in TASKS}


def _p(frag: dict[str, np.ndarray], dose: float, slope: float = 1.0) -> dict[str, np.ndarray]:
    return {t: 1 / (1 + np.exp(-(dose + slope * (f + 3.0) - 3.0))) for t, f in frag.items()}


@pytest.fixture
def planted(tmp_path):
    rng = np.random.default_rng(0)
    frag = _fragility(rng)
    rows = []
    arms = {"plain": (None, 1.0, 1.0), "low": (3e-5, -1.0, 1.0), "mid": (5e-5, -0.5, 1.0),
            "targeted": (2e-5, 0.0, 0.2)}
    for lr, dose, slope in arms.values():
        for seed in range(4):
            spec = _spec(seed, lr)
            _write(tmp_path, spec, _p(frag, dose, slope), rng, margin=True, drift=True)
            rows.append(_row(spec))
    return tmp_path, rows


def test_loads_only_the_selection_half_and_groups_by_arm(planted):
    root, rows = planted
    arms = fg.load_arms(rows, root)
    assert sorted(arms) == ["no-mitigation", "no-mitigation@lr2e-05", "no-mitigation@lr3e-05", "no-mitigation@lr5e-05"]
    run = arms["no-mitigation"][0]
    assert run.correct.shape == (3, sum(len(half_indices(t, N)[0]) for t in TASKS))
    assert run.n_options[:3].tolist() == [4, 4, 4]  # arc_easy from the manifest's chance level


def test_report_half_never_changes_the_result(tmp_path):
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    frag = _fragility(np.random.default_rng(2))
    rows = []
    for seed in range(3):
        spec = _spec(seed)
        _write(tmp_path / "a", spec, _p(frag, 0.0), rng_a, report_lost=False)
        _write(tmp_path / "b", spec, _p(frag, 0.0), rng_b, report_lost=True)
        rows.append(_row(spec))
    a = fg.analyze(rows, tmp_path / "a")
    b = fg.analyze(rows, tmp_path / "b")
    assert a == b


def test_repeatable_losses_score_high_and_random_ones_near_half(tmp_path):
    rng = np.random.default_rng(3)
    frag = _fragility(rng)
    flat = {t: np.full(N, 0.1) for t in TASKS}
    rows_rep, rows_rand = [], []
    for seed in range(4):
        s1, s2 = _spec(seed), _spec(seed, lr=3e-5)
        _write(tmp_path, s1, _p(frag, 0.0), rng)
        _write(tmp_path, s2, flat, rng)
        rows_rep.append(_row(s1))
        rows_rand.append(_row(s2))
    rep = fg.repeatability(fg.load_arms(rows_rep, tmp_path)["no-mitigation"])
    rand = fg.repeatability(fg.load_arms(rows_rand, tmp_path)["no-mitigation@lr3e-05"])
    assert rep["auc"] > 0.65
    assert abs(rand["auc"] - 0.5) < 0.06
    assert rep["lost_by_all"] > 10 * max(rep["lost_by_all_if_random"], 0.1)
    assert fg.repeatability(fg.load_arms(rows_rep[:2], tmp_path)["no-mitigation"]) is None


def test_dose_or_target_flags_the_arm_that_spares_fragile_questions(planted):
    root, rows = planted
    res = {d["arm"]: d for d in fg.dose_and_target(fg.load_arms(rows, root))}
    assert res["no-mitigation@lr2e-05"]["reading"] == "targeted"
    assert res["no-mitigation@lr2e-05"]["relative_slope"] < 0.5
    for arm in ("no-mitigation", "no-mitigation@lr3e-05", "no-mitigation@lr5e-05"):
        assert res[arm]["relative_slope"] > 0.6, arm


def test_timing_drift_margins_and_state(planted):
    root, rows = planted
    runs = fg.load_arms(rows, root)["no-mitigation"]
    t = fg.timing(runs)
    assert t["lost_by_first_domain"] == pytest.approx(0.5, abs=0.06)
    assert t["recovered_after_first_domain"] == 0.0
    d = fg.answer_drift(runs)
    assert d["same_wrong_option"] == 1.0 and d["chance"] < 0.6
    m = fg.margins(runs)
    assert m["lost"] == pytest.approx(0.5) and m["kept"] == pytest.approx(4.0)
    assert m["auc_lower_predicts_loss"] == 1.0
    sd = fg.state_drift(runs)
    assert sorted(sd) == [7, 13] and sd[7]["lost"] == pytest.approx(0.80) and sd[7]["kept"] == pytest.approx(0.97)


def test_report_and_agent_summary_render(planted):
    root, rows = planted
    res = fg.analyze(rows, root)
    text = fg.format_report(res)
    assert "Repeatability" in text and "Dose or target" in text and "State drift" in text
    summary = fg.agent_summary(res)
    assert summary.startswith("Forgetting structure (selection half")
    assert "targeted" in summary


def test_auc_handles_ties():
    assert fg._auc(np.array([1.0, 1.0, 0.0, 0.0]), np.array([True, False, True, False])) == 0.5
    assert fg._auc(np.array([2.0, 1.0]), np.array([True, False])) == 1.0
