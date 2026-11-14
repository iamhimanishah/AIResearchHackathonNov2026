"""The pre-registered screen (analysis/PREREG.md 6.2) must be a resolution IV design over the six
real techniques, contain both controls, and produce specs the real plugins accept."""

import itertools
from pathlib import Path

import numpy as np

from clharness.analysis import design as D
from clharness.spec import load_spec


def _matrix():
    return np.array([[1 if f in run else -1 for f in D.FACTORS] for run in D.factorial()])


def test_factorial_is_a_balanced_orthogonal_quarter_fraction_of_the_six_techniques():
    X = _matrix()
    assert X.shape == (16, 6) and len({tuple(r) for r in X}) == 16
    assert (X.sum(axis=0) == 0).all()  # balanced: every technique on in 8 of 16 runs
    for i, j in itertools.combinations(range(6), 2):
        assert (X[:, i] * X[:, j]).sum() == 0  # main effects orthogonal


def test_no_main_effect_is_aliased_with_a_pair():
    X = _matrix()
    for i in range(6):
        for j, k in itertools.combinations([c for c in range(6) if c != i], 2):
            assert abs((X[:, i] * X[:, j] * X[:, k]).sum()) < 16


def test_both_controls_are_runs_of_the_factorial_and_the_halves_are_not():
    runs = set(D.factorial())
    assert D.NO_MITIGATION in runs and D.REFERENCE in runs
    assert not any(h in runs for h in D.HALVES)


def test_alias_chains_cover_every_pair_once():
    chains = D.alias_chains()
    pairs = [p for c in chains for p in c]
    assert len(pairs) == 15 and len(set(pairs)) == 15
    assert len(chains) == 7
    assert any(("lr_rewarm", "replay") in c for c in chains)


def test_screen_specs_are_distinct_and_valid_for_the_real_plugins():
    from clharness.executor.run import config_from_spec
    from clharness.techniques import _validated

    specs = D.screen()
    assert len(specs) == 22 and len({s.spec_hash() for s in specs}) == 22
    for s in specs:
        _validated(s.techniques)  # raises on an unknown technique or bad params
        config_from_spec(s)  # config overrides of the techniques do not conflict
        assert D.is_design_spec(s)


def test_design_controls_are_the_same_experiments_as_the_committed_specs():
    for path, techs in [("no-mitigation-s0", D.NO_MITIGATION), ("reference-s0", D.REFERENCE),
                        ("replay-only-s0", frozenset({"replay"})), ("rewarm-only-s0", frozenset({"lr_rewarm"}))]:
        assert load_spec(f"configs/bench/specs/{path}.json").spec_hash() == D.spec(techs, 0).spec_hash(), path


def test_selection_is_a_parameter_of_replay():
    s = D.spec(D.REFERENCE, 0, ("act_nov", "low"))
    replay = next(t for t in s.techniques if t.name == "replay")
    assert replay.params["select"] == "act_nov" and replay.params["keep"] == "low"
    assert D.selection_of(s) == ("act_nov", "low") and D.selection_of(D.spec(D.REFERENCE, 0)) is None
    try:
        D.spec(frozenset({"lora"}), 0, ("loss", "high"))
    except ValueError:
        pass
    else:
        raise AssertionError("selection without replay must be refused")


def test_lr_controls_seed_0_are_the_a5_calibration_runs():
    for lr, path in [(5e-5, "calibration-code-legal-lr5e-5-s0"), (3e-5, "calibration-code-legal-lr3e-5-s0")]:
        committed = load_spec(f"configs/bench/specs/{path}.json")
        assert committed.spec_hash() == D.spec(D.NO_MITIGATION, 0, lr=lr).spec_hash(), path
        assert D.lr_of(committed) == lr and D.is_design_spec(committed)
    specs = D.lr_controls()
    assert len(specs) == 6 and len({s.spec_hash() for s in specs}) == 6
    try:
        D.spec(frozenset({"replay"}), 0, lr=5e-5)
    except ValueError:
        pass
    else:
        raise AssertionError("an LR override with techniques must be refused")


def test_phase0_list_is_exactly_the_preregistered_controls():
    """configs/bench/phase0.txt (what A launches) must equal design.phase0(); REUSE lines are the
    seed-0 calibration runs, which need re-measuring, not a GPU."""
    lines = [ln.strip() for ln in Path(D.PHASE0_LIST).read_text().splitlines() if ln.strip()]
    gpu = [ln for ln in lines if not ln.startswith("#")]
    reuse = [ln.removeprefix("# REUSE ").strip() for ln in lines if ln.startswith("# REUSE ")]
    assert {load_spec(p).spec_hash() for p in gpu} == {s.spec_hash() for s in D.phase0_new()}
    assert {load_spec(p).spec_hash() for p in gpu + reuse} == {s.spec_hash() for s in D.phase0()}
    assert sorted(reuse) == sorted(D.PHASE0_REUSE)
    assert len(gpu) == 7 and len(reuse) == 7
    # no technique other than the reference arm runs in phase 0
    assert all(frozenset(s.technique_names()) in (D.NO_MITIGATION, D.REFERENCE) for s in D.phase0())
