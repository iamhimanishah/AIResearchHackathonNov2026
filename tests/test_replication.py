"""The replication set follows the pre-registered rule and agrees with the analysis report."""

from clharness.analysis import attribution as A
from clharness.analysis import design as D
from clharness.analysis.replication import replication_specs
from clharness.fake import fake_rows

ROWS = fake_rows(seed=0)
SCREEN = [r for r in ROWS if r.spec.seed == 0]


def test_it_returns_three_factorial_configurations_and_one_direction_per_signal_at_seeds_1_and_2():
    plan = replication_specs(SCREEN, allow_simulated=True)
    assert len(plan["promoted"]) == 3 and len(plan["promoted_selection"]) == 2
    assert len(plan["specs"]) == plan["planned"] == 10
    assert sorted({s.seed for s in plan["specs"]}) == [1, 2]
    controls = {D.NO_MITIGATION, D.REFERENCE, *D.HALVES}
    for s in plan["specs"]:
        techs = frozenset(s.technique_names())
        assert D.selection_of(s) is not None or techs not in controls  # no control is replicated
        assert D.lr_of(s) is None


def test_each_spec_is_the_design_spec_at_that_seed():
    for s in replication_specs(SCREEN, allow_simulated=True)["specs"]:
        expected = D.spec(frozenset(s.technique_names()), s.seed, D.selection_of(s))
        assert s.spec_hash() == expected.spec_hash(), s.name
        assert s.name == expected.name


def test_the_set_is_the_one_the_analysis_report_calls_promoted():
    plan = replication_specs(SCREEN, allow_simulated=True)
    report = A.analyze(SCREEN, n_boot=100, allow_simulated=True)
    assert plan["promoted"] == [d["config"] for d in report["promotion"] if d["promoted"]]
    assert plan["ranking"] == report["promotion"]


def test_the_ranking_does_not_move_when_replication_rows_arrive():
    """Rank on the seed-0 screen only: a configuration that regresses at seed 1 must stay in."""
    before = replication_specs(SCREEN, allow_simulated=True)
    after = replication_specs(ROWS, allow_simulated=True)  # includes seeds 1 and 2
    assert after["promoted"] == before["promoted"]
    assert after["promoted_selection"] == before["promoted_selection"]
    assert after["already_finished"] > 0
    assert {s.spec_hash() for s in after["specs"]} <= {s.spec_hash() for s in before["specs"]}


def test_duplicate_rows_do_not_change_it_and_simulated_rows_are_refused_by_default():
    assert replication_specs(SCREEN + SCREEN, allow_simulated=True)["promoted"] == \
        replication_specs(SCREEN, allow_simulated=True)["promoted"]
    assert replication_specs(SCREEN)["specs"] == []  # fake rows are not real results


def test_design_exposes_it():
    assert [s.name for s in D.replication_specs(SCREEN, allow_simulated=True)["specs"]] == \
        [s.name for s in replication_specs(SCREEN, allow_simulated=True)["specs"]]
