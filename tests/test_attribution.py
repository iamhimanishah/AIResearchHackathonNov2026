"""C2 analysis on fake rows: it must recover the planted effects, keep the report half closed
unless asked, and apply the PREREG decision rule exactly."""

import math

import pytest

from clharness.analysis import attribution as A
from clharness.fake import PLANTED, fake_rows


@pytest.fixture(scope="module")
def rows():
    return fake_rows(seed=0)


@pytest.fixture(scope="module")
def screening(rows):
    return A.analyze(rows, confirmatory=False, n_boot=500)


@pytest.fixture(scope="module")
def confirmatory(rows):
    return A.analyze(rows, confirmatory=True, n_boot=2000)


def test_retention_main_effects_and_planted_pairs_are_recovered(screening):
    checks = A.check_planted(screening, PLANTED)
    mains = [c for c in checks if c["fit"] == "effects_retention_selection" and "×" not in c["effect"]]
    assert {c["effect"] for c in mains} == set(PLANTED["retention_main_effects"])
    assert all(c["inside"] for c in mains), [c for c in mains if not c["inside"]]
    # the planted pairs are read through their alias chains in the factorial
    chains = screening["alias_chains_retention_selection"]["effects"]
    for pair in ("distill×replay", "lora×replay"):
        name = next(n for n, e in chains.items() if n == pair or pair in e["aliased_with"])
        check = next(c for c in checks if c["fit"] == "alias_chains_retention_selection" and c["effect"] == name)
        assert check["inside"], check


def test_main_fit_is_six_techniques_plus_the_reference_interaction(screening):
    names = set(screening["effects_retention_selection"]["effects"])
    assert names == {"(baseline)", *PLANTED["retention_main_effects"], "lr_rewarm×replay"}
    assert len(screening["alias_chains_retention_selection"]["aliases"]) == 7


def test_selection_variants_and_off_design_rows_stay_out_of_the_fit(rows, screening):
    fitted = screening["effects_retention_selection"]["n_rows"]
    excluded = [r for r in rows if r.status == "finished"
                and (A.selection_of(r.spec) is not None or not A.is_design_spec(r.spec))]
    assert fitted == screening["rows"]["finished"] - len(excluded)
    assert screening["off_design_rows"] == ["fake-distill-10.0-s1"]


def test_most_planted_effects_overall_are_recovered(screening):
    checks = A.check_planted(screening, PLANTED)
    # 95% intervals: about 1 in 20 misses by chance; require at least 85% inside
    assert sum(c["inside"] for c in checks) >= math.ceil(0.85 * len(checks))


def test_screening_never_reads_the_report_half(screening):
    assert screening["halves_read"] == ["selection"]
    assert "confirmatory" not in screening and "sigma_report" not in screening["noise_floor"]
    assert all("report" not in k for k in screening)


def test_failed_and_invalid_rows_are_excluded_and_listed(rows, screening):
    assert screening["rows"]["failed"] == 2 and screening["rows"]["invalid"] == 1
    used = sum(len(c.rows) for c in A.group(rows).values())
    assert used == screening["rows"]["finished"]
    assert {d["status"] for d in screening["not_finished"]} == {"failed", "invalid"}


def test_answer_collapse_is_flagged_but_kept(rows, screening):
    """PREREG A9 (decided): flag a task above 0.90 answer share, keep the run in the analysis."""
    collapsed = [r for r in rows if max(r.answer_share.values(), default=0) > A.COLLAPSE_SHARE]
    assert len(collapsed) == 2 and all(r.status == "finished" for r in collapsed)
    assert set(screening["collapse_flags"]) == {r.spec.name for r in collapsed}
    assert all(flags == ["pubmedqa_pqal"] for flags in screening["collapse_flags"].values())
    used = {r.spec.name for c in A.group(rows).values() for r in c.rows}
    assert {r.spec.name for r in collapsed} <= used


def test_noise_floor_is_the_sd_over_no_mitigation_seeds(rows, screening):
    base = A.find(A.group(rows), A.NO_MITIGATION)
    assert len(base.rows) == 5
    assert screening["noise_floor"]["sigma_selection"] == pytest.approx(A.noise_sd(base, "selection"))


def test_promotion_respects_the_acquisition_gate(screening):
    promoted = [d for d in screening["promotion"] if d["promoted"]]
    assert len(promoted) == 3
    assert all(d["passes_acquisition_gate"] for d in promoted)
    labels = {d["config"] for d in screening["promotion"]}
    assert "no-mitigation" not in labels and "lr_rewarm+replay" not in labels


def test_selection_variants_get_readable_labels(rows):
    labels = {c.label for c in A.group(rows).values()}
    assert {f"lr_rewarm+replay+select-{s}-{k}" for s in ("loss", "act_nov") for k in ("high", "low")} <= labels
    assert "lr_rewarm+replay" in labels  # plain replay stays the reference arm


def test_selection_family_promotes_one_keep_direction_per_signal(screening):
    promoted = [d for d in screening["selection_screen"] if d["promoted"]]
    assert sorted(d["signal"] for d in promoted) == ["act_nov", "loss"]
    assert all(d["passes_acquisition_gate"] for d in promoted)


def test_promotion_excludes_the_reference_halves_and_selection_variants(screening):
    labels = {d["config"] for d in screening["promotion"]}
    assert not labels & {"replay", "lr_rewarm", "no-mitigation", "lr_rewarm+replay"}
    assert not any("select-" in label for label in labels)


def test_confirmatory_selection_family_is_tested_against_the_reference(confirmatory):
    fam = confirmatory["confirmatory_selection"]
    assert len(fam) == 2 and all("select-" in d["config"] for d in fam)
    assert all(d["p_holm"] >= d["p"] for d in fam)


def test_planted_selection_effects_are_recovered(rows):
    checks = A.check_planted_selection(rows, PLANTED, n_boot=1000)
    assert checks and all(c["inside"] for c in checks), checks


def test_confirmatory_needs_three_paired_seeds(confirmatory):
    for d in confirmatory["confirmatory"]:
        if d["n_pairs"] < A.MIN_PAIRS:
            assert not d["enough_seeds"] and not d["beats_reference"]


def test_reference_beats_no_mitigation_in_the_sanity_check(confirmatory):
    s = confirmatory["sanity_reference_vs_no_mitigation"]
    # planted: replay 0.025 + lr_rewarm 0.008, and no planted interaction between them
    # (only distill+replay and lora+replay interact), so the reference gains 0.033
    assert s["lo"] > 0 and s["delta"] == pytest.approx(0.033, abs=0.015)


def test_paired_bootstrap_identical_arms_include_zero(rows):
    cfg = A.find(A.group(rows), A.REFERENCE)
    b = A.paired_bootstrap(cfg, cfg, "selection", n_boot=1000)
    assert b["delta"] == 0 and b["lo"] <= 0 <= b["hi"]


def test_holm_matches_hand_computation():
    adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)  # 3 x 0.01
    assert adj["c"] == pytest.approx(0.06)  # 2 x 0.03
    assert adj["b"] == pytest.approx(0.06)  # max(0.06, 1 x 0.04): step-down stays monotone


def test_decision_rule_needs_all_three_parts(rows):
    configs = A.group(rows)
    ref = A.find(configs, A.REFERENCE)
    x = A.find(configs, frozenset({"lora", "replay", "wavg"}))  # a LoRA run: learns less
    boot = {"delta": 0.05, "lo": 0.02, "hi": 0.08, "p": 0.001, "mode": "parametric", "n_pairs": 3}
    d = A.decide(x, ref, boot, p_adj=0.003, sigma=0.006)
    # with LoRA it learns less than 1.05 x the reference arm allows, so rule 3 fails it
    assert d["rule1_significant"] and d["rule2_large_enough"] and not d["rule3_learns_enough"]
    assert not d["beats_reference"]
    d = A.decide(ref, ref, boot, p_adj=0.003, sigma=0.03)  # 0.05 < 2 x 0.03: too small
    assert not d["rule2_large_enough"] and not d["beats_reference"]


def test_acquisition_uses_own_boundary_and_falls_back_for_old_rows(rows, screening):
    """PREREG 4: the gate reads the own-boundary ratio; rows without it (written before the
    field existed, like the first calibration runs) fall back to the final ratio."""
    r = next(r for r in rows if r.status == "finished")
    assert A.acquisition_of(r) == r.acquisition.mean_own_boundary_ppl_ratio
    old = r.model_copy(update={"acquisition": r.acquisition.model_copy(
        update={"own_boundary_ppl_ratio": {}, "mean_own_boundary_ppl_ratio": None})})
    assert A.acquisition_of(old) == r.acquisition.mean_ppl_ratio
    assert screening["acquisition_measure"] == {"own_boundary": screening["rows"]["finished"], "final_fallback": 0}


def test_markdown_renders(confirmatory):
    md = A.to_markdown(confirmatory)
    assert "Confirmatory tests" in md and "Failed and invalid runs" in md
