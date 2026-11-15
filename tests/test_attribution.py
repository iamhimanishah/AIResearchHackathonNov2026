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
    return A.analyze(rows, confirmatory=False, n_boot=500, allow_simulated=True, frozen=False)


@pytest.fixture(scope="module")
def confirmatory(rows):
    return A.analyze(rows, confirmatory=True, n_boot=2000, allow_simulated=True, frozen=False)


def test_retention_main_effects_and_planted_pairs_are_recovered(screening):
    checks = A.check_planted(screening, PLANTED)
    mains = [c for c in checks if c["fit"] == "effects_retention_selection" and "×" not in c["effect"]]
    assert {c["effect"] for c in mains} == set(PLANTED["retention_main_effects"])
    # the planted pairs are read through their alias chains in the factorial
    chains = screening["alias_chains_retention_selection"]["effects"]
    for pair in ("distill×replay", "lora×replay"):
        name = next(n for n, e in chains.items() if n == pair or pair in e["aliased_with"])
        check = next(c for c in checks if c["fit"] == "alias_chains_retention_selection" and c["effect"] == name)
        assert check["inside"], check


def test_main_fit_is_six_techniques_plus_the_reference_interaction(screening):
    names = set(screening["effects_retention_selection"]["effects"])
    # the reference interaction's alias partners get their own columns: the halves runs make
    # them separately estimable, so their effects cannot leak into replay x lr_rewarm
    chain = next(c for c in A.alias_chains() if ("lr_rewarm", "replay") in c)
    chain_names = {"×".join(sorted(p)) for p in chain}
    assert {"(baseline)", *PLANTED["retention_main_effects"], "lr_rewarm×replay"} <= names
    assert names <= {"(baseline)", *PLANTED["retention_main_effects"]} | chain_names
    assert len(screening["alias_chains_retention_selection"]["aliases"]) == 7


def test_selection_variants_and_off_design_rows_stay_out_of_the_fit(rows, screening):
    fitted = screening["effects_retention_selection"]["n_rows"]
    in_fit = set(A.factorial()) | set(A.HALVES)
    excluded = [r for r in rows if r.status == "finished"
                and (A.selection_of(r.spec) is not None or A.lr_of(r.spec) is not None
                     or not A.is_design_spec(r.spec)
                     or frozenset(r.spec.technique_names()) not in in_fit)]
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
    # sigma_noise is FROZEN (PREREG 0.4); the measured value is reported alongside for comparison
    assert screening["noise_floor"]["sigma_noise"] == pytest.approx(A.SIGMA_NOISE) == 0.0115
    assert screening["noise_floor"]["sigma_selection_measured"] == pytest.approx(
        A.noise_sd(base, "selection"))


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


def test_lr_controls_are_not_the_noise_floor_or_in_the_fit(rows, screening):
    configs = A.group(rows)
    assert A.find(configs, A.NO_MITIGATION).lr is None
    lr_rows = [r for r in rows if r.status == "finished" and A.lr_of(r.spec) is not None]
    assert len(lr_rows) == 6
    assert screening["effects_retention_selection"]["effects"]["(baseline)"]["estimate"] == pytest.approx(
        PLANTED["baseline_retention_mean_delta"], abs=0.01)


def test_lr_control_is_the_best_arm_that_passes_the_acquisition_gate(screening):
    """PREREG A13: 3e-5 learns too little (planted), so 5e-5 is the control."""
    lc = screening["lr_control"]
    gate = {a["lr"]: a["passes_acquisition_gate"] for a in lc["arms"]}
    assert gate == {5e-5: True, 3e-5: False}
    assert lc["chosen_label"] == "no-mitigation-lr5e-05"


def test_rule4_must_hold_to_win_and_is_vacuous_without_a_control(rows):
    configs = A.group(rows)
    ref = A.find(configs, A.REFERENCE)
    x = A.find(configs, frozenset({"distill", "freeze", "replay"}))
    lr = next(c for c in configs.values() if c.lr == 5e-5)
    strong = {"delta": 0.05, "lo": 0.02, "hi": 0.08, "p": 0.001, "mode": "parametric", "n_pairs": 3}
    weak = {"delta": 0.004, "lo": -0.01, "hi": 0.02, "p": 0.6, "mode": "parametric", "n_pairs": 3}
    d = A.decide(x, ref, strong, 0.003, 0.006, lr, weak, 0.6)
    assert d["rule1_significant"] and d["rule2_large_enough"] and d["rule3_learns_enough"]
    assert not d["rule4_beats_lr_control"] and not d["beats_reference"]
    d = A.decide(x, ref, strong, 0.003, 0.006, lr, strong, 0.003)
    assert d["rule4_beats_lr_control"] and d["beats_reference"]
    d = A.decide(x, ref, strong, 0.003, 0.006)  # no lower-LR arm passed the gate
    assert d["rule4_beats_lr_control"] and d["vs_lr_control"] is None


def test_confirmatory_reports_the_lr_comparison(confirmatory):
    assert all(d["vs_lr_control"]["config"] == "no-mitigation-lr5e-05" for d in confirmatory["confirmatory"])
    assert all(d["rule4_beats_lr_control"] for d in confirmatory["confirmatory_selection"])  # n/a there


def _with_acquisition(cfg, code, legal):
    """A copy of cfg whose rows have the given own-boundary ratios."""
    rows = []
    for r in cfg.rows:
        acq = r.acquisition.model_copy(update={"own_boundary_ppl_ratio": {"code": code, "legal": legal},
                                               "mean_own_boundary_ppl_ratio": math.sqrt(code * legal)})
        rows.append(r.model_copy(update={"acquisition": acq}))
    return A.Config(key=cfg.key, techniques=cfg.techniques, rows=rows, label=cfg.label)


def test_acquisition_gate_needs_both_domains(rows):
    """PREREG 5, rule 3: code and legal must each pass; better legal cannot hide worse code."""
    configs = A.group(rows)
    ref = _with_acquisition(A.find(configs, A.REFERENCE), code=0.90, legal=0.78)
    ok = _with_acquisition(A.find(configs, frozenset({"distill", "freeze", "replay"})), 0.93, 0.80)
    hides = _with_acquisition(A.find(configs, frozenset({"distill", "freeze", "replay"})), 0.99, 0.70)
    assert A.acquisition_gate(ok, ref)["passes"]
    g = A.acquisition_gate(hides, ref)
    assert math.sqrt(0.99 * 0.70) < 1.05 * math.sqrt(0.90 * 0.78)  # the average alone would pass
    assert not g["passes"] and not g["domains"]["code"]["passes"] and g["domains"]["legal"]["passes"]


def test_acquisition_margin_widens_to_twice_the_noise(rows):
    configs = A.group(rows)
    base = A.find(configs, A.NO_MITIGATION)
    assert all(m == pytest.approx(A.ACQ_GATE) for m in A.acquisition_margins(base).values())  # quiet fake noise
    noisy_rows = []
    for i, r in enumerate(base.rows):
        v = 0.78 * math.exp(0.08 * (i - 2))  # log ratio spread SD ~0.13 over 5 seeds
        acq = r.acquisition.model_copy(update={"own_boundary_ppl_ratio": {"code": 0.9, "legal": v}})
        noisy_rows.append(r.model_copy(update={"acquisition": acq}))
    noisy = A.Config(key=base.key, techniques=base.techniques, rows=noisy_rows, label=base.label)
    m = A.acquisition_margins(noisy)
    assert m["legal"] > A.ACQ_GATE and m["code"] == pytest.approx(A.ACQ_GATE)


def test_claims_table_covers_every_test_with_a_verdict(confirmatory):
    from clharness.analysis import design as D

    tests = [(c["id"], t[0], t[1], t[2]) for c in D.CLAIMS for t in c["tests"]]
    got = [(c["id"], c["arm"], c["comparator"], c["outcome"]) for c in confirmatory["claims"]]
    assert got == tests
    allowed = {"reproduced", "opposite", "not reproduced (interval includes 0)", "too few seeds", "reported"}
    assert all(c["verdict"] in allowed for c in confirmatory["claims"])
    assert set(confirmatory["claims_elsewhere"]) == {4, 8, 9}


def test_planted_lora_acquisition_claims_reproduce(confirmatory):
    """Planted: LoRA r8 learns legal less than no-mitigation (+0.09 log ratio) and r256 learns
    more than r8 (-0.04); both are far outside seed noise, so both must reproduce."""
    rows = {(c["arm"], c["comparator"], c["outcome"]): c for c in confirmatory["claims"]}
    assert rows[("lora-r8", "no-mitigation", "acq_legal")]["verdict"] == "reproduced"
    assert rows[("lora-r256", "lora-r8", "acq_legal")]["verdict"] == "reproduced"


def test_claim_arms_are_not_off_design_and_not_promoted(screening):
    assert screening["off_design_rows"] == ["fake-distill-10.0-s1"]
    promoted_pool = {d["config"] for d in screening["promotion"]}
    assert not promoted_pool & {"lora", "freeze"}  # lora-r8 / freeze-bottom alone are claim arms, not screen runs


def test_claims_only_in_confirmatory_mode(screening):
    assert "claims" not in screening


def test_duplicate_spec_hashes_keep_one_row(rows, screening):
    """Loading the same run twice (e.g. an old-split noise file next to the remeasured one)
    must not double a configuration's seeds."""
    doubled = rows + rows
    out = A.analyze(doubled, n_boot=200, allow_simulated=True, frozen=False)
    assert out["rows"]["duplicates_dropped"] == len(rows)
    assert out["rows"]["finished"] == screening["rows"]["finished"]
    # and it prefers the row measured the current way when both exist
    old = next(r for r in rows if r.status == "finished" and A.lr_of(r.spec) is None
               and r.spec.technique_names() == [])
    newer = old.model_copy(update={"spec_hash": old.spec_hash})
    a = newer.acquisition.model_copy()
    old_v = old.model_copy(update={"acquisition": a.model_copy(
        update={"own_boundary_ppl_ratio": None, "mean_own_boundary_ppl_ratio": None})})
    dedup, n = A.dedupe_rows([newer, old_v])  # current first, stale second: stale must lose
    assert n == 1 and dedup[0].acquisition.mean_own_boundary_ppl_ratio is not None
    dedup, _ = A.dedupe_rows([old_v, newer])  # order does not decide it
    assert dedup[0].acquisition.mean_own_boundary_ppl_ratio is not None


def test_screen_decisions_use_only_seed_0_rows(rows, screening):
    """Once replication rows exist, ranking the family on all seeds can pick configurations
    the screen never promoted — decisions re-derive from the seed-0 rows only (PREREG 6.3)."""
    screen = A.group([r for r in rows if r.status == "finished" and r.spec.seed == A.SCREEN_SEED])
    ref = A.find(screen, A.REFERENCE)
    # margins come from every no-mitigation seed (red-team RT-24), the ranking from seed 0
    margins = A.acquisition_margins(A.find(A.group(rows), A.NO_MITIGATION))
    expected = A.promote(screen, ref, margins=margins)
    assert [(d["config"], d["promoted"]) for d in screening["promotion"]] == [
        (d["config"], d["promoted"]) for d in expected]
    assert screening["acquisition_margins"] == margins
    # and it must actually differ from the all-seeds ranking on this data, or the test is empty
    all_configs = A.group([r for r in rows if r.status == "finished"])
    all_rank = A.promote(all_configs, A.find(all_configs, A.REFERENCE), margins=margins)
    assert [(d["config"], d["rank"]) for d in all_rank] != [(d["config"], d["rank"]) for d in expected]


def test_acquisition_gate_fails_when_a_domain_is_missing(rows):
    """A run that never measured a domain cannot pass the gate on it."""
    configs = A.group(rows)
    ref = A.find(configs, A.REFERENCE)
    x = A.find(configs, frozenset({"distill", "freeze", "replay"}))
    row = x.rows[0]
    acq = row.acquisition.model_copy(update={
        "own_boundary_ppl_ratio": {"code": 0.9}, "mean_own_boundary_ppl_ratio": 0.9})
    missing = A.Config(key=x.key, techniques=x.techniques,
                       rows=[row.model_copy(update={"acquisition": acq})], label=x.label)
    gate = A.acquisition_gate(missing, ref)
    assert not gate["passes"] and gate["domains"]["legal"]["passes"] is False
    full = A.acquisition_gate(x, ref)
    assert full["passes"]  # sanity: the complete run does pass


def test_rule4_fails_when_lower_lr_rows_are_missing(rows):
    """Trivially holding only applies when lower-LR arms ran but all fail the gate; absent rows
    are missing evidence, and rule 4 must not pass."""
    no_lr = [r for r in rows if A.lr_of(r.spec) is None]
    out = A.analyze(no_lr, confirmatory=True, n_boot=300, allow_simulated=True, frozen=False)
    assert out["lr_control"]["arms"] == [] and out["lr_control"]["chosen"] is None
    assert out["confirmatory"], "expected a promoted family on the fake rows"
    assert all(not d["rule4_beats_lr_control"] for d in out["confirmatory"])
    # present-but-gated-out still holds trivially: 3e-5 fails the gate on the fake rows, so
    # a world with ONLY 3e-5 rows yields arms-without-a-chosen and rule 4 holds
    only_bad = [r for r in rows if A.lr_of(r.spec) in (None, 3e-5)]
    out2 = A.analyze(only_bad, confirmatory=True, n_boot=300, allow_simulated=True, frozen=False)
    assert out2["lr_control"]["arms"] and out2["lr_control"]["chosen"] is None
    assert all(d["rule4_beats_lr_control"] for d in out2["confirmatory"])


def test_non_real_backend_rows_are_excluded(rows):
    n_finished = len([r for r in rows if r.status == "finished"])
    simulated = [r.model_copy(update={"backend": "local"}) for r in rows]
    out = A.analyze(simulated, n_boot=100)
    assert out["rows"]["finished"] == 0
    assert out["rows"]["excluded_backend"] == n_finished
    assert len(out["excluded_backend_rows"]) == n_finished
    out = A.analyze(simulated, n_boot=100, allow_simulated=True)
    assert out["rows"]["finished"] == n_finished


def test_sigma_is_pinned_at_the_frozen_value(screening, confirmatory):
    """2 sigma is the rule-2 size threshold; it must be the frozen 1.15 points, not whatever
    the rows at hand happen to measure."""
    assert A.SIGMA_NOISE == pytest.approx(0.0115)
    assert screening["noise_floor"]["sigma_noise"] == pytest.approx(0.0115)
    assert all(d["size_threshold"] == pytest.approx(2 * 0.0115) for d in confirmatory["confirmatory"])


def test_cost_ranks_by_extra_forward_tokens(screening):
    cost = screening["cost_selection"]
    assert "gain_per_extra_forward_token" in cost[0]
    keyed = [c for c in cost if c["gain_per_extra_forward_token"] is not None]
    assert [c["gain_per_extra_forward_token"] for c in keyed] == sorted(
        (c["gain_per_extra_forward_token"] for c in keyed), reverse=True)
    assert keyed[0]["config"] != "no-mitigation"


def test_acquisition_claim_uses_a_paired_t_interval(confirmatory):
    """3-seed bootstrap intervals reject ~25% of the time under the null; the paired t interval
    is the honest small-sample one (PREREG 6.5 amendment)."""
    row = next(c for c in confirmatory["claims"] if c["outcome"].startswith("acq")
               and math.isfinite(c.get("delta", float("nan"))))
    pairs = row["n_pairs"]
    width = row["hi"] - row["delta"]
    # the half-width must equal t(0.975, n-1) * se, i.e. be a t interval, not a percentile cut
    assert pairs >= 2 and width > 0
    assert row["lo"] < row["delta"] < row["hi"]


def test_planted_effects_recover_across_fake_seeds():
    """The recovery check must hold across seeds, not pass on one lucky seed."""
    ok = 0
    for s in range(8):
        rep = A.analyze(fake_rows(seed=s), n_boot=200, allow_simulated=True)
        checks = A.check_planted(rep, PLANTED)
        ok += sum(c["inside"] for c in checks) >= math.ceil(0.85 * len(checks))
    assert ok >= 6, "recovery should hold on most seeds, not 46% of them"


# ---------------------------------------------------------------------------------------------
# Red-team fixes (docs/reviews/red-team-57fa138.md)
# ---------------------------------------------------------------------------------------------


def test_main_effects_cover_the_planted_values_across_datasets():
    """Coverage over several fake datasets, not one: on a single dataset all six 95% intervals
    contain the planted value only about half the time (0.9^6)."""
    inside = []
    for s in range(10):
        finished = [r for r in fake_rows(seed=s) if r.status == "finished"]
        fit = A.fit_effects(A.fit_rows(finished), lambda r: r.retention.selection.mean_delta,
                            sigma=A.SIGMA_NOISE, interactions=A.main_interactions())
        for name, value in PLANTED["retention_main_effects"].items():
            e = fit["effects"][name]
            inside.append(e["lo"] <= value <= e["hi"])
    assert sum(inside) / len(inside) >= 0.8, sum(inside) / len(inside)


def test_rt01_gate_fails_closed_without_a_reference(rows):
    configs = A.group(rows)
    x = A.find(configs, frozenset({"distill", "freeze", "replay"}))
    assert not A.acquisition_gate(x, None)["passes"]
    no_ref0 = [r for r in rows if not (r.spec.seed == 0 and A.find(A.group([r]), A.REFERENCE))]
    out = A.analyze(no_ref0, n_boot=100, allow_simulated=True, frozen=False)
    assert out["screen_reference"] is None and out["warnings"]
    assert not any(d["promoted"] for d in out["promotion"])
    assert not any(d.get("promoted") for d in out["selection_screen"])


def test_rt01_replication_refuses_without_a_reference(rows):
    from clharness.analysis.replication import replication_specs

    no_ref0 = [r for r in rows if r.spec.seed == 0 and not A.find(A.group([r]), A.REFERENCE)]
    with pytest.raises(ValueError, match="reference"):
        replication_specs(no_ref0, allow_simulated=True)


def test_rt05_items_path_is_read_from_the_repo_root(tmp_path, monkeypatch):
    from clharness.config import REPO_ROOT
    from clharness.spec import ResultRow

    path = REPO_ROOT / "results" / "rows.reference.remeasured.jsonl"
    row = ResultRow.model_validate_json(path.read_text().splitlines()[0])
    assert row.items_path and not row.items_path.startswith("/")
    if not (REPO_ROOT / row.items_path).exists():
        pytest.skip("item files not in this checkout")
    monkeypatch.chdir(tmp_path)  # anywhere but the repo root
    got = A.checkpoint_items(row, "arc_easy", "selection")
    assert got is not None and len(got[0]) == 250


def test_rt05_markdown_prints_the_bootstrap_mode(confirmatory):
    md = A.to_markdown(confirmatory)
    assert "| Bootstrap |" in md and "| parametric |" in md


def test_rt08_a_local_copy_never_replaces_a_real_row(rows):
    real = next(r for r in rows if r.status == "finished").model_copy(update={"backend": "modal"})
    local = real.model_copy(update={"backend": "local"})
    for order in ([real, local], [local, real]):
        kept, n = A.dedupe_rows(order)
        assert n == 1 and kept[0].backend == "modal"


def test_rt09_rows_off_their_token_budget_are_invalid(rows, screening):
    r = next(r for r in rows if r.status == "finished" and A.lr_of(r.spec) is None
             and r.spec.technique_names() == [])
    off = r.model_copy(update={"cost": r.cost.model_copy(
        update={"gradient_tokens": int(r.spec.gradient_tokens() * 1.02)})})
    assert A.token_problem(r) is None and "over 1%" in A.token_problem(off)
    out = A.analyze([off if x is r else x for x in rows], n_boot=100, allow_simulated=True, frozen=False)
    assert out["rows"]["invalid_tokens"] == 1
    assert out["rows"]["finished"] == screening["rows"]["finished"] - 1
    assert any(d["name"] == r.spec.name and "gradient tokens" in d["reason"] for d in out["not_finished"])


def test_rt10_claim_arms_outside_the_design_stay_out_of_the_fit(rows, screening):
    names = {r.spec.name for r in A.fit_rows([r for r in rows if r.status == "finished"])}
    assert not any(n.startswith(("lora-r8", "freeze-bottom")) for n in names)
    assert screening["effects_retention_selection"]["n_rows"] == len(names)


def test_rt11_distill_lora_is_labelled_as_aliased_with_freeze_wavg(screening):
    eff = screening["effects_retention_selection"]["effects"]
    assert eff["distill×lora"]["aliased_with"] == ["freeze×wavg"]
    assert eff["lr_rewarm×replay"]["aliased_with"] == []
    assert "freeze×wavg" not in screening["effects_retention_selection"]["dropped"]


def test_rt24_frozen_margins_and_lr_control_are_the_default(rows):
    out = A.analyze(rows, confirmatory=True, n_boot=200, allow_simulated=True)
    assert out["frozen_decision_values"] and out["acquisition_margins"] == A.FROZEN_MARGINS
    lc = out["lr_control"]
    # the fake rows plant 3e-5 as learning too little, so the rule would pick 5e-5; frozen L wins
    assert lc["chosen_label"] == "no-mitigation-lr3e-05" and lc["derived_label"] == "no-mitigation-lr5e-05"
    assert all(d["vs_lr_control"]["config"] == "no-mitigation-lr3e-05" for d in out["confirmatory"])
    # frozen L absent: missing evidence, rule 4 fails
    no_l = [r for r in rows if A.lr_of(r.spec) != 3e-5]
    out2 = A.analyze(no_l, confirmatory=True, n_boot=200, allow_simulated=True)
    assert out2["lr_control"]["chosen"] is None
    assert all(not d["rule4_beats_lr_control"] for d in out2["confirmatory"])


def test_rt25_collapse_sensitivity_majority_rate_and_secondary_outcomes(rows, confirmatory):
    sens = confirmatory["collapse_sensitivity"]
    assert set(sens) == set(confirmatory["collapse_flags"])
    by_name = {r.spec.name: r for r in rows}
    for name, s in sens.items():
        h = by_name[name].retention.report
        kept = [p.delta for t, p in h.per_task.items() if t not in s["collapsed"]]
        assert s["report"]["without_collapsed"] == pytest.approx(sum(kept) / len(kept))
        assert s["selection"]["primary"] == by_name[name].retention.selection.mean_delta
    assert confirmatory["majority_rate"] == {"pubmedqa_pqal": 0.58}
    md = A.to_markdown(confirmatory)
    assert "Majority-class rate: pubmedqa_pqal 0.58" in md
    sec = confirmatory["configs_selection"][0]["secondary"]
    assert set(sec) == {"acquisition_end_of_stream", "general_ppl_ratio"}


def test_rt27_markdown_without_no_mitigation_rows(rows):
    no_base = [r for r in rows if not (r.spec.technique_names() == [] and A.lr_of(r.spec) is None)]
    out = A.analyze(no_base, confirmatory=True, n_boot=100, allow_simulated=True, frozen=False)
    assert "sanity_reference_vs_no_mitigation" not in out
    assert "not computed" in A.to_markdown(out)


def test_rt24_a_frozen_control_that_fails_the_gate_stays_but_warns(rows):
    """Frozen means frozen: on the fake rows 3e-5 is planted to fail the gate, so the frozen
    control is still the comparator and the report says it fails."""
    out = A.analyze(rows, n_boot=100, allow_simulated=True)
    assert out["lr_control"]["chosen_label"] == "no-mitigation-lr3e-05"
    assert any("fails the acquisition gate" in w for w in out["warnings"])
    derived = A.analyze(rows, n_boot=100, allow_simulated=True, frozen=False)
    assert not any("fails the acquisition gate" in w for w in derived["warnings"])


def test_markdown_renders_an_older_confirmatory_json(confirmatory):
    """A confirmatory JSON written before bootstrap_mode existed still renders."""
    old = dict(confirmatory)
    old["confirmatory"] = [{k: v for k, v in d.items() if k != "bootstrap_mode"} for d in old["confirmatory"]]
    assert "| ? |" in A.to_markdown(old)
