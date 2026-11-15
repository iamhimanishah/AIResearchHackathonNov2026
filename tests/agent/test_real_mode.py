"""Real mode of the loop: it reads real rows, launches nothing, and logs proposals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.agents import (
    ArmTechnique,
    ClaimDraft,
    ClaimPredictionDraft,
    LLMScout,
    ScoutDraft,
)
from clharness.agent.catalog_real import RANGES, bench_problems, is_explore, real_catalog
from clharness.agent.catalog_real import stream as bench_stream
from clharness.agent.llm import FakeLLM
from clharness.agent.loop import build_real, main
from clharness.agent.readonly import LaunchRefused, ReadOnlyExecutor
from clharness.agent.roles import FakeClock
from clharness.agent.search import FakeSearch, source_id
from clharness.agent.states import DONE, FREEZE, REPORT, SCOUT, Step, check_transition, st

REPO = Path(__file__).resolve().parents[2]
REAL_ROWS = [REPO / "results" / "rows.phase0.jsonl", REPO / "results" / "rows.reference.jsonl"]
Q = "q"
SRC = c.Source(id=source_id(Q, "https://arxiv.org/abs/2601.00001"), query=Q,
               url="https://arxiv.org/abs/2601.00001", title="Replay ratios", snippet="s",
               retrieved_at="2026-11-14T00:00:00Z")


def _rows(tmp_path) -> Path:
    """A copy of committed rows from real runs (the phase-0 controls and the reference arm)."""
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(p.read_text() for p in REAL_ROWS))
    return path


def _scout() -> LLMScout:
    claim = ClaimDraft(claim="Replay at 10% halves forgetting", source_ids=[SRC.id],
                       test_arm=[ArmTechnique(name="replay", params={"share": 0.1})],
                       prediction=ClaimPredictionDraft(direction=1, size_points=2.0))
    llm = FakeLLM().on("scout", lambda *a: ScoutDraft(claims=[claim]))
    return LLMScout(llm, FakeSearch({Q: [SRC]}), clock=FakeClock(), queries=((Q, "o"),),
                    runnable=bench_problems)


def _states(out) -> list[str]:
    return [d.state for d in c.DecisionLog(out / "decisions.jsonl").entries()]


def test_the_read_only_executor_refuses_every_launch(tmp_path):
    rows = _rows(tmp_path)
    executor = ReadOnlyExecutor(rows, minutes_per_run=20.0, catalog=real_catalog())
    spec = c.build_spec("x", 0, bench_stream(), {"replay": {"share": 0.1}})
    with pytest.raises(LaunchRefused, match="launches nothing"):
        executor.submit([spec])
    assert executor.poll() == [] and executor.healthy() and executor.estimate_cost(spec) > 0
    assert not ReadOnlyExecutor(tmp_path / "missing.jsonl", minutes_per_run=20.0).healthy()


def test_real_mode_reads_real_rows_logs_proposals_and_launches_nothing(tmp_path, monkeypatch):
    rows, out = _rows(tmp_path), tmp_path / "out"
    before = rows.read_bytes()
    monkeypatch.setattr(ReadOnlyExecutor, "submit", lambda self, specs: pytest.fail("submit called"))
    conductor = build_real(out, sim.default_prereg(), rows, compute_cap_usd=700.0,
                           clock=FakeClock(), scout=_scout())
    assert not conductor.launches and not conductor.claims_in_combine
    assert conductor.explore_share == 0.0
    state = conductor.run(stop_before=REPORT)
    assert state.state == REPORT and state.pending_spec_hashes == []
    assert rows.read_bytes() == before  # the rows file is read, never written
    assert not (out / "report.json").exists()  # the report half of real rows stays closed

    states = _states(out)
    assert states.index("replicate/advance") < states.index("scout") < states.index("combine/plan")
    launches = [d for d in c.DecisionLog(out / "decisions.jsonl").entries()
                if d.state.endswith("/launch")]
    assert launches and all(d.data["choice"]["submitted"] == 0 for d in launches)
    proposals = [json.loads(x) for x in (out / "proposed_specs.jsonl").read_text().splitlines()]
    named = {n for d in launches for n in d.data["proposed_specs"]}
    assert proposals and {p["name"] for p in proposals} == named
    assert len({p["spec_hash"] for p in proposals}) == len(proposals)
    have = {r.spec_hash for r in c.load_rows(rows).rows}
    assert not have & {p["spec_hash"] for p in proposals}
    assert all(c.spec_hash(c.RunSpec.model_validate(p["spec"])) == p["spec_hash"]
               for p in proposals)

    # the claim card takes the one path that is not this loop: the explore inbox, in the
    # fields and file name justin's reader takes
    from clharness.agent import explore

    assert conductor.explore_inbox.name == explore.INBOX_NAME
    cards, problems = explore.read_inbox(conductor.explore_inbox, [])
    assert problems == [] and len(cards) == 1
    assert cards[0][1].origin == "literature" and cards[0][1].techniques[0].name == "replay"
    assert not any(p["exploratory"] for p in proposals)
    [line] = [json.loads(x) for x in (out / "explore" / "literature.jsonl").read_text().splitlines()]
    assert line["techniques"] == [{"name": "replay", "params": {"share": 0.1}}]
    [scout] = [d for d in c.DecisionLog(out / "decisions.jsonl").entries() if d.state == "scout"]
    assert scout.data["choice"]["claims_tested_by"] == "explore_inbox"

    again = build_real(out, sim.default_prereg(), rows, compute_cap_usd=700.0,
                       clock=FakeClock(), scout=_scout())
    assert again.run(stop_before=REPORT).state == REPORT  # resumes where it stopped
    assert _states(out) == states


def test_real_mode_refuses_a_rows_file_of_exploratory_runs(tmp_path, capsys):
    explore = tmp_path / "rows.explore.jsonl"
    explore.write_text("")
    with pytest.raises(ValueError, match="exploratory"):
        build_real(tmp_path / "out", sim.default_prereg(), explore, compute_cap_usd=700.0)
    prereg = tmp_path / "t.yaml"
    prereg.write_text("agent_thresholds: {}\n")
    code = main(["--executor", "real", "--out", str(tmp_path / "out"), "--rows", str(explore),
                 "--prereg", str(prereg), "--compute-cap-usd", "80"])
    assert code == 2


def test_real_mode_cli_needs_thresholds_and_has_no_launch_flag(tmp_path, capsys):
    assert main(["--executor", "real", "--out", str(tmp_path / "out"),
                 "--compute-cap-usd", "80"]) == 2
    assert "needs --prereg" in capsys.readouterr().out
    for gone in (["--executor", "modal"], ["--executor", "real", "--go"], ["--executor", "real"]):
        with pytest.raises(SystemExit):
            main([*gone, "--out", str(tmp_path / "out")])


def test_a_run_named_explore_is_exploratory_and_freeze_never_picks_it(tmp_path):
    named = c.build_spec("explore-nomit-lr6e-5-s0", 0, bench_stream(), {})
    noted = c.build_spec("x", 0, bench_stream(), {}, notes="explore:claim-1")
    plain = c.build_spec("screen-replay-s0", 0, bench_stream(), {"replay": {"share": 0.05}})
    assert is_explore(named) and is_explore(noted) and not is_explore(plain)

    from clharness.agent.loop import build_sim

    clean = build_sim(tmp_path / "clean", 0, 700.0, clock=FakeClock())
    clean.run()
    rows = {r.spec_hash: r for r in c.load_rows(tmp_path / "clean" / "rows.jsonl").rows}
    winner = {c.config_hash(rows[h].spec) for h in clean.s.frozen_spec_hashes
              if rows[h].spec.techniques and not set(c.technique_names(rows[h].spec))
              <= {e.name for e in clean.catalog.controls()}}
    assert len(winner) == 1  # the arm FREEZE picks when nothing is exploratory

    out = tmp_path / "renamed"
    conductor = build_sim(out, 0, 700.0, clock=FakeClock())
    conductor.run(stop_before=FREEZE)
    assert conductor.s.state == FREEZE
    real_latest = conductor._latest

    def renamed():  # the same rows, with the winning arm's runs named explore-*
        return [r.model_copy(update={"spec": r.spec.model_copy(
                    update={"name": "explore-" + r.spec.name})})
                if c.config_hash(r.spec) in winner else r for r in real_latest()]

    conductor._latest = renamed
    assert any(is_explore(r.spec) for r in renamed())
    conductor.step()
    frozen = {c.config_hash(r.spec) for r in renamed()
              if r.spec_hash in conductor.s.frozen_spec_hashes}
    assert not winner & frozen


def test_ranges_is_the_one_table_from_explore():
    from clharness.agent import explore

    assert RANGES is explore.RANGES


def test_an_all_duplicate_replicate_gate_may_still_go_to_the_scout():
    check_transition(st(Step.GATE, c.Phase.REPLICATE), SCOUT)
    assert FREEZE != DONE


# ---- red-team review of 57fa138 (docs/reviews/red-team-57fa138.md) ----------------------------


def test_rt02_published_claims_arms_outside_the_screen_are_exploratory_by_config():
    from clharness.agent.catalog_real import design_exploratory_configs
    from clharness.analysis import design

    configs = design_exploratory_configs()
    key = lambda name: c.config_hash(design.claim_spec(name, 0))
    assert {key("replay-25"), key("lora-r256"), key("freeze-top")} <= configs
    assert key("replay-5") not in configs  # the replay-only half of the screen: confirmatory
    assert not configs & {c.config_hash(s) for s in design.screen()}
    assert not any(is_explore(design.claim_spec(n, 0)) for n in design.CLAIM_ARMS)  # no marker


def test_rt02_freeze_never_picks_an_arm_that_is_exploratory_by_config(tmp_path):
    from clharness.agent.loop import build_sim

    clean = build_sim(tmp_path / "clean", 0, 700.0, clock=FakeClock())
    clean.run()
    rows = {r.spec_hash: r for r in c.load_rows(tmp_path / "clean" / "rows.jsonl").rows}
    controls = {e.name for e in clean.catalog.controls()}
    winner = {c.config_hash(rows[h].spec) for h in clean.s.frozen_spec_hashes
              if not set(c.technique_names(rows[h].spec)) <= controls}
    assert len(winner) == 1

    again = build_sim(tmp_path / "again", 0, 700.0, clock=FakeClock())
    again.run(stop_before=FREEZE)
    again.exploratory_configs = frozenset(winner)  # the same arm, now exploratory by design
    again.step()
    rows = {r.spec_hash: r for r in c.load_rows(tmp_path / "again" / "rows.jsonl").rows}
    assert not winner & {c.config_hash(rows[h].spec) for h in again.s.frozen_spec_hashes}

    real = build_real(tmp_path / "real", sim.default_prereg(), _rows(tmp_path),
                      compute_cap_usd=80.0, clock=FakeClock())
    assert real.exploratory_configs  # real mode is given the design's exploratory arms


def test_rt03_the_cap_comes_from_the_config_unless_the_flag_is_given(tmp_path, monkeypatch):
    from clharness.agent import loop
    from clharness.agent.config import load_agent_config

    config = REPO / "configs" / "agent.yaml"
    cfg = load_agent_config(config)
    seen = {}

    def fake_build_real(out, prereg, rows, **kw):
        seen.update(kw)
        raise ValueError("stop here")

    monkeypatch.setattr(loop, "build_real", fake_build_real)
    monkeypatch.setattr(loop, "load_prereg", lambda path: sim.default_prereg())
    base = ["--executor", "real", "--out", str(tmp_path / "out"), "--prereg", "x.yaml"]
    assert main([*base, "--config", str(config)]) == 2
    assert seen["compute_cap_usd"] == cfg.compute_budget_usd != 700.0
    assert seen["reserve_fraction"] == cfg.compute_reserve_fraction
    assert main([*base, "--config", str(config), "--compute-cap-usd", "25"]) == 2
    assert seen["compute_cap_usd"] == 25.0


@pytest.mark.parametrize("cap", [10.0, 28.0, 40.0, 50.0, 61.0])
def test_rt04_a_loop_that_runs_out_of_budget_freezes_instead_of_halting(tmp_path, cap):
    from clharness.agent.loop import build_sim

    state = build_sim(tmp_path, 0, cap, clock=FakeClock(), infra_fail_rate=0.0).run()
    assert state.state == DONE and state.halted_reason is None
    assert (tmp_path / "report.json").exists()
    assert state.compute_usd_spent <= cap


def test_rt13_the_scout_cli_uses_the_real_catalog_and_prints_both_lists():
    from clharness import techniques
    from clharness.agent import scout as scout_cli

    request = scout_cli.real_request()
    assert list(request.catalog.names()) == sorted(techniques.available())
    assert request.stream == bench_stream() and request.design_arms
    result = _scout().scout(request)
    text = scout_cli.summary(result)
    assert "claim claim-" in text and "Replay at 10% halves forgetting" in text
    assert SRC.url in text
    from clharness.agent.roles import ScoutResult

    assert "skipped: nothing new" in scout_cli.summary(ScoutResult(skipped_reason="nothing new"))


def test_rt33_the_fallback_planner_replicates_the_top_three():
    from clharness.agent.roles import StubPlanner

    assert StubPlanner().top_k == 3
