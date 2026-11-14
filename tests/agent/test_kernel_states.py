"""Kernel state machine: the transition table cannot trap the loop, and state.json round-trips."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clharness.agent import states as s
from clharness.agent.contracts import FREEZE_STATE_ID, Phase

NOW = "2026-11-14T20:00:00+00:00"


def test_every_state_of_the_architecture_exists():
    ids = {state.id for state in s.all_states()}
    cycle = {f"{p.value}/{step.value}" for p in Phase for step in s.CYCLE_STEPS}
    assert ids == cycle | {"init", "preflight", "scout", "freeze", "report", "done", "halted"}
    assert len(ids) == 28
    assert set(s.ACTORS) == set(s.Step)
    assert s.FREEZE.id == FREEZE_STATE_ID
    for state in s.all_states():
        assert s.State.from_id(state.id) == state


def test_every_non_terminal_state_has_an_exit():
    assert s.prove_exits() == []
    for state in s.all_states():
        assert bool(s.allowed_next(state)) != s.is_terminal(state)


def test_prove_exits_goes_red_on_a_broken_table():
    plan = s.st(s.Step.PLAN, Phase.SCREEN)
    wait = s.st(s.Step.WAIT, Phase.REPLICATE)
    stranded = dict(s.TRANSITIONS)
    stranded[wait] = (s.Transition(to=s.HALTED, when="only way out"),)
    assert f"{wait.id}: no exit other than HALTED" in s.prove_exits(stranded)
    looped = dict(s.TRANSITIONS)
    looped[s.FREEZE] = (s.Transition(to=plan, when="loop"), s.Transition(to=s.HALTED, when="x"))
    assert any("DONE is not reachable" in p for p in s.prove_exits(looped))
    orphaned = dict(s.TRANSITIONS)
    orphaned[s.PREFLIGHT] = (s.Transition(to=s.FREEZE, when="skip"),
                             s.Transition(to=s.HALTED, when="x"))
    assert f"{plan.id}: not reachable from INIT" in s.prove_exits(orphaned)
    no_halt = dict(s.TRANSITIONS)
    no_halt[s.INIT] = (s.Transition(to=s.PREFLIGHT, when="ok"),)
    assert "init: cannot halt" in s.prove_exits(no_halt)


def test_the_documented_paths_are_in_the_table():
    def nxt(step, phase=None):
        return {t.id for t in s.allowed_next(s.st(step, phase))}

    assert nxt(s.Step.GATE, Phase.SCREEN) >= {"screen/launch", "screen/plan",
                                              "screen/fallback_plan"}
    assert nxt(s.Step.PLAN, Phase.SCREEN) >= {"screen/gate", "screen/fallback_plan"}
    assert "screen/plan" not in nxt(s.Step.FALLBACK_PLAN, Phase.SCREEN)
    assert nxt(s.Step.ADVANCE, Phase.SCREEN) >= {"replicate/plan", "screen/plan"}
    assert nxt(s.Step.ADVANCE, Phase.REPLICATE) >= {"scout", "combine/plan"}
    assert "freeze" in nxt(s.Step.ADVANCE, Phase.COMBINE)
    assert "combine/plan" in nxt(s.Step.SCOUT)
    assert nxt(s.Step.FREEZE) == {"report", "halted"}
    assert nxt(s.Step.REPORT) == {"done", "halted"}
    for phase in Phase:
        for step in s.CYCLE_STEPS:
            assert "freeze" in nxt(step, phase), "budget or deadline must reach FREEZE"
    assert "report" not in nxt(s.Step.ADVANCE, Phase.COMBINE), "no report before the freeze"


def test_state_needs_a_phase_exactly_when_it_is_in_the_cycle():
    with pytest.raises(ValidationError):
        s.State(step=s.Step.PLAN)
    with pytest.raises(ValidationError):
        s.State(step=s.Step.FREEZE, phase=Phase.SCREEN)
    assert s.next_phase(Phase.SCREEN) is Phase.REPLICATE and s.next_phase(Phase.COMBINE) is None


def test_loop_state_moves_only_along_the_table_and_resumes(tmp_path):
    loop = s.LoopState()
    with pytest.raises(s.IllegalTransition):
        loop.move(s.FREEZE, NOW)
    loop.move(s.PREFLIGHT, NOW)
    plan = s.st(s.Step.PLAN, Phase.SCREEN)
    loop.move(plan, NOW)
    loop.move(s.HALTED, NOW, reason="executor unreachable")
    assert loop.resume_from == plan and loop.halted_reason == "executor unreachable"
    path = tmp_path / "state.json"
    loop.save(path)
    loaded = s.LoopState.load(path)
    assert loaded == loop and loaded.state == s.HALTED
    assert loaded.resume(NOW) == plan and loaded.halted_reason is None
    with pytest.raises(s.IllegalTransition):
        loaded.resume(NOW)
