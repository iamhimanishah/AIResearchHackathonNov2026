"""The Conductor, Gatekeeper and Operator on the simulator: ARCHITECTURE section 8."""

from __future__ import annotations

import json

import pytest

from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.gatekeeper import Gatekeeper
from clharness.agent.loop import build_sim
from clharness.agent.operator import Operator
from clharness.agent.roles import FakeClock, GateRequest
from clharness.agent.states import DONE, HALTED

CAP = 700.0


class Killed(Exception):
    pass


def _run(out, seed=0, **kw):
    conductor = build_sim(out, seed, CAP, **kw)
    return conductor, conductor.run()


def _choices(out):
    return [(d.key, d.data.get("choice")) for d in c.DecisionLog(out / "decisions.jsonl").entries()]


def test_sim_run_reaches_done_with_a_verified_chain(tmp_path):
    _, state = _run(tmp_path)
    assert state.state == DONE
    log = c.DecisionLog(tmp_path / "decisions.jsonl")
    report = log.verify(expected_head=state.decision_head)
    assert report.ok and report.count == state.decision_count
    freeze = next(d for d in log.entries() if d.state == "freeze")
    assert freeze.data["frozen_set_hash"] == state.frozen_set_hash
    assert any(d.proposal is not None for d in log.entries())
    assert any(d.verdict is not None for d in log.entries())
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["frozen_set_hash"] == state.frozen_set_hash and written["arms"]
    assert len(c.load_rows(tmp_path / "rows.jsonl").rows) == len(
        sim.a1.read_rows(tmp_path / "rows.jsonl"))


def test_sim_run_is_deterministic(tmp_path):
    _run(tmp_path / "a")
    _run(tmp_path / "b")
    assert _choices(tmp_path / "a") == _choices(tmp_path / "b")


def test_kill_at_every_state_and_point_then_resume_ends_in_the_same_place(tmp_path):
    _run(tmp_path / "clean")
    killed: set[tuple[str, str]] = set()

    def fault(state_id, point):
        if (state_id, point) not in killed:
            killed.add((state_id, point))
            raise Killed(f"{state_id}/{point}")

    clock, out = FakeClock(), tmp_path / "killed"
    for _ in range(500):
        try:
            state = build_sim(out, 0, CAP, clock=clock, fault=fault).run()
            break
        except Killed:
            continue
    assert state.state == DONE
    assert {p for _, p in killed} == {"work_done", "logged", "saved"}
    assert len({s for s, _ in killed}) >= 15
    assert _choices(out) == _choices(tmp_path / "clean")
    clean = json.loads((tmp_path / "clean" / "report.json").read_text())
    assert json.loads((out / "report.json").read_text())["arms"] == clean["arms"]


def test_a_cut_or_edited_log_stops_the_restart(tmp_path):
    _run(tmp_path)
    path = tmp_path / "decisions.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))  # cut the last entry: the chain still verifies
    with pytest.raises(c.ContractError, match="lost or changed"):
        build_sim(tmp_path, 0, CAP)
    edited = json.loads(lines[3])
    edited["reason"] = "rewritten"
    path.write_text("".join(lines[:3] + [json.dumps(edited) + "\n"] + lines[4:]))
    with pytest.raises(c.ContractError, match="does not verify"):
        build_sim(tmp_path, 0, CAP)


@pytest.mark.parametrize("seed", range(6))
def test_the_trap_technique_is_never_frozen(tmp_path, seed):
    _, state = _run(tmp_path, seed=seed)
    world = sim.World(seed)
    assert state.state == DONE
    rows = {r.spec_hash: r for r in c.load_rows(tmp_path / "rows.jsonl").rows}
    frozen = [rows[h] for h in state.frozen_spec_hashes]
    assert frozen and all(world.trap not in c.technique_names(r.spec) for r in frozen)


def test_report_half_needs_the_freeze_decision(tmp_path):
    conductor = build_sim(tmp_path, 0, CAP)
    conductor.run(max_transitions=3)
    with pytest.raises(c.FirewallError):
        c.mint_freeze_token(conductor.log, c.frozen_set_hash(["x"]))


# ---- Gatekeeper: each rule goes red ----------------------------------------------------------


def _request(rows=(), pending=(), left=100.0, estimate=1.5, specs=()):
    return GateRequest(
        phase=c.Phase.SCREEN, prereg=sim.default_prereg(), catalog=sim.default_catalog(),
        budget=c.BudgetView(compute_usd_left=left, compute_usd_reserve=0.0, model_usd_left=1.0,
                            minutes_left=None, concurrency=8, minutes_per_run=30.0,
                            baseline_run_usd=1.5),
        stream=sim.default_stream(), rows=tuple(rows), pending_spec_hashes=tuple(pending),
        estimated_usd={c.spec_hash(s): estimate for s in specs},
    )


def _proposal(*specs, role="treatment"):
    items = [c.ProposedSpec(spec=s, role=role, reason="r",
                            predicted=c.Predicted(retention_mean_delta=0.0, acquisition_ratio=1.0))
             for s in specs]
    return c.Proposal(phase=c.Phase.SCREEN, specs=tuple(items), rationale="test")


def test_gate_accepts_a_clean_proposal():
    spec = sim.make_spec(["replay"])
    verdict = Gatekeeper().review(_proposal(spec), _request(specs=[spec]))
    assert verdict.token == "ACCEPT" and verdict.kept == (c.spec_hash(spec),)


def test_gate_matched_compute_drops_other_streams_and_optim_overrides():
    short = c.build_spec("s", 0, [c.StreamDomain(name="code", tokens=1000)], {"replay": {}})
    lr = sim.make_spec(["lora"], optim={"lr": 1e-3})
    good = sim.make_spec(["replay"])
    verdict = Gatekeeper().review(_proposal(short, lr, good), _request(specs=[short, lr, good]))
    assert verdict.token == "TRIM" and verdict.kept == (c.spec_hash(good),)
    assert {d.rule for d in verdict.dropped} == {"matched_compute"}


def test_gate_duplicate_drops_settled_pending_and_repeated_but_not_infra_failures():
    world = sim.World(0, infra_fail_rate=0.0)
    done, pend, infra = (sim.make_spec([n]) for n in ("replay", "lora", "distill"))
    rows = c.selection_views([world.simulate(done), world.failed_row(infra, "infra", "preempted")])
    verdict = Gatekeeper().review(
        _proposal(done, pend, infra, infra),
        _request(rows=rows, pending=[c.spec_hash(pend)], specs=[done, pend, infra]))
    assert verdict.kept == (c.spec_hash(infra),)
    assert sorted(d.reason.split()[-1] for d in verdict.dropped) == ["launched", "proposal", "row"]


def test_gate_budget_trims_treatments_and_rejects_an_unaffordable_control():
    a, b = sim.make_spec(["replay"]), sim.make_spec(["lora"])
    verdict = Gatekeeper().review(_proposal(a, b), _request(left=4.0, specs=[a, b]))
    assert verdict.token == "TRIM" and verdict.kept == (c.spec_hash(a),)
    control = sim.make_spec()
    verdict = Gatekeeper().review(_proposal(control, role="control"),
                                  _request(left=1.0, specs=[control]))
    assert verdict.token == "REJECT" and verdict.dropped[0].spec_hash == c.WHOLE_PROPOSAL


def test_gate_rejects_an_empty_proposal():
    verdict = Gatekeeper().review(c.Proposal(phase=c.Phase.SCREEN, specs=(), rationale="x"),
                                  _request())
    assert verdict.token == "REJECT" and verdict.dropped[0].rule == "empty"


# ---- Operator: one retry for infrastructure, none for science ---------------------------------


def _operator(tmp_path, world=None):
    clock = FakeClock()
    world = world or sim.World(0, infra_fail_rate=0.0)
    executor = sim.SimExecutor(world, tmp_path / "rows.jsonl", clock)
    return executor, Operator(executor, clock, tmp_path / "rows.jsonl"), clock


def test_operator_retries_an_infra_failure_once_with_the_same_spec(tmp_path):
    executor, operator, clock = _operator(tmp_path)
    spec = sim.make_spec(["replay"])
    key = c.spec_hash(spec)
    executor.fail_spec(key, "infra", times=1)
    operator.launch([spec])
    report = operator.wait([spec], clock.now() + sim.timedelta(days=1))
    assert report.requeued == (key,) and report.rows[0].status == "finished"
    assert report.rows[0].attempt == 2 and report.rows[0].spec == spec
    assert executor.launches[key] == 2


def test_operator_does_not_retry_a_science_failure(tmp_path):
    executor, operator, clock = _operator(tmp_path)
    spec = sim.make_spec(["replay"])
    executor.fail_spec(c.spec_hash(spec), "science", times=1)
    operator.launch([spec])
    report = operator.wait([spec], clock.now() + sim.timedelta(days=1))
    assert report.requeued == () and report.rows[0].failure_kind == "science"


def test_operator_launch_is_idempotent(tmp_path):
    _, operator, _ = _operator(tmp_path)
    spec = sim.make_spec(["replay"])
    first = operator.launch([spec, spec])
    again = operator.launch([spec], pending=first.submitted)
    assert len(first.submitted) == 1 and again.submitted == ()


def test_halted_loop_records_its_reason(tmp_path):
    def reject_all(specs, request):
        return [c.Dropped(spec_hash=c.WHOLE_PROPOSAL, rule="test", reason="no")]

    conductor = build_sim(tmp_path, 0, CAP, gatekeeper=Gatekeeper({"test": reject_all}))
    state = conductor.run()
    assert state.state == HALTED and "bug" in state.halted_reason
