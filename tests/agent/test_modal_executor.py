"""ModalExecutor against a fake Modal backend: no network, no `modal` package needed."""

from __future__ import annotations

import subprocess
import sys

import pytest

from clharness import spec as a1
from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.conductor import Conductor
from clharness.agent.gatekeeper import Gatekeeper
from clharness.agent.modal_executor import ModalExecutor, RunLost
from clharness.agent.operator import Operator
from clharness.agent.roles import ExecutorUnreachable, FakeClock, StubAnalyst, StubPlanner
from clharness.agent.states import DONE


class FakeModal:
    """Spawns return ids; a call's result is a simulated A1 row line, after `delay` polls."""

    def __init__(self, world: sim.World, delay: int = 0) -> None:
        self.world, self.delay = world, delay
        self.spawned: list[str] = []
        self.calls: dict[str, dict] = {}
        self.lost: set[str] = set()  # spec hashes whose next call is lost
        self.down = False

    def spawn(self, spec_json, git_sha, gpu):
        if self.down:
            raise ConnectionError("modal is down")
        spec = c.RunSpec.model_validate_json(spec_json)
        call_id = f"fc-{len(self.spawned)}"
        self.spawned.append(c.spec_hash(spec))
        self.calls[call_id] = {"spec": spec, "polls": 0, "git_sha": git_sha}
        return call_id

    def result(self, call_id):
        if self.down:
            raise ConnectionError("modal is down")
        call = self.calls[call_id]
        key = c.spec_hash(call["spec"])
        if key in self.lost:
            self.lost.discard(key)
            raise RunLost("container lost")
        call["polls"] += 1
        if call["polls"] <= self.delay:
            return None
        row = self.world.simulate_a1(call["spec"], 1, False)
        return a1.row_to_line(row.model_copy(update={"backend": "modal", "git_sha": call["git_sha"]}))

    def healthy(self):
        return not self.down


def _spec(seed=0, techniques=None):
    return c.build_spec(name=f"t{seed}", seed=seed, techniques=techniques or {"replay": {}},
                        stream=sim.default_stream())


def _executor(tmp_path, backend):
    return ModalExecutor(tmp_path / "rows.jsonl", backend, git_sha="abc123")


def test_rows_land_in_the_rows_file_and_validate_as_a1(tmp_path):
    backend = FakeModal(sim.World(0), delay=1)
    ex = _executor(tmp_path, backend)
    ex.submit([_spec(0), _spec(1)])
    assert ex.poll() == [] and len(ex.in_flight()) == 2
    landed = ex.poll()
    assert len(landed) == 2 and ex.in_flight() == []
    rows = a1.read_rows(tmp_path / "rows.jsonl")
    assert {r.spec_hash for r in rows} == {c.spec_hash(_spec(0)), c.spec_hash(_spec(1))}


def test_submit_is_idempotent_by_spec_hash(tmp_path):
    backend = FakeModal(sim.World(0), delay=5)
    ex = _executor(tmp_path, backend)
    ex.submit([_spec(0), _spec(0)])
    ex.submit([_spec(0)])
    assert len(backend.spawned) == 1
    backend.delay = 0
    ex.poll()
    ex.submit([_spec(0)])  # finished: never rerun
    assert len(backend.spawned) == 1


def test_a_lost_call_is_a_retryable_infrastructure_failure(tmp_path):
    backend = FakeModal(sim.World(0))
    ex = _executor(tmp_path, backend)
    backend.lost.add(c.spec_hash(_spec(0)))
    ex.submit([_spec(0)])
    (row,) = ex.poll()
    assert row.status == "failed" and row.failure_kind == "infra" and row.backend == "modal"
    ex.submit([_spec(0)])
    (row,) = ex.poll()
    assert row.status == "finished" and row.attempt == 2 and len(backend.spawned) == 2


def test_a_dead_backend_raises_unreachable_and_records_nothing(tmp_path):
    backend = FakeModal(sim.World(0))
    backend.down = True
    ex = _executor(tmp_path, backend)
    with pytest.raises(ExecutorUnreachable):
        ex.submit([_spec(0)])
    assert ex.in_flight() == [] and not ex.healthy()


def test_a_restarted_executor_collects_calls_it_did_not_spawn(tmp_path):
    backend = FakeModal(sim.World(0), delay=3)
    _executor(tmp_path, backend).submit([_spec(0)])
    backend.delay = 0
    again = _executor(tmp_path, backend)
    assert len(again.poll()) == 1 and len(backend.spawned) == 1


def test_a_row_for_another_spec_is_refused(tmp_path):
    backend = FakeModal(sim.World(0))
    ex = _executor(tmp_path, backend)
    ex.submit([_spec(0)])
    backend.calls["fc-0"]["spec"] = _spec(7)
    (row,) = ex.poll()
    assert row.status == "failed" and row.failure_kind == "infra"


def test_the_loop_runs_to_done_on_modal(tmp_path):
    world, clock = sim.World(0, infra_fail_rate=0.0), FakeClock()
    rows = tmp_path / "rows.jsonl"
    ex = ModalExecutor(rows, FakeModal(world, delay=1), git_sha="abc123")
    conductor = Conductor(
        tmp_path, rows, executor=ex, clock=clock, planner=StubPlanner(),
        analyst=StubAnalyst(clock), gatekeeper=Gatekeeper(), operator=Operator(ex, clock, rows),
        prereg=sim.default_prereg(world), catalog=sim.default_catalog(),
        stream=sim.default_stream(), compute_cap_usd=700.0,
    )
    assert conductor.run().state == DONE
    assert all(r.backend == "modal" for r in a1.read_rows(rows))


def test_importing_the_adapter_does_not_import_modal():
    code = ("import sys, clharness.agent.modal_executor, clharness.agent.loop; "
            "sys.exit('modal' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
