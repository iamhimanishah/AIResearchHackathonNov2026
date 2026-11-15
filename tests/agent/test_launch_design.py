"""The pre-registered launcher against a fake Modal backend: no network, no GPU."""

import json

import pytest

from clharness import spec as a1
from clharness.agent.contracts import DecisionLog
from clharness.agent.launch_design import Halt, launch, planned_specs

SHA = "a" * 40


def _row(spec: a1.RunSpec, sha: str = SHA, status: str = "finished", reason: str | None = None) -> a1.ResultRow:
    common = dict(spec=spec, spec_hash=spec.spec_hash(), git_sha=sha, backend="modal")
    if status != "finished":
        return a1.ResultRow(status=status, reason=reason, **common)
    half = a1.HalfRetention(per_task={"arc_easy": a1.TaskRetention(base=0.7, final=0.65, delta=-0.05, n_items=250)},
                            mean_delta=-0.05)
    return a1.ResultRow(
        status="finished", retention=a1.Retention(selection=half, report=half),
        acquisition=a1.Acquisition(ppl_ratio={"legal": 0.78}, mean_ppl_ratio=0.78),
        cost=a1.Cost(gradient_tokens=20_000_000, wall_clock_s=1100.0, gpu_type="H100", est_usd=1.2), **common)


class FakeModal:
    """Each call finishes on the second time it is polled. Records how many run at once."""

    def __init__(self, sha: str = SHA, fail_once: set[str] | None = None):
        self.sha, self.calls, self.polls, self.peak, self.spawned = sha, {}, {}, 0, []
        self.fail_once = set(fail_once or ())

    def spawn(self, spec_json: str, git_sha: str, gpu: str) -> str:
        spec = a1.RunSpec.model_validate_json(spec_json)
        call_id = f"fc-{len(self.spawned)}"
        self.calls[call_id], self.polls[call_id] = spec, 0
        self.spawned.append(spec.name)
        self.peak = max(self.peak, len(self.calls))
        return call_id

    def result(self, call_id: str) -> str | None:
        self.polls[call_id] += 1
        if self.polls[call_id] < 2:
            return None
        spec = self.calls.pop(call_id)
        if spec.name in self.fail_once:
            self.fail_once.discard(spec.name)
            from clharness.agent.modal_executor import RunLost

            raise RunLost("container lost")
        return a1.row_to_line(_row(spec, self.sha))

    def healthy(self) -> bool:
        return True


def _launch(tmp_path, backend, specs, **kw):
    said = []
    counts = launch(specs, tmp_path / "agent", backend, SHA, sleep=lambda s: None, say=said.append, **kw)
    return counts, said


def test_the_plan_is_the_pre_registered_design_with_unique_specs():
    specs = planned_specs()
    assert len(specs) == len({s.spec_hash() for s in specs}) == len(planned_specs("screen")) + len(planned_specs("claims"))
    assert all(s.bench == "fast" for s in specs)


def test_runs_everything_in_waves_and_logs_a_chain_that_verifies(tmp_path):
    specs = planned_specs("screen")[:7]
    backend = FakeModal()
    counts, _ = _launch(tmp_path, backend, specs, concurrency=3)
    assert counts == {"finished": 7}
    assert backend.peak <= 3 and len(backend.spawned) == 7
    rows = a1.read_rows(tmp_path / "agent" / "rows.jsonl")
    assert {r.spec_hash for r in rows} == {s.spec_hash() for s in specs}
    log = DecisionLog(tmp_path / "agent" / "decisions.jsonl")
    assert log.verify().ok
    keys = [d.key for d in log.entries()]
    assert keys[0] == "design/plan" and keys[-1] == "design/done"
    assert sum(k.startswith("design/launch#") for k in keys) >= 3


def test_specs_with_a_finished_row_elsewhere_are_not_run_again(tmp_path):
    specs = planned_specs("screen")[:4]
    have = tmp_path / "have.jsonl"
    a1.write_rows(have, [_row(specs[0]), _row(specs[1], status="failed", reason="oom")])
    backend = FakeModal()
    counts, _ = _launch(tmp_path, backend, specs, have=[have])
    assert specs[0].name not in backend.spawned  # finished elsewhere: skipped
    assert specs[1].name in backend.spawned  # a failed row elsewhere does not count as done
    assert counts == {"finished": 3}


def test_running_it_again_starts_nothing(tmp_path):
    specs = planned_specs("screen")[:3]
    _launch(tmp_path, FakeModal(), specs)
    again = FakeModal()
    counts, _ = _launch(tmp_path, again, specs)
    assert again.spawned == [] and counts == {"finished": 3}


def test_a_run_lost_to_the_machine_is_launched_once_more(tmp_path):
    specs = planned_specs("screen")[:2]
    backend = FakeModal(fail_once={specs[0].name})
    counts, _ = _launch(tmp_path, backend, specs)
    assert backend.spawned.count(specs[0].name) == 2 and counts == {"finished": 2}


def test_over_the_cap_launches_nothing(tmp_path):
    backend = FakeModal()
    with pytest.raises(Halt, match="over the cap"):
        _launch(tmp_path, backend, planned_specs("screen")[:5], cap_usd=3.0)
    assert backend.spawned == []


def test_a_deployment_at_another_commit_halts_the_launch(tmp_path):
    specs = planned_specs("screen")[:4]
    backend = FakeModal(sha="b" * 40)  # the app answers with rows built from other code
    with pytest.raises(Halt, match="other code"):
        _launch(tmp_path, backend, specs, concurrency=2)
    assert len(backend.spawned) == 2  # the first wave only; nothing after the refusal
    plan = json.loads((tmp_path / "agent" / "decisions.jsonl").read_text().splitlines()[0])
    assert plan["data"]["to_run"] == 4
