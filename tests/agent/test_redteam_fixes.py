"""Fixes for the open red-team findings (docs/reviews/red-team-57fa138.md), one block each."""

from __future__ import annotations

import math

import pytest

from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.loop import build_sim
from clharness.agent.roles import FakeClock

CAP = 700.0


# ---- RT-16: rows without a price are charged the estimate -------------------------------------


def _infra_row(spec: c.RunSpec, usd: float | None = None) -> c.ResultRow:
    cost = c.a1_cost(0, 0.0, "H100", usd) if usd is not None else None
    return c.from_a1_row(c.a1_failed_row(spec, "infra", "lost the call", cost=cost))


def test_rt16_an_unpriced_infra_row_is_charged_the_estimate(tmp_path):
    conductor = build_sim(tmp_path, 0, CAP, clock=FakeClock())
    spec = sim.make_spec(["replay"])
    before = conductor._budget().compute_usd_left
    (tmp_path / "rows.jsonl").write_text(
        c.a1_failed_row(spec, "infra", "lost the call").model_dump_json() + "\n")
    after = conductor._budget().compute_usd_left
    assert before - after == pytest.approx(conductor._estimate(spec))


@pytest.mark.parametrize("usd", [math.nan, math.inf, 0.0, -3.0])
def test_rt16_a_nan_or_negative_price_is_charged_the_estimate(tmp_path, usd):
    conductor = build_sim(tmp_path, 0, CAP, clock=FakeClock())
    spec = sim.make_spec(["replay"])
    row = _infra_row(spec)
    row = row.model_copy(update={"cost": row.cost.model_copy(update={"usd": usd})})
    assert conductor._spent([row]) == pytest.approx(conductor._estimate(spec))
    priced = _infra_row(spec, 0.42)
    assert conductor._spent([priced]) == pytest.approx(0.42)  # a real price is kept


def test_rt16_the_explore_share_counts_an_unpriced_infra_row(tmp_path):
    conductor = build_sim(tmp_path, 0, CAP, clock=FakeClock())
    spec = sim.make_spec(["replay"]).model_copy(update={"notes": "explore:claim-1"})
    assert conductor._explore_used([_infra_row(spec)]) == pytest.approx(
        conductor._estimate(spec))


# ---- RT-30: a timed-out run is re-queued once -------------------------------------------------


class _Scripted:
    """An executor whose runs end as scripted: one failure kind (or "ok") per attempt."""

    def __init__(self, rows_path, script, unreachable_submits=0):
        self.rows_path, self.script = rows_path, list(script)
        self.queue, self.submits, self.unreachable = [], 0, unreachable_submits

    def submit(self, specs):
        from clharness.agent.roles import ExecutorUnreachable

        self.submits += 1
        if self.submits > 1 and self.unreachable > 0:  # the first launch always answers
            self.unreachable -= 1
            raise ExecutorUnreachable("down")
        self.queue += list(specs)

    def poll(self):
        out = []
        for spec in self.queue:
            kind = self.script.pop(0)
            if kind == "ok":
                row = sim.World(0, infra_fail_rate=0.0).simulate_a1(spec)
            else:
                row = c.a1_failed_row(spec, kind, "lost")
            prior = len(c.load_rows(self.rows_path).rows) if self.rows_path.exists() else 0
            row = row.model_copy(update={"attempt": prior + 1})
            with open(self.rows_path, "a") as handle:
                handle.write(row.model_dump_json() + "\n")
            out.append(c.from_a1_row(row))
        self.queue = []
        return out

    def estimate_cost(self, spec):
        return 1.0

    def healthy(self):
        return True


def _operate(tmp_path, script, unreachable=0):
    from datetime import timedelta

    from clharness.agent.operator import Operator

    clock = FakeClock()
    rows = tmp_path / "rows.jsonl"
    executor = _Scripted(rows, script, unreachable)
    operator = Operator(executor, clock, rows)
    spec = sim.make_spec(["replay"])
    operator.launch([spec])
    report = operator.wait([spec], clock.now() + timedelta(hours=2))
    return report, executor, c.load_rows(rows).rows


def test_rt30_a_timeout_is_retried_once_and_settles_on_attempt_two(tmp_path):
    report, executor, rows = _operate(tmp_path, ["timeout", "ok"])
    assert executor.submits == 2 and len(rows) == 2
    assert [r.status for r in report.rows] == ["finished"] and report.rows[0].attempt == 2
    assert report.requeued and not report.timed_out


def test_rt30_two_timeouts_settle_and_the_gate_drops_the_spec(tmp_path):
    from clharness.agent.gatekeeper import duplicate

    report, executor, rows = _operate(tmp_path, ["timeout", "timeout", "ok"])
    assert executor.submits == 2 and len(rows) == 2  # never a third run
    assert report.rows[0].failure_kind == "timeout" and not report.timed_out
    conductor = build_sim(tmp_path / "loop", 0, CAP, clock=FakeClock())
    conductor.rows_path = tmp_path / "rows.jsonl"
    exhausted = conductor._retry_exhausted()
    assert exhausted == (rows[0].spec_hash,)
    item = c.ProposedSpec(spec=rows[0].spec, role="treatment", reason="r",
                          predicted=c.Predicted(retention_mean_delta=0.0, acquisition_ratio=1.0))
    from clharness.agent.roles import GateRequest

    request = GateRequest(
        phase=c.Phase.REPLICATE, prereg=sim.default_prereg(), catalog=sim.default_catalog(),
        budget=conductor._budget(), stream=sim.default_stream(),
        rows=c.selection_views(rows[-1:]), retry_exhausted=exhausted)
    [drop] = duplicate([item], request)
    assert "infrastructure or timeout" in drop.reason


def test_rt30_infra_then_timeout_share_the_cap(tmp_path):
    report, executor, rows = _operate(tmp_path, ["infra", "timeout", "ok"])
    assert executor.submits == 2 and len(rows) == 2
    assert report.rows[0].failure_kind == "timeout"


def test_rt30_a_requeue_survives_an_unreachable_executor(tmp_path):
    report, executor, _ = _operate(tmp_path, ["timeout", "ok"], unreachable=1)
    assert executor.submits == 3  # launch, the refused re-queue, the re-queue that answered
    assert [r.status for r in report.rows] == ["finished"]


def test_rt30_one_constant_for_retryable_failures():
    from pathlib import Path

    from clharness.agent import contracts

    assert contracts.RETRYABLE_KINDS == frozenset({"infra", "timeout"})
    src = Path(contracts.__file__).parent
    for path in src.glob("*.py"):
        text = path.read_text()
        assert '("infra", "timeout")' not in text, path.name
        assert 'failure_kind == "infra"' not in text, path.name


# ---- RT-15 + RT-17: one writer per --out ------------------------------------------------------

_HOLD = """
import sys, time
from clharness.agent.lock import single_writer
with single_writer(sys.argv[1]):
    print("held", flush=True)
    time.sleep(60)
"""


def _holder(out):
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", _HOLD, str(out)], stdout=subprocess.PIPE,
                            text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_rt15_a_held_lock_refuses_the_loop_and_writes_nothing(tmp_path, capsys):
    from clharness.agent.loop import main

    out = tmp_path / "out"
    proc = _holder(out)
    try:
        assert main(["--executor", "sim", "--out", str(out)]) == 2
    finally:
        proc.kill()
        proc.wait()
    assert "refused:" in capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == [".agent.lock"]


def test_rt15_the_lock_is_released_when_the_holder_is_killed(tmp_path):
    from clharness.agent.lock import single_writer

    proc = _holder(tmp_path)
    proc.kill()
    proc.wait()
    with single_writer(tmp_path) as path:
        assert path.name == ".agent.lock" and path.exists()
    assert path.exists()  # never unlinked


def test_rt15_the_lock_is_reentrant_in_one_process_and_per_directory(tmp_path):
    from clharness.agent.lock import AlreadyRunning, single_writer

    with single_writer(tmp_path / "a") as first, single_writer(tmp_path / "a") as again, \
            single_writer(tmp_path / "b") as other:
        assert first == again != other
    proc = _holder(tmp_path / "a")
    try:
        with pytest.raises(AlreadyRunning, match="pid"), single_writer(tmp_path / "a"):
            pass
        with single_writer(tmp_path / "b"):
            pass
    finally:
        proc.kill()
        proc.wait()


def test_rt15_launch_design_and_explore_take_the_lock(tmp_path, capsys):
    from clharness.agent import explore, launch_design

    out = tmp_path / "out"
    proc = _holder(out)
    try:
        with pytest.raises(SystemExit) as stop:
            launch_design.main(["--go", "--only", "screen", "--out", str(out), "--have"])
        assert stop.value.code == 2
        with pytest.raises(SystemExit) as stop:
            explore.main(["--out", str(out), "--have"])
        assert stop.value.code == 2
    finally:
        proc.kill()
        proc.wait()
    assert capsys.readouterr().err.count("refused:") == 2


def test_rt15_the_lock_file_is_git_ignored_and_uv_lock_is_not():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / ".gitignore").read_text().splitlines()
    assert "results/**/.agent.lock" in text and "*.lock" not in text
