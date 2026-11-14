"""The Operator: launch accepted specs, wait for their rows, re-queue an infra failure once.

It keeps no state of its own. What has run is read from `rows.jsonl` every time, so a process
killed anywhere can call `launch` and `wait` again with the same specs and nothing runs twice.
The executor is also idempotent by spec hash; this is the second lock, not the only one.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .contracts import ResultRow, RunSpec, load_rows, spec_hash
from .roles import Clock, Executor, ExecutorUnreachable, LaunchReport, WaitReport

MAX_INFRA_ATTEMPTS = 2  # the first run plus one re-queue


def _attempts(rows_path: Path) -> dict[str, list[ResultRow]]:
    by_spec: dict[str, list[ResultRow]] = {}
    for row in load_rows(rows_path).rows:
        by_spec.setdefault(row.spec_hash, []).append(row)
    return by_spec


def _settled(attempts: list[ResultRow]) -> ResultRow | None:
    """The final row of a spec, or None if it should still run (or run again)."""
    if not attempts:
        return None
    last = attempts[-1]
    if last.failure_kind == "infra" and len(attempts) < MAX_INFRA_ATTEMPTS:
        return None
    return last


class Operator:
    def __init__(self, executor: Executor, clock: Clock, rows_path: str | Path,
                 poll_seconds: float = 60.0) -> None:
        self.executor = executor
        self.clock = clock
        self.rows_path = Path(rows_path)
        self.poll_seconds = poll_seconds

    def launch(self, specs: Sequence[RunSpec], pending: Sequence[str] = ()) -> LaunchReport:
        """Submit the specs that have no settled row and are not already in flight."""
        attempts = _attempts(self.rows_path)
        fresh, skipped = {}, []
        for spec in specs:
            key = spec_hash(spec)
            if key in pending or key in fresh or _settled(attempts.get(key, [])) is not None:
                skipped.append(key)
            else:
                fresh[key] = spec
        if fresh:
            self.executor.submit(list(fresh.values()))
        return LaunchReport(submitted=tuple(fresh), skipped=tuple(skipped))

    def wait(self, specs: Sequence[RunSpec], deadline: datetime) -> WaitReport:
        """Poll until every spec has a settled row or the deadline passes."""
        wanted = {spec_hash(s): s for s in specs}
        requeued: set[str] = set()
        while True:
            try:
                self.executor.poll()  # the executor appends new rows to rows.jsonl
            except ExecutorUnreachable:
                pass  # the rows file is the truth; poll again on the next tick
            attempts = _attempts(self.rows_path)
            final = {}
            for key, spec in wanted.items():
                tries = attempts.get(key, [])
                row = _settled(tries)
                if row is not None:
                    final[key] = row
                elif tries and key not in requeued:  # one infra failure so far: run it again
                    self.executor.submit([spec])  # no-op if it is already in flight
                    requeued.add(key)
            missing = [k for k in wanted if k not in final]
            if not missing or self.clock.now() >= deadline:
                retried = {k for k in wanted if any(
                    r.failure_kind in ("infra", "timeout") for r in attempts.get(k, []))}
                return WaitReport(  # read from the rows file, so a resumed wait reports the same
                    rows=tuple(final.values()), requeued=tuple(sorted(requeued | retried)),
                    timed_out=tuple(missing),
                )
            self.clock.sleep(self.poll_seconds)
