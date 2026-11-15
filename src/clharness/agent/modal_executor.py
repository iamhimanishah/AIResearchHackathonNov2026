"""`Executor` on Modal: each spec runs as one call of A6's deployed `clharness` app.

The remote function (`src/clharness/executor/modal_app.py::run`, task A6) trains one spec and
returns its A1 result row as a JSON line. This adapter spawns one call per spec, remembers the
call ids in `<rows>.modal.json`, and on `poll` appends every row that came back to `rows.jsonl`.

    run raised inside the container  -> A6 returns a `failed` row: final, never retried
    machine lost after Modal's retries -> `failed`, reason "infrastructure: ...": retryable
    the function hit its timeout       -> `failed`, reason "timeout: ...": retryable

Needs `modal deploy src/clharness/executor/modal_app.py` once, so the function can be looked
up by name. `modal` is imported lazily: the agent package must import without it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .. import spec as a1
from .contracts import (
    Catalog,
    FailureKind,
    ResultRow,
    RunSpec,
    a1_failed_row,
    append_jsonl,
    atomic_write_json,
    from_a1_row,
    latest_by_spec,
    load_rows,
    retryable,
    spec_hash,
    technique_names,
)
from .roles import ExecutorUnreachable

GPU_USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "A100": 2.10, "L40S": 1.95, "A10G": 1.10}


class RunLost(Exception):
    """The call will never return a row. `kind` says whether rerunning may help."""

    def __init__(self, message: str, kind: FailureKind = "infra") -> None:
        super().__init__(message)
        self.kind = kind


class ModalBackend(Protocol):
    def spawn(self, spec_json: str, git_sha: str, gpu: str) -> str:
        """Start one run; return the call id."""
        ...

    def result(self, call_id: str) -> str | None:
        """The row line, or None while running. Raises `RunLost` for a call that is gone."""
        ...

    def healthy(self) -> bool: ...


class ModalFunctionBackend:
    """The real backend: `modal.Function.from_name(app, function)`."""

    def __init__(self, app: str = "clharness", function: str = "run") -> None:
        self.app, self.function = app, function
        self._fn: Any = None

    def _lookup(self) -> Any:
        if self._fn is None:
            import modal

            self._fn = modal.Function.from_name(self.app, self.function)
        return self._fn

    def spawn(self, spec_json: str, git_sha: str, gpu: str) -> str:
        return self._lookup().spawn(spec_json, git_sha, gpu).object_id

    def result(self, call_id: str) -> str | None:
        import modal

        timeout_error = getattr(modal.exception, "FunctionTimeoutError", ())
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except timeout_error as exc:
            raise RunLost(f"function timeout: {exc}", kind="timeout") from exc
        except TimeoutError:
            return None
        except modal.exception.Error as exc:
            raise RunLost(f"{type(exc).__name__}: {exc}") from exc

    def healthy(self) -> bool:
        try:
            self._lookup()
            return True
        except Exception:  # noqa: BLE001 - any failure to look the app up means not healthy
            return False


class ModalExecutor:
    """`roles.Executor` on Modal. Idempotent by spec hash; `rows.jsonl` stays the truth."""

    def __init__(self, rows_path: str | Path, backend: ModalBackend, *, git_sha: str,
                 gpu: str = "H100", minutes_per_run: float = 30.0,
                 catalog: Catalog | None = None) -> None:
        self.rows_path = Path(rows_path)
        self.catalog = catalog
        self.backend, self.git_sha, self.gpu = backend, git_sha, gpu
        self.minutes_per_run = minutes_per_run
        self._state_path = self.rows_path.with_name(self.rows_path.name + ".modal.json")
        state = (json.loads(self._state_path.read_text(encoding="utf-8"))
                 if self._state_path.exists() else {})
        self._calls: dict[str, dict[str, Any]] = state.get("calls", {})
        self.launches: dict[str, int] = state.get("launches", {})

    def in_flight(self) -> list[str]:
        return sorted(self._calls)

    def submit(self, specs: Sequence[RunSpec]) -> None:
        latest = latest_by_spec(load_rows(self.rows_path).rows)
        for spec in specs:
            key = spec_hash(spec)
            last = latest.get(key)
            if key in self._calls or (last and not retryable(last)):
                continue
            try:
                call_id = self.backend.spawn(spec.model_dump_json(), self.git_sha, self.gpu)
            except Exception as exc:
                self._save()
                raise ExecutorUnreachable(f"modal spawn failed: {exc}") from exc
            self.launches[key] = self.launches.get(key, 0) + 1
            self._calls[key] = {"call_id": call_id, "spec": spec.model_dump(mode="json"),
                                "attempt": self.launches[key]}
            self._save()

    def poll(self) -> list[ResultRow]:
        landed = []
        for key, call in sorted(self._calls.items()):
            spec = RunSpec.model_validate(call["spec"])
            try:
                line = self.backend.result(call["call_id"])
            except RunLost as exc:
                row = self._failed(spec, exc.kind, str(exc), call["attempt"])
            except Exception as exc:
                self._save()
                raise ExecutorUnreachable(f"modal poll failed: {exc}") from exc
            else:
                if line is None:
                    continue
                row = self._parse(spec, key, line, call["attempt"])
            append_jsonl(self.rows_path, row)
            del self._calls[key]
            self._save()
            landed.append(from_a1_row(row))
        return landed

    def estimate_cost(self, spec: RunSpec) -> float:
        base = self.minutes_per_run / 60 * GPU_USD_PER_HOUR.get(self.gpu, 4.0)
        if self.catalog is not None:
            names = [n for n in technique_names(spec) if self.catalog.has(n)]
            base *= self.catalog.cost_multiplier(names)
        return round(base, 2)

    def healthy(self) -> bool:
        return self.backend.healthy()

    def _parse(self, spec: RunSpec, key: str, line: str, attempt: int) -> a1.ResultRow:
        try:
            row = a1.ResultRow.model_validate_json(line)
        except ValueError as exc:
            return self._failed(spec, "infra", f"row did not validate: {exc}"[:600], attempt)
        if row.spec_hash != key:
            return self._failed(spec, "infra", f"row is for spec {row.spec_hash[:12]}", attempt)
        if row.git_sha != self.git_sha:
            why = f"row was built from git {row.git_sha[:12]}, not {self.git_sha[:12]}"
            return self._failed(spec, "infra", why, attempt)
        return row.model_copy(update={"attempt": attempt})

    def _failed(self, spec: RunSpec, kind: FailureKind, reason: str,
                attempt: int) -> a1.ResultRow:
        return a1_failed_row(spec, kind, reason, git_sha=self.git_sha, backend="modal",
                             attempt=attempt)

    def _save(self) -> None:
        atomic_write_json(self._state_path, {"calls": self._calls, "launches": self.launches})
