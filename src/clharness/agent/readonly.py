"""An executor that only reads rows. It never launches a run.

Real mode of the loop (`loop.py --executor real`) uses it: the Conductor reads the rows the
pre-registered grid produced, plans, gates and scouts on the real catalog, and logs every
accepted spec as a proposal. Every launch goes through `launch_design`. `submit` raises, so no
code path can spend GPU money through this executor.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .contracts import Catalog, ResultRow, RunSpec, technique_names
from .modal_executor import GPU_USD_PER_HOUR


class LaunchRefused(RuntimeError):
    """Something tried to launch a run through the read-only executor."""


class ReadOnlyExecutor:
    def __init__(self, rows_path: str | Path, *, minutes_per_run: float, gpu: str = "H100",
                 catalog: Catalog | None = None) -> None:
        self.rows_path = Path(rows_path)
        self.minutes_per_run, self.gpu, self.catalog = minutes_per_run, gpu, catalog

    def submit(self, specs: Sequence[RunSpec]) -> None:
        raise LaunchRefused(
            f"the read-only executor launches nothing ({len(specs)} spec(s) refused); "
            "launch through clharness.agent.launch_design")

    def poll(self) -> list[ResultRow]:
        return []

    def estimate_cost(self, spec: RunSpec) -> float:
        """What the run would cost if someone launched it (same price as `ModalExecutor`)."""
        base = self.minutes_per_run / 60 * GPU_USD_PER_HOUR.get(self.gpu, 4.0)
        if self.catalog is not None:
            names = [n for n in technique_names(spec) if self.catalog.has(n)]
            base *= self.catalog.cost_multiplier(names)
        return round(base, 2)

    def healthy(self) -> bool:
        return self.rows_path.is_file()
