"""The pre-registered replication set, for the Conductor's replicate phase in real mode.

PREREG 6.3 says "the agent applies this rule mechanically, and its decision log shows the ranking
it used". So in real mode the replicate phase does not plan: it proposes exactly
`clharness.analysis.design.replication_specs(rows)["specs"]`, the same call `launch_design --only
replication` makes, and logs the ranking that call returns.

The analysis code works on A1 rows, which carry both halves. Before they reach it, every row is
blinded here: the report half, its boundary numbers and the report-only tasks are replaced by NaN.
`analyze(confirmatory=False)` reads the selection half only, so the result does not change; if it
ever read the report half, the NaN would show in the ranking (a test checks both).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from clharness import spec as a1

from .contracts import Predicted, ProposedSpec, RunSpec, read_jsonl

NAN = math.nan
RULE = "PREREG 6.3 / 6.3b"


def blind(row: a1.ResultRow) -> a1.ResultRow:
    """The row with every report-half number replaced by NaN (selection half untouched)."""
    if row.retention is None:
        return row
    ret = row.retention
    report = ret.report.model_copy(update={
        "mean_delta": NAN,
        "per_task": {k: t.model_copy(update={"base": NAN, "final": NAN, "delta": NAN})
                     for k, t in ret.report.per_task.items()},
    })
    bounds = [b.model_copy(update={"report_mean_delta": NAN}) for b in ret.boundaries]
    return row.model_copy(update={"retention": ret.model_copy(update={
        "report": report, "boundaries": bounds, "report_only": {}})})


def blinded_rows(rows_path: str | Path) -> list[a1.ResultRow]:
    """Every A1 row of the file that validates, blinded. Bad lines are skipped, as `load_rows`."""
    out = []
    for record in read_jsonl(rows_path).records:
        try:
            out.append(blind(a1.ResultRow.model_validate(record)))
        except ValidationError:
            continue
    return out


def replicate_plan(rows_path: str | Path) -> tuple[list[ProposedSpec], dict[str, Any]]:
    """The replicate proposal and the data to log with it.

    The proposal is `replication_specs(rows)["specs"]`. When that is empty because every
    replication run already has a finished row, the whole planned set is proposed instead (from
    the seed-0 rows, which rank the same), so the gate drops each one as a duplicate and the loop
    moves on with "nothing new to run" rather than an empty proposal.
    """
    from clharness.analysis.design import replication_specs
    from clharness.analysis.replication import REPLICATION_SEEDS

    rows = blinded_rows(rows_path)
    plan = replication_specs(rows)
    specs: Sequence[RunSpec] = plan["specs"]
    all_done = not specs and plan["planned"] > 0
    if all_done:
        screen_only = [r for r in rows if r.spec.seed not in REPLICATION_SEEDS]
        specs = replication_specs(screen_only)["specs"]
    by_label = {d["config"]: d for d in [*plan["ranking"], *plan["selection"]]}
    proposed = []
    for s in specs:
        seen = by_label.get(s.name.removesuffix(f"-s{s.seed}"), {})
        proposed.append(ProposedSpec(
            spec=s, role="replicate",
            reason=f"{RULE}: promoted on the seed-0 selection half, replicated at seed {s.seed}",
            predicted=Predicted(
                retention_mean_delta=_finite(seen.get("selection_primary"), 0.0),
                acquisition_ratio=_finite(seen.get("acquisition"), 1.0))))
    data = {
        "source": "clharness.analysis.design.replication_specs",
        "rule": plan["rule"],
        "promoted": plan["promoted"],
        "promoted_selection": plan["promoted_selection"],
        "planned": plan["planned"],
        "already_finished": plan["already_finished"],
        "ranking": [_slim(d) for d in plan["ranking"]],
        "selection_ranking": [_slim(d) for d in plan["selection"]],
        "halves_read": ["selection"],
    }
    return proposed, data


def _finite(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _slim(entry: dict[str, Any]) -> dict[str, Any]:
    """One ranking line for the decision log: plain JSON values only."""
    out = {}
    for k, v in entry.items():
        if isinstance(v, float):
            out[k] = round(v, 6) if math.isfinite(v) else None
        elif isinstance(v, (str, int, bool)) or v is None:
            out[k] = v
    return out
