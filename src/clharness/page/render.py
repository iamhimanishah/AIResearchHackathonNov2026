"""Render an agent output directory as one HTML page.

Reads `rows.jsonl`, `state.json`, `decisions.jsonl`, `memory/*.jsonl` and `report.json`. It
computes no statistic: every number shown is in a row or in `report.json`. The report half of a
row stays sealed until the row appears in `report.json`, which only exists after FREEZE.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from clharness.agent.contracts import (
    DecisionLog,
    ResultRow,
    arm_of,
    latest_by_spec,
    load_rows,
    read_jsonl,
)

STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1200px;color:#1b1f24;padding:0 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #d0d7de}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:4px 8px;border-bottom:1px solid #eaeef2;text-align:left;vertical-align:top}
th{background:#f6f8fa}td.n{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:#1a7f37;font-weight:600}.bad{color:#cf222e;font-weight:600}.muted{color:#656d76}
.tag{display:inline-block;padding:0 6px;border-radius:10px;background:#eaeef2;font-size:12px}
.ref{background:#fff8c5}code{font-size:12px}
"""


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(value: Any, fmt: str = "+.4f") -> str:
    return format(value, fmt) if isinstance(value, int | float) else "–"


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset(),
           classes: list[str] | None = None) -> str:
    if not rows:
        return '<p class="muted">None yet.</p>'
    head = "".join(f"<th>{escape(h)}</th>" for h in headers) if any(headers) else ""
    body = []
    for i, cells in enumerate(rows):
        cls = f' class="{classes[i]}"' if classes and classes[i] else ""
        tds = "".join(f'<td class="n">{c}</td>' if j in numeric else f"<td>{c}</td>"
                      for j, c in enumerate(cells))
        body.append(f"<tr{cls}>{tds}</tr>")
    head_row = f"<tr>{head}</tr>" if head else ""
    return f"<table>{head_row}{''.join(body)}</table>"


def _summary(state: dict[str, Any], out: Path) -> str:
    log = DecisionLog(out / "decisions.jsonl")
    chain = log.verify(expected_head=state.get("decision_head"))
    status = (f'<span class="ok">intact</span> ({chain.count} entries)' if chain.ok else
              f'<span class="bad">BROKEN</span>: {escape("; ".join(chain.problems[:3]))}')
    where = state.get("state") or {}
    label = "/".join(str(where[k]) for k in ("phase", "step") if where.get(k)) or "not started"
    if state.get("halted_reason"):
        label += f" ({state['halted_reason']})"
    facts = [
        ("State", escape(label)),
        ("Decision log", status),
        ("Compute spent (USD)", _num(state.get("compute_usd_spent"), ".2f")),
        ("Model spend (USD)", _num(state.get("model_usd_spent"), ".2f")),
        ("Frozen set", escape(str(state.get("frozen_set_hash") or "not frozen"))[:16]),
    ]
    return _table(["", ""], [[escape(k), v] for k, v in facts])


def _report(report: dict[str, Any]) -> str:
    if not report:
        return ('<p class="muted">Sealed. The report half opens only after FREEZE, for the '
                "frozen set.</p>")
    reference = report.get("reference_arm")
    rows, classes = [], []
    for arm, v in report.get("arms", {}).items():
        rows.append([
            escape(arm), escape(", ".join(str(s) for s in v.get("seeds", []))),
            _num(v.get("report_mean_delta")), _num(v.get("report_sd"), ".4f"),
            _num(v.get("report_vs_reference")), _num(v.get("selection_mean_delta")),
            _num(v.get("acquisition_mean"), ".4f"),
        ])
        classes.append("ref" if arm == reference else "")
    note = (f'<p class="muted">Reference arm: <b>{escape(str(reference))}</b>. Deltas are '
            "fractions of base accuracy; SD is across seeds.</p>")
    return note + _table(
        ["Arm", "Seeds", "Report Δ", "SD", "vs reference", "Selection Δ", "Acquisition ppl"],
        rows, numeric={2, 3, 4, 5, 6}, classes=classes)


def _runs(latest: list[ResultRow], opened: set[str]) -> str:
    rows = []
    for r in latest:
        report = (_num(r.retention_report.mean_delta)
                  if r.row_hash in opened and r.retention_report else '<span class="muted">sealed</span>')
        rows.append([
            escape(r.spec.name), escape(arm_of(r.spec)), str(r.spec.seed),
            f'<span class="tag">{escape(r.status)}</span>', str(r.attempt), escape(r.backend),
            _num(r.retention_selection.mean_delta if r.retention_selection else None), report,
            _num(r.acquisition_mean, ".4f"), _num(r.cost.usd, ".2f"),
        ])
    return _table(["Run", "Arm", "Seed", "Status", "Attempt", "Backend", "Selection Δ",
                   "Report Δ", "Acquisition ppl", "USD"], rows, numeric={2, 4, 6, 7, 8, 9})


def _failures(rows: list[ResultRow]) -> str:
    return _table(["Run", "Status", "Kind", "Attempt", "Reason"], [
        [escape(r.spec.name), escape(r.status), escape(r.failure_kind or ""), str(r.attempt),
         escape(r.reason)] for r in rows if r.status != "finished"], numeric={3})


def _findings(out: Path) -> str:
    records = read_jsonl(out / "memory" / "findings.jsonl").records
    return _table(["Id", "Kind", "Confidence", "Claim", "Author", "Evidence rows"], [
        [f"<code>{escape(str(f.get('id', '')))}</code>", escape(str(f.get("kind", ""))),
         escape(str(f.get("confidence", ""))), escape(str(f.get("claim", ""))),
         escape(str(f.get("author", ""))), str(len(f.get("evidence_row_hashes", []))),
         ] for f in records], numeric={5})


def _decisions(out: Path) -> str:
    rows = []
    for d in DecisionLog(out / "decisions.jsonl").entries():
        choice = d.data.get("choice")
        choice_text = choice if isinstance(choice, str) else json.dumps(choice, sort_keys=True)
        rows.append([
            str(d.seq), escape(d.state), escape(d.actor),
            escape(d.verdict.token if d.verdict else ""), escape(d.reason),
            escape((choice_text or "")[:160]), escape(d.model or ""),
            escape(", ".join(d.cites_findings)),
        ])
    return _table(["#", "State", "Actor", "Verdict", "Reason", "Choice", "Model", "Cites"],
                  rows, numeric={0})


def render(out_dir: str | Path, rows_path: str | Path | None = None,
           refresh_s: int | None = 30) -> str:
    out = Path(out_dir)
    state = _json(out / "state.json")
    report = _json(out / "report.json")
    all_rows = load_rows(Path(rows_path) if rows_path else out / "rows.jsonl").rows
    latest = sorted(latest_by_spec(all_rows).values(), key=lambda r: (r.spec.name, r.spec_hash))
    opened = set(report.get("row_hashes", []))
    refresh = f'<meta http-equiv="refresh" content="{refresh_s}">' if refresh_s else ""
    return "".join([
        f'<!doctype html><html><head><meta charset="utf-8">{refresh}',
        f"<title>clharness agent results</title><style>{STYLE}</style></head><body>",
        "<h1>Continual-learning agent: results</h1>",
        f'<p class="muted">Read-only view of <code>{escape(str(out))}</code>.</p>',
        "<h2>Status</h2>", _summary(state, out),
        "<h2>Report (frozen set)</h2>", _report(report),
        f"<h2>Runs ({len(latest)})</h2>", _runs(latest, opened),
        "<h2>Failed and invalid runs</h2>", _failures(all_rows),
        "<h2>Findings</h2>", _findings(out),
        "<h2>Decisions</h2>", _decisions(out),
        "</body></html>",
    ])
