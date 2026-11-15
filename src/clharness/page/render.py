"""Render an agent output directory as one HTML page.

Reads `rows.jsonl`, `state.json`, `decisions.jsonl`, `memory/*.jsonl`, `report.json`, the
Scout's `claim_cards.jsonl` and `candidates/*.json`, and `veto.jsonl` (candidate reviews). It
computes no statistic: every number shown is in a row or in `report.json`. The report half of a
row stays sealed until the row appears in `report.json`, which only exists after FREEZE.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any

from clharness.agent.agents import candidate_queue
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


def _safe_url(url: str) -> str:
    """A link target from the web is only used if it is plain http(s); anything else becomes #."""
    return url if str(url).lower().startswith(("http://", "https://")) else "#"


def _arm(spec: Any) -> str:
    """The arm a run belongs to. An optimizer override is part of the arm: no mitigation at a
    lower learning rate is a control of its own (PREREG rule 4), not the 1e-4 baseline."""
    name = arm_of(spec)
    if spec.optim:
        name += " @ " + ", ".join(
            f"{k}={v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}={v}"
            for k, v in sorted(spec.optim.items()))
    return name


def _own_boundary(rows_file: Path) -> dict[str, dict[str, float]]:
    """spec hash -> {domain: held-out perplexity right after that domain's training, / base}.

    This is the acquisition measure the pre-registration gates on. The end-of-stream ratio mixes
    in what later domains made the model forget, so it is not shown. Read from the row file
    directly, because the agent's row view carries only the end-of-stream values."""
    from clharness import spec as a1

    out: dict[str, dict[str, float]] = {}
    if not rows_file.exists():
        return out
    for line in rows_file.read_text(encoding="utf-8").splitlines():
        try:
            row = a1.ResultRow.model_validate_json(line)
        except ValueError:
            continue
        if row.acquisition is not None:  # a later line for the same spec wins
            out[row.spec_hash] = dict(row.acquisition.own_boundary_ppl_ratio or {})
    return out


def _runs(latest: list[ResultRow], opened: set[str], own: dict[str, dict[str, float]]) -> str:
    domains: list[str] = []
    for r in latest:  # columns in stream order
        domains += [d.name for d in r.spec.stream if d.name not in domains]
    rows = []
    for r in latest:
        report = (_num(r.retention_report.mean_delta)
                  if r.row_hash in opened and r.retention_report else '<span class="muted">sealed</span>')
        acq = own.get(r.spec_hash, {})
        rows.append([
            escape(r.spec.name), escape(_arm(r.spec)), str(r.spec.seed),
            f'<span class="tag">{escape(r.status)}</span>', str(r.attempt), escape(r.backend),
            _num(r.retention_selection.mean_delta if r.retention_selection else None), report,
            *[_num(acq.get(d), ".3f") for d in domains], _num(r.cost.usd, ".2f"),
        ])
    n = len(domains)
    note = ('<p class="muted">Perplexity columns: held-out perplexity on a domain right after '
            "training on it, divided by the base model's. Below 1 means the domain was learned; "
            "above 1 means training on it made the model worse at it.</p>")
    return note + _table(
        ["Run", "Arm", "Seed", "Status", "Attempt", "Backend", "Selection Δ", "Report Δ",
         *[f"{d} ppl" for d in domains], "USD"], rows, numeric={2, 4, 6, 7, *range(8, 9 + n)})


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


def _claim_cards(out: Path) -> str:
    records = read_jsonl(out / "claim_cards.jsonl").records
    return _table(["Id", "Claim", "Test arm", "Control arm", "Predicted", "Status"], [
        [f"<code>{escape(str(c.get('id', '')))}</code>", escape(str(c.get("claim", ""))),
         f"<code>{escape(json.dumps(c.get('test_arm', {}), sort_keys=True))}</code>"
         + (f" lr={escape(str(c['test_lr']))}" if c.get("test_lr") is not None else ""),
         f"<code>{escape(json.dumps(c.get('control_arm', {}), sort_keys=True))}</code>"
         + (f" lr={escape(str(c['control_lr']))}" if c.get("control_lr") is not None else ""),
         escape(f"{c.get('prediction', {}).get('direction', 0):+d} "
                f"~{c.get('prediction', {}).get('size_points', '')} pts"),
         escape(f"repeats PREREG claim {c['duplicate_of']}" if c.get("duplicate_of") is not None
                else str(c.get("status", ""))),
         ] for c in records])


def _candidates(out: Path) -> str:
    queue = candidate_queue(out / "candidates", out / "veto.jsonl")
    cls = {"approved": "ok", "rejected": "bad", "pending": "muted"}
    note = ('<p class="muted">Review: append <code>{"kind": "approve_candidate" | '
            '"reject_candidate", "target": "&lt;name&gt;", ...}</code> to '
            '<code>veto.jsonl</code>. The loop does not wait.</p>')
    return note + _table(["#", "Candidate", "Status", "What it does", "Sources", "File"], [
        [str(c["rank"]), escape(c["name"]),
         f'<span class="{cls[c["status"]]}">{escape(c["status"])}</span>',
         escape(c["description"]),
         "<br>".join(f'<a href="{escape(_safe_url(s["url"]))}">{escape(s["title"][:80] or s["url"])}</a>'
                     for s in c["sources"]),
         f"<code>{escape(c['file'])}</code>"] for c in queue], numeric={0})


_URL = re.compile(r"https?://[^\s)\]<>'\"]+")


def _explore_dir(path: Path) -> str:
    records = read_jsonl(path / "hypotheses.jsonl").records
    searches = []
    pending = _json(path / "next_search.json")
    if pending.get("query"):
        searches.append([escape(str(pending["query"])),
                         "run" if pending.get("used") else "not run yet", ""])
    for d in DecisionLog(path / "decisions.jsonl").entries():
        if d.key.startswith("explore/search"):
            searches.append([escape(str(d.data.get("query", ""))), "run",
                             ", ".join(f'<a href="{escape(_safe_url(str(u)))}">link</a>'
                                       for u in d.data.get("urls") or [])])
    if not records and not searches:
        return ""
    rows = []
    for h in records:
        source = str(h.get("source") or "")
        match = _URL.search(source)
        src = (f'<a href="{escape(_safe_url(match.group(0)))}">{escape(match.group(0)[:60])}</a>'
               if match else "")
        pw = h.get("prior_work")
        pw = pw if isinstance(pw, dict) else {}
        if pw.get("error"):
            prior = f'Exa error: {escape(str(pw["error"])[:120])}'
        elif pw:
            cites = " ".join(
                f'<a href="{escape(_safe_url(str(c.get("url", ""))))}">[{i + 1}]</a>'
                for i, c in enumerate(pw.get("citations") or []))
            prior = escape(str(pw.get("answer", ""))[:300]) + " " + cites
        else:
            prior = '<span class="muted">not asked</span>'
        verdict = escape(str(h.get("status", "")))
        effect, threshold = h.get("effect_points"), h.get("threshold_points")
        if effect is not None and threshold is not None:
            verdict += f' {_num(effect, "+.2f")} / {_num(threshold, ".2f")} pts'
        own = h.get("new_technique")
        own = own if isinstance(own, dict) else {}
        rows.append([f"<code>{escape(str(h.get('id', '')))}</code>", verdict,
                     escape(str(h.get("claim", ""))[:300]),
                     escape(str(h.get("origin", ""))) + (" " + src if src else ""),
                     escape(str(own.get("title", ""))[:80]), prior])
    search_html = ("<h3>Searches the agent chose</h3>" +
                   _table(["Query", "Ran through Exa", "URLs"], searches)) if searches else ""
    return (f"<h2>Explore hypotheses — <code>{escape(path.name)}</code></h2>" +
            _table(["Id", "Verdict (effect / threshold)", "Claim", "Origin / source",
                    "Agent technique", "Prior work (Exa answer)"], rows) +
            search_html)


def _explore(out: Path) -> str:
    """Explore-loop evidence: hypotheses with the source and the Exa prior-work check, and the
    searches the model chose itself. Covers the rendered dir and each `explore-*` subdir.
    Read-only."""
    if not out.is_dir():
        return ""
    return "".join(_explore_dir(path) for path in
                   [out, *(p for p in sorted(out.iterdir()) if p.is_dir()
                           and p.name.startswith("explore-"))] if path.is_dir())


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
    rows_file = Path(rows_path) if rows_path else out / "rows.jsonl"
    all_rows = load_rows(rows_file).rows
    latest = sorted(latest_by_spec(all_rows).values(), key=lambda r: (r.spec.name, r.spec_hash))
    opened = set(report.get("row_hashes", []))
    refresh = f'<meta http-equiv="refresh" content="{refresh_s}">' if refresh_s else ""
    return "".join([
        f'<!doctype html><html><head><meta charset="utf-8">{refresh}',
        f"<title>Remember: results</title><style>{STYLE}</style></head><body>",
        "<h1>Remember: continual-learning results</h1>",
        f'<p class="muted">Read-only view of <code>{escape(str(out))}</code>.</p>',
        "<h2>Status</h2>", _summary(state, out),
        "<h2>Report (frozen set)</h2>", _report(report),
        f"<h2>Runs ({len(latest)})</h2>", _runs(latest, opened, _own_boundary(rows_file)),
        "<h2>Failed and invalid runs</h2>", _failures(all_rows),
        "<h2>Findings</h2>", _findings(out),
        "<h2>Claim cards (exploratory tests)</h2>", _claim_cards(out),
        "<h2>Candidate review queue</h2>", _candidates(out),
        _explore(out),
        "<h2>Decisions</h2>", _decisions(out),
        "</body></html>",
    ])
