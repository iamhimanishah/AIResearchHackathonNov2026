"""The results page: shows what the files say, keeps the report half sealed until FREEZE."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from clharness.agent import contracts as c
from clharness.agent.loop import build_sim
from clharness.agent.states import DONE
from clharness.page import render
from clharness.page.__main__ import make_server


@pytest.fixture(scope="module")
def done_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("done")
    assert build_sim(out, seed=0, compute_cap_usd=700.0).run().state == DONE
    return out


def test_finished_run_shows_report_reference_and_findings(done_dir):
    html = render(done_dir)
    report = json.loads((done_dir / "report.json").read_text())
    assert report["reference_arm"] in report["arms"]
    assert report["arms"][report["reference_arm"]]["report_vs_reference"] == 0.0
    for arm, v in report["arms"].items():
        assert arm in html and format(v["report_mean_delta"], "+.4f") in html
    assert "intact" in html and "BROKEN" not in html
    for f in c.read_jsonl(done_dir / "memory" / "findings.jsonl").records:
        assert f["id"] in html


def test_report_half_is_sealed_before_freeze(tmp_path):
    conductor = build_sim(tmp_path, seed=0, compute_cap_usd=700.0)
    while conductor.s.frozen_set_hash is None and conductor.s.state != DONE:
        conductor.step()
        if conductor.s.state.phase == "replicate":
            break
    rows = [r for r in c.load_rows(tmp_path / "rows.jsonl").rows if r.status == "finished"]
    assert rows and not (tmp_path / "report.json").exists()
    html = render(tmp_path)
    assert html.count("sealed") >= len(c.latest_by_spec(rows))
    for r in rows:
        report, selection = r.retention_report.mean_delta, r.retention_selection.mean_delta
        if format(report, "+.4f") != format(selection, "+.4f"):
            assert format(report, "+.4f") not in html


def test_a_tampered_log_is_shown_broken(done_dir, tmp_path):
    for name in ("state.json", "rows.jsonl"):
        (tmp_path / name).write_bytes((done_dir / name).read_bytes())
    lines = (done_dir / "decisions.jsonl").read_text().splitlines()
    lines[3] = lines[3].replace('"reason":"', '"reason":"edited ', 1)
    (tmp_path / "decisions.jsonl").write_text("\n".join(lines) + "\n")
    assert "BROKEN" in render(tmp_path)


def test_text_from_files_is_escaped(tmp_path):
    finding = {"id": "F-x", "kind": "anomaly", "claim": "<script>alert(1)</script>",
               "confidence": "low", "author": "analyst", "evidence_row_hashes": []}
    c.append_jsonl(tmp_path / "memory" / "findings.jsonl", finding)
    html = render(tmp_path)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_empty_directory_renders(tmp_path):
    assert "None yet" in render(tmp_path)


def test_server_serves_get_and_refuses_writes(done_dir):
    server = make_server(done_dir, None, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as resp:
            assert resp.status == 200 and b"Decisions" in resp.read()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(urllib.request.Request(base + "/", data=b"x", method="POST"))
        assert err.value.code == 501
    finally:
        server.shutdown()
