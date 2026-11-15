"""Real mode's replicate phase follows the pre-registered design (PREREG 6.3), blinded."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from clharness import spec as a1
from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.loop import DEFAULT_PREREG, build_real, build_sim, main
from clharness.agent.readonly import ReadOnlyExecutor
from clharness.agent.roles import FakeClock
from clharness.agent.states import HALTED, REPORT

pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
AGENT_ROWS = REPO / "results" / "agent" / "rows.jsonl"


def _a1_rows(path: Path) -> list[a1.ResultRow]:
    return [a1.ResultRow.model_validate(json.loads(x)) for x in path.read_text().splitlines()
            if x.strip()]


def _copy(tmp_path: Path, extra: list[a1.ResultRow] = ()) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text(AGENT_ROWS.read_text() + "".join(r.model_dump_json() + "\n" for r in extra))
    return path


def _decisions(out: Path) -> list[c.Decision]:
    return c.DecisionLog(out / "decisions.jsonl").entries()


def _no_report(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the report half was opened")
    from clharness.agent import conductor

    monkeypatch.setattr(conductor, "mint_freeze_token", refuse)
    monkeypatch.setattr(conductor, "to_report_view", refuse)
    monkeypatch.setattr(ReadOnlyExecutor, "submit", lambda self, specs: pytest.fail("submit"))


def test_real_mode_replicate_proposes_exactly_the_prereg_set(tmp_path, monkeypatch):
    from clharness.analysis.design import replication_specs

    rows, out = _copy(tmp_path), tmp_path / "out"
    before = rows.read_bytes()
    plan = replication_specs(_a1_rows(rows))
    want = [s.spec_hash() for s in plan["specs"]]
    assert want  # the committed rows promote something to replicate
    _no_report(monkeypatch)
    conductor = build_real(out, c.load_prereg(DEFAULT_PREREG), rows, compute_cap_usd=700.0,
                           clock=FakeClock())
    state = conductor.run(stop_before=REPORT)
    assert state.state == REPORT
    assert rows.read_bytes() == before and not (out / "report.json").exists()

    [fallback] = [d for d in _decisions(out) if d.state == "replicate/fallback_plan"]
    assert fallback.data["proposed_spec_hashes"] == want
    assert list(fallback.proposal.spec_hashes()) == want
    assert [d["config"] for d in fallback.data["ranking"]] == [
        d["config"] for d in plan["ranking"]]
    assert fallback.data["promoted"] == plan["promoted"]
    assert fallback.data["halves_read"] == ["selection"]
    [plan_step] = [d for d in _decisions(out) if d.state == "replicate/plan"]
    assert plan_step.data["choice"] == "use the pre-registered replication set"

    launches = [d for d in _decisions(out) if d.state.endswith("/launch")]
    assert all(d.data["choice"]["submitted"] == 0 for d in launches)
    proposed = [json.loads(x) for x in (out / "proposed_specs.jsonl").read_text().splitlines()]
    assert sorted(p["spec_hash"] for p in proposed if p["phase"] == "replicate") == sorted(want)
    # launched elsewhere but no row yet: read-only mode does not repeat the phase
    [advance] = [d for d in _decisions(out) if d.state == "replicate/advance"]
    assert advance.data["choice"] == "go to combine"


def test_blinding_hides_the_report_half_and_keeps_the_ranking():
    from clharness.agent.prereg_plan import blind, blinded_rows
    from clharness.analysis.design import replication_specs

    rows = _a1_rows(AGENT_ROWS)
    blinded = blinded_rows(AGENT_ROWS)
    assert len(blinded) == len(rows)
    for row in blinded:
        assert math.isnan(row.retention.report.mean_delta)
        assert all(math.isnan(t.delta) for t in row.retention.report.per_task.values())
        assert all(math.isnan(b.report_mean_delta) for b in row.retention.boundaries)
        assert not row.retention.report_only
    assert blind(rows[0]).retention.selection == rows[0].retention.selection
    open_, closed = replication_specs(rows), replication_specs(blinded)
    assert [s.spec_hash() for s in open_["specs"]] == [s.spec_hash() for s in closed["specs"]]
    for key in ("ranking", "selection"):  # by JSON, so NaN == NaN (the 'vs_reference' column)
        assert json.dumps(open_[key], default=str) == json.dumps(closed[key], default=str)


def _finished_replications(rows: list[a1.ResultRow]) -> list[a1.ResultRow]:
    """A finished row for every replication spec: its seed-0 row with the new spec."""
    from clharness.analysis.design import replication_specs

    by_hash = {r.spec_hash: r for r in rows}
    out = []
    for spec in replication_specs(rows)["specs"]:
        parent = by_hash[spec.parent_spec_hash]
        out.append(a1.ResultRow.model_validate(
            {**parent.model_dump(mode="json"), "spec": spec.model_dump(mode="json"),
             "spec_hash": spec.spec_hash()}))
    return out


def test_replicate_with_every_run_finished_moves_on_without_repeating(tmp_path, monkeypatch):
    from clharness.agent.prereg_plan import replicate_plan

    done = _finished_replications(_a1_rows(AGENT_ROWS))
    rows, out = _copy(tmp_path, done), tmp_path / "out"
    specs, data = replicate_plan(rows)
    assert data["already_finished"] == data["planned"] == len(done)
    assert sorted(p.spec.spec_hash() for p in specs) == sorted(r.spec_hash for r in done)
    _no_report(monkeypatch)
    state = build_real(out, c.load_prereg(DEFAULT_PREREG), rows, compute_cap_usd=700.0,
                       clock=FakeClock()).run(stop_before=REPORT)
    assert state.state == REPORT and state.state != HALTED
    [gate] = [d for d in _decisions(out) if d.state == "replicate/gate"]
    assert gate.reason == "nothing new to run in this phase: every spec already has a row"
    assert all(": duplicate: " in x for x in gate.data["choice"]["dropped"])
    states = [d.state for d in _decisions(out)]
    assert "replicate/launch" not in states and states.count("replicate/plan") == 1
    assert states[states.index("replicate/gate") + 1] == "combine/plan"


def test_the_simulator_keeps_its_planner(tmp_path):
    conductor = build_sim(tmp_path, 0, 700.0, clock=FakeClock())
    assert conductor.replicate_plan is None
    conductor.run()
    [fallback] = [d for d in _decisions(tmp_path) if d.state == "replicate/fallback_plan"]
    assert fallback.reason == "stub: deterministic replicate plan"
    assert "ranking" not in fallback.data


def test_thresholds_draft_transcribes_prereg_and_is_the_default(tmp_path, monkeypatch):
    from clharness.agent import loop

    assert DEFAULT_PREREG == REPO / "configs" / "agent" / "thresholds.yaml"
    text = DEFAULT_PREREG.read_text()
    assert text.splitlines()[0] == "# DRAFT: himani to confirm."
    p = c.load_prereg(DEFAULT_PREREG)
    assert p.primary_metric == "mean_delta"
    assert p.noise_sd == pytest.approx(0.0115)  # PREREG 0.4: 1.15 points
    assert p.decision_multiplier == 2.0 and p.decision_threshold == pytest.approx(0.023)
    assert p.acquisition_floor_delta == pytest.approx(0.05)  # PREREG 0.2/0.4: margin 1.05
    assert p.collapse_share_limit == pytest.approx(0.9)  # PREREG 8, A9
    assert p.screen_seeds == (0,) and p.replicate_seeds == (0, 1, 2)
    for key in ("noise_sd", "decision_multiplier", "acquisition_floor_delta",
                "collapse_share_limit", "screen_seeds", "replicate_seeds", "primary_metric"):
        line = next(i for i, x in enumerate(text.splitlines()) if x.strip().startswith(key))
        assert "PREREG" in "\n".join(text.splitlines()[max(0, line - 4):line])

    seen = {}

    def fake_build_real(out, prereg, rows, **kw):
        seen["prereg"] = prereg
        raise ValueError("stop here")

    monkeypatch.setattr(loop, "build_real", fake_build_real)
    assert main(["--executor", "real", "--out", str(tmp_path / "o"),
                 "--compute-cap-usd", "80"]) == 2
    assert seen["prereg"] == p


def test_out_inside_the_explore_directory_is_refused(tmp_path):
    rows = _copy(tmp_path)
    inbox = tmp_path / "explore" / "literature.jsonl"
    with pytest.raises(ValueError, match="explore directory"):
        build_real(tmp_path / "explore" / "loop", sim.default_prereg(), rows,
                   compute_cap_usd=80.0, explore_inbox=inbox)
    build_real(tmp_path / "loop", sim.default_prereg(), rows, compute_cap_usd=80.0,
               explore_inbox=inbox)  # beside it is fine
