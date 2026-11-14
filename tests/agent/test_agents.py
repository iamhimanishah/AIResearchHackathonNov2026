"""The model-backed roles (LLM Planner, LLM Analyst) and file-backed Memory, on the simulator."""

from __future__ import annotations

import re

import pytest

from clharness.agent import contracts as c
from clharness.agent.agents import AnalystDraft, DraftFinding, FileMemory, LLMAnalyst, LLMPlanner
from clharness.agent.context import scan_for_report_markers
from clharness.agent.gatekeeper import Gatekeeper
from clharness.agent.llm import FAKE_MODEL, FakeLLM, InvalidOutput
from clharness.agent.loop import build_sim
from clharness.agent.roles import FakeClock, ReadRequest, StubPlanner
from clharness.agent.states import DONE

CAP = 700.0


def _scripted(box: dict) -> FakeLLM:
    """A fake model that plans like the deterministic planner and cites one true and one false
    number per read, so the checker has something to drop."""

    def plan(agent, system, user, schema):
        phase = c.Phase(re.search(r"Current phase: (\w+)", user).group(1))
        return StubPlanner().fallback(box["c"]._plan_request(phase))

    def read(agent, system, user, schema):
        rows = [r for r in c.selection_views(box["c"]._latest()) if r.status == "finished"]
        out = []
        for row in rows[-1:]:
            value = row.retention_selection.mean_delta
            for label, cited in (("true", value), ("false", value + 0.5)):
                number = c.CitedNumber(label="mean_delta", value=cited, row_hash=row.row_hash[:12],
                                       field_path="retention_selection.mean_delta")
                out.append(DraftFinding(kind="effect", claim=f"{label} claim on {row.spec.name}",
                                        numbers=[number]))
        return AnalystDraft(findings=out)

    return FakeLLM().on("planner", plan).on("analyst", read)


def _build(out, **kw):
    box: dict = {}
    llm = kw.pop("llm", None) or _scripted(box)
    conductor = build_sim(out, 0, CAP, llm=llm, **kw)
    box["c"] = conductor
    return conductor, llm


def test_model_loop_reaches_done_and_logs_prompt_metadata(tmp_path):
    conductor, _ = _build(tmp_path)
    assert conductor.run().state == DONE
    log = c.DecisionLog(tmp_path / "decisions.jsonl")
    assert log.verify().ok
    plans = [d for d in log.entries() if d.state.endswith("/plan")]
    assert plans and all(d.model == FAKE_MODEL and d.prompt_hash and d.card_hash for d in plans)
    assert all(d.proposal is not None for d in plans)
    assert not [d for d in log.entries() if d.state.endswith("/fallback_plan")]
    reads = [d for d in log.entries() if d.state.endswith("/read")]
    assert reads and all(d.model == FAKE_MODEL for d in reads)


def test_analyst_claims_are_checked_and_stored_in_memory(tmp_path):
    conductor, _ = _build(tmp_path)
    conductor.run()
    findings = FileMemory(tmp_path / "memory").findings()
    assert findings and all(f.author == "analyst" for f in findings)
    assert all(f.claim.startswith("true") for f in findings)
    assert all(len(f.evidence_row_hashes[0]) == 64 for f in findings)
    tried = FileMemory(tmp_path / "memory").tried()
    rows = c.load_rows(tmp_path / "rows.jsonl").rows
    assert set(tried) == {r.spec_hash for r in rows}


def test_no_prompt_ever_contains_the_report_half(tmp_path):
    conductor, llm = _build(tmp_path)
    conductor.run()
    assert llm.prompts("planner") and llm.prompts("analyst")
    assert scan_for_report_markers(llm.prompts()) == []


def test_a_dead_model_falls_back_to_deterministic_plan_and_findings(tmp_path):
    conductor, _ = _build(tmp_path, llm=FakeLLM())
    assert conductor.run().state == DONE
    log = list(c.DecisionLog(tmp_path / "decisions.jsonl").entries())
    plans = [d for d in log if d.state.endswith("/plan")]
    assert plans and all(d.reason.startswith("planner failed") for d in plans)
    assert [d for d in log if d.state.endswith("/fallback_plan")]
    assert FileMemory(tmp_path / "memory").findings()  # deterministic findings were stored


class RejectFirst(Gatekeeper):
    def __init__(self, times: int) -> None:
        super().__init__()
        self.left = times

    def review(self, proposal, request):
        if self.left:
            self.left -= 1
            drop = c.Dropped(spec_hash=c.WHOLE_PROPOSAL, rule="test", reason="rejected by test")
            return c.Verdict(token="REJECT", dropped=(drop,))
        return super().review(proposal, request)


def test_one_rejection_replans_with_the_verdict(tmp_path):
    conductor, llm = _build(tmp_path, gatekeeper=RejectFirst(1))
    assert conductor.run().state == DONE
    keys = [d.key for d in c.DecisionLog(tmp_path / "decisions.jsonl").entries()]
    assert keys[:4] == ["init#0", "preflight#0", "screen/plan#0", "screen/gate#0"]
    assert "screen/fallback_plan#0" not in keys
    assert any("rejected by test" in p for p in llm.prompts("planner"))


def test_max_rejections_switch_to_the_deterministic_plan(tmp_path):
    conductor, _ = _build(tmp_path, gatekeeper=RejectFirst(2))
    assert conductor.run().state == DONE
    log = list(c.DecisionLog(tmp_path / "decisions.jsonl").entries())
    assert any(d.state == "screen/fallback_plan" for d in log)


def test_kill_and_resume_with_a_model_matches_a_clean_run(tmp_path):
    clean, _ = _build(tmp_path / "clean")
    clean.run()
    killed: set[tuple[str, str]] = set()

    def fault(state_id, point):
        if (state_id, point) not in killed and state_id.endswith(("/plan", "/read")):
            killed.add((state_id, point))
            raise KeyboardInterrupt

    clock = FakeClock()
    for _ in range(200):
        try:
            conductor, _ = _build(tmp_path / "killed", clock=clock, fault=fault)
            if conductor.run().state == DONE:
                break
        except KeyboardInterrupt:
            continue
    choices = [
        [(d.key, d.data.get("choice")) for d in c.DecisionLog(p / "decisions.jsonl").entries()]
        for p in (tmp_path / "clean", tmp_path / "killed")
    ]
    assert choices[0] == choices[1]
    assert {p for _, p in killed} == {"work_done", "logged", "saved"}


def test_planner_citing_an_unknown_finding_is_invalid(tmp_path):
    conductor, _ = _build(tmp_path)
    request = conductor._plan_request(c.Phase.SCREEN)
    proposal = StubPlanner().fallback(request).model_copy(update={"cites_findings": ("nope",)})
    planner = LLMPlanner(FakeLLM().queue("planner", proposal))
    with pytest.raises(InvalidOutput):
        planner.plan(request)


def test_analyst_drops_numbers_for_unknown_rows(tmp_path):
    conductor, _ = _build(tmp_path)
    conductor.run()
    rows = c.selection_views(conductor._latest())
    bad = c.CitedNumber(label="x", value=0.0, row_hash="deadbeefdeadbeef",
                        field_path="retention_selection.mean_delta")
    llm = FakeLLM().queue("analyst", AnalystDraft(findings=[
        DraftFinding(kind="effect", claim="made up", numbers=[bad]),
        DraftFinding(kind="anomaly", claim="no numbers"),
    ]))
    analyst = LLMAnalyst(llm, clock=FakeClock())
    request = ReadRequest(phase=c.Phase.SCREEN, prereg=conductor.prereg, new_rows=rows[:3])
    findings, meta = analyst.read(request)
    assert findings == () and len(analyst.dropped) == 2 and meta.model == FAKE_MODEL


def test_file_memory_survives_a_restart_and_refuses_report_rows(tmp_path):
    conductor, _ = _build(tmp_path / "run")
    conductor.run()
    rows = c.selection_views(conductor._latest())
    memory = FileMemory(tmp_path / "mem")
    memory.record_rows(rows)
    again = FileMemory(tmp_path / "mem")
    assert again.tried() == memory.tried() and again.failures() == memory.failures()
    with pytest.raises(TypeError):
        memory.record_rows([object()])
