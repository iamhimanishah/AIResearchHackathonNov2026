"""Scout edges (fix plan S1-S3): an Exa timeout, kept search errors, per-item validation.
No network: a fake Exa client, a fake search and a fake model."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from clharness.agent import catalog_real
from clharness.agent.agents import MAX_SCOUT_ITEMS, LLMScout, ScoutDraft
from clharness.agent.contracts import Source
from clharness.agent.llm import FakeLLM
from clharness.agent.roles import FakeClock, ScoutRequest
from clharness.agent.search import ExaSearch, FakeSearch, SearchError, source_id

Q1, O1 = "q one", "objective one"
Q2, O2 = "q two", "objective two"
A = Source(id=source_id(Q1, "https://arxiv.org/abs/2601.00001"), query=Q1,
           url="https://arxiv.org/abs/2601.00001", title="Replay ratios", snippet="s",
           retrieved_at="2026-11-14T00:00:00Z")


class _Exa:
    """A fake `exa_py.Exa`: records the keyword arguments, or raises what it is given."""

    def __init__(self, raises: Exception | None = None):
        self.raises, self.kwargs = raises, []

    def search(self, query, **kwargs):
        self.kwargs.append(kwargs)
        if self.raises:
            raise self.raises
        return SimpleNamespace(results=[{"url": "https://x.org/a", "title": "t",
                                         "highlights": ["h"]}])


def test_s1_exa_gets_a_timeout_and_a_timeout_is_a_search_error():
    client = _Exa()
    ExaSearch(client=client, clock=FakeClock()).search("q", "o")
    assert client.kwargs[0]["timeout"] == 30.0
    ExaSearch(client=client, clock=FakeClock(), timeout=5.0).search("q", "o")
    assert client.kwargs[1]["timeout"] == 5.0
    slow = ExaSearch(client=_Exa(raises=TimeoutError("read timed out")), clock=FakeClock())
    with pytest.raises(SearchError, match="TimeoutError: read timed out"):
        slow.search("q", "o")


def _claim(text: str, share: float, **kw) -> dict:
    return {"claim": text, "source_ids": [A.id],
            "test_arm": [{"name": "replay", "params": {"share": share}}],
            "prediction": {"direction": 1, "size_points": 1.0}, **kw}


def _scout(llm: FakeLLM, search: FakeSearch) -> LLMScout:
    return LLMScout(llm, search, clock=FakeClock(), queries=((Q1, O1), (Q2, O2)),
                    runnable=catalog_real.bench_problems)


def _request() -> ScoutRequest:
    return ScoutRequest(catalog=catalog_real.real_catalog(), stream=catalog_real.stream(),
                        seeds=(0, 1), design_arms=catalog_real.design_claim_arms())


class _HalfDown(FakeSearch):
    def search(self, query, objective):
        if query == Q2:
            self.calls.append((query, objective))
            raise SearchError("TimeoutError: q two timed out")
        return super().search(query, objective)


def test_s2_partial_search_errors_are_kept_and_logged(tmp_path):
    from clharness.agent import sim
    from clharness.agent.contracts import DecisionLog
    from clharness.agent.loop import build_sim

    llm = FakeLLM().on("scout", lambda *a: {"claims": [_claim("Replay at 10% helps", 0.1)]})
    result = _scout(llm, _HalfDown({Q1: [A]})).scout(_request())
    assert len(result.claims) == 1
    assert result.searches == 2 and result.search_errors == ("TimeoutError: q two timed out",)

    sim_scout = LLMScout(FakeLLM().on("scout", lambda *a: {"skip_reason": "nothing new"}),
                         _HalfDown({Q1: [A]}), clock=FakeClock(), queries=((Q1, O1), (Q2, O2)))
    conductor = build_sim(tmp_path, 0, 700.0, clock=FakeClock(), scout=sim_scout)
    conductor.run()
    [scout] = [d for d in DecisionLog(tmp_path / "decisions.jsonl").entries() if d.state == "scout"]
    assert scout.data["searches"] == 2
    assert scout.data["search_errors"] == ["TimeoutError: q two timed out"]
    assert sim.default_prereg()  # the loop finished on the simulator


def test_s2_a_skip_still_records_the_search_errors():
    result = _scout(FakeLLM(), FakeSearch(fail=True)).scout(_request())
    assert result.skipped_reason.startswith("search returned nothing")
    assert result.searches == 2 and len(result.search_errors) == 2


def test_s3_one_bad_item_rejects_only_itself_and_the_list_is_cut_to_five():
    claims = [_claim(f"Replay at {s:g} helps", s) for s in (0.05, 0.1, 0.15, 0.2, 0.3, 0.35)]
    claims.insert(2, {"claim": "no prediction and no sources"})  # invalid: missing fields
    assert len(claims) == 7
    candidates = [{"name": f"cand-{i}", "description": "d", "mechanism": "m",
                   "why_not_in_catalog": "w", "expected_effect": "e",
                   "est_cost_multiplier": 1.1, "source_ids": [A.id]} for i in range(6)]
    candidates[0]["est_cost_multiplier"] = -1  # invalid: must be > 0
    llm = FakeLLM().on("scout", lambda *a: {"claims": claims, "candidates": candidates})
    result = _scout(llm, FakeSearch({Q1: [A]})).scout(_request())
    assert len(result.claims) == MAX_SCOUT_ITEMS == 5
    assert [c.claim for c in result.claims] == [c["claim"] for c in claims if "prediction" in c][:5]
    assert [c.name for c in result.candidates] == [f"cand-{i}" for i in range(1, 6)]
    joined = "\n".join(result.rejected)
    assert "claim 3: invalid" in joined and "claim 7: beyond the 5-item limit" in joined
    assert "candidate 1: invalid" in joined
    assert result.skipped_reason is None


def test_s3_the_model_facing_schema_is_unchanged():
    schema = json.dumps(ScoutDraft.model_json_schema(), sort_keys=True)
    # sha256 of the schema before S3 (main at 4691ed4): the prompt and card hashes do not move
    assert hashlib.sha256(schema.encode()).hexdigest() == (
        "db9bdb6b0c77ee9e892dc944ea8fcf14ce1407869d4e245066a227bc0301ecad")
    assert '"maxItems": 5' in schema  # the model is still told the limit
