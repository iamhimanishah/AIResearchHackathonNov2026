"""Kernel model access and configuration: the fake, the budget guard, the call log, the config."""

from __future__ import annotations

import sys
import types

import pytest
import yaml
from pydantic import BaseModel

from clharness.agent import config as cfg
from clharness.agent import llm
from clharness.agent.contracts import read_jsonl

PRICES = {"fake": cfg.ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)}


class Answer(BaseModel):
    choice: str
    score: float


def _guard(tmp_path, fake, cap=1.0, **kw) -> llm.BudgetedLLM:
    return llm.BudgetedLLM(fake, cap_usd=cap, log_path=tmp_path / "llm_calls.jsonl", prices=PRICES,
                           max_output_tokens=1000, **kw)


# ---- FakeLLM --------------------------------------------------------------------------------


def test_fake_llm_follows_its_script_and_records_full_prompts():
    fake = llm.FakeLLM()
    fake.queue("planner", {"choice": "a", "score": 1}, Answer(choice="b", score=2.0))
    fake.on("analyst", lambda agent, system, user, schema: {"choice": user[-1], "score": 0})
    first, usage = fake.complete("planner", "SYSTEM", "USER one", Answer)
    second, _ = fake.complete("planner", "SYSTEM", "USER two", Answer)
    third, _ = fake.complete("analyst", "S", "xyz", Answer)
    assert (first.choice, second.choice, third.choice) == ("a", "b", "z")
    assert usage.model == "fake" and usage.input_tokens > 0
    assert [c.agent for c in fake.calls] == ["planner", "planner", "analyst"]
    assert fake.prompts("planner") == ["SYSTEM\nUSER one", "SYSTEM\nUSER two"]
    assert isinstance(fake, llm.LLMClient)


def test_fake_llm_can_return_garbage_raise_or_run_dry():
    fake = llm.FakeLLM().queue("planner", llm.GARBAGE, {"choice": 1}, RuntimeError("boom"))
    with pytest.raises(llm.InvalidOutput) as caught:
        fake.complete("planner", "s", "u", Answer)
    assert caught.value.raw == llm.GARBAGE and caught.value.usage.input_tokens > 0
    with pytest.raises(llm.InvalidOutput):
        fake.complete("planner", "s", "u", Answer)  # valid JSON, wrong shape
    with pytest.raises(RuntimeError):
        fake.complete("planner", "s", "u", Answer)
    with pytest.raises(llm.LLMError):
        fake.complete("planner", "s", "u", Answer)  # nothing scripted
    assert len(fake.calls) == 4


def test_parse_structured_accepts_a_fenced_block():
    parsed = llm.parse_structured('```json\n{"choice": "a", "score": 1}\n```', Answer)
    assert parsed == Answer(choice="a", score=1.0)


# ---- BudgetedLLM ----------------------------------------------------------------------------


def test_budget_cap_blocks_before_the_call(tmp_path):
    fake = llm.FakeLLM().queue("planner", {"choice": "a", "score": 1})
    guard = _guard(tmp_path, fake, cap=0.003)  # one attempt is estimated at about 0.004
    assert guard.estimate_usd("planner", "s", "u") > 0.003
    with pytest.raises(llm.BudgetExceeded):
        guard.complete("planner", "s", "u", Answer)
    assert fake.calls == [], "the model must not be called once the cap would be passed"
    assert not (tmp_path / "llm_calls.jsonl").exists() and guard.spent_usd == 0.0


def test_budget_counts_spending_and_survives_a_restart(tmp_path):
    fake = llm.FakeLLM(output_tokens=1000)
    fake.on("planner", lambda *a: {"choice": "a", "score": 1})
    guard = _guard(tmp_path, fake, cap=0.0101)
    guard.complete("planner", "s", "u", Answer)
    guard.complete("planner", "s", "u", Answer)
    assert guard.spent_usd == pytest.approx(2 * (1 * 1.0 + 1000 * 4.0) / 1e6)
    with pytest.raises(llm.BudgetExceeded):
        guard.complete("planner", "s", "u", Answer)
    assert len(fake.calls) == 2
    restarted = _guard(tmp_path, fake, cap=0.0101)
    assert restarted.spent_usd == pytest.approx(guard.spent_usd)
    with pytest.raises(llm.BudgetExceeded):
        restarted.complete("planner", "s", "u", Answer)
    assert len(fake.calls) == 2 and restarted.remaining_usd < 0.0101


def test_invalid_output_is_retried_once_and_every_attempt_is_logged(tmp_path):
    fake = llm.FakeLLM().queue("planner", llm.GARBAGE, {"choice": "a", "score": 1})
    guard = _guard(tmp_path, fake)
    answer, usage = guard.complete("planner", "SYS", "USER", Answer)
    assert answer.choice == "a" and usage.attempts == 2 and usage.cost_usd > 0
    assert len(fake.calls) == 2 and "did not match the output schema" in fake.calls[1].user
    records = read_jsonl(tmp_path / "llm_calls.jsonl").records
    assert [(r["agent"], r["ok"], r["attempt"]) for r in records] == [
        ("planner", False, 1), ("planner", True, 2)]
    assert records[0]["prompt_hash"] == llm.prompt_hash("SYS", "USER")
    assert records[1]["prompt_hash"] == usage.prompt_hash != records[0]["prompt_hash"]
    assert set(records[0]) >= {"ts", "model", "input_tokens", "output_tokens", "cost_usd"}
    assert guard.spent_usd == pytest.approx(sum(r["cost_usd"] for r in records))


def test_invalid_output_twice_raises_after_exactly_two_calls(tmp_path):
    fake = llm.FakeLLM().queue("planner", llm.GARBAGE, llm.GARBAGE, {"choice": "a", "score": 1})
    guard = _guard(tmp_path, fake)
    with pytest.raises(llm.InvalidOutput):
        guard.complete("planner", "s", "u", Answer)
    assert len(fake.calls) == 2
    assert [r["ok"] for r in read_jsonl(tmp_path / "llm_calls.jsonl").records] == [False, False]


def test_provider_errors_become_one_error_family_and_are_logged(tmp_path):
    fake = llm.FakeLLM().queue("planner", RuntimeError("socket closed"))
    guard = _guard(tmp_path, fake)
    with pytest.raises(llm.LLMUnavailable):
        guard.complete("planner", "s", "u", Answer)
    record = read_jsonl(tmp_path / "llm_calls.jsonl").records[0]
    assert record["ok"] is False and "socket closed" in record["error"] and len(fake.calls) == 1


def test_an_unpriced_model_is_refused_before_the_call(tmp_path):
    fake = llm.FakeLLM(model="mystery").queue("planner", {"choice": "a", "score": 1})
    with pytest.raises(llm.ModelNotConfigured):
        _guard(tmp_path, fake).complete("planner", "s", "u", Answer)
    assert fake.calls == []


# ---- OpenAIClient (never touches the network here) -------------------------------------------


class _FakeSDK:
    def __init__(self, content: str) -> None:
        self.requests: list[dict] = []
        message = types.SimpleNamespace(content=content)
        self._response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._response


def test_openai_client_shapes_the_request_and_parses_the_answer():
    sdk = _FakeSDK('{"choice": "a", "score": 0.5}')
    client = llm.OpenAIClient({"planner": "some-model", "analyst": None}, client=sdk)
    answer, usage = client.complete("planner", "SYS", "USER", Answer)
    assert answer == Answer(choice="a", score=0.5)
    assert (usage.model, usage.input_tokens, usage.output_tokens) == ("some-model", 11, 7)
    request = sdk.requests[0]
    assert request["model"] == "some-model"
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["response_format"]["json_schema"]["name"] == "Answer"
    with pytest.raises(llm.ModelNotConfigured):
        client.complete("analyst", "s", "u", Answer)
    with pytest.raises(llm.InvalidOutput):
        llm.OpenAIClient({"planner": "m"}, client=_FakeSDK("nope")).complete("planner", "s", "u",
                                                                             Answer)


def test_openai_sdk_is_imported_lazily_with_a_clear_error(monkeypatch):
    assert "openai" not in sys.modules, "importing the agent package must not import a client"
    monkeypatch.setitem(sys.modules, "openai", None)  # makes `import openai` fail
    client = llm.OpenAIClient({"planner": "some-model"})
    with pytest.raises(llm.ModelNotConfigured, match="not installed"):
        client.complete("planner", "s", "u", Answer)


# ---- AgentConfig ----------------------------------------------------------------------------


def test_config_for_tests_is_complete_and_models_are_unset(tmp_path):
    conf = cfg.config_for_tests(tmp_path)
    assert conf.models == {"planner": None, "analyst": None, "scout": None, "reporter": None}
    assert conf.max_gate_rejections == 2 and conf.autonomy == "veto"
    assert conf.spendable_compute_usd == pytest.approx(170.0)
    assert conf.path("decisions") == tmp_path.resolve() / "results/agent/decisions.jsonl"
    assert conf.memory_path("findings").name == "findings.jsonl"
    assert conf.path("cards_dir").name == "cards"
    with pytest.raises(cfg.ConfigError, match="planner, analyst, scout, reporter"):
        conf.require_models()
    with pytest.raises(cfg.ConfigError):
        conf.path("catalog")


def test_config_requires_what_a_run_spends_and_forbids_unknown_keys(tmp_path):
    raw = cfg.config_for_tests(tmp_path).model_dump(mode="json")
    for key in ("compute_budget_usd", "model_budget_usd", "deadline", "autonomy", "executor",
                "scout_enabled", "phase_timeout_minutes", "compute_reserve_fraction"):
        with pytest.raises(ValueError):
            cfg.AgentConfig.model_validate({k: v for k, v in raw.items() if k != key})
    for bad in ({"typo_key": 1}, {"models": {"critic": "m"}}, {"autonomy": "approval"},
                {"phase_timeout_minutes": {"screen": 1.0}}, {"compute_reserve_fraction": 1.0}):
        with pytest.raises(ValueError):
            cfg.AgentConfig.model_validate({**raw, **bad})


def test_config_loads_from_yaml_and_parts_validate_their_own_settings(tmp_path):
    class GateSettings(BaseModel):
        max_specs: int = 16

    raw = cfg.config_for_tests(tmp_path).model_dump(mode="json")
    raw["models"] = {"planner": "model-a", "analyst": "model-a"}
    raw["prices"]["model-a"] = {"input_per_mtok": 2.0, "output_per_mtok": 8.0}
    raw["parts"] = {"gatekeeper": {"max_specs": 12}, "page": {"max_specs": "many"}}
    raw["deadline"] = "2026-11-15T07:00:00-08:00"
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(raw))
    conf = cfg.load_agent_config(path, root=tmp_path / "elsewhere")
    assert conf.root.endswith("elsewhere") and conf.deadline.utcoffset() is not None
    assert conf.require_models(("planner", "analyst")) == {"planner": "model-a",
                                                           "analyst": "model-a"}
    with pytest.raises(cfg.ConfigError, match="scout"):
        conf.require_models()
    assert conf.part("gatekeeper", GateSettings).max_specs == 12
    assert conf.part("operator", GateSettings).max_specs == 16
    with pytest.raises(cfg.ConfigError):
        conf.part("page", GateSettings)
