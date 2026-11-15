"""The explore cycle against a scripted model and a fake Modal: no network, no GPU."""

import json

import pytest

from clharness import spec as a1
from clharness.agent import explore as ex
from clharness.agent.contracts import DecisionLog
from clharness.agent.llm import FakeLLM
from clharness.analysis import design

SHA = "a" * 40
REPORT_SENTINEL = -0.7777  # planted in every report-half field; must never reach a prompt
STREAM = [a1.StreamDomain(name=n, tokens=t) for n, t in design.STREAM]


def _row(spec: a1.RunSpec, selection_points: float, legal: float = 0.78, sha: str = SHA) -> a1.ResultRow:
    def half(value: float) -> a1.HalfRetention:
        return a1.HalfRetention(per_task={"arc_easy": a1.TaskRetention(base=0.7, final=0.7 + value, delta=value, n_items=250)},
                                mean_delta=value)

    return a1.ResultRow(
        spec=spec, spec_hash=spec.spec_hash(), git_sha=sha, backend="modal", status="finished",
        retention=a1.Retention(selection=half(selection_points / 100), report=half(REPORT_SENTINEL)),
        acquisition=a1.Acquisition(ppl_ratio={"legal": legal}, mean_ppl_ratio=legal,
                                   own_boundary_ppl_ratio={"legal": legal, "code": 1.1}),
        cost=a1.Cost(gradient_tokens=20_000_000, wall_clock_s=1100.0, gpu_type="H100", est_usd=1.2))


def _spec(seed: int, techniques=(), lr=None) -> a1.RunSpec:
    return a1.RunSpec(name=f"x-s{seed}", seed=seed, bench="fast", stream=STREAM,
                      optim={"lr": lr} if lr else {}, techniques=list(techniques))


@pytest.fixture
def have(tmp_path):
    """No mitigation at 1e-4 (five seeds, about -6) and at 3e-5 (three seeds, about -1.5)."""
    rows = [_row(_spec(s), v) for s, v in enumerate([-5.8, -5.0, -5.8, -5.4, -6.6])]
    rows += [_row(_spec(s, lr=3e-5), v, legal=0.754) for s, v in enumerate([-1.6, -1.3, -1.5])]
    path = tmp_path / "have.jsonl"
    a1.write_rows(path, rows)
    return path


class FakeModal:
    """Finishes each call on its second poll, with a selection score chosen per technique set."""

    def __init__(self, score, sha: str = SHA):
        self.score, self.sha, self.calls, self.polls, self.spawned = score, sha, {}, {}, []

    def spawn(self, spec_json: str, git_sha: str, gpu: str) -> str:
        spec = a1.RunSpec.model_validate_json(spec_json)
        call_id = f"fc-{len(self.spawned)}"
        self.calls[call_id], self.polls[call_id] = spec, 0
        self.spawned.append(spec)
        return call_id

    def result(self, call_id: str) -> str | None:
        self.polls[call_id] += 1
        if self.polls[call_id] < 2:
            return None
        spec = self.calls.pop(call_id)
        return a1.row_to_line(_row(spec, *self.score(spec), sha=self.sha))

    def healthy(self) -> bool:
        return True


def _hyp(**kw) -> dict:
    base = dict(question="q?", claim="replay at a low rate forgets less", origin="surprise",
                prediction={"direction": "less_forgetting", "size_points": 0.5},
                techniques=[{"name": "replay", "params": {"share": 0.05}}], lr=3e-5,
                control="no-mitigation@lr3e-05")
    return {**base, **kw}


def _cycle(tmp_path, have, llm=None, backend=None, **kw):
    said = []
    out = ex.cycle(tmp_path / "explore", [have], llm=llm, backend=backend, git_sha=SHA,
                   sleep=lambda s: None, say=said.append, **kw)
    return out, said


def test_arms_hold_selection_numbers_only_and_the_prompt_never_sees_the_report_half(tmp_path, have):
    arms = ex.read_arms([have])
    assert {a.name for a in arms} == {"no-mitigation", "no-mitigation@lr3e-05"}
    assert ex.noise_sd_points(arms) == pytest.approx(0.59, abs=0.01)
    llm = FakeLLM().queue("explorer", {"hypotheses": [_hyp()]})
    _cycle(tmp_path, have, llm=llm)
    prompt = "\n".join(llm.prompts("explorer"))
    assert "no-mitigation@lr3e-05" in prompt and "-1.47" in prompt
    assert "7777" not in prompt and "77.77" not in prompt


def test_the_gate_refuses_what_the_bench_cannot_or_should_not_run(tmp_path, have):
    bad = [
        _hyp(claim="unknown technique", techniques=[{"name": "no_mitigation", "params": {}}]),
        _hyp(claim="rank not allowed", techniques=[{"name": "lora", "params": {"rank": 1024}}], lr=None),
        _hyp(claim="lr too high", lr=3e-4),
        _hyp(claim="control missing", control="reference"),
        _hyp(claim="distill on replay without replay",
             techniques=[{"name": "distill", "params": {"rows": "replay"}}]),
        _hyp(claim="nothing tested", techniques=[], lr=None, control="no-mitigation"),
        _hyp(claim="same as control", techniques=[], lr=3e-5),
        _hyp(claim="lora would replace the spec's lr", techniques=[{"name": "lora", "params": {"rank": 8}}]),
        _hyp(claim="fine"),
        _hyp(claim="a repeat of the fine one"),
    ]
    llm = FakeLLM().queue("explorer", {"hypotheses": bad})
    out, _ = _cycle(tmp_path, have, llm=llm, max_hypotheses=20)
    by_claim = {h.claim: h for h in out}
    assert by_claim["fine"].status == "accepted" and len(by_claim["fine"].test_spec_hashes) == 2
    reasons = {c: h.reason for c, h in by_claim.items() if h.status == "rejected"}
    assert len(reasons) == 9
    assert "unknown technique" in reasons["unknown technique"]
    assert "outside" in reasons["rank not allowed"] and "outside" in reasons["lr too high"]
    assert "not an arm" in reasons["control missing"]
    assert "needs replay" in reasons["distill on replay without replay"]
    assert "does not build" in reasons["lora would replace the spec's lr"]
    assert "already" in reasons["a repeat of the fine one"]
    assert not (tmp_path / "explore" / "hypotheses.jsonl").exists()  # a dry run saves nothing


def test_the_budget_cap_keeps_the_first_hypotheses_and_refuses_the_rest(tmp_path, have):
    many = [_hyp(claim=f"share {s}", techniques=[{"name": "replay", "params": {"share": s}}])
            for s in (0.02, 0.05, 0.1, 0.25)]
    llm = FakeLLM().queue("explorer", {"hypotheses": many})
    out, _ = _cycle(tmp_path, have, llm=llm, cap_usd=2 * 2 * ex.USD_PER_RUN + 0.01)
    assert [h.status for h in out] == ["accepted", "accepted", "rejected", "rejected"]
    assert "cap" in out[-1].reason


def test_a_full_cycle_launches_resolves_and_logs(tmp_path, have):
    props = [
        _hyp(claim="helps", techniques=[{"name": "replay", "params": {"share": 0.25}}]),
        _hyp(claim="predicted help, none found", techniques=[{"name": "replay", "params": {"share": 0.02}}]),
        _hyp(claim="predicted no difference, it hurt", prediction={"direction": "no_difference", "size_points": 0.2},
             techniques=[{"name": "wavg", "params": {"alpha": 0.5}}]),
    ]

    def score(spec):  # (selection points, legal ratio); the control sits near -1.45
        share = dict(spec.techniques[0].params).get("share")
        if share == 0.25:
            return (0.6, 0.76)
        if share == 0.02:
            return (-1.4, 0.76)
        return (-4.0, 0.95)

    backend = FakeModal(score)
    out, _ = _cycle(tmp_path, have, llm=FakeLLM().queue("explorer", {"hypotheses": props}),
                    backend=backend, go=True)
    by_claim = {h.claim: h for h in out}
    assert by_claim["helps"].status == "confirmed" and by_claim["helps"].effect_points == pytest.approx(2.05)
    assert by_claim["predicted help, none found"].status == "inconclusive"
    hurt = by_claim["predicted no difference, it hurt"]
    assert hurt.status == "refuted" and hurt.acquisition_ok is False
    assert len(backend.spawned) == 6
    assert all(s.notes.startswith(ex.NOTE_PREFIX) and s.bench == "fast" for s in backend.spawned)
    stored = ex.load_hypotheses(tmp_path / "explore" / "hypotheses.jsonl")
    assert [h.status for h in stored] == [h.status for h in out]
    log = DecisionLog(tmp_path / "explore" / "decisions.jsonl")
    assert log.verify().ok
    states = [d.state for d in log.entries()]
    assert states[0] == "explore/gate" and states[-1] == "explore/read"
    assert json.loads((tmp_path / "explore" / "rows.jsonl").read_text().splitlines()[0])["spec"]["notes"].startswith("exploratory:")


def test_a_second_cycle_does_not_repeat_a_recorded_hypothesis(tmp_path, have):
    first = FakeLLM().queue("explorer", {"hypotheses": [_hyp(claim="once")]})
    _cycle(tmp_path, have, llm=first, backend=FakeModal(lambda s: (-1.0, 0.76)), go=True)
    second = FakeLLM().queue("explorer", {"hypotheses": [_hyp(claim="again, same configuration")]})
    backend = FakeModal(lambda s: (-1.0, 0.76))
    out, _ = _cycle(tmp_path, have, llm=second, backend=backend, go=True)
    assert backend.spawned == [] and out[-1].status == "rejected"
    assert "once" in "\n".join(second.prompts("explorer"))  # the model is told what exists


def test_with_no_model_or_a_failing_one_the_fixed_list_is_used(tmp_path, have):
    out, _ = _cycle(tmp_path, have)
    assert [h.author for h in out] == ["fixed-list", "fixed-list"] and all(h.status == "accepted" for h in out)
    out, _ = _cycle(tmp_path, have, llm=FakeLLM())  # nothing queued: the provider is dead
    assert out and out[0].author == "fixed-list"
