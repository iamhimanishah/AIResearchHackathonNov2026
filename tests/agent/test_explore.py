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
    assert json.loads((tmp_path / "explore" / "rows.jsonl").read_text().splitlines()[0])["spec"]["notes"].startswith(ex.NOTE_PREFIX)


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


# -- the Scout's claim-card inbox --------------------------------------------------------------


def _card(card_id: str, **kw) -> str:
    """One inbox line as the Conductor's Scout step writes it (conductor._write_claim_cards)."""
    return json.dumps({"card_id": card_id, **_hyp(origin="literature",
                       source="Simple and Scalable Strategies https://arxiv.org/abs/2403.08763",
                       control="no_mitigation-lr3e-05", **kw)}) + "\n"


def test_claim_cards_become_literature_hypotheses_with_their_source(tmp_path, have):
    out = tmp_path / "explore"
    out.mkdir()
    (out / ex.INBOX_NAME).write_text(_card("c1") + "not json\n" + _card("c2", lr=5.0))
    said = []
    hyps = ex.cycle(out, [have], say=said.append, max_hypotheses=4)
    by_card = {h.card_id: h for h in hyps if h.card_id}
    assert by_card["c1"].status == "accepted" and by_card["c1"].author == "scout"
    assert by_card["c1"].origin == "literature" and "arxiv.org/abs/2403.08763" in by_card["c1"].source
    assert by_card["c1"].control == "no-mitigation@lr3e-05"  # the Scout's spelling, normalised
    assert by_card["c2"].status == "rejected"  # the gate applies to a card like any proposal
    assert any("skipped" in line and ":2:" in line for line in said)  # the bad line is reported
    assert sum(h.card_id is None for h in hyps) == 2  # the fixed list fills what is left


def test_a_card_is_used_once_and_cards_take_the_allowance_first(tmp_path, have):
    out = tmp_path / "explore"
    out.mkdir()
    (out / ex.INBOX_NAME).write_text(_card("c1"))
    backend = FakeModal(lambda spec: (-1.0, 0.75))
    first = ex.cycle(out, [have], backend=backend, git_sha=SHA, go=True, sleep=lambda s: None,
                     say=lambda s: None, max_hypotheses=1)
    assert [h.card_id for h in first] == ["c1"]  # allowance of one: the card, not the fixed list
    again = ex.cycle(out, [have], say=lambda s: None, max_hypotheses=4)  # inbox rewritten whole
    assert [h.card_id for h in again].count("c1") == 1


def test_control_names_from_the_scout_map_onto_the_arm_table():
    assert ex._control_name("no_mitigation") == "no-mitigation"
    assert ex._control_name("no_mitigation-lr3e-05") == "no-mitigation@lr3e-05"
    assert ex._control_name("lr_rewarm+replay") == "lr_rewarm+replay"
    assert ex._control_name("no-mitigation@lr3e-05") == "no-mitigation@lr3e-05"


# -- review fixes --------------------------------------------------------------------------------


def test_gate_refuses_a_freeze_that_leaves_nothing_to_train(tmp_path, have):
    arms = ex.read_arms([have])
    specs, why = ex.gate_one(ex.ProposedHypothesis.model_validate(_hyp(
        techniques=[{"name": "freeze", "params": {"bottom_layers": 13, "top_layers": 13}}], lr=None,
        control="no-mitigation")), "h001", arms, set())
    assert specs == [] and "no layer would be left" in why


def test_a_failed_test_run_closes_its_hypothesis_as_inconclusive(tmp_path, have):
    class Failing(FakeModal):
        def result(self, call_id):
            self.polls[call_id] += 1
            if self.polls[call_id] < 2:
                return None
            spec = self.calls.pop(call_id)
            return a1.row_to_line(a1.ResultRow(spec=spec, spec_hash=spec.spec_hash(), git_sha=SHA,
                                               status="failed", reason="ValueError: nothing to train", backend="modal"))

    out = tmp_path / "explore"
    hyps = ex.cycle(out, [have], backend=Failing(lambda s: (0, 0)), git_sha=SHA, go=True,
                    sleep=lambda s: None, say=lambda s: None, max_hypotheses=1)
    assert [h.status for h in hyps] == ["inconclusive"] and "did not finish" in hyps[0].reason
    again = ex.cycle(out, [have], resolve_only=True, say=lambda s: None)
    assert [h.status for h in again] == ["inconclusive"]


def test_the_total_cap_holds_across_cycles(tmp_path, have):
    out = tmp_path / "explore"
    backend = FakeModal(lambda spec: (-1.0, 0.75))
    kw = dict(backend=backend, git_sha=SHA, go=True, sleep=lambda s: None, say=lambda s: None)
    two_runs = 2 * ex.USD_PER_RUN
    first = ex.cycle(out, [have], max_hypotheses=1, total_cap_usd=two_runs + 0.01, **kw)
    assert sum(bool(h.test_spec_hashes) for h in first) == 1
    second = ex.cycle(out, [have], max_hypotheses=2, total_cap_usd=two_runs + 0.01, **kw)
    assert sum(bool(h.test_spec_hashes) for h in second) == 1  # nothing more fits under the total
    assert any("left under the caps" in h.reason for h in second if h.status == "rejected")


# -- agent-written techniques ----------------------------------------------------------------------

GOOD_SOURCE = '''import torch
from clharness.train.hooks import BaseHook


class Pull(BaseHook):
    """Keep the trainable weights near where they started."""

    def __init__(self, weight=0.01):
        self.weight = weight
        self.start = None

    def on_run_start(self, ctx):
        self.start = [p.detach().clone() for p in ctx.model.parameters() if p.requires_grad]
        return None

    def extra_loss(self, ctx, batch, logits, labels):
        total = logits.new_zeros(())
        for p, s in zip([q for q in ctx.model.parameters() if q.requires_grad], self.start):
            total = total + (p.float() - s.float()).pow(2).sum()
        return self.weight * total
'''


def _own(**kw) -> dict:
    base = {"new_technique": {"title": "pull to start", "source": GOOD_SOURCE, "params": {"weight": 0.01}}}
    return _hyp(techniques=[], lr=None, control="no-mitigation", **{**base, **kw})


def test_source_checker_accepts_a_plain_hook_and_refuses_dangerous_code():
    from clharness.techniques.agent_code import validate_source

    assert validate_source(GOOD_SOURCE) == []
    head = "from clharness.train.hooks import BaseHook\n"
    bad = {
        "os": "import os\n" + head + "class A(BaseHook):\n    pass\n",
        "open": head + "class A(BaseHook):\n    def on_run_start(self, ctx):\n        open('/etc/passwd')\n",
        "dunder": head + "class A(BaseHook):\n    def on_run_start(self, ctx):\n        return ctx.__class__\n",
        "save": "import torch\n" + head + "class A(BaseHook):\n    def on_run_start(self, ctx):\n        torch.save(ctx.model, 'x')\n",
        "while": head + "class A(BaseHook):\n    def on_run_start(self, ctx):\n        while True:\n            pass\n",
        "two classes": head + "class A(BaseHook):\n    pass\nclass B(BaseHook):\n    pass\n",
        "top-level call": head + "print('hi')\nclass A(BaseHook):\n    pass\n",
        "getattr": head + "class A(BaseHook):\n    def on_run_start(self, ctx):\n        return getattr(ctx, 'model')\n",
        "string dunder": head + "class A(BaseHook):\n    name = '__import__'\n",
        "paths": head + "class A(BaseHook):\n    def on_run_start(self, ctx):\n        return ctx.output_dir\n",
        "not a hook": "class A:\n    pass\n",
    }
    for name, source in bad.items():
        assert validate_source(source), name


def test_the_tiny_model_check_passes_good_code_and_catches_bad_behaviour():
    from clharness.techniques.agent_check import tiny_check

    ok = tiny_check(GOOD_SOURCE, {"weight": 0.01})
    assert ok["ok"] and ok["gradient_tokens"] == 1024
    not_finite = tiny_check(GOOD_SOURCE.replace("return self.weight * total", "return total / 0.0 * 0.0"), {})
    assert not not_finite["ok"] and "not finite" in not_finite["problems"][0]
    drops = ("from clharness.train.hooks import BaseHook\nclass Drop(BaseHook):\n"
             "    def select_batch(self, ctx, batch):\n        return batch[:1]\n")
    assert "optimizer steps" in tiny_check(drops, {})["problems"][0]
    assert tiny_check("import os\n", {})["stage"] == "static"


def test_an_agent_written_technique_is_probed_before_its_full_runs(tmp_path, have):
    backend = FakeModal(lambda spec: (-2.0, 0.78))
    out = tmp_path / "explore"
    hyps = ex.cycle(out, [have], llm=FakeLLM().queue("explorer", {"hypotheses": [_own()]}), backend=backend,
                    git_sha=SHA, go=True, sleep=lambda s: None, say=lambda s: None,
                    tiny_check=lambda source, params: {"ok": True, "stage": "tiny-model", "problems": []})
    names = [s.name for s in backend.spawned]
    assert names[0] == "explore-h001-probe" and sorted(names[1:]) == ["explore-h001-s0", "explore-h001-s1"]
    assert backend.spawned[0].stream[0].tokens == ex.PROBE_TOKENS  # the probe is the short stream
    assert all(s.technique_names() == ["agent_code"] for s in backend.spawned)
    (h,) = hyps
    assert h.checks == ["static allow-list", "tiny-model training", "four-step run on a GPU"]
    assert h.new_technique.source == GOOD_SOURCE and h.status in ("confirmed", "refuted", "inconclusive")


def test_a_failed_check_or_probe_stops_the_full_runs(tmp_path, have):
    refuse = lambda source, params: {"ok": False, "stage": "tiny-model", "problems": ["a loss was not finite"]}  # noqa: E731
    backend = FakeModal(lambda spec: (-2.0, 0.78))
    hyps = ex.cycle(tmp_path / "a", [have], llm=FakeLLM().queue("explorer", {"hypotheses": [_own()]}),
                    backend=backend, git_sha=SHA, go=True, sleep=lambda s: None, say=lambda s: None, tiny_check=refuse)
    assert hyps[0].status == "rejected" and "tiny-model check" in hyps[0].reason and backend.spawned == []

    class ProbeFails(FakeModal):
        def result(self, call_id):
            self.polls[call_id] += 1
            if self.polls[call_id] < 2:
                return None
            spec = self.calls.pop(call_id)
            return a1.row_to_line(a1.ResultRow(spec=spec, spec_hash=spec.spec_hash(), git_sha=SHA, status="failed",
                                               reason="RuntimeError: CUDA out of memory", backend="modal"))

    backend = ProbeFails(lambda spec: (0, 0))
    hyps = ex.cycle(tmp_path / "b", [have], llm=FakeLLM().queue("explorer", {"hypotheses": [_own()]}),
                    backend=backend, git_sha=SHA, go=True, sleep=lambda s: None, say=lambda s: None,
                    tiny_check=lambda source, params: {"ok": True})
    assert [s.name for s in backend.spawned] == ["explore-h001-probe"]  # no full run was paid for
    assert hyps[0].status == "rejected" and "probe run did not finish" in hyps[0].reason


def test_one_agent_written_technique_per_cycle_and_the_lora_limit(tmp_path, have):
    ok = lambda source, params: {"ok": True}  # noqa: E731
    two = [_own(claim="first"), _own(claim="second", new_technique={
        "title": "other", "source": GOOD_SOURCE.replace("Pull", "Other"), "params": {}})]
    hyps = ex.cycle(tmp_path / "e", [have], llm=FakeLLM().queue("explorer", {"hypotheses": two}),
                    say=lambda s: None, tiny_check=ok)
    assert [h.status for h in hyps] == ["accepted", "rejected"] and "one agent-written" in hyps[1].reason
    arms = ex.read_arms([have])
    wild = ex.ProposedHypothesis.model_validate(_hyp(
        techniques=[{"name": "lora", "params": {"rank": 256, "lr": 1e-3}}], lr=None, control="no-mitigation"))
    assert "diverged" in ex.gate_one(wild, "h9", arms, set())[1]
    tame = ex.ProposedHypothesis.model_validate(_hyp(
        techniques=[{"name": "lora", "params": {"rank": 256, "lr": 1.5e-4}}], lr=None, control="no-mitigation"))
    assert ex.gate_one(tame, "h9", arms, set())[0]


def test_no_grad_is_the_one_decorator_allowed():
    from clharness.techniques.agent_code import validate_source

    head = "import torch\nfrom clharness.train.hooks import BaseHook\nclass A(BaseHook):\n"
    assert validate_source(head + "    @torch.no_grad()\n    def on_step_end(self, ctx, stats):\n        return None\n") == []
    for decorator in ("@staticmethod", "@torch.jit.script", "@torch.no_grad", "@torch.enable_grad()"):
        assert validate_source(head + f"    {decorator}\n    def on_step_end(self, ctx, stats):\n        return None\n")


# ---------------------------------------------------------------------------------------------
# Exa in the loop: Scout candidates reach the proposer (A), prior work before launch (B), and the
# model chooses the next search (C)
# ---------------------------------------------------------------------------------------------

from clharness.agent.contracts import Source
from clharness.agent.search import FakeSearch


def _src(i: int, query: str = "q") -> Source:
    return Source(id=f"src-{i}", query=query, url=f"https://arxiv.org/abs/2608.{i:05d}",
                  title=f"Paper {i} <b>", snippet="spaced review lowers forgetting", published="2026-08-18",
                  retrieved_at="2026-11-15T00:00:00+00:00")


def test_scout_candidates_are_quoted_with_their_papers(tmp_path):
    result = {"candidates": [{"name": "Spaced Repetition Training", "description": "review old rows",
                              "source_ids": ["src-1"]}],
              "sources": [_src(1).model_dump(mode="json"), _src(2).model_dump(mode="json")]}
    (tmp_path / "cands").mkdir()
    (tmp_path / "cands" / "a.json").write_text(json.dumps(result))
    path = ex.scout_candidates_path(tmp_path / "cands")
    block = ex.scout_sources(path)
    assert block.startswith("Scout candidates, found through Exa search:")
    assert "Spaced Repetition Training" in block and "https://arxiv.org/abs/2608.00001" in block
    assert "2608.00002" not in block  # only the papers the candidate cites
    assert "<b>" not in block and "&lt;b&gt;" in block  # web text cannot close the quote tag


def test_the_proposer_sees_scout_candidates_and_the_exa_instructions(tmp_path, have):
    llm = FakeLLM().queue("explorer", {"hypotheses": [_hyp()]})
    _cycle(tmp_path, have, llm=llm, sources="Scout candidates, found through Exa search:\nX")
    assert "Scout candidates" in llm.prompts("explorer")[0]
    assert "next_search" in json.dumps(ex.Proposal.model_json_schema())


def test_prior_work_is_asked_before_launch_and_stored_with_citations(tmp_path, have):
    props = [_hyp(claim="helps", techniques=[{"name": "replay", "params": {"share": 0.25}}]),
             _hyp(claim="lr too high", lr=3e-4)]
    search = FakeSearch()
    question = ex.ask_prior_work("helps", lambda q: ("", ())).question
    search.results[question] = (_src(7, question),)
    backend = FakeModal(lambda spec: (0.6, 0.76))
    out, _ = _cycle(tmp_path, have, llm=FakeLLM().queue("explorer", {"hypotheses": props}),
                    backend=backend, go=True, prior_work=search.answer)
    by_claim = {h.claim: h for h in out}
    pw = by_claim["helps"].prior_work
    assert pw is not None and pw.answer.startswith("fake answer") and pw.citations[0].url.endswith("00007")
    assert by_claim["lr too high"].prior_work is None  # refused by the gate: nothing to check
    assert [c for c in search.calls if c[1] == "answer"] == [(question, "answer")]
    stored = ex.load_hypotheses(tmp_path / "explore" / "hypotheses.jsonl")
    assert stored[0].prior_work.citations[0].url.endswith("00007")
    log = DecisionLog(tmp_path / "explore" / "decisions.jsonl")
    assert log.verify().ok
    entry = next(d for d in log.entries() if d.key.startswith("explore/prior_work"))
    assert entry.data["prior_work"][0]["citations"][0]["url"].endswith("00007")
    # gate first, then prior work, then the launch
    states = [d.key.split("#")[0] for d in log.entries()]
    assert states.index("explore/gate") < states.index("explore/prior_work")


def test_a_failing_prior_work_search_never_stops_the_cycle(tmp_path, have):
    backend = FakeModal(lambda spec: (0.6, 0.76))
    out, _ = _cycle(tmp_path, have, llm=FakeLLM().queue("explorer", {"hypotheses": [_hyp()]}),
                    backend=backend, go=True, prior_work=FakeSearch(fail=True).answer)
    assert out[0].prior_work.error and out[0].status in ("confirmed", "inconclusive", "refuted")
    assert len(backend.spawned) == 2


def test_the_model_chooses_the_next_search_and_it_runs_once(tmp_path, have):
    backend = FakeModal(lambda spec: (0.6, 0.76))
    llm = FakeLLM().queue("explorer", {"hypotheses": [_hyp()], "next_search": "spaced repetition replay LLM"})
    _cycle(tmp_path, have, llm=llm, backend=backend, go=True)
    out = tmp_path / "explore"
    req = json.loads((out / ex.NEXT_SEARCH_NAME).read_text())
    assert req["query"] == "spaced repetition replay LLM" and req["used"] is False
    asked = []

    def search(query, where):
        asked.append(query)
        return '<source>\n{"url": "https://arxiv.org/abs/2608.17530"}\n</source>'

    text = ex.run_next_search(out, say=lambda s: None, search=search)
    assert asked == ["spaced repetition replay LLM"] and "2608.17530" in text
    assert json.loads((out / ex.NEXT_SEARCH_NAME).read_text())["used"] is True
    assert ex.run_next_search(out, say=lambda s: None, search=search) == ""  # once only
    log = DecisionLog(out / "decisions.jsonl")
    assert log.verify().ok
    keys = [d.key.split("#")[0] for d in log.entries()]
    assert "explore/next_search" in keys and "explore/search" in keys
    entry = next(d for d in log.entries() if d.key.startswith("explore/search"))
    assert entry.data["urls"] == ["https://arxiv.org/abs/2608.17530"]
