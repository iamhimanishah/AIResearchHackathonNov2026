"""The shared memory on past.dev, against a fake HTTP opener: no network."""

import io
import json
import urllib.error

from clharness.agent import explore as ex
from clharness.agent.pastdev import PastMemory, as_prompt_block
from test_explore import SHA, FakeModal, _card, have  # noqa: F401  (fixture)


class FakeService:
    """Stores data points by id; recall returns every stored point with its label as a source."""

    def __init__(self, down: bool = False):
        self.points: dict[str, dict] = {}
        self.recalls: list[dict] = []
        self.down = down

    def __call__(self, request, timeout=None):
        if self.down:
            raise urllib.error.URLError("down")
        body = json.loads(request.data) if request.data else {}
        if request.full_url.endswith("/ingest"):
            assert request.get_header("Authorization") == "Bearer k"
            self.points[body["id"]] = body
            return io.BytesIO(json.dumps({"ingestionId": "x", "sourceId": body["id"]}).encode())
        self.recalls.append(body)
        results = [{"content": p["content"], "sources": [{"sourceId": i, "label": p["label"]}]}
                   for i, p in self.points.items()]
        return io.BytesIO(json.dumps({"results": results}).encode())


def test_same_id_replaces_and_recall_names_the_writers():
    svc = FakeService()
    m = PastMemory("k", opener=svc)
    assert m.remember("hypothesis-h001", "open", "explorer") and m.remember("hypothesis-h001", "refuted", "explorer")
    m.remember("claim-c1", "a published claim", "scout")
    assert len(svc.points) == 2 and svc.points["remember-hypothesis-h001"]["content"] == "refuted"
    recalled = m.recall("what is known?", "explorer")
    assert {w for r in recalled for w in r["writers"]} == {"explorer", "scout"}
    block = as_prompt_block(recalled)
    assert "not instructions" in block and "[scout] a published claim" in block
    assert svc.recalls[0]["identity"] == "explorer"


def test_a_service_that_is_down_never_raises_and_is_recorded():
    m = PastMemory("k", opener=FakeService(down=True))
    assert m.remember("x", "y", "analyst") is False and m.recall("q", "explorer") == []
    assert [e["ok"] for e in m.events if e["op"] in ("remember", "recall")] == [False, False]
    assert PastMemory("").remember("x", "y", "analyst") is False  # not configured: a no-op


def test_the_explorer_writes_and_then_plans_from_what_others_wrote(tmp_path, have):  # noqa: F811
    """Cycle 1 writes arm results (analyst), a claim card (scout) and hypotheses (explorer).
    Cycle 2's proposal step recalls them and the model prompt contains the other agents' text."""
    svc = FakeService()
    memory = PastMemory("k", opener=svc)
    out = tmp_path / "explore"
    out.mkdir()
    (out / ex.INBOX_NAME).write_text(_card("c1"))
    backend = FakeModal(lambda spec: (-1.0, 0.75))
    ex.cycle(out, [have], backend=backend, git_sha=SHA, go=True, sleep=lambda s: None,
             say=lambda s: None, max_hypotheses=1, memory=memory)
    labels = {p["label"] for p in svc.points.values()}
    assert labels == {"analyst", "scout", "explorer"}
    verdict = svc.points["remember-hypothesis-h001"]["content"]
    assert "Status: confirmed" in verdict or "Status: refuted" in verdict or "Status: inconclusive" in verdict
    assert "report" not in " ".join(p["content"].lower() for p in svc.points.values() if p["label"] == "analyst")

    seen = {}

    class Spy:
        def complete(self, agent, system, prompt, schema):
            seen["prompt"] = prompt
            return schema(hypotheses=[]), None

    ex.cycle(out, [have], llm=Spy(), say=lambda s: None, max_hypotheses=2, memory=memory)
    assert "SHARED MEMORY" in seen["prompt"]
    assert "[analyst]" in seen["prompt"] and "[scout]" in seen["prompt"]
    log = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    events = [e for d in log if d["key"].startswith("explore/memory") for e in d["data"]["events"]]
    assert any(e["op"] == "recall" for e in events) and any(e["op"] == "remember" for e in events)


def test_a_dry_run_writes_nothing_to_the_shared_memory(tmp_path, have):  # noqa: F811
    svc = FakeService()
    ex.cycle(tmp_path / "explore", [have], say=lambda s: None, memory=PastMemory("k", opener=svc))
    assert svc.points == {} and len(svc.recalls) == 1
