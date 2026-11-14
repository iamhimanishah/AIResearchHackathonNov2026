"""Kernel contracts: hashes, the selection/report firewall, the decision log, JSONL files."""

from __future__ import annotations

import copy
import json
import pickle
import types
import typing

import pytest
from pydantic import BaseModel, ValidationError

from clharness import spec as a1
from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.context import REPORT_SENTINEL

TS = "2026-11-14T20:00:00+00:00"


def _row(techniques=(), seed=0, world_seed=7, **world_kwargs) -> c.ResultRow:
    world = sim.World(world_seed, infra_fail_rate=0.0, **world_kwargs)
    return world.simulate(sim.make_spec(techniques, seed=seed))


def _log_with(tmp_path, n=3) -> c.DecisionLog:
    log = c.DecisionLog(tmp_path / "decisions.jsonl")
    for i in range(n):
        log.append(key=f"k{i}", state="screen/plan", actor="conductor", reason=f"r{i}", ts=TS)
    return log


# ---- canonical JSON and spec hashes ---------------------------------------------------------


def test_canonical_json_is_stable_and_refuses_nan():
    assert c.canonical_json({"b": 1.0, "a": [2, 3.5]}) == '{"a":[2,3.5],"b":1}'
    assert c.hash_obj({"x": 1}) == c.hash_obj({"x": 1.0})
    with pytest.raises(c.ContractError):
        c.canonical_json({"x": float("nan")})


def test_spec_hash_ignores_name_notes_and_lineage_but_not_seed_or_params():
    base = sim.make_spec(["replay"], seed=1)
    renamed = base.model_copy(update={"name": "x", "notes": "y", "parent_spec_hash": "abc"})
    assert c.spec_hash(base) == c.spec_hash(renamed)
    assert c.spec_hash(base) != c.spec_hash(sim.make_spec(["replay"], seed=2))
    assert c.spec_hash(base) != c.spec_hash(sim.make_spec({"replay": {"rate": 0.2}}, seed=1))
    assert c.config_hash(base) == c.config_hash(sim.make_spec(["replay"], seed=2))


def test_no_mitigation_has_one_representation():
    explicit = sim.make_spec({"no_mitigation": {}})
    assert explicit.techniques == [] and c.arm_of(explicit) == "no_mitigation"
    assert c.spec_hash(explicit) == c.spec_hash(sim.make_spec())
    with pytest.raises(c.ContractError):
        sim.make_spec({"no_mitigation": {}, "replay": {}})


def test_models_are_frozen_and_reject_unknown_keys():
    spec = sim.make_spec()
    with pytest.raises(ValidationError):
        spec.seed = 5
    with pytest.raises(ValidationError):
        c.RunSpec(name="x", seed=0, stream=sim.default_stream(), lr=1e-4)


# ---- rows and the firewall ------------------------------------------------------------------


def _field_names(model: type[BaseModel], seen=None) -> set[str]:
    """Every field name reachable from a model, through nested models and containers."""
    seen = seen if seen is not None else set()
    names = set()
    for name, info in model.model_fields.items():
        names.add(name)
        stack = [info.annotation]
        while stack:
            item = stack.pop()
            stack.extend(typing.get_args(item))
            is_class = isinstance(item, type) and not isinstance(item, types.GenericAlias)
            if is_class and issubclass(item, BaseModel) and item not in seen:
                seen.add(item)
                names |= _field_names(item, seen)
    return names


def test_selection_row_has_no_report_field_by_introspection():
    names = _field_names(c.SelectionRow)
    assert "retention_selection" in names and "mean_delta" in names
    assert [n for n in names if "report" in n] == []
    schema = c.SelectionRow.model_json_schema()
    properties = set(schema["properties"])
    for definition in schema["$defs"].values():
        properties |= set(definition.get("properties", {}))
    assert [n for n in properties if "report" in n] == []
    assert "retention_report" in _field_names(c.ResultRow)  # the walk does see report fields
    assert "retention_report" in c.ResultRow.model_json_schema()["properties"]
    assert not issubclass(c.SelectionRow, c.ResultRow)
    assert not issubclass(c.ResultRow, c.SelectionRow)


def test_selection_view_drops_the_report_half_values():
    row = _row(["replay"], report_sentinel=REPORT_SENTINEL)
    assert row.retention_report.mean_delta == REPORT_SENTINEL
    view = c.to_selection_view(row)
    assert type(view) is c.SelectionRow
    assert "918273" not in view.model_dump_json()
    assert view.row_hash == row.row_hash and view.retention_selection == row.retention_selection
    with pytest.raises(ValidationError):
        c.SelectionRow(**row.model_dump())  # the full row does not fit the selection class
    with pytest.raises(c.FirewallError):
        c.to_selection_view(view)


def test_rows_are_a1_rows_and_failure_kind_comes_from_the_reason():
    world = sim.World(7, infra_fail_rate=0.0)
    spec = sim.make_spec(["replay"])
    stored = world.simulate_a1(spec)
    assert isinstance(stored, a1.ResultRow) and stored.spec_hash == spec.spec_hash()
    row = c.from_a1_row(stored)
    assert row.row_hash == c.a1_row_hash(stored) and row.failure_kind is None
    assert row.retention_selection.mean_delta == stored.retention.selection.mean_delta
    assert row.cost.usd == stored.cost.est_usd and row.acquisition_mean is not None
    assert c.failure_kind_of("failed", "infrastructure: preempted") == "infra"
    assert c.failure_kind_of("failed", "Timeout: no row") == "timeout"
    assert c.failure_kind_of("failed", "loss went NaN") == "science"
    assert c.failure_kind_of("invalid", "infrastructure: x") == "science"
    assert c.failure_kind_of("finished", None) is None
    failed = c.a1_failed_row(spec, "timeout", "no row by the deadline")
    assert failed.reason == "timeout: no row by the deadline"
    assert c.from_a1_row(failed).failure_kind == "timeout"
    with pytest.raises(ValidationError):
        a1.ResultRow(spec=spec, spec_hash=spec.spec_hash(), git_sha="x", status="finished",
                     backend="local")


def test_a1_fake_rows_load_as_agent_views():
    read = c.load_rows("results/fake/rows.jsonl")
    assert read.problems == [] and len(read.rows) == len(a1.read_rows("results/fake/rows.jsonl"))
    views = c.selection_views(read.rows)
    assert {v.status for v in views} == {"finished", "failed", "invalid"}
    assert all(v.retention_selection for v in views if v.status == "finished")


def test_freeze_token_cannot_be_built_copied_or_faked(tmp_path):
    with pytest.raises(c.FirewallError):
        c.FreezeToken("abc")
    with pytest.raises(c.FirewallError):
        c.FreezeToken("abc", object())
    row = _row(["replay"])

    class Lookalike:
        frozen_set_hash = "abc"

    for fake in (None, "abc", Lookalike()):
        with pytest.raises(c.FirewallError):
            c.to_report_view(row, fake)
    log = _log_with(tmp_path)
    frozen = c.frozen_set_hash([row.spec_hash])
    log.append(key="freeze", state="freeze", actor="conductor", reason="done", ts=TS,
               data={"frozen_set_hash": frozen})
    token = c.mint_freeze_token(log, frozen)
    for clone in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(c.FirewallError):
            clone(token)
    with pytest.raises(c.FirewallError):
        token._frozen_set_hash = "other"


def test_freeze_token_needs_the_conductors_freeze_decision(tmp_path):
    row = _row(["replay"])
    frozen = c.frozen_set_hash([row.spec_hash])
    log = _log_with(tmp_path)
    with pytest.raises(c.FirewallError):
        c.mint_freeze_token(log, frozen)  # no freeze decision at all
    log.append(key="f1", state="freeze", actor="planner", reason="x", ts=TS,
               data={"frozen_set_hash": frozen})
    log.append(key="f2", state="screen/plan", actor="conductor", reason="x", ts=TS,
               data={"frozen_set_hash": frozen})
    log.append(key="f3", state="freeze", actor="conductor", reason="x", ts=TS,
               data={"frozen_set_hash": "another set"})
    with pytest.raises(c.FirewallError):
        c.mint_freeze_token(log, frozen)  # wrong actor, wrong state, wrong hash
    log.append(key="f4", state="freeze", actor="conductor", reason="frozen", ts=TS,
               data={"frozen_set_hash": frozen})
    token = c.mint_freeze_token(log, frozen)
    report = c.to_report_view(row, token)
    assert type(report) is c.ReportRow and report.frozen_set_hash == frozen
    assert report.retention_report == row.retention_report


def test_freeze_token_is_refused_when_the_log_was_tampered(tmp_path):
    log = _log_with(tmp_path)
    log.append(key="freeze", state="freeze", actor="conductor", reason="x", ts=TS,
               data={"frozen_set_hash": "h"})
    c.mint_freeze_token(log, "h")
    text = log.path.read_text().replace('"reason":"r1"', '"reason":"edited"')
    log.path.write_text(text)
    with pytest.raises(c.FirewallError):
        c.mint_freeze_token(log, "h")


# ---- decision log ---------------------------------------------------------------------------


def test_decision_log_chain_verifies_and_links(tmp_path):
    log = _log_with(tmp_path)
    entries = log.entries()
    report = log.verify()
    assert report.ok and report.count == 3 and report.head == entries[-1].hash
    assert [d.seq for d in entries] == [0, 1, 2]
    assert entries[0].prev_hash == c.GENESIS_HASH and entries[2].prev_hash == entries[1].hash
    assert log.verify(expected_head=entries[-1].hash).ok


@pytest.mark.parametrize("damage", ["edit", "delete_middle", "swap", "drop_last", "rehash"])
def test_decision_log_detects_tampering(tmp_path, damage):
    log = _log_with(tmp_path)
    head = log.head()
    lines = log.path.read_text().splitlines()
    if damage == "edit":
        lines[1] = lines[1].replace('"reason":"r1"', '"reason":"better story"')
    elif damage == "delete_middle":
        del lines[1]
    elif damage == "swap":
        lines[0], lines[1] = lines[1], lines[0]
    elif damage == "drop_last":
        del lines[2]
    else:  # edit an entry and recompute its own hash: the next link still breaks
        record = json.loads(lines[1])
        record["reason"] = "better story"
        record["hash"] = c.hash_obj({k: v for k, v in record.items() if k != "hash"})
        lines[1] = json.dumps(record)
    log.path.write_text("\n".join(lines) + "\n")
    assert not log.verify(expected_head=head).ok
    if damage != "drop_last":
        assert not log.verify().ok  # visible without the stored head as well


def test_decision_log_append_is_idempotent_by_key(tmp_path):
    log = _log_with(tmp_path, n=2)
    first = log.append(key="gate#1", state="screen/gate", actor="gatekeeper", reason="a", ts=TS)
    again = log.append(key="gate#1", state="screen/gate", actor="gatekeeper", reason="b", ts=TS)
    assert again == first and again.reason == "a"
    assert len(log.path.read_text().splitlines()) == 3
    assert log.verify().ok and log.has("gate#1") and not log.has("gate#2")


def test_decision_log_tolerates_and_repairs_a_truncated_last_line(tmp_path):
    log = _log_with(tmp_path)
    whole = log.path.read_text()
    last = whole.splitlines()[-1]
    with open(log.path, "a") as handle:
        handle.write(last[: len(last) // 2])  # a writer died mid-line
    report = log.verify()
    assert report.ok and report.truncated_tail and report.count == 3
    assert len(log.entries()) == 3
    log.append(key="k3", state="screen/gate", actor="conductor", reason="after crash", ts=TS)
    report = log.verify()
    assert report.ok and not report.truncated_tail and report.count == 4
    assert log.path.read_text().startswith(whole) and log.path.read_text().endswith("\n")


def test_decision_carries_versions_proposal_and_verdict(tmp_path):
    spec = sim.make_spec(["replay"])
    predicted = c.Predicted(retention_mean_delta=-8.0, acquisition_ratio=0.83)
    proposal = c.Proposal(
        phase=c.Phase.SCREEN, rationale="try replay", cites_findings=("F-1",),
        specs=(c.ProposedSpec(spec=spec, role="treatment", reason="r", predicted=predicted),),
    )
    verdict = c.Verdict(token="ACCEPT", kept=proposal.spec_hashes())
    log = c.DecisionLog(tmp_path / "d.jsonl")
    made = log.append(
        key="screen/gate#1", state="screen/gate", actor="gatekeeper", reason="ok", ts=TS,
        prompt_hash="p", card_hash="c", skill_hashes={"plan_screen": "s"}, model="m",
        proposal=proposal, verdict=verdict, cites_findings=("F-1",), input_row_hashes=("r1",),
    )
    read = log.entries()[0]
    assert read == made and read.proposal == proposal and read.verdict.token == "ACCEPT"
    assert log.verify().ok
    assert c.accepted_specs(proposal, verdict) == (spec,)


# ---- verdicts, prereg, catalog, citations ---------------------------------------------------


def test_verdict_token_must_match_its_content():
    drop = c.Dropped(spec_hash="h", rule="budget", reason="too expensive")
    assert next(iter(c.Verdict.model_fields)) == "token"
    c.Verdict(token="TRIM", kept=("a",), dropped=(drop,))
    for bad in (
        {"token": "ACCEPT", "kept": ("a",), "dropped": (drop,)},
        {"token": "REJECT", "kept": ("a",), "dropped": (drop,)},
        {"token": "REJECT"},
        {"token": "TRIM", "kept": ("a",)},
        {"token": "MAYBE"},
    ):
        with pytest.raises(ValidationError):
            c.Verdict(**bad)


PREREG_MD = """# PREREG

Some prose first.

```yaml
split: {seed: 3}
```

```yaml
agent_thresholds:
  primary_metric: mean_delta
  noise_sd: 0.008
  decision_multiplier: 2.0
  acquisition_floor_delta: 0.05
  collapse_share_limit: 0.9
  screen_seeds: [0]
  replicate_seeds: [0, 1, 2]
```
"""


def test_prereg_is_read_from_a_yaml_block_and_missing_is_an_error(tmp_path):
    prereg = c.parse_prereg(PREREG_MD)
    assert prereg == sim.default_prereg() and prereg.decision_threshold == pytest.approx(0.016)
    with pytest.raises(c.PreregError):
        c.parse_prereg("# PREREG\n\nno thresholds yet\n")
    with pytest.raises(c.PreregError):
        c.parse_prereg(PREREG_MD.replace("  noise_sd: 0.008\n", ""))
    with pytest.raises(c.PreregError):
        c.load_prereg(tmp_path / "PREREG.md")
    path = tmp_path / "prereg.yaml"
    path.write_text(PREREG_MD.split("```yaml")[2].split("```")[0])
    assert c.load_prereg(path) == prereg


def test_thresholds_have_one_definition():
    assert c.passes_acquisition_floor(0.855, 0.815, 0.05)
    assert not c.passes_acquisition_floor(0.857, 0.815, 0.05)
    assert c.is_collapsed({"a": 0.5, "b": 0.91}, 0.9)
    assert not c.is_collapsed({"a": 0.5, "b": 0.9}, 0.9)


def test_catalog_reports_unknown_techniques_and_bad_parameters():
    catalog = sim.default_catalog()
    assert catalog.problems({"replay": {"rate": 0.2}, "lora": {}}) == []
    problems = catalog.problems({"replay": {"rate": 5, "speed": 1}, "ewc": {}})
    assert len(problems) == 3 and "unknown technique: ewc" in problems
    assert {e.name for e in catalog.controls()} == {"no_mitigation", "reference_arm",
                                                    "select_random"}
    assert catalog.by_role("reference")[0].name == "reference_arm"
    assert catalog.cost_multiplier(["replay", "distill"]) == pytest.approx(1.76)


def test_check_numbers_confirms_citations_against_rows():
    row = c.to_selection_view(_row(["replay"]))
    rows = {row.row_hash: row}
    value = row.retention_selection.mean_delta
    path = "retention_selection.mean_delta"

    def cite(**kw):
        base = {"label": "x", "value": value, "row_hash": row.row_hash, "field_path": path}
        return c.CitedNumber(**{**base, **kw})

    assert c.check_numbers([cite(), cite(value=round(value, 3))], rows) == []
    assert c.check_numbers([cite(row_hash=row.row_hash[:12])], rows) == []
    task = "retention_selection.per_task_delta.piqa"
    assert c.check_numbers([cite(field_path=task, value=value)], rows) != []
    assert len(c.check_numbers([cite(value=value + 0.5)], rows)) == 1
    assert len(c.check_numbers([cite(row_hash="feedbeef")], rows)) == 1
    report_path = "retention_report.mean_delta"
    assert "no field" in c.check_numbers([cite(field_path=report_path)], rows)[0]


# ---- files ----------------------------------------------------------------------------------


def test_read_jsonl_skips_and_reports_a_partial_trailing_line(tmp_path):
    path = tmp_path / "x.jsonl"
    assert c.read_jsonl(path).records == []
    path.write_text('{"a":1}\nnot json\n\n{"a":2}\n{"a":3,"b":')
    read = c.read_jsonl(path)
    assert read.records == [{"a": 1}, {"a": 2}]
    assert read.bad_lines == [2] and read.partial == '{"a":3,"b":'
    path.write_text('{"a":1}\n{"a":2}')  # complete last line, newline missing
    read = c.read_jsonl(path)
    assert read.records == [{"a": 1}, {"a": 2}] and read.partial is None


def test_append_jsonl_never_glues_a_line_to_a_partial_one(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a":1}\n{"a":2,"b"')
    c.append_jsonl(path, {"a": 3})
    assert path.read_text() == '{"a":1}\n{"a":3}\n'
    path.write_text('{"a":1}\n{"a":2}')
    c.append_jsonl(path, {"a": 3})
    assert c.read_jsonl(path).records == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_atomic_write_replaces_whole_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "sub" / "state.json"
    c.atomic_write_json(path, {"n": 1})
    c.atomic_write_json(path, {"n": 2})
    assert json.loads(path.read_text()) == {"n": 2}
    assert [p.name for p in path.parent.iterdir()] == ["state.json"]


def test_load_rows_skips_rows_that_do_not_verify(tmp_path):
    path = tmp_path / "rows.jsonl"
    world = sim.World(7, infra_fail_rate=0.0)
    good = world.simulate_a1(sim.make_spec(["replay"]))
    other = world.simulate_a1(sim.make_spec(["lora"]))
    c.append_jsonl(path, good)
    edited = other.model_dump(mode="json")
    edited["spec"]["seed"] = 99  # spec no longer matches spec_hash
    c.append_jsonl(path, edited)
    c.append_jsonl(path, {**other.model_dump(mode="json"), "status": "bogus"})
    retry = good.model_copy(update={"attempt": 2})
    c.append_jsonl(path, retry)
    read = c.load_rows(path)
    assert [r.spec_hash for r in read.rows] == [good.spec_hash, good.spec_hash]
    assert len(read.problems) == 2
    assert c.latest_by_spec(read.rows)[good.spec_hash].row_hash == c.a1_row_hash(retry)
