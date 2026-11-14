"""The run-spec and result-row formats: hashing, validation rules, and the committed fixtures."""

import json

import pytest
from pydantic import ValidationError

from clharness.cli import main
from clharness.config import REPO_ROOT
from clharness.fake import PLANTED, fake_rows
from clharness.spec import (
    ResultRow,
    RunSpec,
    append_row,
    json_schemas,
    read_rows,
    row_to_line,
    write_rows,
)

FAKE = REPO_ROOT / "results" / "fake" / "rows.jsonl"
SCHEMA_DIR = REPO_ROOT / "configs" / "schema"


def spec(**kw) -> RunSpec:
    base = {
        "name": "x",
        "seed": 0,
        "stream": [{"name": "code", "tokens": 1000}, {"name": "biomedical", "tokens": 1000}],
        "techniques": [{"name": "replay", "params": {"share": 0.05}}],
    }
    return RunSpec.model_validate({**base, **kw})


def test_hash_ignores_labels_only():
    a = spec()
    assert a.spec_hash() == spec(name="renamed", notes="why", parent_spec_hash="abc").spec_hash()
    assert a.spec_hash() != spec(seed=1).spec_hash()
    assert a.spec_hash() != spec(optim={"lr": 1e-3}).spec_hash()
    assert a.spec_hash() != spec(techniques=[]).spec_hash()
    changed = spec(techniques=[{"name": "replay", "params": {"share": 0.25}}])
    assert a.spec_hash() != changed.spec_hash()


def test_hash_is_stable_across_key_order_and_round_trip():
    a = spec(optim={"lr": 1e-3, "warmup_steps": 5})
    b = spec(optim={"warmup_steps": 5, "lr": 1e-3})
    assert a.spec_hash() == b.spec_hash()
    assert RunSpec.model_validate_json(a.model_dump_json()).spec_hash() == a.spec_hash()


def test_spec_rejects_mistakes():
    with pytest.raises(ValidationError, match="unknown optim override"):
        spec(optim={"learning_rate": 1e-3})
    with pytest.raises(ValidationError, match="listed twice"):
        spec(techniques=[{"name": "replay"}, {"name": "replay"}])
    with pytest.raises(ValidationError):
        spec(stream=[])
    with pytest.raises(ValidationError):
        spec(sead=1)  # unknown keys are errors, so a typo cannot silently change a run


def test_row_rules():
    s = spec()
    common = {"spec": s, "spec_hash": s.spec_hash(), "git_sha": "abc", "backend": "local"}
    with pytest.raises(ValidationError, match="finished row is missing"):
        ResultRow(status="finished", **common)
    with pytest.raises(ValidationError, match="needs a reason"):
        ResultRow(status="failed", **common)
    with pytest.raises(ValidationError, match="spec_hash does not match"):
        ResultRow(status="failed", reason="x", **{**common, "spec_hash": "0" * 64})
    assert ResultRow(status="failed", reason="oom", **common).retention is None


def test_fake_rows_cover_the_cases_and_round_trip(tmp_path):
    rows = fake_rows(0)
    assert 35 <= len(rows) <= 45
    assert {r.status for r in rows} == {"finished", "failed", "invalid"}
    assert len({r.spec_hash for r in rows}) == len(rows)  # every row is a distinct experiment
    assert sum(1 for r in rows if not r.spec.techniques) == 5  # noise-floor seeds
    path = tmp_path / "rows.jsonl"
    write_rows(path, rows[:-1])
    append_row(path, rows[-1])
    assert read_rows(path) == rows


def test_fake_rows_carry_the_planted_effects():
    def mean(names: set[str]) -> float:
        hits = [
            r.retention.report.mean_delta
            for r in fake_rows(0)
            if r.status == "finished" and set(r.spec.technique_names()) == names
        ]
        return sum(hits) / len(hits)

    base = mean(set())
    assert base == pytest.approx(PLANTED["baseline_retention_mean_delta"], abs=0.01)
    assert mean({"replay"}) - base == pytest.approx(0.025, abs=0.015)
    # the fake study follows the pre-registered screen, which has no lora-only run: planted,
    # distill+lora retains 0.050 more than base and lr_rewarm alone 0.008
    assert mean({"distill", "lora"}) > mean({"lr_rewarm"}) + 0.02


def test_bad_line_is_reported_with_its_number(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(row_to_line(fake_rows(0)[0]) + '{"status": "finished"}\n')
    with pytest.raises(ValueError, match=r"rows\.jsonl:2"):
        read_rows(path)
    assert main(["validate-row", str(path)]) == 1


def test_committed_fake_rows_are_current():
    """results/fake/rows.jsonl must be what `clh fake-rows` writes; regenerate if this fails."""
    assert main(["validate-row", str(FAKE)]) == 0
    assert FAKE.read_text() == "".join(row_to_line(r) for r in fake_rows(0))
    assert json.loads(FAKE.with_name("planted_effects.json").read_text()) == PLANTED


def test_committed_schemas_are_current():
    """configs/schema/ must be what `clh export-schema` writes; regenerate if this fails."""
    for name, schema in json_schemas().items():
        assert json.loads((SCHEMA_DIR / name).read_text()) == schema, name
