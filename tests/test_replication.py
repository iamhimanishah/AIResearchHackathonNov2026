"""PREREG 6.3/6.3b replication planning: promoted configurations and selection variants are
replicated at seeds 1 and 2. The helper reads the selection-half promotion tables only, and every
spec it emits is the canonical design.spec at the new seed."""

import json
import math

import pytest

from clharness.analysis import attribution as A
from clharness.analysis import design as D
from clharness.cli import main
from clharness.executor.run import config_from_spec
from clharness.fake import fake_rows
from clharness.spec import load_spec, write_rows
from clharness.techniques import _validated


@pytest.fixture(scope="module")
def screen():
    return A.analyze(fake_rows(seed=0), confirmatory=False, n_boot=500)


@pytest.fixture(scope="module")
def specs(screen):
    return D.replication_specs(screen["promotion"], screen["selection_screen"])


def _expected_hashes(screen, seeds=D.REPLICATION_SEEDS):
    """The design.spec hash of every promoted configuration and selection variant, per seed."""
    out = set()
    for d in screen["promotion"]:
        if d["promoted"]:
            out.update(D.spec(frozenset(d["techniques"]), s).spec_hash() for s in seeds)
    for d in screen["selection_screen"]:
        if d.get("promoted"):
            out.update(
                D.spec(frozenset(d["techniques"]), s, selection=(d["signal"], d["keep"])).spec_hash()
                for s in seeds)
    return out


def test_promoted_factorial_configs_replicated_at_seeds_1_and_2(screen, specs):
    assert D.REPLICATION_SEEDS == (1, 2)
    assert {s.seed for s in specs} == set(D.REPLICATION_SEEDS)
    promoted = [d for d in screen["promotion"] if d["promoted"]]
    for d in promoted:
        for seed in D.REPLICATION_SEEDS:
            expected = D.spec(frozenset(d["techniques"]), seed)
            assert expected.spec_hash() in {s.spec_hash() for s in specs}
    assert len(promoted) == 3  # the screen promotes its top three factorial configurations


def test_promoted_selection_variants_replicated_at_seeds_1_and_2(screen, specs):
    promoted = [d for d in screen["selection_screen"] if d.get("promoted")]
    assert promoted  # each signal promotes one `keep` direction
    assert {d["signal"] for d in promoted} == {r["signal"] for r in screen["selection_screen"]}
    for d in promoted:
        for seed in D.REPLICATION_SEEDS:
            expected = D.spec(frozenset(d["techniques"]), seed, selection=(d["signal"], d["keep"]))
            assert expected.spec_hash() in {s.spec_hash() for s in specs}


def test_every_spec_is_exactly_design_spec(screen, specs):
    assert {s.spec_hash() for s in specs} == _expected_hashes(screen)
    assert all(s.notes.startswith("replication") for s in specs)


def test_specs_validate_and_build(specs):
    for sp in specs:
        _validated(sp.techniques)  # unknown technique or bad params raises
        cfg = config_from_spec(sp)
        assert cfg.train.seed == sp.seed


def test_only_promoted_entries_produce_specs(screen, specs):
    not_promoted = [d["config"] for d in screen["promotion"] if not d["promoted"]]
    names = {s.name for s in specs}
    for label in not_promoted:
        assert all(not n.startswith(label) for n in names), label
    assert D.replication_specs([], []) == []
    empty = [{"promoted": True, "techniques": ["replay"]}]
    assert {s.spec_hash() for s in D.replication_specs(empty, [])} == {
        D.spec(frozenset({"replay"}), s).spec_hash() for s in D.REPLICATION_SEEDS}


def test_deduplicated_by_spec_hash():
    entry = {"promoted": True, "techniques": ["replay"]}
    specs = D.replication_specs([entry, dict(entry)], [])
    assert len(specs) == len(D.REPLICATION_SEEDS)


def test_promotion_without_techniques_field_is_rejected():
    with pytest.raises(ValueError, match="techniques"):
        D.replication_specs([{"promoted": True}], [])


def test_report_half_never_read(screen):
    """Rows whose report half is NaN must still produce the promoted specs — the helper may run
    as soon as the screen finishes, when no report-half data exists."""
    nan_rows = []
    for r in fake_rows(seed=0):
        if r.retention is None:  # failed/invalid rows carry no retention at all
            nan_rows.append(r)
            continue
        report_half = r.retention.report
        ret = r.retention.model_copy(update={
            "report": report_half.model_copy(
                update={"per_task": {t: v.model_copy(update={"delta": float("nan")})
                                     for t, v in report_half.per_task.items()},
                        "mean_delta": float("nan")}),
            "boundaries": [b.model_copy(update={"report_mean_delta": float("nan")})
                           for b in r.retention.boundaries],
        })
        nan_rows.append(r.model_copy(update={"retention": ret}))
    report = A.analyze(nan_rows, confirmatory=False, n_boot=500)
    assert math.isfinite(report["selection_screen"][0]["vs_reference"])  # still finite
    specs = D.replication_specs(report["promotion"], report["selection_screen"])
    assert {s.spec_hash() for s in specs} == _expected_hashes(screen)


def test_cli_writes_specs_and_list(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    write_rows(rows_path, fake_rows(seed=0))
    out = tmp_path / "replication"
    assert main(["plan-replication", str(rows_path), "--n-boot", "200", "--out", str(out)]) == 0
    lines = [ln for ln in (out / "list.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) > 0
    loaded = [load_spec(ln) for ln in lines]
    # every list.txt path is a spec file the runner accepts, unique by hash
    assert len({s.spec_hash() for s in loaded}) == len(loaded)
    for sp in loaded:
        _validated(sp.techniques)
        config_from_spec(sp)
        assert sp.seed in D.REPLICATION_SEEDS
    # on-disk JSON round-trips to the same spec
    files = sorted(out.glob("*.json"))
    assert len(files) == len(lines)
    assert {load_spec(f).spec_hash() for f in files} == {s.spec_hash() for s in loaded}
    header = json.loads(files[0].read_text())
    assert "selection" not in json.dumps(header.get("notes", ""))  # notes say replication, not screen
