"""Spec -> config, run directory -> result row, and the technique registry. No model, no GPU."""

import json

import numpy as np
import pytest

from clharness.config import REPO_ROOT
from clharness.executor.rows import half_indices, measure_run
from clharness.executor.run import config_from_spec, run_name, run_spec
from clharness.spec import RunSpec, load_spec
from clharness.techniques import build_hooks
from clharness.train.measure import default_hooks

SPECS = REPO_ROOT / "configs" / "bench" / "specs"
TASKS = ["arc_easy", "piqa"]


def test_committed_specs_resolve_against_the_fast_bench():
    for path in sorted(SPECS.glob("*.json")):
        spec = load_spec(path)
        cfg = config_from_spec(spec)
        assert cfg.name == run_name(spec) == spec.spec_hash()[:16]
        assert [(d.name, d.tokens) for d in cfg.data.domains] == [(d.name, d.tokens) for d in spec.stream]


def test_spec_overrides_reach_the_config():
    spec = RunSpec(name="x", seed=7, stream=[{"name": "code", "tokens": 4096}],
                   optim={"lr": 3e-5, "batch_size": 16})
    cfg = config_from_spec(spec, output_dir="/tmp/somewhere")
    assert (cfg.train.seed, cfg.optim.lr, cfg.optim.batch_size) == (7, 3e-5, 16)
    assert cfg.train.output_dir == "/tmp/somewhere"
    assert cfg.optim.weight_decay == 0.1  # untouched bench default


def test_fast_bench_switches_the_expensive_instruments_off():
    cfg = config_from_spec(load_spec(SPECS / "no-mitigation-s0.json"))
    assert [h.name for h in default_hooks(cfg)] == ["eval"]
    assert len(default_hooks()) == 4  # the pilot setup is unchanged
    assert {t.task for t in cfg.eval.lm_eval_tasks} == {"arc_easy", "hellaswag", "piqa", "pubmedqa_pqal"}


def test_unknown_bench_and_unknown_technique_become_failed_rows():
    row = run_spec(RunSpec(name="x", seed=0, bench="nope", stream=[{"name": "code", "tokens": 1}]))
    assert row.status == "failed" and "unknown bench" in row.reason
    row = run_spec(RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 1}],
                           techniques=[{"name": "no_such_technique"}]))
    assert row.status == "failed" and "unknown technique" in row.reason


def test_techniques_reach_the_config_with_their_overrides():
    spec = RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 4096}],
                   techniques=[{"name": "replay"}, {"name": "lr_rewarm"}])
    assert spec.technique_names() == ["lr_rewarm", "replay"]  # sorted, so order never matters
    flipped = RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 4096}],
                      techniques=[{"name": "lr_rewarm"}, {"name": "replay"}])
    assert spec.spec_hash() == flipped.spec_hash()
    cfg = config_from_spec(spec)
    assert [t.name for t in cfg.techniques] == ["lr_rewarm", "replay"]
    assert cfg.optim.schedule_scope == "per_domain"  # lr_rewarm's own config override
    assert [h.name for h in build_hooks(cfg.techniques)] == ["lr_rewarm", "replay"]


def test_halves_are_disjoint_stable_and_cover_everything():
    sel, rep = half_indices("arc_easy", 500)
    assert len(sel) == len(rep) == 250
    assert sorted([*sel, *rep]) == list(range(500))
    assert (half_indices("arc_easy", 500)[0] == sel).all()
    assert not (half_indices("piqa", 500)[0] == sel).all()  # each task gets its own shuffle


def _ckpt(run, index, domain, boundary, tokens, correct, ppl, share=0.4):
    d = run / "checkpoints" / f"ckpt_{index:04d}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"index": index, "domain": domain, "is_domain_boundary": boundary, "gradient_tokens": tokens}))
    metrics = {f"eval/ppl_doc/{k}": v for k, v in ppl.items()}
    for task in TASKS:
        np.save(d / f"items_{task}.npy", correct[task].astype(np.float32))
        metrics[f"eval/lm_eval/{task}/answer_share_max"] = share
    (d / "metrics.json").write_text(json.dumps(metrics))


def _run_dir(tmp_path, final_share=0.4):
    """Base: every item right. After code: the selection half of arc_easy all wrong.
    After biomedical: additionally the report half of piqa all wrong."""
    run = tmp_path / "run"
    ones = {t: np.ones(100) for t in TASKS}
    after_code = {t: v.copy() for t, v in ones.items()}
    after_code["arc_easy"][half_indices("arc_easy", 100)[0]] = 0
    after_bio = {t: v.copy() for t, v in after_code.items()}
    after_bio["piqa"][half_indices("piqa", 100)[1]] = 0
    _ckpt(run, 0, "base", True, 0, ones, {"code": 10.0, "biomedical": 20.0, "general": 30.0})
    _ckpt(run, 1, "code", True, 1000, after_code, {"code": 5.0, "biomedical": 22.0, "general": 33.0})
    _ckpt(run, 2, "biomedical", True, 2000, after_bio,
          {"code": 8.0, "biomedical": 10.0, "general": 36.0}, share=final_share)
    return run


def test_measure_run_reads_halves_boundaries_and_perplexity(tmp_path):
    m = measure_run(_run_dir(tmp_path), ["code", "biomedical"])
    sel, rep = m["retention"].selection, m["retention"].report
    assert sel.per_task["arc_easy"].delta == pytest.approx(-1.0)
    assert rep.per_task["arc_easy"].delta == pytest.approx(0.0)
    assert rep.per_task["piqa"].delta == pytest.approx(-1.0)
    assert (sel.mean_delta, rep.mean_delta) == (pytest.approx(-0.5), pytest.approx(-0.5))
    b = m["retention"].boundaries
    assert [(x.domain, x.gradient_tokens) for x in b] == [("code", 1000), ("biomedical", 2000)]
    assert (b[0].selection_mean_delta, b[0].report_mean_delta) == (pytest.approx(-0.5), pytest.approx(0.0))
    assert m["acquisition"].ppl_ratio == {"code": pytest.approx(0.8), "biomedical": pytest.approx(0.5)}
    assert m["acquisition"].mean_ppl_ratio == pytest.approx((0.8 * 0.5) ** 0.5)
    assert m["acquisition"].general_ppl_ratio == pytest.approx(1.2)
    # own boundary: code after code (5 / 10), biomedical after biomedical (10 / 20)
    assert m["acquisition"].own_boundary_ppl_ratio == {"code": pytest.approx(0.5), "biomedical": pytest.approx(0.5)}
    assert m["acquisition"].mean_own_boundary_ppl_ratio == pytest.approx(0.5)
    assert m["gradient_tokens"] == 2000 and m["invalid_reason"] is None


def test_answer_collapse_is_recorded_but_keeps_the_run(tmp_path):
    """PREREG A9: collapse is flagged from `answer_share` by the analysis, not invalidated."""
    m = measure_run(_run_dir(tmp_path, final_share=0.97), ["code", "biomedical"])
    assert m["invalid_reason"] is None
    assert max(m["answer_share"].values()) == pytest.approx(0.97)


def test_measure_run_agrees_with_a_september_baseline():
    """pilot-chosen-200M lost 7.4 points on the multiple-choice mean (docs/baselines.md).
    The other pilot runs are not used here: their boundary item files were written after the
    recoverability probe, so they differ slightly from the logged scores."""
    run = REPO_ROOT / "results" / "runs" / "pilot-chosen-200M"
    m = measure_run(run, ["code", "biomedical", "legal", "math"])
    r = m["retention"]
    mc = [t for t in r.report.per_task if t != "mbpp"]
    both = sum(r.selection.per_task[t].delta + r.report.per_task[t].delta for t in mc) / (2 * len(mc))
    assert both == pytest.approx(-0.074, abs=0.002)
    assert len(r.boundaries) == 4
    assert m["acquisition"].ppl_ratio["legal"] == pytest.approx(0.79, abs=0.01)
    assert m["acquisition"].general_ppl_ratio == pytest.approx(1.43, abs=0.01)


def test_every_fake_row_names_real_techniques_with_valid_params():
    """The fake rows must have the shape real rows will have, or analysis and the agent are
    built against the wrong thing."""
    from clharness.fake import fake_rows

    for row in fake_rows(0):
        build_hooks(row.spec.techniques)  # raises on an unknown technique or a bad parameter


def test_remeasure_keeps_the_run_and_replaces_only_measurements(tmp_path):
    from clharness.executor.run import remeasure_row
    from clharness.fake import fake_rows

    old = next(r for r in fake_rows(0) if r.status == "finished")
    old = old.model_copy(update={"spec": old.spec.model_copy(update={
        "stream": old.spec.stream[:1] + [old.spec.stream[1].model_copy(update={"name": "biomedical"})]})})
    old = old.model_copy(update={"spec_hash": old.spec.spec_hash()})
    new = remeasure_row(old, _run_dir(tmp_path))
    assert (new.spec, new.cost, new.git_sha, new.spec_hash) == (old.spec, old.cost, old.git_sha, old.spec_hash)
    assert new.acquisition.ppl_ratio == {"code": pytest.approx(0.8), "biomedical": pytest.approx(0.5)}
    assert new.acquisition.own_boundary_ppl_ratio == {"code": pytest.approx(0.5), "biomedical": pytest.approx(0.5)}
    assert new.retention.selection.mean_delta == pytest.approx(-0.5)
    assert new.remeasured_at is not None and old.remeasured_at is None


def test_extra_eval_bench_trains_identically_and_keeps_the_protected_items():
    """fast-extra may differ from the frozen bench only in its eval list, and the four protected
    tasks must be scored on exactly the frozen items."""
    import json

    base = config_from_spec(load_spec(SPECS / "no-mitigation-s0.json"))
    extra = config_from_spec(load_spec(SPECS / "plumbing-extra-evals.json").model_copy(
        update={"stream": load_spec(SPECS / "no-mitigation-s0.json").stream}))
    for section in ("model", "data", "optim", "train"):
        assert getattr(base, section) == getattr(extra, section), section
    assert [t.task for t in extra.eval.lm_eval_tasks][:4] == [t.task for t in base.eval.lm_eval_tasks]
    assert [t.task for t in extra.eval.lm_eval_tasks][4:] == ["winogrande", "arc_challenge"]
    frozen = json.loads((REPO_ROOT / "manifests" / "eval-protected-v1.json").read_text())["tasks"]
    wider = json.loads((REPO_ROOT / "manifests" / "eval-protected-v1-extra.json").read_text())["tasks"]
    for task in ("arc_easy", "hellaswag", "piqa", "pubmedqa_pqal"):
        assert wider[task]["indices"] == frozen[task]["indices"], task
        assert wider[task]["docs_sha256"] == frozen[task]["docs_sha256"], task


def test_a_run_that_died_part_way_is_not_measured_as_finished(tmp_path):
    """Remove the last domain's checkpoint: the directory must be refused, not scored."""
    import shutil

    run = _run_dir(tmp_path)
    shutil.rmtree(run / "checkpoints" / "ckpt_0002")
    with pytest.raises(ValueError, match="incomplete run"):
        measure_run(run, ["code", "biomedical"])


def test_a_technique_may_not_silently_replace_a_spec_optim_override():
    spec = RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 4096}],
                   optim={"lr": 5e-5}, techniques=[{"name": "lora", "params": {"rank": 8}}])
    with pytest.raises(ValueError, match="replaced by a technique"):
        config_from_spec(spec)
    keeps = RunSpec(name="x", seed=0, stream=[{"name": "code", "tokens": 4096}],
                    optim={"lr": 5e-5}, techniques=[{"name": "lora", "params": {"rank": 8, "lr": None}}])
    assert config_from_spec(keeps).optim.lr == 5e-5


def test_tasks_outside_the_protected_set_never_enter_the_halves_or_the_mean(tmp_path):
    """A bench may score extra tasks (fast-extra). They are reported apart, over all items."""
    run = _run_dir(tmp_path)
    for ckpt, score in (("ckpt_0000", 1.0), ("ckpt_0001", 1.0), ("ckpt_0002", 0.25)):
        items = np.zeros(40, dtype=np.float32)
        items[: int(40 * score)] = 1
        np.save(run / "checkpoints" / ckpt / "items_winogrande.npy", items)
    m = measure_run(run, ["code", "biomedical"])
    r = m["retention"]
    assert set(r.selection.per_task) == set(r.report.per_task) == set(TASKS)
    assert (r.selection.mean_delta, r.report.mean_delta) == (pytest.approx(-0.5), pytest.approx(-0.5))
    assert r.report_only["winogrande"].delta == pytest.approx(-0.75)
    assert r.report_only["winogrande"].n_items == 40
