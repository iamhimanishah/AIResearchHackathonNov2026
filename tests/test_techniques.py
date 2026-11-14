"""The technique registry: a run's `techniques` list becomes hooks that the trainer calls."""

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from test_loop_hooks import _cfg
from test_loop_tiny import VOCAB, RecordingHook, _tiny_model

from clharness.config import REPO_ROOT, RunConfig
from clharness.spec import TechniqueSpec
from clharness.techniques import (
    REGISTRY,
    TechniqueParams,
    apply_config_overrides,
    available,
    build_hooks,
    register,
)
from clharness.train.hooks import BaseHook
from clharness.train.loop import Trainer


@pytest.fixture
def probe_technique():
    """Register a throwaway technique for one test, then remove it again."""

    @register
    class Probe(BaseHook):
        name = "_probe"

        class Params(TechniqueParams):
            scale: float = 1.0

        def __init__(self, params):
            self.p = params
            self.domains, self.batches = [], 0

        def on_domain_start(self, ctx, domain):
            self.domains.append(domain)

        def select_batch(self, ctx, batch):
            self.batches += 1
            return batch

    yield Probe
    del REGISTRY["_probe"]


def test_frozen_pilot_configs_do_not_need_the_field():
    # `clh reeval` re-validates these, and they predate the techniques field. Some of them
    # also predate other required fields and fail for that reason; that is not this test's concern.
    paths = sorted((REPO_ROOT / "results" / "runs").glob("pilot-*/config.resolved.json"))
    assert paths
    validated = 0
    for p in paths:
        try:
            cfg = RunConfig.model_validate(json.loads(p.read_text()))
        except ValidationError as e:
            assert all(err["loc"][0] != "techniques" for err in e.errors()), p
        else:
            assert cfg.techniques == ()
            validated += 1
    assert validated >= 1


def test_build_hooks_validates(probe_technique):
    (hook,) = build_hooks([TechniqueSpec(name="_probe", params={"scale": 2})])
    assert isinstance(hook, probe_technique) and hook.p.scale == 2.0
    assert build_hooks([]) == []
    with pytest.raises(ValueError, match="unknown technique 'nope'"):
        build_hooks([TechniqueSpec(name="nope")])
    with pytest.raises(ValueError, match="scael"):
        build_hooks([TechniqueSpec(name="_probe", params={"scael": 2})])
    with pytest.raises(ValueError, match="listed twice"):
        build_hooks([TechniqueSpec(name="_probe"), TechniqueSpec(name="_probe")])


def test_register_needs_params():
    with pytest.raises(TypeError, match="Params"):

        @register
        class NoParams(BaseHook):
            name = "_noparams"

    assert "_noparams" not in REGISTRY


def test_config_techniques_reach_the_trainer(tmp_path: Path, monkeypatch, probe_technique):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    raw = _cfg(tmp_path, "tiny-technique").dump()  # code then math, 4 steps each, 2 micro-batches per step
    raw["techniques"] = [{"name": "_probe", "params": {"scale": 0.5}}]
    cfg = RunConfig.model_validate(raw)

    hooks = build_hooks(cfg.techniques)
    Trainer(cfg, hooks, eval_manifest=None, model_override=_tiny_model(vocab=VOCAB, dtype=torch.float32)).run()

    assert hooks[0].p.scale == 0.5
    assert hooks[0].domains == ["code", "math"]
    assert hooks[0].batches == 16  # 2 domains x 4 steps x 2 micro-batches
    # the technique list is written with the run, so a result can be traced to it
    resolved = json.loads((tmp_path / cfg.name / "config.resolved.json").read_text())
    assert resolved["techniques"] == [{"name": "_probe", "params": {"scale": 0.5}}]


def test_available_lists_param_schemas(probe_technique):
    assert available()["_probe"]["properties"]["scale"]["type"] == "number"


def _with(cfg: RunConfig, *techniques: dict) -> RunConfig:
    return RunConfig.model_validate({**cfg.dump(), "techniques": list(techniques)})


def test_listing_order_does_not_matter(probe_technique):
    a, b = TechniqueSpec(name="_probe"), TechniqueSpec(name="lr_rewarm")
    assert [h.name for h in build_hooks([a, b])] == [h.name for h in build_hooks([b, a])] == ["_probe", "lr_rewarm"]


def test_lr_rewarm_switches_on_the_per_domain_schedule(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    base = _cfg(tmp_path, "tiny-rewarm")  # constant schedule, global scope, 4 steps per domain
    assert apply_config_overrides(base) is base  # no techniques: nothing changes

    cfg = apply_config_overrides(_with(base, {"name": "lr_rewarm", "params": {"rewarm_steps": 2, "min_lr_ratio": 0.5}}))
    o = cfg.optim
    assert (o.schedule_scope, o.rewarm_steps, o.min_lr_ratio, o.schedule) == ("per_domain", 2, 0.5, "cosine")
    assert cfg.optim.lr == base.optim.lr and cfg.data == base.data  # nothing else moved

    rec = RecordingHook()
    Trainer(cfg, [*build_hooks(cfg.techniques), rec], eval_manifest=None,
            model_override=_tiny_model(vocab=VOCAB, dtype=torch.float32)).run()
    lrs = [s.lr for s in rec.steps]
    assert lrs[:4] == pytest.approx(lrs[4:])  # the second domain repeats the first domain's schedule
    assert lrs[0] == pytest.approx(o.lr / 2) and lrs[1] == pytest.approx(o.lr)  # two steps of warmup
    # then a cosine over the 2 remaining steps, from lr toward the floor at lr * 0.5
    assert lrs[2] == pytest.approx(o.lr) and lrs[3] == pytest.approx(0.75 * o.lr)


def test_conflicting_overrides_are_an_error(tmp_path: Path):
    @register
    class Clash(BaseHook):
        name = "_clash"

        class Params(TechniqueParams):
            pass

        def __init__(self, params):
            pass

        @classmethod
        def config_overrides(cls, params):
            return {"optim": {"rewarm_steps": 99}}

    try:
        cfg = _with(_cfg(tmp_path, "tiny-clash"), {"name": "_clash"}, {"name": "lr_rewarm"})
        with pytest.raises(ValueError, match="both set optim.rewarm_steps"):
            apply_config_overrides(cfg)
    finally:
        del REGISTRY["_clash"]
