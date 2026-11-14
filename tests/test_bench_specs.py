"""Every committed run spec is valid and its techniques exist with valid parameters."""

from pathlib import Path

import pytest
from test_loop_hooks import _cfg

from clharness.config import REPO_ROOT, RunConfig
from clharness.executor.run import config_from_spec
from clharness.spec import load_spec
from clharness.techniques import apply_config_overrides, build_hooks

SPECS = sorted((REPO_ROOT / "configs" / "bench" / "specs").glob("*.json"))


ALL_SPECS = sorted((REPO_ROOT / "configs" / "bench" / "specs").rglob("*.json"))  # includes sanity/


@pytest.mark.parametrize("path", ALL_SPECS, ids=lambda p: p.stem)
def test_spec_loads_and_its_techniques_build(path: Path):
    spec = load_spec(path)
    hooks = build_hooks(spec.techniques)
    assert sorted(h.name for h in hooks) == sorted(spec.technique_names())
    config_from_spec(spec)  # the fast bench plus this spec's overrides is a valid run config


def test_reference_arm_is_replay_plus_rewarm(tmp_path: Path):
    spec = load_spec(REPO_ROOT / "configs" / "bench" / "specs" / "reference-s0.json")
    assert sorted(spec.technique_names()) == ["lr_rewarm", "replay"]
    replay = next(h for h in build_hooks(spec.techniques) if h.name == "replay")
    assert (replay.p.share, replay.p.source, replay.p.select) == (0.05, "general", "random")
    # on any config, the arm switches on the per-domain schedule and leaves the learning rate alone
    base = _cfg(tmp_path, "tiny-reference")
    techniques = [t.model_dump() for t in spec.techniques]
    cfg = apply_config_overrides(RunConfig.model_validate({**base.dump(), "techniques": techniques}))
    assert (cfg.optim.schedule_scope, cfg.optim.schedule, cfg.optim.rewarm_steps) == ("per_domain", "cosine", 8)
    assert cfg.optim.lr == base.optim.lr


def test_the_two_halves_differ_from_the_arm_by_one_technique():
    names = {p.stem: set(load_spec(p).technique_names()) for p in SPECS}
    assert names["replay-only-s0"] == {"replay"} and names["rewarm-only-s0"] == {"lr_rewarm"}
    assert names["reference-s0"] == names["replay-only-s0"] | names["rewarm-only-s0"]
