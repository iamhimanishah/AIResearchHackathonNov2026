"""Every technique combination the study launches runs end to end on the tiny model.

Each technique has its own tests. This one is about combinations: the screening grid switches up
to six techniques on at once, and a pair that breaks only together would otherwise first fail on
a GPU, overnight. The specs come from the analysis design itself, so a new arm is covered as soon
as it is added there.
"""

import math
from pathlib import Path

import pytest
import torch
from test_loop_hooks import _cfg
from test_loop_tiny import VOCAB, RecordingHook, _tiny_model
from test_replay import _add_general

from clharness.analysis import design
from clharness.config import RunConfig
from clharness.spec import RunSpec
from clharness.techniques import apply_config_overrides, build_hooks
from clharness.train.loop import Trainer

SPECS: dict[str, RunSpec] = {s.name: s for s in design.screen()}
if hasattr(design, "claims_specs"):  # the published-claims arms, once that design is merged
    SPECS.update({s.name: s for s in design.claims_specs() if s.seed == 0})


def _for_tiny_model(name: str, params: dict) -> dict:
    """The spec's params, with layer counts scaled from Gemma 3 1B's 26 layers to the tiny model's 2."""
    p = dict(params)
    if name == "freeze":
        p["bottom_layers"] = 1 if p.get("bottom_layers", 13) else 0
        p["top_layers"] = 1 if p.get("top_layers", 0) else 0
    if name == "replay" and "layer" in p:
        p["layer"] = 1
    return p


@pytest.mark.parametrize("name", sorted(SPECS))
def test_combination_trains_on_the_tiny_model(tmp_path: Path, monkeypatch, name: str):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    spec = SPECS[name]
    _add_general(tmp_path)  # replay's source domain
    techniques = [{"name": t.name, "params": _for_tiny_model(t.name, t.params)} for t in spec.techniques]
    base = _cfg(tmp_path, "combo")  # code then math, 4 optimizer steps each, 32 rows of 32 tokens
    cfg = apply_config_overrides(RunConfig.model_validate({**base.dump(), "techniques": techniques}))
    hooks, rec = build_hooks(cfg.techniques), RecordingHook()
    assert [h.name for h in hooks] == sorted(spec.technique_names())

    lm = _tiny_model(vocab=VOCAB, dtype=torch.float32)
    trainer = Trainer(cfg, [*hooks, rec], eval_manifest=None, model_override=lm)
    trainer.run()

    assert len(rec.steps) == 8
    assert trainer.gradient_tokens == 32 * 32  # no combination changes the gradient-token budget
    assert all(math.isfinite(s.loss) and math.isfinite(s.grad_norm) and s.grad_norm > 0 for s in rec.steps)
    assert all(torch.isfinite(p).all() for p in lm.model.parameters())
