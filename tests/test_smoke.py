"""End-to-end smoke: downloads Gemma 3 1B + 3 SAEs on first run; a few minutes on Apple silicon
or a small GPU. `pytest -m smoke` or `make smoke` (the Makefile path prints progress)."""

import json
import shutil

import numpy as np
import pytest

from clharness.cli import main
from clharness.config import REPO_ROOT

SMOKE_CFG = "configs/smoke.yaml"
RUN_DIR = REPO_ROOT / "results" / "runs" / "smoke"


@pytest.mark.smoke
def test_smoke_pipeline():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    if not (REPO_ROOT / "manifests" / "smoke.json").exists():
        assert main(["build-data", "--set", "smoke"]) == 0
    if not (REPO_ROOT / "manifests" / "eval-smoke.json").exists():
        assert main(["build-eval-manifest", SMOKE_CFG]) == 0
    assert main(["verify", SMOKE_CFG]) == 0
    assert main(["train", SMOKE_CFG]) == 0

    ckpts = sorted((RUN_DIR / "checkpoints").iterdir())
    # base + 4 domain boundaries at minimum
    assert len(ckpts) >= 5
    metas = [json.loads((c / "meta.json").read_text()) for c in ckpts]
    assert metas[0]["domain"] == "base" and metas[0]["gradient_tokens"] == 0
    assert sum(m["is_domain_boundary"] for m in metas) == 5
    # gradient tokens are monotone and end at (or just past) the configured budget
    from clharness.config import load_run_config
    budget = sum(d.tokens for d in load_run_config(REPO_ROOT / SMOKE_CFG).data.domains)
    gt = [m["gradient_tokens"] for m in metas]
    assert gt == sorted(gt) and gt[-1] >= budget
    # every checkpoint has metrics + compact feature arrays for each layer
    for c in ckpts:
        m = json.loads((c / "metrics.json").read_text())
        assert "eval/ppl/code" in m and "probe/L13/fvu" in m and "eval/lm_eval/arc_easy/acc" in m
        for L in (7, 13, 22):
            z = np.load(c / f"features_L{L}.npz")
            assert z["freq"].shape == (16384,) and 0.0 <= float(z["freq"].max()) <= 1.0
    # CKA vs base is 1 at checkpoint 0 and <= 1 afterwards
    m0 = json.loads((ckpts[0] / "metrics.json").read_text())
    assert m0["probe/L13/cka_vs_base"] == 1.0 and m0["probe/L13/valid"] == 1.0
    mN = json.loads((ckpts[-1] / "metrics.json").read_text())
    assert "eval/lm_eval/arc_easy/flip_1to0" in mN and mN["eval/lm_eval/arc_easy/flip_1to0"] >= 0.0
    assert 0.0 < mN["probe/L13/cka_vs_base"] <= 1.0 and 0.0 < mN["probe/L13/mknn"] <= 1.0
    assert (ckpts[-1] / "items_arc_easy.npy").exists()
    # readout probes: base-model readout on base activations should separate the 5 domains well
    assert m0["probe/L13/readout_base_probe_acc"] > 0.6 and 0.0 <= mN["probe/L13/readout_fresh_probe_acc"] <= 1.0
    # recoverability probe ran at the boundary checkpoints (not at checkpoint 0) and restored the weights
    boundary = [json.loads((c / "metrics.json").read_text()) for c, m in zip(ckpts, metas) if m["is_domain_boundary"] and m["index"] > 0]
    assert boundary and all("recover/ppl/general" in b and "recover/ppl/general_delta" in b for b in boundary)
    assert all(b["recover/tokens_used"] >= 4096 for b in boundary)
    # per-example signals logged for every trained sequence
    z = np.load(RUN_DIR / "signals.npz")
    assert z["data"].shape[0] == gt[-1] // 256 and "act_nov_L13" in list(z["columns"]) and np.isfinite(z["data"]).all()
