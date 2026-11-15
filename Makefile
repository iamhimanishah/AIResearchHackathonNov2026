.PHONY: setup env smoke test lint clean-smoke reproduce reproduce-figures reproduce-runs

UV ?= uv
SMOKE_CFG = configs/smoke.yaml

setup:            ## create venv + install locked deps
	$(UV) sync --extra dev

env:              ## print detected device / versions
	$(UV) run clh env

manifests/smoke.json:
	$(UV) run clh build-data --set smoke

manifests/eval-smoke.json: $(SMOKE_CFG) configs/eval/smoke.yaml
	$(UV) run clh build-eval-manifest $(SMOKE_CFG)

smoke: manifests/smoke.json manifests/eval-smoke.json   ## end-to-end pipeline on a tiny config
	$(UV) run clh verify $(SMOKE_CFG)
	$(UV) run clh sae-baseline $(SMOKE_CFG)
	$(UV) run clh train $(SMOKE_CFG)

test:             ## fast tests (no model download)
	$(UV) run pytest -m "not smoke" -q

test-all:         ## includes the smoke test (downloads model + SAEs)
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests

clean-smoke:
	rm -rf results/runs/smoke

# ---- reproduce -------------------------------------------------------------------------------
# `make reproduce-figures` rebuilds every reported table and the results page from the result
# rows committed under results/. No GPU, no downloads, about a minute.
# `make reproduce-runs` retrains the control runs on Modal GPUs (about 20 minutes, about $18 for
# the 14 phase 0 runs) and writes fresh rows next to the committed ones, for comparison.
#
# REPORTED_ROWS is the list of row files the reported numbers are computed from. Re-measured
# files replace the originals they were derived from (analysis/PREREG.md, section 0.4).
REPORTED_ROWS ?= results/rows.phase0.remeasured.jsonl results/rows.phase0.jsonl \
                 results/rows.reference.remeasured.jsonl $(wildcard results/agent/rows.jsonl)
REPRO_OUT ?= results/analysis

reproduce: reproduce-figures   ## what a reader should run: tables and page from committed rows

reproduce-figures:             ## tables, analysis report and results page from committed rows
	$(UV) run clh validate-row $(REPORTED_ROWS)
	$(UV) run clh grid-analysis $(REPORTED_ROWS) --out $(REPRO_OUT)
	@echo "analysis written to $(REPRO_OUT)"

reproduce-runs:                ## retrain the phase 0 controls on Modal (needs `modal setup` and the huggingface secret)
	$(UV) run clh run --modal $$(grep -v '^#' configs/bench/phase0.txt) \
		$$(grep '^# REUSE' configs/bench/phase0.txt | sed 's/^# REUSE //') \
		--out results/rows.reproduced.jsonl
