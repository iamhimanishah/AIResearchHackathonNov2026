.PHONY: setup env smoke test lint clean-smoke

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
