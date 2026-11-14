#!/usr/bin/env bash
# CPU/MPS-only work that is worth doing before the GPU session. Sequential; each step logs its own
# section so a failure in one does not hide the others. Run: nohup bash scripts/overnight_mac.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export WANDB_MODE=offline
step() { echo; echo "=== $1  $(date '+%F %T') ==="; }

step "1/3 smoke end-to-end test (regression on the precision refactor)"
uv run pytest -m smoke -q -x 2>&1 | grep -vE "it/s|it\]|Warning|warn\(" | tail -5

step "2/3 frozen suite (protected-v1) on base model, MPS cross-check"
uv run clh baseline-evals configs/baseline-protected-v1.yaml 2>&1 | grep -vE "it/s|it\]|Warning|warn\(|Overwriting|already-initialized|pretrained. model" | tail -12

step "3/3 SAE drift baseline on the real probe set"
uv run clh sae-baseline configs/pilot/base.yaml 2>&1 | grep -vE "Warning|warn\(|Loading weights" | tail -6

step "done"
