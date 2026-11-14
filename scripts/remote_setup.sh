#!/usr/bin/env bash
# Bootstrap a fresh Lambda / RunPod box (Ubuntu, CUDA driver present) for clharness.
# Usage: from the Mac, `scripts/remote_sync.sh push <ip>` (rsyncs the repo incl. .env), then
#        `scripts/remote_sync.sh ssh <ip> bash scripts/remote_setup.sh`. wandb runs offline on the box.
set -euo pipefail

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12
uv sync --extra dev
uv run clh env

# Data: rebuild from pinned sources (~6 GB download; a few minutes on datacenter bandwidth).
# The manifest is checked in, so `clh verify` afterwards is a byte-for-byte reproducibility test
# of the build across machines.
if [ ! -f data/real/math/stream.tokens.npy ]; then
  set -a; source .env; set +a
  uv run clh build-data --set real --seq-len 1024
fi
uv run clh verify configs/pilot/base.yaml

cat <<'EOF'

Next (each is a separate command; ask before anything over ~1 GPU-hour):
  uv run clh baseline-evals configs/baseline-protected-v1.yaml      # official base numbers, ~5 min
  uv run clh sae-baseline configs/pilot/base.yaml                   # drift baseline on real probe set, ~3 min
  uv run clh train configs/pilot/cuda-check.yaml                    # ~25 min; then rerun with name cuda-check-b
  uv run clh compare-runs results/runs/cuda-check-a results/runs/cuda-check-b
EOF
