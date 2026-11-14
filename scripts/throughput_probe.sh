#!/usr/bin/env bash
# Measure training tokens/s for micro-batch x compile combinations. ~3 min each on an H100.
# Usage on the box: bash scripts/throughput_probe.sh        (env MBS="4 6 8" COMPILES="false true")
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH; set -a; source .env; set +a; export WANDB_MODE=disabled
MBS="${MBS:-4 6 8}"; COMPILES="${COMPILES:-false true}"
for c in $COMPILES; do for mb in $MBS; do
  name="tp-mb${mb}-c${c}"
  printf 'include:\n  - %s/configs/pilot/throughput-probe.yaml\nname: %s\noptim:\n  micro_batch_size: %s\ntrain:\n  compile: %s\n' "$(pwd)" "$name" "$mb" "$c" > /tmp/$name.yaml
  rm -rf results/runs/$name
  t0=$(date +%s)
  uv run clh train /tmp/$name.yaml > /tmp/$name.log 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then echo "$name: FAILED (rc=$rc) $(grep -m1 -E 'OutOfMemory|Error' /tmp/$name.log | cut -c1-90)"; continue; fi
  uv run python scripts/throughput_report.py "$name" "$t0" "$t1"
done; done
