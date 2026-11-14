#!/usr/bin/env bash
# Push the repo to a GPU box (no git remote needed) and pull results back.
#   scripts/remote_sync.sh push  <host>      rsync repo -> ~/ContinualLearningResearch on the box
#   scripts/remote_sync.sh pull  <host>      rsync results/ manifests/ back from the box
#   scripts/remote_sync.sh ssh   <host> CMD  run a command on the box inside the repo
# <host> is user@ip (default user ubuntu). Key: ~/.ssh/lambda_cl. The remote runs wandb OFFLINE;
# offline runs are synced from the Mac after `pull` (wandb sync results/runs/<run>/wandb/offline-run-*).
set -euo pipefail
KEY="${LAMBDA_KEY:-$HOME/.ssh/lambda_cl}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cmd="${1:?push|pull|ssh}"; host="${2:?user@ip}"; [[ "$host" == *@* ]] || host="ubuntu@$host"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
case "$cmd" in
  push)
    rsync -az -e "$SSH" \
      --exclude .venv --exclude data/real --exclude results \
      --exclude .git --exclude '__pycache__' --exclude '.pytest_cache' --exclude wandb \
      "$REPO/" "$host:ContinualLearningResearch/"
    # .env is gitignored but the box needs HF_TOKEN; force offline wandb there
    $SSH "$host" "cd ContinualLearningResearch && sed -i 's/^WANDB_MODE=.*/WANDB_MODE=offline/' .env && grep -c HF_TOKEN .env"
    ;;
  pull)
    rsync -az -e "$SSH" \
      --exclude '*.pt' \
      "$host:ContinualLearningResearch/results/" "$REPO/results/"
    rsync -az -e "$SSH" "$host:ContinualLearningResearch/manifests/" "$REPO/manifests/"
    ;;
  ssh)
    shift 2
    $SSH "$host" "cd ContinualLearningResearch && export PATH=\$HOME/.local/bin:\$PATH && $*"
    ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
