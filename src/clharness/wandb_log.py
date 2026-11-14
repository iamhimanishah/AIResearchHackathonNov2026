"""wandb logging. Offline unless WANDB_PROJECT (or cfg.wandb.project) is set and WANDB_MODE=online.

Every run logs: the fully-resolved config, the data + eval manifests as artifacts, the git
commit, and per-checkpoint metrics. Feature-level arrays (per-feature freq / magnitude) are
saved as .npz files under the run's output dir and attached as artifacts, not logged as
scalars (16k x 3 layers of them per checkpoint would swamp the UI).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from clharness.config import REPO_ROOT


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


class RunLogger:
    def __init__(self, cfg: dict[str, Any], run_name: str | None, project: str | None, tags: list[str], output_dir: Path):
        import wandb

        project = project or os.environ.get("WANDB_PROJECT") or None
        mode = os.environ.get("WANDB_MODE", "offline")
        if project is None and mode == "online":
            mode = "offline"
        self.output_dir = output_dir
        self.run = wandb.init(
            project=project or "clharness-offline",
            name=run_name,
            tags=tags,
            config={**cfg, "git_commit": git_commit()},
            mode=mode,
            dir=str(output_dir),
        )
        self._metrics_file = open(output_dir / "metrics.jsonl", "a")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.run.log(flat, step=step)
        self._metrics_file.write(json.dumps({"step": step, **metrics}, default=float) + "\n")
        self._metrics_file.flush()

    def attach_file(self, path: Path, kind: str) -> None:
        import wandb

        art = wandb.Artifact(name=f"{kind}-{path.stem}", type=kind)
        art.add_file(str(path))
        self.run.log_artifact(art)

    def finish(self) -> None:
        self._metrics_file.close()
        self.run.finish()
