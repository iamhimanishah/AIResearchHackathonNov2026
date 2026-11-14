"""Checkpoint writing: weights (optional) + JSON sidecar with step / token counts / config hash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    out_dir: Path,
    index: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    meta: dict[str, Any],
    save_weights: bool,
    save_optimizer: bool = False,
) -> Path:
    ckpt_dir = out_dir / "checkpoints" / f"ckpt_{index:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if save_weights:
        # bare state_dict, not save_pretrained: smaller, faster, and we reload into the same class
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_dir / "model.pt")
        if optimizer is not None and save_optimizer:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    return ckpt_dir


def load_weights(model: torch.nn.Module, ckpt_dir: Path, device: str) -> None:
    sd = torch.load(ckpt_dir / "model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
