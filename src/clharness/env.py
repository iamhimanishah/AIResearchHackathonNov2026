"""Environment: secrets, device detection, seeding.

Importing this module loads `.env` from the repo root (if present) into os.environ
without overriding variables that are already set.
"""

from __future__ import annotations

import os
import platform
import random
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)
# reduce fragmentation from the large transient logits tensors (must be set before CUDA init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def pick_device(preferred: str = "auto") -> str:
    """Return a torch device string. `preferred` may be auto|cuda|mps|cpu."""
    import torch

    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch (all devices).

    Determinism caveats (documented, not hidden):
      * CUDA: `torch.use_deterministic_algorithms(True)` needs CUBLAS_WORKSPACE_CONFIG=:4096:8
        set before CUDA init; we set it here if unset. Some ops (scatter/index_add on bf16)
        may still be nondeterministic and will raise if strictly deterministic mode is on;
        we use warn_only=True so runs proceed and log the warning.
      * MPS: no determinism guarantees from PyTorch; results are bitwise-repeatable in
        practice for the ops used here but this is not promised.
    """
    import numpy as np
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def describe_env() -> dict[str, str]:
    import torch

    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": pick_device(),
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(torch.backends.mps.is_available()),
        "hf_token_present": str(hf_token() is not None),
        "wandb_mode": os.environ.get("WANDB_MODE", "unset"),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_mem_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}"
    for mod in ("transformers", "sae_lens", "lm_eval", "peft", "datasets", "wandb"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"IMPORT FAILED: {e}"
    return info
