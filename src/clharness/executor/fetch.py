"""Copy finished runs' per-item files from the Modal work volume into results/items/.

One helper, used by `clh fetch-items` and by the explore loop after each launch.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from clharness.executor.run import items_path, run_name
from clharness.spec import read_rows, write_rows

DEFAULT_VOLUME = "clh-work"


def fetch_items(files: Sequence[str | Path], volume: str, repo_root: Path, *,
                say: Callable[[str], None] = print) -> int:
    """For every finished Modal row in `files` whose per-item files are not under
    `repo_root`/results/items yet, download them with `modal volume get`, and point the row's
    `items_path` at the local copy. Returns how many runs could not be fetched."""
    missing = 0
    for path in files:
        if not Path(path).exists():
            continue
        rows, changed = read_rows(path), False
        for i, row in enumerate(rows):
            if row.retention is None or row.backend != "modal":
                continue
            local = Path(repo_root) / items_path(row.spec)
            if not list(local.glob("ckpt_*/items_*.npy")):
                local.parent.mkdir(parents=True, exist_ok=True)
                remote = f"runs/{run_name(row.spec)}/checkpoints"
                res = subprocess.run(["modal", "volume", "get", "--force", volume, remote, str(local.parent)],
                                     capture_output=True, text=True, check=False)
                if res.returncode != 0 or not list(local.glob("ckpt_*/items_*.npy")):
                    missing += 1
                    print(f"MISSING {row.spec.name} {row.spec_hash[:12]}: {res.stderr.strip()[-200:]}", file=sys.stderr)
                    continue
                say(f"fetched {row.spec.name} -> {items_path(row.spec)}")
            if row.items_path != items_path(row.spec):  # older rows pointed into the cloud volume
                rows[i] = row.model_copy(update={"items_path": items_path(row.spec)})
                changed = True
        if changed:
            write_rows(path, rows)
            say(f"updated items_path in {path}")
    return missing
