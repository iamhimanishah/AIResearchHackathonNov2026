"""One writer per output directory (RT-15, RT-17).

    from clharness.agent.lock import single_writer, lock_or_exit

    with single_writer(out):      # raises AlreadyRunning if another process holds it
        ...
    lock_or_exit(args.out)        # in a CLI main: hold it for the life of the process

The lock is `fcntl.flock(LOCK_EX | LOCK_NB)` on `<out>/.agent.lock`. The kernel drops it when
the holder exits, even by SIGKILL, so a crash never leaves a stale lock. The file is never
unlinked (unlinking a flock file lets two processes lock two different inodes). The holder's
pid is written into it only for a person to read. Nested use in one process is a no-op, so
code that builds several Conductors in one process, or a CLI that also calls a locked helper,
does not lock itself out. POSIX only, and per host.
"""

from __future__ import annotations

import fcntl
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .contracts import ContractError

LOCK_NAME = ".agent.lock"
_HELD: dict[Path, int] = {}  # lock path -> open file descriptor, for this process


class AlreadyRunning(ContractError):
    """Another process holds the lock on this output directory."""


def lock_path(out_dir: str | Path) -> Path:
    return Path(out_dir).resolve() / LOCK_NAME


def _acquire(out_dir: str | Path) -> tuple[Path, bool]:
    """Take the lock. Returns (path, True) when this call took it, (path, False) when this
    process already held it."""
    path = lock_path(out_dir)
    if path in _HELD:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 64, 0).decode(errors="replace").strip() or "unknown"
        os.close(fd)
        raise AlreadyRunning(f"{path.parent} is in use by another process (pid {holder}); "
                             "one writer per output directory") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
    _HELD[path] = fd
    return path, True


def _release(path: Path) -> None:
    fd = _HELD.pop(path, None)
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def single_writer(out_dir: str | Path) -> Iterator[Path]:
    """Hold the lock on `out_dir` for the block. Raises `AlreadyRunning` when it is taken."""
    path, mine = _acquire(out_dir)
    try:
        yield path
    finally:
        if mine:
            _release(path)


def lock_or_exit(out_dir: str | Path) -> Path:
    """For a CLI main: hold the lock on `out_dir` until the process exits, or print `refused:`
    to stderr and exit 2 when another process holds it."""
    try:
        return _acquire(out_dir)[0]
    except AlreadyRunning as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
