"""C5 arms: the training schedule each arm produces.

A schedule is a list of stages; a stage is a list of batches; a batch is a list of
`BatchItem`s — (task, index into that task's frozen train subsample). Examples are never
duplicated within an epoch: with 2 epochs per stage every scheduled example is presented
exactly twice, so example-presentations are token-matched across arms by construction.

  naive       stage t trains on task t's examples only
  replay-200  stage t mixes a fixed seeded buffer of `replay_per_task` examples per earlier
              task in BY REPLACEMENT: each replayed example replaces a new-task example, so
              the stage's size (and its token count, up to length differences) equals naive's
  joint       one stage on the shuffled union of all tasks' examples — its total equals the
              naive run's total across stages

Replayed examples are marked `replay=True` so the runner can record the realized replay
share (of tokens) in the row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from clharness.tinker.data import TASK_ORDER

ARMS = ("naive", "replay-200", "joint")
REPLAY_PER_TASK = 200


def _arm_code(arm: str) -> int:
    # SeedSequence entropy must be ints; crc32 keeps schedules stable across processes.
    import zlib

    return zlib.crc32(arm.encode())


@dataclass(frozen=True)
class BatchItem:
    task: str
    index: int  # position in the task's frozen train subsample
    replay: bool = False


@dataclass
class Stage:
    index: int
    batches: list[list[BatchItem]] = field(default_factory=list)
    replay_share: float = 0.0  # share of the stage's items that are replay

    def n_items(self) -> int:
        return sum(len(b) for b in self.batches)


@dataclass
class Schedule:
    arm: str
    seed: int
    epochs: int
    batch_size: int
    n_train: int
    stages: list[Stage]

    def tasks_seen_at(self, stage_index: int) -> list[str]:
        """Tasks trained on at or before this stage, in stream order — what stage `s` evals."""
        seen: set[str] = set()
        for st in self.stages[: stage_index + 1]:
            seen.update(it.task for b in st.batches for it in b)
        return [t for t in TASK_ORDER if t in seen]

    def n_items(self) -> int:
        return sum(s.n_items() for s in self.stages)


def _batches(items: list[BatchItem], epochs: int, batch_size: int,
             rng: np.random.Generator) -> list[list[BatchItem]]:
    out: list[list[BatchItem]] = []
    for _ in range(epochs):
        perm = rng.permutation(len(items))
        for lo in range(0, len(items), batch_size):
            out.append([items[int(j)] for j in perm[lo : lo + batch_size]])
    return out


def build_schedule(
    arm: str,
    n_train: int,
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 0,
    replay_per_task: int = REPLAY_PER_TASK,
) -> Schedule:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}: choose from {ARMS}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _arm_code(arm)]))
    n_tasks = len(TASK_ORDER)
    sched = Schedule(arm=arm, seed=seed, epochs=epochs, batch_size=batch_size, n_train=n_train,
                     stages=[])

    if arm == "joint":
        items = [BatchItem(t, i) for t in TASK_ORDER for i in range(n_train)]
        sched.stages.append(Stage(index=0, batches=_batches(items, epochs, batch_size, rng)))
        return sched

    # The replay buffers are fixed for the whole run: `per` examples per task, seeded.
    # Capped so a stage never replaces more examples than it has (matters only for --smoke).
    per = min(replay_per_task, n_train // n_tasks) if arm == "replay-200" else 0
    buffers = {
        t: [int(x) for x in rng.choice(n_train, size=per, replace=n_train < per)]
        for t in TASK_ORDER
    } if per else {}

    for t, task in enumerate(TASK_ORDER):
        replay_items = [BatchItem(et, i, replay=True)
                        for et in TASK_ORDER[:t] for i in buffers.get(et, [])]
        n_replay = len(replay_items)
        # `n_replay` new-task examples are dropped so the stage's size stays n_train.
        keep = sorted(set(range(n_train))
                      - {int(x) for x in rng.choice(n_train, size=n_replay, replace=False)})
        items = [BatchItem(task, i) for i in keep] + replay_items
        stage = Stage(index=t)
        stage.batches = _batches(items, epochs, batch_size, rng)
        stage.replay_share = n_replay / max(1, len(items))
        sched.stages.append(stage)
    return sched
