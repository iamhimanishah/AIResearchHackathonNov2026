"""The loop's states and the transition table (ARCHITECTURE.md section 3), as data.

A state is a `(step, phase)` pair. The cycle steps PLAN, FALLBACK_PLAN, GATE, LAUNCH, WAIT, READ
and ADVANCE exist once per phase. INIT, PREFLIGHT, SCOUT, FREEZE, REPORT, DONE and HALTED have no
phase. The Conductor may only move along `TRANSITIONS`; `check_transition` raises otherwise.

`prove_exits()` returns the list of ways the table could trap the loop. A kernel test asserts
that the list is empty, so a change that strands a state fails the suite.

`LoopState` is the content of `results/agent/state.json`: written by the Conductor after every
transition, read by the Conductor on resume and by the page.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    FREEZE_STATE_ID,
    GENESIS_HASH,
    Frozen,
    Phase,
    Proposal,
    Verdict,
    atomic_write_json,
)

__all__ = [
    "ACTORS", "CYCLE_STEPS", "DONE", "FREEZE", "HALTED", "INIT", "PHASE_ORDER", "PREFLIGHT",
    "REPORT", "SCOUT", "TRANSITIONS", "IllegalTransition", "LoopState", "Phase", "State", "Step",
    "Transition", "all_states", "allowed_next", "check_transition", "is_terminal", "next_phase",
    "prove_exits", "st",
]


class Step(StrEnum):
    INIT = "init"
    PREFLIGHT = "preflight"
    PLAN = "plan"
    FALLBACK_PLAN = "fallback_plan"
    GATE = "gate"
    LAUNCH = "launch"
    WAIT = "wait"
    READ = "read"
    ADVANCE = "advance"
    SCOUT = "scout"
    FREEZE = "freeze"
    REPORT = "report"
    DONE = "done"
    HALTED = "halted"


CYCLE_STEPS: tuple[Step, ...] = (
    Step.PLAN, Step.FALLBACK_PLAN, Step.GATE, Step.LAUNCH, Step.WAIT, Step.READ, Step.ADVANCE,
)
PHASE_ORDER: tuple[Phase, ...] = (Phase.SCREEN, Phase.REPLICATE, Phase.COMBINE)

# Who acts in each step. "code" is the deterministic fallback planner.
ACTORS: dict[Step, str] = {
    Step.INIT: "conductor", Step.PREFLIGHT: "conductor", Step.PLAN: "planner",
    Step.FALLBACK_PLAN: "code", Step.GATE: "gatekeeper", Step.LAUNCH: "operator",
    Step.WAIT: "operator", Step.READ: "analyst", Step.ADVANCE: "conductor",
    Step.SCOUT: "scout", Step.FREEZE: "conductor", Step.REPORT: "reporter",
    Step.DONE: "-", Step.HALTED: "-",
}


class IllegalTransition(RuntimeError):
    """The Conductor tried a move that is not in the transition table."""


class State(Frozen):
    """One state of the loop. Hashable, so it can key a dict. `id` is the form used in logs."""

    step: Step
    phase: Phase | None = None

    @model_validator(mode="after")
    def _phase_matches_step(self) -> State:
        in_cycle = self.step in CYCLE_STEPS
        if in_cycle and self.phase is None:
            raise ValueError(f"step {self.step} needs a phase")
        if not in_cycle and self.phase is not None:
            raise ValueError(f"step {self.step} has no phase")
        return self

    @property
    def id(self) -> str:
        return f"{self.phase}/{self.step}" if self.phase else str(self.step)

    @classmethod
    def from_id(cls, text: str) -> State:
        phase, _, step = text.rpartition("/")
        return cls(step=Step(step), phase=Phase(phase) if phase else None)

    def __str__(self) -> str:
        return self.id


def st(step: Step, phase: Phase | None = None) -> State:
    """Short constructor: `st(Step.PLAN, Phase.SCREEN)`."""
    return State(step=step, phase=phase)


INIT = st(Step.INIT)
PREFLIGHT = st(Step.PREFLIGHT)
SCOUT = st(Step.SCOUT)
FREEZE = st(Step.FREEZE)
REPORT = st(Step.REPORT)
DONE = st(Step.DONE)
HALTED = st(Step.HALTED)
assert FREEZE.id == FREEZE_STATE_ID


class Transition(Frozen):
    to: State
    when: str  # the condition, in words; the Conductor implements it


def next_phase(phase: Phase) -> Phase | None:
    """The phase after `phase`, or None after the last one."""
    index = PHASE_ORDER.index(phase)
    return PHASE_ORDER[index + 1] if index + 1 < len(PHASE_ORDER) else None


def _edge(to: State, when: str) -> Transition:
    return Transition(to=to, when=when)


_STOP = "compute budget, model budget or deadline reached, or a freeze_now veto"


def _cycle(phase: Phase) -> dict[State, list[Transition]]:
    plan, fallback = st(Step.PLAN, phase), st(Step.FALLBACK_PLAN, phase)
    following = {Phase.SCREEN: Phase.REPLICATE, Phase.REPLICATE: Phase.COMBINE}.get(phase)
    skip = ([_edge(st(Step.PLAN, following), "the deterministic plan has nothing new to run")]
            if following else [])
    if phase is Phase.REPLICATE:
        skip.append(_edge(SCOUT, "nothing new to run in replicate and scout is enabled"))
    gate, launch = st(Step.GATE, phase), st(Step.LAUNCH, phase)
    wait, read, advance = st(Step.WAIT, phase), st(Step.READ, phase), st(Step.ADVANCE, phase)
    return {
        plan: [
            _edge(gate, "a schema-valid proposal exists"),
            _edge(fallback, "model error or invalid output twice"),
        ],
        fallback: [_edge(gate, "always: the deterministic plan cannot fail")],
        gate: [
            _edge(launch, "verdict ACCEPT or TRIM"),
            _edge(plan, "verdict REJECT and fewer than max_gate_rejections so far"),
            _edge(fallback, "verdict REJECT and max_gate_rejections reached"),
            *skip,
        ],
        launch: [_edge(wait, "accepted specs not already in rows are submitted")],
        wait: [_edge(read, "every spec has a row, or the phase timeout passed")],
        read: [_edge(advance, "findings stored (the list may be empty)")],
        advance: _advance_edges(phase, plan),
    }


def _advance_edges(phase: Phase, plan: State) -> list[Transition]:
    repeat = _edge(plan, "too many failures: repeat this phase once")
    if phase is Phase.SCREEN:
        return [_edge(st(Step.PLAN, Phase.REPLICATE), "screen done"), repeat]
    if phase is Phase.REPLICATE:
        combine = st(Step.PLAN, Phase.COMBINE)
        return [
            _edge(SCOUT, "replicate done, scout enabled and budget allows"),
            _edge(combine, "replicate done, scout disabled or not affordable"),
            repeat,
        ]
    return [_edge(FREEZE, "combine done"), repeat]


def _build_table() -> dict[State, tuple[Transition, ...]]:
    table: dict[State, list[Transition]] = {
        INIT: [_edge(PREFLIGHT, "config and state loaded, decision-log chain verifies")],
        PREFLIGHT: [_edge(st(Step.PLAN, Phase.SCREEN), "every preflight check passes")],
        SCOUT: [_edge(st(Step.PLAN, Phase.COMBINE), "candidate written, or skipped on failure")],
        FREEZE: [_edge(REPORT, "frozen set hash written and logged")],
        REPORT: [_edge(DONE, "report written (template-only if the checker fails twice)")],
        DONE: [],
        HALTED: [],
    }
    for phase in PHASE_ORDER:
        table.update(_cycle(phase))
    early_freeze = [s for s in table if s.step in CYCLE_STEPS or s == SCOUT]
    for state in early_freeze:
        if FREEZE not in [t.to for t in table[state]]:
            table[state].append(_edge(FREEZE, _STOP))
    for state, edges in table.items():
        if state not in (DONE, HALTED):
            edges.append(_edge(HALTED, "unrecoverable: needs a human"))
    return {state: tuple(edges) for state, edges in table.items()}


TRANSITIONS: dict[State, tuple[Transition, ...]] = _build_table()


def all_states() -> tuple[State, ...]:
    return tuple(TRANSITIONS)


def is_terminal(state: State) -> bool:
    """DONE and HALTED. HALTED is left only by a human restart (`LoopState.resume`)."""
    return state in (DONE, HALTED)


def allowed_next(state: State) -> tuple[State, ...]:
    """Every state the loop may move to from `state`."""
    return tuple(t.to for t in TRANSITIONS[state])


def check_transition(src: State, dst: State) -> None:
    if dst not in allowed_next(src):
        raise IllegalTransition(f"{src.id} -> {dst.id} is not in the transition table")


def _reachable(start: State, table: dict[State, tuple[Transition, ...]]) -> set[State]:
    seen, stack = {start}, [start]
    while stack:
        for edge in table.get(stack.pop(), ()):
            if edge.to not in seen:
                seen.add(edge.to)
                stack.append(edge.to)
    return seen


def prove_exits(table: dict[State, tuple[Transition, ...]] | None = None) -> list[str]:
    """Ways the table could trap the loop. An empty list is the proof that it cannot.

    Checks, for the given table (default: `TRANSITIONS`):
      1. every target of a transition is itself a state in the table;
      2. every non-terminal state has an exit other than HALTED;
      3. every non-terminal state can halt;
      4. DONE is reachable from every non-terminal state;
      5. every state is reachable from INIT.
    """
    table = TRANSITIONS if table is None else table
    problems = []
    for state, edges in table.items():
        targets = [e.to for e in edges]
        problems.extend(f"{state.id}: unknown target {t.id}" for t in targets if t not in table)
        if is_terminal(state):
            continue
        if not [t for t in targets if t != HALTED]:
            problems.append(f"{state.id}: no exit other than HALTED")
        if HALTED not in targets:
            problems.append(f"{state.id}: cannot halt")
        if DONE not in _reachable(state, table):
            problems.append(f"{state.id}: DONE is not reachable")
    from_init = _reachable(INIT, table)
    problems.extend(f"{s.id}: not reachable from INIT" for s in table if s not in from_init)
    return problems


class LoopState(BaseModel):
    """Content of `state.json`. The Conductor is the only writer.

    `move` enforces the transition table. Entering HALTED records where to resume; `resume`
    returns there. Parts that need more persisted state use `extra` (JSON values only).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state: State = INIT
    resume_from: State | None = None
    halted_reason: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    gate_rejections: int = 0  # rejections in the current PLAN/GATE turn
    phase_repeats: dict[str, int] = Field(default_factory=dict)  # phase -> times repeated
    compute_usd_spent: float = 0.0  # from rows that exist
    compute_usd_committed: float = 0.0  # estimated cost of launched runs without a row yet
    model_usd_spent: float = 0.0
    proposal: Proposal | None = None  # the proposal waiting for or past the gate this turn
    verdict: Verdict | None = None
    pending_spec_hashes: list[str] = Field(default_factory=list)  # launched, no row yet
    seen_row_hashes: list[str] = Field(default_factory=list)  # rows the Analyst has read
    scout_done: bool = False
    frozen_set_hash: str | None = None
    frozen_spec_hashes: list[str] = Field(default_factory=list)
    decision_count: int = 0
    decision_head: str = GENESIS_HASH
    extra: dict[str, Any] = Field(default_factory=dict)

    def move(self, to: State, now: str, reason: str | None = None) -> None:
        """Move along the table. Raises `IllegalTransition` for a move that is not in it."""
        check_transition(self.state, to)
        if to == HALTED:
            self.resume_from = self.state
            self.halted_reason = reason or "unspecified"
        self.state = to
        self.updated_at = now

    def resume(self, now: str) -> State:
        """Leave HALTED for the state that was interrupted. Only a human restart calls this."""
        if self.state != HALTED or self.resume_from is None:
            raise IllegalTransition("resume is only possible from HALTED")
        self.state, self.resume_from, self.halted_reason = self.resume_from, None, None
        self.updated_at = now
        return self.state

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self)

    @classmethod
    def load(cls, path: str | Path) -> LoopState:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
