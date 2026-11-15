"""The interface of every part, as a Protocol, plus a trivial deterministic stub of each.

The Conductor is written against these Protocols only. Each real part (planner.py, gatekeeper.py,
operator.py, analyst.py, scout.py, reporter.py, memory.py, and workstream 2's executor) implements
one of them. The stubs here let the Conductor and the integration tests run before the real parts
exist. Stubs call no model and make no judgement worth keeping: they are scaffolding.

The request models are the typed inputs the Conductor assembles. They carry the firewall by
type: `PlanRequest`, `GateRequest` and `ReadRequest` hold `SelectionRow`; only `ReportRequest`
holds `ReportRow`. `Memory` accepts only `SelectionRow`, so the report half can never be stored
in memory or mirrored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from .context import Prompt
from .contracts import (
    BudgetView,
    Candidate,
    Catalog,
    CitedNumber,
    ClaimCard,
    Decision,
    Dropped,
    FailureEntry,
    Finding,
    Frozen,
    Lesson,
    Phase,
    Predicted,
    Prereg,
    Proposal,
    ProposedSpec,
    ReportRow,
    ResultRow,
    RunSpec,
    SelectionRow,
    Source,
    StreamDomain,
    TriedEntry,
    Verdict,
    arm_label,
    arm_of,
    build_spec,
    config_hash,
    finding_id,
    passes_acquisition_floor,
    retryable,
    spec_hash,
    technique_map,
    technique_names,
    usable,
    utc_iso,
)
from .llm import LLMClient, Usage

T = TypeVar("T", bound=BaseModel)


class ExecutorUnreachable(RuntimeError):
    """The executor did not answer. LAUNCH halts after `max_unreachable` of these in a row."""


# --------------------------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        import time

        time.sleep(seconds)


class FakeClock:
    """A clock tests control. `sleep` advances it instantly, so a WAIT loop costs no real time."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 11, 14, 20, 0, tzinfo=UTC)
        self.slept_seconds = 0.0

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float = 0.0, minutes: float = 0.0) -> None:
        self._now += timedelta(seconds=seconds, minutes=minutes)

    def sleep(self, seconds: float) -> None:
        self.slept_seconds += seconds
        self.advance(seconds=seconds)


# --------------------------------------------------------------------------------------------
# Typed requests and results
# --------------------------------------------------------------------------------------------


class CallMeta(Frozen):
    """Which prompt version and model produced an output. Copied into the decision log."""

    agent: str
    model: str
    prompt_hash: str
    card_hash: str
    skill_hashes: dict[str, str] = Field(default_factory=dict)
    usage: Usage | None = None


class PlanRequest(Frozen):
    phase: Phase
    prereg: Prereg
    catalog: Catalog
    budget: BudgetView
    stream: tuple[StreamDomain, ...]  # the frozen bench stream every spec must use
    rows: tuple[SelectionRow, ...] = ()  # latest row per spec hash, all phases so far
    findings: tuple[Finding, ...] = ()
    lessons: tuple[Lesson, ...] = ()
    rejections: tuple[Verdict, ...] = ()  # this turn's rejections, oldest first
    claim_cards: tuple[ClaimCard, ...] = ()  # accepted, not a PREREG repeat; combine only
    claim_seeds: tuple[int, ...] = ()  # seeds every claim test runs at; () = first two replicate


class GateRequest(Frozen):
    phase: Phase
    prereg: Prereg
    catalog: Catalog
    budget: BudgetView
    stream: tuple[StreamDomain, ...]
    rows: tuple[SelectionRow, ...] = ()
    pending_spec_hashes: tuple[str, ...] = ()  # launched, no row yet
    estimated_usd: dict[str, float] = Field(default_factory=dict)  # spec hash -> executor estimate
    bench: str = "fast"  # the frozen bench every spec must use
    retry_exhausted: tuple[str, ...] = ()  # spec hashes whose infra retries are used up
    lr_range: tuple[float, float] | None = None  # optim.lr an exploratory spec may set; None = none
    explore_usd_left: float | None = None  # what exploratory specs may still spend; None = nothing
    claim_arms: dict[str, tuple[str, ...]] = Field(default_factory=dict)  # accepted claim id ->
    # config hashes (seed left out) of its test and control arms


class LaunchReport(Frozen):
    submitted: tuple[str, ...] = ()  # spec hashes handed to the executor now
    skipped: tuple[str, ...] = ()  # already had a row or were already in flight


class WaitReport(Frozen):
    rows: tuple[ResultRow, ...] = ()  # final row per resolved spec
    requeued: tuple[str, ...] = ()  # infrastructure failures submitted a second time
    timed_out: tuple[str, ...] = ()  # no row by the deadline


class ReadRequest(Frozen):
    phase: Phase
    prereg: Prereg
    new_rows: tuple[SelectionRow, ...]
    all_rows: tuple[SelectionRow, ...] = ()
    prior_findings: tuple[Finding, ...] = ()
    attribution: dict[str, Any] | None = None  # workstream 4 output, selection half only


class ScoutRequest(Frozen):
    """What the Scout's code checks need. Only `catalog` and `findings` reach the model; the
    rest (hashes, seeds, the stream) is for the deterministic checks of each claim card."""

    catalog: Catalog
    findings: tuple[Finding, ...] = ()
    stream: tuple[StreamDomain, ...] = ()  # the frozen stream claim specs are built on
    seeds: tuple[int, ...] = (0,)  # seeds a claim test would run at
    finished_config_hashes: tuple[str, ...] = ()  # configurations (seed left out) with a row
    design_arms: dict[str, int] = Field(default_factory=dict)  # arm key -> PREREG claim id


class ScoutResult(Frozen):
    """The ranked shortlist: claim cards the loop may test, candidates a person reviews."""

    claims: tuple[ClaimCard, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    rejected: tuple[str, ...] = ()  # one line per draft item the checks dropped, with the reason
    sources: tuple[Source, ...] = ()
    skipped_reason: str | None = None  # set when both lists are empty
    meta: CallMeta | None = None
    searches: int = 0  # web searches attempted (one per query; for the Exa budget count)
    search_errors: tuple[str, ...] = ()  # one line per query that failed, even if others worked
    memory_kept: int = 0  # past.dev statements shown to the Scout (after the firewall filter)
    memory_dropped: int = 0  # recalled statements the filter withheld (analyst, measured effects)


class ReportRequest(Frozen):
    prereg: Prereg
    catalog: Catalog
    frozen_set_hash: str
    rows: tuple[ReportRow, ...]
    findings: tuple[Finding, ...] = ()
    decisions: tuple[Decision, ...] = ()


class ReportResult(Frozen):
    markdown: str
    data: dict[str, Any] = Field(default_factory=dict)  # content of report.json
    template_only: bool = False  # True when no model prose was used
    meta: CallMeta | None = None


# --------------------------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------------------------


@runtime_checkable
class Executor(Protocol):
    """Workstream 2's side of the contract. Idempotent by `spec_hash`."""

    def submit(self, specs: Sequence[RunSpec]) -> None:
        """Queue runs. A spec that already has a final row or is in flight is ignored.
        Raises `ExecutorUnreachable` when the backend does not answer."""
        ...

    def poll(self) -> list[ResultRow]:
        """Rows that landed since the last poll of this instance. `rows.jsonl` is the truth:
        after a restart, read the file instead of trusting what poll returns."""
        ...

    def estimate_cost(self, spec: RunSpec) -> float:
        """Estimated dollars for one run of `spec`."""
        ...

    def healthy(self) -> bool: ...


@runtime_checkable
class Planner(Protocol):
    def plan(self, request: PlanRequest) -> tuple[Proposal, CallMeta]:
        """Model step. Raises `LLMError` when no valid proposal could be produced."""
        ...

    def fallback(self, request: PlanRequest) -> Proposal:
        """Deterministic plan for the phase. Must not fail and must pass the gate."""
        ...


@runtime_checkable
class Gatekeeper(Protocol):
    def review(self, proposal: Proposal, request: GateRequest) -> Verdict:
        """Pure code. Never repairs a proposal and never consults a model."""
        ...


@runtime_checkable
class Operator(Protocol):
    def launch(self, specs: Sequence[RunSpec]) -> LaunchReport: ...

    def wait(self, spec_hashes: Sequence[str], deadline: datetime) -> WaitReport:
        """Poll until every spec has a final row or `deadline` passes. Re-queues an
        infrastructure failure once. Never retries a scientific failure."""
        ...


@runtime_checkable
class Analyst(Protocol):
    def read(self, request: ReadRequest) -> tuple[tuple[Finding, ...], CallMeta | None]:
        """Findings whose every number passed the checker. Raises `LLMError` on model failure."""
        ...

    def deterministic_findings(self, request: ReadRequest) -> tuple[Finding, ...]:
        """Findings from code alone, used when the model step fails twice."""
        ...


@runtime_checkable
class Scout(Protocol):
    def scout(self, request: ScoutRequest) -> ScoutResult:
        """One candidate with sources, or a skip. Never raises: any failure is a skip."""
        ...


@runtime_checkable
class Reporter(Protocol):
    def report(self, request: ReportRequest) -> ReportResult:
        """The final report. Falls back to a template-only report; never raises for the model."""
        ...


@runtime_checkable
class Memory(Protocol):
    """Local files are the truth; the shared mirror is best effort and never raises."""

    def record_rows(self, rows: Sequence[SelectionRow]) -> None:
        """Update `tried` and `failures` from selection-view rows. Idempotent by row hash."""
        ...

    def add_findings(self, findings: Sequence[Finding]) -> None: ...

    def add_lessons(self, lessons: Sequence[Lesson]) -> None: ...

    def tried(self) -> dict[str, TriedEntry]: ...

    def failures(self) -> list[FailureEntry]: ...

    def findings(self) -> list[Finding]: ...

    def lessons(self) -> list[Lesson]: ...

    def sync(self) -> bool:
        """Push local entries to the shared mirror. Returns False when the mirror is down."""
        ...


def run_prompt[T: BaseModel](llm: LLMClient, prompt: Prompt, schema: type[T]) -> tuple[T, CallMeta]:
    """Send an assembled prompt and return the validated output with its `CallMeta`."""
    result, usage = llm.complete(prompt.agent, prompt.system, prompt.user, schema)
    meta = CallMeta(
        agent=prompt.agent, model=usage.model, prompt_hash=usage.prompt_hash or prompt.hash,
        card_hash=prompt.card_hash, skill_hashes=prompt.skill_hashes, usage=usage,
    )
    return result, meta


# --------------------------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------------------------


def _proposed(request: PlanRequest, techniques: Mapping[str, Mapping[str, Any]], seed: int,
              role: str, reason: str, parent: str | None = None) -> ProposedSpec:
    arm = arm_label(techniques)
    spec = build_spec(
        f"{request.phase.value}-{arm}-s{seed}", seed, request.stream, techniques,
        parent_spec_hash=parent,
    )
    predicted = Predicted(retention_mean_delta=0.0, acquisition_ratio=1.0)
    return ProposedSpec(spec=spec, role=role, reason=reason, predicted=predicted)


def _defaults(catalog: Catalog, name: str) -> dict[str, Any]:
    return {key: param.default for key, param in catalog.get(name).params.items()}


def _control_arms(catalog: Catalog) -> list[dict[str, dict[str, Any]]]:
    arms: list[dict[str, dict[str, Any]]] = [{}]
    arms += [{e.name: _defaults(catalog, e.name)} for e in catalog.controls() if e.role != "none"]
    return arms


def _valid_treatments(request: PlanRequest) -> list[SelectionRow]:
    """Finished non-control rows that pass the acquisition floor, best retention first."""
    controls = {e.name for e in request.catalog.controls()}
    base = [r.acquisition_mean for r in request.rows
            if usable(r, request.prereg) and not r.spec.techniques and r.acquisition_mean]
    baseline = sum(base) / len(base) if base else None
    keep = []
    for row in request.rows:
        if not usable(row, request.prereg) or not row.spec.techniques:
            continue
        if row.spec.notes.startswith("explore:") or row.spec.name.startswith("explore-"):
            # exploratory rows (claim tests, hand-launched explore-* runs) never seed a phase
            continue
        if set(technique_names(row.spec)) <= controls or row.acquisition_mean is None:
            continue
        delta = request.prereg.acquisition_floor_delta
        if baseline is None or passes_acquisition_floor(row.acquisition_mean, baseline, delta):
            keep.append(row)
    return sorted(keep, key=lambda r: (-r.retention_selection.mean_delta, r.spec_hash))


class StubPlanner:
    """Deterministic plans, no model. `plan` and `fallback` return the same proposal.

    screen: every catalog entry alone, at the screen seeds. replicate: the controls and the top
    `top_k` screened treatments, at the replicate seeds. combine: the pair of the top two
    replicated treatments plus the controls, at the replicate seeds.
    """

    def __init__(self, top_k: int = 3) -> None:  # PREREG 6.3 promotes the top 3
        self.top_k = top_k

    def plan(self, request: PlanRequest) -> tuple[Proposal, CallMeta]:
        meta = CallMeta(agent="planner", model="stub", prompt_hash="", card_hash="")
        return self.fallback(request), meta

    def fallback(self, request: PlanRequest) -> Proposal:
        builders = {
            Phase.SCREEN: self._screen, Phase.REPLICATE: self._replicate,
            Phase.COMBINE: self._combine,
        }
        specs = builders[request.phase](request)
        return Proposal(
            phase=request.phase, specs=tuple(specs),
            rationale=f"stub: deterministic {request.phase.value} plan",
        )

    def _controls(self, request: PlanRequest, seeds: Sequence[int]) -> list[ProposedSpec]:
        return [
            _proposed(request, arm, seed, "control", "control arm")
            for arm in _control_arms(request.catalog) for seed in seeds
        ]

    def _screen(self, request: PlanRequest) -> list[ProposedSpec]:
        seeds = request.prereg.screen_seeds
        out = self._controls(request, seeds)
        for entry in request.catalog.entries:
            if entry.is_control:
                continue
            arm = {entry.name: _defaults(request.catalog, entry.name)}
            out += [_proposed(request, arm, seed, "treatment", "screen it alone") for seed in seeds]
        return out

    def _top_arms(self, request: PlanRequest) -> list[SelectionRow]:
        seen, top = set(), []
        for row in _valid_treatments(request):
            key = config_hash(row.spec)
            if key not in seen and len(row.spec.techniques) == 1:
                seen.add(key)
                top.append(row)
        return top[: self.top_k]

    def _replicate(self, request: PlanRequest) -> list[ProposedSpec]:
        seeds = request.prereg.replicate_seeds
        out = self._controls(request, seeds)
        for row in self._top_arms(request):
            out += [
                _proposed(request, technique_map(row.spec), seed, "replicate", "top of the screen",
                          parent=row.spec_hash)
                for seed in seeds
            ]
        return out

    def _combine(self, request: PlanRequest) -> list[ProposedSpec]:
        seeds = request.prereg.replicate_seeds
        out = self._controls(request, seeds)
        top = self._top_arms(request)
        if len(top) >= 2:
            combo = {**technique_map(top[0].spec), **technique_map(top[1].spec)}
            out += [_proposed(request, combo, seed, "combo", "top two together") for seed in seeds]
        return out + self._claim_tests(request)

    def _claim_tests(self, request: PlanRequest) -> list[ProposedSpec]:
        """Each claim card's test arm paired with its control at two replicate seeds, labelled
        `explore:<card id>`. Exploratory: never part of the win rule (SCOUT-HANDOFF 3.5)."""
        from .catalog_real import claim_specs

        out = []
        for card in request.claim_cards:
            expect = card.prediction.direction * card.prediction.size_points / 100
            seeds = request.claim_seeds or request.prereg.replicate_seeds[:2]
            for spec, which in claim_specs(card, seeds, request.stream):
                role = "control" if which == "control" and not spec.techniques else "treatment"
                out.append(ProposedSpec(
                    spec=spec, role=role, reason=f"claim {card.id} {which} arm",
                    predicted=Predicted(retention_mean_delta=expect if which == "test" else 0.0,
                                        acquisition_ratio=1.0)))
        return out


class StubGatekeeper:
    """Accepts everything except exact duplicates. Give it `script` to force verdicts in tests.

    Each scripted item is a `Verdict` or a callable `(proposal, request) -> Verdict`; items are
    used in order, then the default rule applies.
    """

    def __init__(self, script: Sequence[Any] = ()) -> None:
        self._script = list(script)
        self.reviewed: list[Proposal] = []

    def review(self, proposal: Proposal, request: GateRequest) -> Verdict:
        self.reviewed.append(proposal)
        if self._script:
            item = self._script.pop(0)
            return item(proposal, request) if callable(item) else item
        known = {r.spec_hash for r in request.rows if not retryable(r)}
        known |= set(request.pending_spec_hashes)
        kept, dropped = [], []
        for key in dict.fromkeys(proposal.spec_hashes()):
            if key in known:
                dropped.append(Dropped(spec_hash=key, rule="duplicate", reason="already run"))
            else:
                kept.append(key)
        if not dropped:
            return Verdict(token="ACCEPT", kept=tuple(kept))
        if not kept:
            return Verdict(token="REJECT", dropped=tuple(dropped))
        return Verdict(token="TRIM", kept=tuple(kept), dropped=tuple(dropped))


class StubOperator:
    """Submits, then polls on the injected clock. Re-queues an infrastructure failure once."""

    def __init__(self, executor: Executor, clock: Clock, poll_seconds: float = 60.0) -> None:
        self.executor = executor
        self.clock = clock
        self.poll_seconds = poll_seconds
        self._specs: dict[str, RunSpec] = {}

    def launch(self, specs: Sequence[RunSpec]) -> LaunchReport:
        fresh = {spec_hash(s): s for s in specs}
        self._specs.update(fresh)
        self.executor.submit(list(fresh.values()))
        return LaunchReport(submitted=tuple(fresh))

    def wait(self, spec_hashes: Sequence[str], deadline: datetime) -> WaitReport:
        final: dict[str, ResultRow] = {}
        requeued: list[str] = []
        while True:
            for row in self.executor.poll():
                self._settle(row, final, requeued)
            missing = [h for h in spec_hashes if h not in final]
            if not missing or self.clock.now() >= deadline:
                return WaitReport(
                    rows=tuple(final.values()), requeued=tuple(requeued), timed_out=tuple(missing)
                )
            self.clock.sleep(self.poll_seconds)

    def _settle(self, row: ResultRow, final: dict[str, ResultRow], requeued: list[str]) -> None:
        retry = retryable(row) and row.spec_hash not in requeued
        if retry and row.spec_hash in self._specs:
            requeued.append(row.spec_hash)
            self.executor.submit([self._specs[row.spec_hash]])
        else:
            final[row.spec_hash] = row


class StubAnalyst:
    """One checked finding per finished new row, straight from the row. No model."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or FakeClock()

    def read(self, request: ReadRequest) -> tuple[tuple[Finding, ...], CallMeta | None]:
        return self.deterministic_findings(request), None

    def deterministic_findings(self, request: ReadRequest) -> tuple[Finding, ...]:
        return tuple(self._finding(row) for row in request.new_rows if row.status == "finished")

    def _finding(self, row: SelectionRow) -> Finding:
        value = row.retention_selection.mean_delta
        arm, seed = arm_of(row.spec), row.spec.seed
        claim = f"{arm} seed {seed}: selection-half mean retention changed by {value:.3f}"
        number = CitedNumber(
            label="mean_delta", value=value, row_hash=row.row_hash,
            field_path="retention_selection.mean_delta",
        )
        return Finding(
            id=finding_id("effect", claim, [row.row_hash]), kind="effect", claim=claim,
            evidence_row_hashes=(row.row_hash,), numbers=(number,), confidence="low",
            author="analyst:stub", created_at=utc_iso(self.clock.now()),
        )


class StubScout:
    """Always skips."""

    def scout(self, request: ScoutRequest) -> ScoutResult:
        return ScoutResult(skipped_reason="stub scout: search is not wired")


class StubReporter:
    """A template-only report: one line per report-view row. No model."""

    def report(self, request: ReportRequest) -> ReportResult:
        lines = [f"# Report (stub)\n\nFrozen set: `{request.frozen_set_hash}`\n"]
        table = {}
        for row in request.rows:
            half = row.retention_report
            value = half.mean_delta if half else None
            table[row.row_hash] = value
            lines.append(f"- {arm_of(row.spec)} seed {row.spec.seed}: {row.status}, report {value}")
        data = {"frozen_set_hash": request.frozen_set_hash, "report_mean_delta": table}
        return ReportResult(markdown="\n".join(lines) + "\n", data=data, template_only=True)


class InMemoryMemory:
    """`Memory` kept in dictionaries. Nothing is written to disk or mirrored."""

    def __init__(self) -> None:
        self._tried: dict[str, TriedEntry] = {}
        self._failures: dict[str, FailureEntry] = {}
        self._findings: dict[str, Finding] = {}
        self._lessons: dict[str, Lesson] = {}

    def record_rows(self, rows: Sequence[SelectionRow]) -> None:
        for row in rows:
            if type(row) is not SelectionRow:
                raise TypeError("memory stores selection-view rows only")
            half = row.retention_selection
            self._tried[row.spec_hash] = TriedEntry(
                spec_hash=row.spec_hash, arm=arm_of(row.spec), seed=row.spec.seed,
                status=row.status, mean_delta=half.mean_delta if half else None,
                acquisition_mean=row.acquisition_mean, row_hash=row.row_hash,
            )
            if row.failure_kind:
                self._failures[row.row_hash] = FailureEntry(
                    spec_hash=row.spec_hash, arm=arm_of(row.spec), seed=row.spec.seed,
                    failure_kind=row.failure_kind, reason=row.reason, row_hash=row.row_hash,
                )

    def add_findings(self, findings: Sequence[Finding]) -> None:
        self._findings.update({f.id: f for f in findings})

    def add_lessons(self, lessons: Sequence[Lesson]) -> None:
        self._lessons.update({item.id: item for item in lessons})

    def tried(self) -> dict[str, TriedEntry]:
        return dict(self._tried)

    def failures(self) -> list[FailureEntry]:
        return list(self._failures.values())

    def findings(self) -> list[Finding]:
        return list(self._findings.values())

    def lessons(self) -> list[Lesson]:
        return list(self._lessons.values())

    def sync(self) -> bool:
        return True


class InstantExecutor:
    """An `Executor` that finishes every run on the next poll, using a row function you give it.

    For a simulated bench with timing, failures and a rows file, use `sim.SimExecutor`.
    """

    def __init__(self, make_row: Callable[[RunSpec], ResultRow], usd_per_run: float = 1.0) -> None:
        self._make_row = make_row
        self._usd = usd_per_run
        self._queue: list[RunSpec] = []
        self.launched: list[str] = []

    def submit(self, specs: Sequence[RunSpec]) -> None:
        for spec in specs:
            if spec_hash(spec) not in self.launched:
                self.launched.append(spec_hash(spec))
                self._queue.append(spec)

    def poll(self) -> list[ResultRow]:
        rows, self._queue = [self._make_row(spec) for spec in self._queue], []
        return rows

    def estimate_cost(self, spec: RunSpec) -> float:
        return self._usd

    def healthy(self) -> bool:
        return True
