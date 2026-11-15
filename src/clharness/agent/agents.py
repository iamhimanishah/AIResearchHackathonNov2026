"""Model-backed roles: the LLM Planner, the LLM Analyst, and file-backed Memory.

Each role assembles its prompt with `context.build_*_prompt` (which enforces the firewall),
calls the model through an `LLMClient` (normally `BudgetedLLM`, which retries invalid output
and enforces the model budget), and validates the answer deterministically. A role never
moves the loop: on `LLMError` the Conductor takes the deterministic path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from .cards import CardStore
from .context import build_analyst_prompt, build_planner_prompt, build_scout_prompt
from .contracts import (
    Candidate,
    CitedNumber,
    FailureEntry,
    Finding,
    Lesson,
    ParamSpec,
    Prereg,
    Proposal,
    SelectionRow,
    Source,
    TriedEntry,
    atomic_write_text,
    check_numbers,
    finding_id,
    read_jsonl,
    utc_iso,
)
from .llm import InvalidOutput, LLMClient, LLMError
from .roles import (
    CallMeta,
    Clock,
    InMemoryMemory,
    PlanRequest,
    ReadRequest,
    ScoutRequest,
    ScoutResult,
    StubAnalyst,
    StubPlanner,
    SystemClock,
    run_prompt,
)
from .search import SearchClient, SearchError


class LLMPlanner:
    """Proposes specs with a model. `fallback` is the deterministic plan."""

    def __init__(self, llm: LLMClient, cards: CardStore | None = None,
                 fallback_planner: StubPlanner | None = None) -> None:
        self.llm = llm
        self.cards = cards or CardStore()
        self._fallback = fallback_planner or StubPlanner()

    def plan(self, request: PlanRequest) -> tuple[Proposal, CallMeta]:
        state_id = f"{request.phase.value}/plan"
        prompt = build_planner_prompt(
            card=self.cards.card("planner"), skills=self.cards.skills_for("planner", state_id),
            phase=request.phase, prereg=request.prereg, catalog=request.catalog,
            budget=request.budget, rows=request.rows, findings=request.findings,
            lessons=request.lessons, rejections=request.rejections,
        )
        proposal, meta = run_prompt(self.llm, prompt, Proposal)
        if proposal.phase != request.phase:
            raise InvalidOutput(f"proposal is for {proposal.phase}, not {request.phase}")
        known = {f.id for f in request.findings}
        unknown = [f for f in proposal.cites_findings if f not in known]
        if unknown:
            raise InvalidOutput(f"proposal cites unknown findings: {unknown[:3]}")
        return _clip(proposal), meta

    def fallback(self, request: PlanRequest) -> Proposal:
        return self._fallback.fallback(request)


MAX_RATIONALE = 2000
MAX_REASON = 300
MAX_FINDINGS_PER_READ = 20


def _clip(proposal: Proposal) -> Proposal:
    """Cut the model's free text so one long answer cannot bloat the log, state and prompts."""
    specs = tuple(p.model_copy(update={"reason": p.reason[:MAX_REASON]}) for p in proposal.specs)
    return proposal.model_copy(update={
        "rationale": proposal.rationale[:MAX_RATIONALE], "specs": specs,
        "cites_findings": tuple(proposal.cites_findings)[:MAX_FINDINGS_PER_READ],
    })


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.(\d+))?(%?)")


def unbacked_numbers(claim: str, allowed: Sequence[float]) -> list[str]:
    """Decimals and percentages in `claim` that no allowed value rounds to.

    Plain integers (seeds, counts, ranks) are not checked. A percentage is compared with 100x
    the value, and the sign is ignored ("cut forgetting by 0.08" for a delta of -0.08).
    """
    bad = []
    for match in _NUMBER.finditer(claim):
        decimals, percent = match.group(1), match.group(2)
        if decimals is None and not percent:
            continue
        shown = abs(float(match.group(0).rstrip("%")))
        slack = 0.5 * 10 ** -len(decimals or "") + 1e-9
        scale = 100.0 if percent else 1.0
        if not any(abs(shown - abs(v) * scale) <= slack for v in allowed):
            bad.append(match.group(0))
    return bad


def _allowed_values(numbers: Sequence[CitedNumber], rows: Mapping[str, SelectionRow],
                    prereg: Prereg) -> list[float]:
    """The cited numbers plus what the claim may also name: the cited rows' technique
    parameters and the preregistered constants."""
    out = [n.value for n in numbers]
    for n in numbers:
        row = rows.get(n.row_hash)
        if row is not None:
            out += [float(v) for t in row.spec.techniques for v in t.params.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out += [float(v) for v in prereg.model_dump().values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return out


def _cap_confidence(asked: str, evidence: Sequence[str], prereg: Prereg) -> str:
    """`high` needs a row for every replicate seed, `medium` at least two rows."""
    if asked == "high" and len(evidence) >= len(prereg.replicate_seeds):
        return "high"
    if asked in ("high", "medium") and len(evidence) >= 2:
        return "medium"
    return "low"


class DraftFinding(BaseModel):
    kind: Literal["effect", "interaction", "failure", "anomaly", "null_result"]
    claim: str
    numbers: list[CitedNumber] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class AnalystDraft(BaseModel):
    findings: list[DraftFinding]


class LLMAnalyst:
    """Writes findings with a model. Claims whose cited numbers do not match a row are dropped."""

    def __init__(self, llm: LLMClient, cards: CardStore | None = None,
                 clock: Clock | None = None) -> None:
        self.llm = llm
        self.cards = cards or CardStore()
        self.clock = clock or SystemClock()
        self._stub = StubAnalyst(self.clock)
        self.dropped: list[str] = []

    def read(self, request: ReadRequest) -> tuple[tuple[Finding, ...], CallMeta | None]:
        if not request.new_rows:
            return (), None
        prompt = build_analyst_prompt(
            card=self.cards.card("analyst"), skills=self.cards.skills_for("analyst", "read"),
            phase=request.phase, prereg=request.prereg, new_rows=request.new_rows,
            prior_findings=request.prior_findings, attribution=request.attribution,
            schema=AnalystDraft,
        )
        draft, meta = run_prompt(self.llm, prompt, AnalystDraft)
        rows = {r.row_hash: r for r in (*request.all_rows, *request.new_rows)}
        found, self.dropped = [], []
        for item in draft.findings:
            if len(found) >= MAX_FINDINGS_PER_READ:
                self.dropped.append(f"more than {MAX_FINDINGS_PER_READ} findings in one read")
                break
            numbers = tuple(n.model_copy(update={"row_hash": _full(n.row_hash, rows)})
                            for n in item.numbers)
            problems = check_numbers(numbers, rows) if numbers else ["no cited number"]
            if not problems:
                stray = unbacked_numbers(item.claim, _allowed_values(numbers, rows, request.prereg))
                if stray:
                    problems = [f"claim states {stray[:3]} with no cited row behind it"]
            if problems:
                self.dropped.append(f"{item.claim[:80]}: {problems[0]}")
                continue
            evidence = tuple(sorted({n.row_hash for n in numbers}))
            found.append(Finding(
                id=finding_id(item.kind, item.claim, evidence), kind=item.kind, claim=item.claim,
                evidence_row_hashes=evidence, numbers=numbers,
                confidence=_cap_confidence(item.confidence, evidence, request.prereg),
                author="analyst", created_at=utc_iso(self.clock.now()),
            ))
        return tuple(found), meta

    def deterministic_findings(self, request: ReadRequest) -> tuple[Finding, ...]:
        return self._stub.deterministic_findings(request)


# Queries the Scout's search step sends. Source preference (papers) is phrased in the query and
# the objective, not as a domain filter. Each query is one Exa search.
SCOUT_QUERIES: tuple[tuple[str, str], ...] = (
    (("research papers on mitigating catastrophic forgetting during continual pre-training "
      "of language models, 2025 and 2026"),
     ("Find papers that propose a concrete training-time method that reduces forgetting of "
      "general ability while a pretrained LLM is further trained on new-domain text. Rank first "
      "papers with a named method and measured results; exclude surveys, fine-tuning-only "
      "instruction tuning and RL papers. Pull the method's mechanism and reported effect.")),
    ("new method to reduce forgetting when continually pretraining an LLM on a new domain",
     ("Find recent arXiv or conference papers with a method that could be built as a hook on a "
      "standard training loop (loss term, gradient or parameter update rule, data mixing or "
      "learning-rate schedule). Rank first methods with ablations; exclude benchmark-only "
      "papers. Pull the update rule, its hyperparameters and the compute overhead.")),
)
MAX_SCOUT_SOURCES = 20


class CandidateDraft(BaseModel):
    name: str
    description: str
    mechanism: str
    why_not_in_catalog: str
    expected_effect: str
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    est_cost_multiplier: float = Field(gt=0)
    source_ids: list[str] = Field(min_length=1)


class ScoutDraft(BaseModel):
    candidate: CandidateDraft | None = None
    skip_reason: str | None = None  # set when candidate is null


class LLMScout:
    """Searches the web with Exa, then asks a model for at most one candidate technique.

    Code runs the search (the model never holds the key or a tool) and the model reads the
    results as untrusted data. The candidate must cite sources that were actually returned and
    must not reuse a catalog name. Never raises: any failure is a skip with a reason.
    """

    def __init__(self, llm: LLMClient, search: SearchClient, cards: CardStore | None = None,
                 clock: Clock | None = None,
                 queries: Sequence[tuple[str, str]] = SCOUT_QUERIES) -> None:
        self.llm = llm
        self.search = search
        self.cards = cards or CardStore()
        self.clock = clock or SystemClock()
        self.queries = tuple(queries)

    def gather(self) -> tuple[tuple[Source, ...], list[str]]:
        """Run every query; return sources deduplicated by URL and one error string per failure."""
        seen: dict[str, Source] = {}
        errors: list[str] = []
        for query, objective in self.queries:
            try:
                found = self.search.search(query, objective)
            except SearchError as exc:
                errors.append(str(exc))
                continue
            for src in found:
                seen.setdefault(src.url, src)
        return tuple(seen.values())[:MAX_SCOUT_SOURCES], errors

    def scout(self, request: ScoutRequest) -> ScoutResult:
        sources, errors = self.gather()
        if not sources:
            reason = "search returned nothing" + (f": {errors[0]}" if errors else "")
            return ScoutResult(skipped_reason=reason)
        prompt = build_scout_prompt(
            card=self.cards.card("scout"), skills=self.cards.skills_for("scout", "scout"),
            catalog=request.catalog, findings=request.findings, search_results=sources,
            schema=ScoutDraft,
        )
        try:
            draft, meta = run_prompt(self.llm, prompt, ScoutDraft)
        except LLMError as exc:
            return ScoutResult(sources=sources, skipped_reason=f"model failed: {exc}")
        item = draft.candidate
        if item is None:
            return ScoutResult(sources=sources, meta=meta,
                               skipped_reason=draft.skip_reason or "model skipped")
        known = {s.id for s in sources}
        cited = [sid for sid in item.source_ids if sid in known]
        problem = None
        if not cited:
            problem = f"candidate cites no returned source: {item.source_ids[:3]}"
        elif item.name.strip().lower() in {n.lower() for n in request.catalog.names()}:
            problem = f"candidate {item.name!r} is already in the catalog"
        if problem:
            return ScoutResult(sources=sources, meta=meta, skipped_reason=problem)
        candidate = Candidate(
            **item.model_dump(exclude={"source_ids", "params"}), params=item.params,
            source_ids=tuple(cited), created_at=utc_iso(self.clock.now()),
        )
        marked = tuple(s.model_copy(update={"used": s.id in cited}) for s in sources)
        return ScoutResult(candidate=candidate, sources=marked, meta=meta)


def write_scout_result(result: ScoutResult, out_dir: str | Path) -> Path:
    """Write the candidate card (or the skip) with every source it saw, as one JSON file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = (result.candidate.created_at if result.candidate else
             (result.sources[0].retrieved_at if result.sources else "none"))
    name = result.candidate.name if result.candidate else "skip"
    slug = re.sub(r"[^a-z0-9]+", "-", f"{stamp}-{name}".lower()).strip("-")[:80]
    path = out / f"{slug}.json"
    atomic_write_text(path, result.model_dump_json(indent=2) + "\n")
    return path


def _full(row_id: str, rows: Mapping[str, object]) -> str:
    """The full row hash for a row id the model copied (the prompt shows a prefix).

    An unknown or ambiguous id is returned as given, so the number checker rejects it.
    """
    matches = [h for h in rows if h.startswith(row_id)] if len(row_id) >= 8 else []
    return matches[0] if len(matches) == 1 else row_id


class FileMemory(InMemoryMemory):
    """`Memory` kept as one JSONL file per store under `root` (ARCHITECTURE section 5).

    Every write rewrites the store atomically, so a kill leaves the old or the new file.
    """

    STORES: ClassVar[dict[str, type[BaseModel]]] = {"tried": TriedEntry, "failures": FailureEntry, "findings": Finding,
              "lessons": Lesson}

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self._tried = {e.spec_hash: e for e in self._load("tried")}
        self._failures = {e.row_hash: e for e in self._load("failures")}
        self._findings = {e.id: e for e in self._load("findings")}
        self._lessons = {e.id: e for e in self._load("lessons")}

    def _load(self, store: str) -> list:
        model = self.STORES[store]
        return [model.model_validate(r) for r in read_jsonl(self.root / f"{store}.jsonl").records]

    def _save(self, store: str, items: Sequence[BaseModel]) -> None:
        lines = "".join(json.dumps(i.model_dump(mode="json"), sort_keys=True) + "\n"
                        for i in items)
        atomic_write_text(self.root / f"{store}.jsonl", lines)

    def record_rows(self, rows) -> None:
        super().record_rows(rows)
        self._save("tried", list(self._tried.values()))
        self._save("failures", list(self._failures.values()))

    def add_findings(self, findings: Sequence[Finding]) -> None:
        super().add_findings(findings)
        self._save("findings", list(self._findings.values()))

    def add_lessons(self, lessons: Sequence[Lesson]) -> None:
        super().add_lessons(lessons)
        self._save("lessons", list(self._lessons.values()))
