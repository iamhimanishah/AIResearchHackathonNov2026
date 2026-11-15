"""Model-backed roles: the LLM Planner, the LLM Analyst, and file-backed Memory.

Each role assembles its prompt with `context.build_*_prompt` (which enforces the firewall),
calls the model through an `LLMClient` (normally `BudgetedLLM`, which retries invalid output
and enforces the model budget), and validates the answer deterministically. A role never
moves the loop: on `LLMError` the Conductor takes the deterministic path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from .cards import CardStore
from .catalog_real import LR_RANGE, arm_key, claim_arm_spec, lr_in_range
from .context import build_analyst_prompt, build_planner_prompt, build_scout_prompt
from .contracts import (
    NO_MITIGATION,
    Candidate,
    CitedNumber,
    ClaimCard,
    FailureEntry,
    Finding,
    Lesson,
    ParamSpec,
    Prereg,
    Proposal,
    RunSpec,
    SelectionRow,
    Source,
    TriedEntry,
    atomic_write_text,
    check_numbers,
    claim_card_id,
    config_hash,
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

BenchCheck = Callable[[RunSpec], list[str]]


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
            claim_cards=request.claim_cards,
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
# the objective, not as a domain filter. Each query is one Exa search. The third asks for claims:
# reported numbers the loop can test with the plugins it already has.
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
    (("measured effect of replay ratio, learning-rate re-warm or LoRA rank on forgetting in "
      "continual pre-training of LLMs"),
     ("Find papers that report a number for how much a replay share, a learning-rate re-warm or "
      "re-decay schedule, a LoRA rank, layer freezing, weight averaging or distillation changes "
      "forgetting when an LLM is further pretrained on a new domain. Pull the reported number, "
      "the parameter values compared, the model size and the training setup.")),
)
MAX_SCOUT_SOURCES = 24
MAX_SCOUT_ITEMS = 5  # per list: claims, candidates

_ARXIV = re.compile(r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|html|pdf)/"
                    r"([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?(?:[?#].*)?$",
                    re.IGNORECASE)


def source_key(url: str) -> str:
    """The deduplication key of a URL: `arxiv:<id>` for every arXiv abs, html or pdf link at any
    version (`abs/2610.22293` and `pdf/2610.22293v1.pdf` are one paper), else the URL."""
    match = _ARXIV.match(url.strip())
    return f"arxiv:{match.group(1).lower()}" if match else url.strip()


class ArmTechnique(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ClaimPredictionDraft(BaseModel):
    direction: Literal[-1, 0, 1]  # +1: the test arm forgets less than the control arm
    size_points: float = Field(ge=0, le=20)


class ClaimDraft(BaseModel):
    claim: str  # one sentence, as the paper states it
    source_ids: list[str] = Field(min_length=1)
    test_arm: list[ArmTechnique] = Field(default_factory=list)  # catalog techniques only
    control_arm: list[ArmTechnique] = Field(default_factory=list)  # empty = no mitigation
    test_lr: float | None = None  # only for a claim about the learning rate
    control_lr: float | None = None
    prediction: ClaimPredictionDraft


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
    claims: list[ClaimDraft] = Field(default_factory=list, max_length=MAX_SCOUT_ITEMS)  # ranked
    candidates: list[CandidateDraft] = Field(default_factory=list,
                                             max_length=MAX_SCOUT_ITEMS)  # ranked
    skip_reason: str | None = None  # set when both lists are empty


def _arm(items: Sequence[ArmTechnique]) -> tuple[dict[str, dict[str, Any]], str | None]:
    arm: dict[str, dict[str, Any]] = {}
    for t in items:
        name = t.name.strip()
        if name == NO_MITIGATION:
            continue
        if name in arm:
            return {}, f"technique {name!r} is listed twice"
        arm[name] = dict(t.params)
    return arm, None


def check_claim(item: ClaimDraft, request: ScoutRequest, returned: Collection[str],
                runnable: BenchCheck | None, created_at: str) -> tuple[ClaimCard | None, str]:
    """Code checks of one drafted claim (SCOUT-HANDOFF 3.3). Returns the card, or None and why.

    A card that repeats a pre-registered claim is kept with `duplicate_of` set; it is not planned.
    Nothing is repaired: a draft that fails a check is dropped with its reason.
    """
    unknown = [sid for sid in item.source_ids if sid not in returned]
    if unknown:
        return None, f"cites sources this search did not return: {unknown[:3]}"
    test, why = _arm(item.test_arm)
    control, why2 = _arm(item.control_arm)
    if why or why2:
        return None, why or why2 or ""
    for label, arm in (("test", test), ("control", control)):
        problems = request.catalog.problems(arm)
        if problems:
            return None, f"{label} arm: {'; '.join(problems)[:240]}"
    for label, lr in (("test", item.test_lr), ("control", item.control_lr)):
        if lr is not None and not lr_in_range(lr):
            low, high = LR_RANGE
            return None, f"{label}_lr={lr!r} is outside the allowed range {low:g} to {high:g}"
    if arm_key(test, item.test_lr) == arm_key(control, item.control_lr):
        return None, "the test arm and the control arm are the same configuration"
    stream, seed = request.stream, request.seeds[0]
    test_spec = claim_arm_spec(test, item.test_lr, seed, stream, "check")
    control_spec = claim_arm_spec(control, item.control_lr, seed, stream, "check")
    if runnable is not None:
        for label, spec in (("test", test_spec), ("control", control_spec)):
            problems = runnable(spec)
            if problems:
                return None, f"{label} arm does not build: {'; '.join(problems)[:240]}"
    duplicate_of = request.design_arms.get(arm_key(test, item.test_lr))
    if duplicate_of is None and config_hash(test_spec) in set(request.finished_config_hashes):
        return None, "the test arm already has a finished row"
    claim = " ".join(item.claim.split())[:400]
    return ClaimCard(
        id=claim_card_id(claim, item.source_ids), claim=claim,
        source_ids=tuple(dict.fromkeys(item.source_ids)), test_arm=test, control_arm=control,
        test_lr=item.test_lr, control_lr=item.control_lr,
        prediction=item.prediction.model_dump(), duplicate_of=duplicate_of, status="accepted",
        created_at=created_at,
    ), ""


def check_candidate(item: CandidateDraft, catalog_names: Collection[str], returned: Collection[str],
                    created_at: str) -> tuple[Candidate | None, str]:
    """A candidate cites at least one returned source and has a name not in the catalog."""
    cited = [sid for sid in item.source_ids if sid in returned]
    if not cited:
        return None, f"cites no returned source: {item.source_ids[:3]}"
    if item.name.strip().lower() in {n.lower() for n in catalog_names}:
        return None, f"{item.name!r} is already in the catalog"
    return Candidate(
        **item.model_dump(exclude={"source_ids", "params"}), params=item.params,
        source_ids=tuple(cited), created_at=created_at,
    ), ""


class LLMScout:
    """Searches the web with Exa, then asks a model for a ranked shortlist: claim cards the loop
    can test with the plugins it has, and candidate techniques that need new plugin code.

    Code runs the search (the model never holds the key or a tool) and the model reads the
    results as untrusted data. Every item is checked in code (`check_claim`, `check_candidate`);
    a failed item goes to `rejected` with its reason. `runnable` is the bench check (real mode:
    `catalog_real.bench_problems`). Never raises: any failure is a skip with a reason.
    """

    def __init__(self, llm: LLMClient, search: SearchClient, cards: CardStore | None = None,
                 clock: Clock | None = None,
                 queries: Sequence[tuple[str, str]] = SCOUT_QUERIES,
                 runnable: BenchCheck | None = None) -> None:
        self.llm = llm
        self.search = search
        self.cards = cards or CardStore()
        self.clock = clock or SystemClock()
        self.queries = tuple(queries)
        self.runnable = runnable

    def gather(self) -> tuple[tuple[Source, ...], list[str]]:
        """Run every query; return sources deduplicated by paper (arXiv versions merged) and one
        error string per failed query."""
        seen: dict[str, Source] = {}
        errors: list[str] = []
        for query, objective in self.queries:
            try:
                found = self.search.search(query, objective)
            except SearchError as exc:
                errors.append(str(exc))
                continue
            for src in found:
                seen.setdefault(source_key(src.url), src)
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
        now = utc_iso(self.clock.now())
        returned = {s.id for s in sources}
        claims: list[ClaimCard] = []
        candidates: list[Candidate] = []
        rejected: list[str] = []
        for item in draft.claims:
            card, why = check_claim(item, request, returned, self.runnable, now)
            if card is None:
                rejected.append(f"claim {item.claim[:80]!r}: {why}")
            elif any(c.id == card.id for c in claims):
                rejected.append(f"claim {item.claim[:80]!r}: repeats an earlier claim card")
            else:
                claims.append(card)
        for item in draft.candidates:
            cand, why = check_candidate(item, request.catalog.names(), returned, now)
            if cand is None:
                rejected.append(f"candidate {item.name[:60]!r}: {why}")
            elif any(c.name.lower() == cand.name.lower() for c in candidates):
                rejected.append(f"candidate {item.name[:60]!r}: repeats an earlier candidate")
            else:
                candidates.append(cand)
        used = {sid for c in claims for sid in c.source_ids}
        used |= {sid for c in candidates for sid in c.source_ids}
        marked = tuple(s.model_copy(update={"used": s.id in used}) for s in sources)
        skipped = None
        if not claims and not candidates:
            skipped = (draft.skip_reason or "model skipped") if not rejected else \
                f"every item failed a check: {rejected[0]}"
        return ScoutResult(claims=tuple(claims), candidates=tuple(candidates),
                           rejected=tuple(rejected), sources=marked, skipped_reason=skipped,
                           meta=meta)


def write_scout_result(result: ScoutResult, out_dir: str | Path) -> Path:
    """Write the shortlist (claim cards, candidates, rejections) with every source it saw, as one
    JSON file: `<out_dir>/<timestamp>-<what>.json`. This is the candidate review queue's input."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [*result.claims, *result.candidates]
    stamp = (items[0].created_at if items else
             (result.sources[0].retrieved_at if result.sources else "none"))
    what = (f"{len(result.claims)}-claims-{len(result.candidates)}-candidates" if items
            else "skip")
    slug = re.sub(r"[^a-z0-9]+", "-", f"{stamp}-{what}".lower()).strip("-")[:80]
    path = out / f"{slug}.json"
    atomic_write_text(path, result.model_dump_json(indent=2) + "\n")
    return path


def candidate_queue(candidates_dir: str | Path, veto_path: str | Path) -> list[dict[str, Any]]:
    """The candidate review queue (SCOUT-HANDOFF 3.6): every candidate the Scout wrote, ranked as
    written, with its status. A person approves or rejects one by appending a `Veto` line
    (`approve_candidate` / `reject_candidate`, target = the candidate name) to `veto.jsonl`; the
    last line for a name wins. The loop never waits for this. An approved candidate enters the
    catalog only when its plugin is merged and has passed a sanity run.
    """
    decided: dict[str, str] = {}
    for record in read_jsonl(veto_path).records:
        kind = record.get("kind")
        if kind in ("approve_candidate", "reject_candidate"):
            decided[str(record.get("target", "")).strip().lower()] = (
                "approved" if kind == "approve_candidate" else "rejected")
    out: list[dict[str, Any]] = []
    root = Path(candidates_dir)
    for path in sorted(root.glob("*.json")) if root.exists() else ():
        try:
            result = ScoutResult.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        by_id = {s.id: s for s in result.sources}
        for rank, cand in enumerate(result.candidates, start=1):
            out.append({
                "name": cand.name, "rank": rank, "file": path.name,
                "status": decided.get(cand.name.strip().lower(), "pending"),
                "description": cand.description, "mechanism": cand.mechanism,
                "expected_effect": cand.expected_effect, "created_at": cand.created_at,
                "sources": [{"title": by_id[s].title, "url": by_id[s].url}
                            for s in cand.source_ids if s in by_id],
            })
    return out


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
