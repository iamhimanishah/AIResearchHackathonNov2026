"""The Conductor: the deterministic state machine of docs/agent/ARCHITECTURE.md section 3.

Thin first version. It plans with the deterministic fallback only (no model call yet), and the
READ and SCOUT steps do the minimum: there is no Analyst or Scout behind them. REPORT opens the
report half through a `FreezeToken` and writes the deterministic numbers (`report.json`); the
Reporter's prose comes later.

Decisions go to the hash-chained `DecisionLog` (`decisions.jsonl`). `state.json` keeps the head
hash, so a log that was cut, edited or reordered stops the loop on restart.

Every step follows one pattern, which is what makes a kill at any point safe:

    1. do the work        every outside effect is idempotent (launch is keyed by spec hash)
    2. log the decision   `DecisionLog.append` returns the existing entry for a logged key
    3. move and save      state.json is replaced atomically

A process killed between any two of these redoes the step on restart and ends in the same place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .contracts import (
    BudgetView,
    Catalog,
    ContractError,
    Decision,
    DecisionLog,
    Phase,
    Prereg,
    ResultRow,
    RunSpec,
    StreamDomain,
    Verdict,
    accepted_specs,
    arm_label,
    arm_of,
    atomic_write_json,
    build_spec,
    config_hash,
    frozen_set_hash,
    latest_by_spec,
    load_rows,
    mint_freeze_token,
    passes_acquisition_floor,
    selection_views,
    spec_hash,
    technique_names,
    to_report_view,
    usable,
    utc_iso,
)
from .llm import LLMError
from .operator import MAX_INFRA_ATTEMPTS
from .roles import Clock, Executor, GateRequest, PlanRequest, ReadRequest
from .states import (
    ACTORS,
    DONE,
    FREEZE,
    HALTED,
    PREFLIGHT,
    REPORT,
    LoopState,
    State,
    Step,
    next_phase,
    st,
)

Fault = Callable[[str, str], None]  # (state id, point) -> may raise; tests use it to kill the loop
Outcome = tuple[State, dict[str, Any]]


class Conductor:
    def __init__(
        self, out_dir: str | Path, rows_path: str | Path, *, executor: Executor, clock: Clock,
        planner: Any, gatekeeper: Any, operator: Any, prereg: Prereg, catalog: Catalog,
        stream: Sequence[StreamDomain], compute_cap_usd: float, reserve_fraction: float = 0.3,
        deadline: datetime | None = None, phase_timeout_minutes: float = 240.0,
        concurrency: int = 8, minutes_per_run: float = 30.0, fault: Fault | None = None,
        analyst: Any = None, memory: Any = None, max_gate_rejections: int = 2,
        bench: str = "fast",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.rows_path = Path(rows_path)
        self.state_path = self.out_dir / "state.json"
        self.log_path = self.out_dir / "decisions.jsonl"
        self.executor, self.clock = executor, clock
        self.planner, self.gatekeeper, self.operator = planner, gatekeeper, operator
        self.analyst, self.memory = analyst, memory
        self.max_gate_rejections = max_gate_rejections
        self.bench = bench
        self.prereg, self.catalog, self.stream = prereg, catalog, tuple(stream)
        self.compute_cap_usd, self.reserve_fraction = compute_cap_usd, reserve_fraction
        self.deadline = deadline
        self.phase_timeout_minutes = phase_timeout_minutes
        self.concurrency, self.minutes_per_run = concurrency, minutes_per_run
        self.fault = fault or (lambda state_id, point: None)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.s = LoopState.load(self.state_path) if self.state_path.exists() else LoopState()
        self.log = DecisionLog(self.log_path)
        self._check_log()

    def _check_log(self) -> None:
        """The log must verify and must still hold the head recorded in `state.json`.

        The log may be one entry ahead of the state (a kill between logging and saving); the
        step is then redone and its `append` returns that entry.
        """
        report = self.log.verify()
        if not report.ok:
            raise ContractError(f"decision log does not verify: {report.problems[0]}")
        hashes = [d.hash for d in self.log.entries()]
        head, count = self.s.decision_head, self.s.decision_count
        if count and (len(hashes) < count or hashes[count - 1] != head):
            raise ContractError("decision log lost or changed entries recorded in state.json")

    # ---- driving ---------------------------------------------------------------------------

    def run(self, max_transitions: int | None = None) -> LoopState:
        count = 0
        while self.s.state not in (DONE, HALTED):
            if max_transitions is not None and count >= max_transitions:
                break
            self.step()
            count += 1
        return self.s

    def step(self) -> None:
        state = self.s.state
        handler = getattr(self, f"_{state.step.value}")
        key = self._key(state)
        nxt, decision = handler(state.phase) if state.phase else handler()
        self.fault(state.id, "work_done")
        self._log(state, key, decision)
        self.fault(state.id, "logged")
        now = utc_iso(self.clock.now())
        self.s.started_at = self.s.started_at or now
        self.s.move(nxt, now, reason=decision.get("reason"))
        self.s.save(self.state_path)
        self.fault(state.id, "saved")

    def _key(self, state: State) -> str:
        """Decision key: state, phase repeat and, after a gate rejection, the replan number."""
        if not state.phase:
            return f"{state.id}#0"
        repeat = self.s.phase_repeats.get(state.phase.value, 0)
        replans = self.s.extra.get("replans", {}).get(f"{state.phase.value}#{repeat}", 0)
        return f"{state.id}#{repeat}" + (f".{replans}" if replans else "")

    def _log(self, state: State, key: str, decision: dict[str, Any]) -> None:
        """Append one chained decision, once per decision key."""
        data = {"choice": decision["choice"], "accepted": decision.get("accepted"),
                **decision.get("data", {})}
        meta = decision.get("meta")
        entry = self.log.append(
            key=key, state=state.id, actor=ACTORS[state.step],
            reason=decision["reason"], ts=self.clock.now(),
            input_row_hashes=decision.get("input_row_hashes", ()),
            proposal=self.s.proposal if state.step in (Step.FALLBACK_PLAN, Step.PLAN) else None,
            verdict=self.s.verdict if state.step == Step.GATE else None,
            prompt_hash=meta.prompt_hash if meta else None,
            card_hash=meta.card_hash if meta else None,
            skill_hashes=meta.skill_hashes if meta else None,
            model=meta.model if meta else None,
            cites_findings=decision.get("cites_findings", ()),
            data=data,
        )
        self.s.decision_count, self.s.decision_head = entry.seq + 1, entry.hash

    # ---- shared reads ----------------------------------------------------------------------

    def _rows(self) -> list[ResultRow]:
        return load_rows(self.rows_path).rows

    def _latest(self) -> list[ResultRow]:
        return list(latest_by_spec(self._rows()).values())

    def _estimate(self, spec: RunSpec) -> float:
        return self.executor.estimate_cost(spec)

    def _spent(self, rows: Sequence[ResultRow]) -> float:
        """Money the rows cost. A finished or failed run with no price is charged the estimate,
        so a bench that forgets `est_usd` cannot make the budget look unspent."""
        return sum(r.cost.usd if r.cost.usd > 0 or r.failure_kind == "infra"
                   else self._estimate(r.spec) for r in rows)

    def _model_usd_left(self) -> float:
        llm = getattr(self.planner, "llm", None)
        return float(getattr(llm, "remaining_usd", 0.0))

    def _retry_exhausted(self) -> tuple[str, ...]:
        attempts: dict[str, int] = {}
        for row in self._rows():
            attempts[row.spec_hash] = attempts.get(row.spec_hash, 0) + 1
        return tuple(sorted(r.spec_hash for r in self._latest()
                            if r.failure_kind in ("infra", "timeout")
                            and attempts[r.spec_hash] >= MAX_INFRA_ATTEMPTS))

    def _budget(self) -> BudgetView:
        rows = self._rows()
        have_row = {r.spec_hash for r in rows}
        spent = self._spent(rows)
        committed = sum(
            cost for key, cost in self.s.extra.get("pending_usd", {}).items()
            if key not in have_row
        )
        reserve = self.compute_cap_usd * self.reserve_fraction
        now = self.clock.now()
        minutes = (self.deadline - now).total_seconds() / 60 if self.deadline else None
        baseline = build_spec("baseline", 0, self.stream)
        return BudgetView(
            compute_usd_left=self.compute_cap_usd - reserve - spent - committed,
            compute_usd_reserve=reserve, model_usd_left=self._model_usd_left(),
            minutes_left=minutes,
            concurrency=self.concurrency, minutes_per_run=self.minutes_per_run,
            baseline_run_usd=self._estimate(baseline),
        )

    def _out_of_room(self) -> str | None:
        budget = self._budget()
        if budget.compute_usd_left < budget.baseline_run_usd:
            return f"compute budget: ${budget.compute_usd_left:.2f} left, one run costs more"
        if budget.minutes_left is not None and budget.minutes_left <= 0:
            return "deadline reached"
        return None

    # ---- states outside the cycle ----------------------------------------------------------

    def _init(self) -> Outcome:
        return PREFLIGHT, {"choice": "start", "reason": "state loaded"}

    def _preflight(self) -> Outcome:
        checks = {
            "executor answers": self.executor.healthy(),
            "catalog has a technique besides the controls":
                any(not e.is_control for e in self.catalog.entries),
            "compute budget is set": self.compute_cap_usd > 0,
            "prereg has replicate seeds": len(self.prereg.replicate_seeds) > 0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            return HALTED, {"choice": "halt", "reason": f"preflight failed: {failed}",
                            "accepted": False}
        return st(Step.PLAN, Phase.SCREEN), {
            "choice": "preflight passed", "reason": "; ".join(checks), "accepted": True,
        }

    # ---- the cycle -------------------------------------------------------------------------

    def _logged(self, state: State) -> Decision | None:
        """The decision already logged for this step, if a kill came after logging."""
        key = self._key(state)
        return next((d for d in self.log.entries() if d.key == key), None)

    def _plan_request(self, phase: Phase, rejections: Sequence[Verdict] = ()) -> PlanRequest:
        return PlanRequest(
            phase=phase, prereg=self.prereg, catalog=self.catalog, budget=self._budget(),
            stream=self.stream, rows=selection_views(self._latest()),
            findings=tuple(self.memory.findings()) if self.memory else (),
            lessons=tuple(self.memory.lessons()) if self.memory else (),
            rejections=tuple(rejections),
        )

    def _plan(self, phase: Phase) -> Outcome:
        if getattr(self.planner, "llm", None) is None:
            return st(Step.FALLBACK_PLAN, phase), {
                "choice": "use the deterministic plan",
                "reason": "no model planner is configured",
            }
        done = self._logged(st(Step.PLAN, phase))
        if done is not None and done.proposal is not None:
            proposal, meta = done.proposal, None
        else:
            rejections = [Verdict.model_validate(v) for v in self.s.extra.get("rejections", [])]
            request = self._plan_request(phase, rejections)
            try:
                proposal, meta = self.planner.plan(request)
            except LLMError as exc:
                return st(Step.FALLBACK_PLAN, phase), {
                    "choice": "use the deterministic plan", "reason": f"planner failed: {exc}",
                }
        self.s.proposal, self.s.verdict = proposal, None
        self.s.extra = {**self.s.extra, "proposal_source": "planner"}
        return st(Step.GATE, phase), {
            "choice": [p.spec.name for p in proposal.specs],
            "reason": proposal.rationale,
            "input_row_hashes": [r.row_hash for r in selection_views(self._latest())],
            "meta": meta, "cites_findings": proposal.cites_findings,
        }

    def _fallback_plan(self, phase: Phase) -> Outcome:
        request = self._plan_request(phase)
        proposal = self.planner.fallback(request)
        self.s.proposal, self.s.verdict = proposal, None
        self.s.extra = {**self.s.extra, "proposal_source": "fallback"}
        return st(Step.GATE, phase), {
            "choice": [p.spec.name for p in proposal.specs],
            "reason": proposal.rationale,
            "input_row_hashes": [r.row_hash for r in request.rows],
        }

    def _gate(self, phase: Phase) -> Outcome:
        proposal = self.s.proposal
        latest = self._latest()
        request = GateRequest(
            phase=phase, prereg=self.prereg, catalog=self.catalog, budget=self._budget(),
            stream=self.stream, rows=selection_views(latest),
            pending_spec_hashes=tuple(self.s.pending_spec_hashes),
            estimated_usd={spec_hash(p.spec): self._estimate(p.spec) for p in proposal.specs},
            bench=self.bench, retry_exhausted=self._retry_exhausted(),
        )
        verdict = self.gatekeeper.review(proposal, request)
        self.s.verdict = verdict
        names = {spec_hash(p.spec): p.spec.name for p in proposal.specs}
        drops = [f"{names.get(d.spec_hash, d.spec_hash)}: {d.rule}: {d.reason}"
                 for d in verdict.dropped]
        decision = {
            "choice": {"verdict": verdict.token, "kept": [names[k] for k in verdict.kept],
                       "dropped": drops},
            "reason": "all rules passed" if not drops else f"{len(drops)} spec(s) dropped",
            "accepted": verdict.token != "REJECT",
            "input_row_hashes": [r.row_hash for r in latest],
        }
        if verdict.token != "REJECT":
            self.s.gate_rejections = 0
            self.s.extra = {**self.s.extra, "rejections": []}
            return st(Step.LAUNCH, phase), decision
        if self.s.extra.get("proposal_source") == "planner":
            self.s.gate_rejections += 1
            rejected = [*self.s.extra.get("rejections", []), verdict.model_dump(mode="json")]
            turn = f"{phase.value}#{self.s.phase_repeats.get(phase.value, 0)}"
            replans = dict(self.s.extra.get("replans", {}))
            replans[turn] = replans.get(turn, 0) + 1
            self.s.extra = {**self.s.extra, "rejections": rejected, "replans": replans}
            if self.s.gate_rejections < self.max_gate_rejections:
                decision["reason"] = "the planner's proposal was rejected: replan with the verdict"
                return st(Step.PLAN, phase), decision
            decision["reason"] = (f"the planner was rejected {self.s.gate_rejections} times: "
                                  "use the deterministic plan")
            self.s.gate_rejections = 0
            self.s.extra = {**self.s.extra, "rejections": []}
            return st(Step.FALLBACK_PLAN, phase), decision
        if all(d.rule == "duplicate" for d in verdict.dropped):
            following = next_phase(phase)
            decision["reason"] = "nothing new to run in this phase: every spec already has a row"
            return (st(Step.PLAN, following) if following else FREEZE), decision
        decision["reason"] = "the deterministic plan was rejected: this is a bug, a human must look"
        return HALTED, decision

    def _launch(self, phase: Phase) -> Outcome:
        specs = accepted_specs(self.s.proposal, self.s.verdict)
        report = self.operator.launch(specs, pending=self.s.pending_spec_hashes)
        pending_usd = dict(self.s.extra.get("pending_usd", {}))
        for spec in specs:
            pending_usd[spec_hash(spec)] = self._estimate(spec)
        self.s.extra = {**self.s.extra, "pending_usd": pending_usd}
        self.s.pending_spec_hashes = sorted({*self.s.pending_spec_hashes, *report.submitted})
        return st(Step.WAIT, phase), {
            "choice": {"submitted": len(report.submitted), "skipped": len(report.skipped)},
            "reason": "accepted specs without a settled row were submitted",
        }

    def _wait(self, phase: Phase) -> Outcome:
        specs = accepted_specs(self.s.proposal, self.s.verdict)
        deadline = self.clock.now() + timedelta(minutes=self.phase_timeout_minutes)
        if self.deadline:
            deadline = min(deadline, self.deadline)
        report = self.operator.wait(specs, deadline)
        self.s.pending_spec_hashes = sorted(report.timed_out)
        counts: dict[str, int] = {}
        for row in report.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return st(Step.READ, phase), {
            "choice": {**counts, "requeued": len(report.requeued),
                       "timed_out": len(report.timed_out)},
            "reason": "every spec has a row" if not report.timed_out else "phase timeout passed",
            "input_row_hashes": [r.row_hash for r in report.rows],
        }

    def _read(self, phase: Phase) -> Outcome:
        rows = self._rows()
        seen = set(self.s.seen_row_hashes)
        new = [r for r in rows if r.row_hash not in seen]
        self.s.seen_row_hashes = [r.row_hash for r in rows]
        self.s.compute_usd_spent = round(self._spent(rows), 4)
        problems = [f"{r.spec.name}: {r.status}: {r.reason}" for r in new if r.status != "finished"]
        decision: dict[str, Any] = {
            "choice": {"new_rows": len(new), "not_finished": problems,
                       "compute_usd_spent": self.s.compute_usd_spent},
            "reason": "rows recorded (no Analyst configured: no findings are written)",
            "input_row_hashes": [r.row_hash for r in new],
        }
        views = selection_views(new)
        if self.memory is not None:
            self.memory.record_rows(views)
        if self.analyst is None or self._logged(st(Step.READ, phase)) is not None:
            return st(Step.ADVANCE, phase), decision
        request = ReadRequest(
            phase=phase, prereg=self.prereg, new_rows=views,
            all_rows=selection_views(self._latest()),
            prior_findings=tuple(self.memory.findings()) if self.memory else (),
        )
        try:
            findings, meta = self.analyst.read(request)
            decision["reason"] = f"rows recorded; analyst wrote {len(findings)} checked finding(s)"
        except LLMError as exc:
            findings, meta = self.analyst.deterministic_findings(request), None
            decision["reason"] = f"rows recorded; analyst failed ({exc}): deterministic findings"
        if self.memory is not None:
            self.memory.add_findings(findings)
        decision["meta"] = meta
        decision["data"] = {"findings": [f.id for f in findings]}
        return st(Step.ADVANCE, phase), decision

    def _advance(self, phase: Phase) -> Outcome:
        wanted = set(self.s.verdict.kept) if self.s.verdict else set()
        done = [r for r in self._latest() if r.spec_hash in wanted and r.status == "finished"]
        hashes = [r.row_hash for r in done]
        stop = self._out_of_room()
        if stop:
            return FREEZE, {"choice": "freeze", "reason": stop, "input_row_hashes": hashes}
        if not done:
            repeats = self.s.phase_repeats.get(phase.value, 0)
            if repeats >= 1:
                return HALTED, {"choice": "halt", "input_row_hashes": hashes,
                                "reason": f"{phase.value} produced no finished run twice"}
            self.s.phase_repeats = {**self.s.phase_repeats, phase.value: repeats + 1}
            return st(Step.PLAN, phase), {"choice": f"repeat {phase.value}",
                                          "reason": "no run finished", "input_row_hashes": hashes}
        following = next_phase(phase)
        if following is None:
            return FREEZE, {"choice": "freeze", "reason": "combine done",
                            "input_row_hashes": hashes}
        return st(Step.PLAN, following), {
            "choice": f"go to {following.value}",
            "reason": f"{len(done)} of {len(wanted)} {phase.value} runs finished",
            "input_row_hashes": hashes,
        }

    # ---- the end ---------------------------------------------------------------------------

    def _freeze(self) -> Outcome:
        """Pick the best replicated arm on the SELECTION half and fix the set."""
        arms: dict[str, list[ResultRow]] = {}
        for row in self._latest():
            if usable(row, self.prereg):
                arms.setdefault(config_hash(row.spec), []).append(row)
        need = len(self.prereg.replicate_seeds)
        controls = {e.name for e in self.catalog.controls()}
        base = [r.acquisition_mean for rows in arms.values() for r in rows
                if not r.spec.techniques and r.acquisition_mean is not None]
        baseline = sum(base) / len(base) if base else None
        best: tuple[float, str, list[ResultRow]] | None = None
        for rows in arms.values():
            spec = rows[0].spec
            names = set(technique_names(spec))
            if len(rows) < need or not names or names <= controls:
                continue
            acq = [r.acquisition_mean for r in rows if r.acquisition_mean is not None]
            if baseline is not None and acq and not passes_acquisition_floor(
                    sum(acq) / len(acq), baseline, self.prereg.acquisition_floor_delta):
                continue
            mean = sum(r.retention_selection.mean_delta for r in rows) / len(rows)
            if best is None or (mean, arm_of(spec)) > (best[0], best[1]):
                best = (mean, arm_of(spec), rows)
        control_rows = [r for rows in arms.values() for r in rows
                        if len(rows) >= need and set(technique_names(rows[0].spec)) <= controls]
        chosen = (best[2] if best else []) + control_rows
        self.s.frozen_spec_hashes = sorted(r.spec_hash for r in chosen)
        self.s.frozen_set_hash = frozen_set_hash(self.s.frozen_spec_hashes)
        if best is None:
            choice: Any = "no replicated arm passed the acquisition floor"
        else:
            choice = {"arm": best[1], "seeds": len(best[2]),
                      "selection_half_mean_delta": round(best[0], 5)}
        return REPORT, {
            "choice": choice,
            "reason": "highest mean retention on the selection half among arms with all "
                      "replicate seeds that pass the acquisition floor; controls kept for "
                      "comparison",
            "input_row_hashes": [r.row_hash for r in chosen],
            "data": {"frozen_set_hash": self.s.frozen_set_hash,
                     "frozen_spec_hashes": self.s.frozen_spec_hashes},
        }

    def _report(self) -> Outcome:
        """Open the report half of the frozen set and write its numbers. No model is called."""
        token = mint_freeze_token(self.log, self.s.frozen_set_hash or "")
        frozen = set(self.s.frozen_spec_hashes)
        rows = [to_report_view(r, token) for r in self._latest()
                if r.spec_hash in frozen and r.status == "finished"]
        arms: dict[str, list[Any]] = {}
        for row in rows:
            arms.setdefault(arm_of(row.spec), []).append(row)
        table = {
            arm: {
                "seeds": sorted(r.spec.seed for r in group),
                "selection_mean_delta": _mean([r.retention_selection.mean_delta for r in group]),
                "report_mean_delta": _mean([r.retention_report.mean_delta for r in group]),
                "report_sd": _sd([r.retention_report.mean_delta for r in group]),
                "acquisition_mean": _mean([r.acquisition_mean for r in group]),
            }
            for arm, group in sorted(arms.items())
        }
        references = [e.name for e in self.catalog.by_role("reference")]
        reference = arm_label(references[:1]) if references else None
        base = table.get(reference, {}).get("report_mean_delta") if reference else None
        for v in table.values():
            mine = v["report_mean_delta"]
            v["report_vs_reference"] = (
                round(mine - base, 6) if mine is not None and base is not None else None)
        atomic_write_json(self.out_dir / "report.json", {
            "frozen_set_hash": token.frozen_set_hash, "reference_arm": reference, "arms": table,
            "row_hashes": sorted(r.row_hash for r in rows),
        })
        return DONE, {
            "choice": {arm: v["report_mean_delta"] for arm, v in table.items()},
            "reason": "report half opened for the frozen set; numbers in report.json",
            "input_row_hashes": sorted(r.row_hash for r in rows),
        }


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 6) if kept else None


def _sd(values: Sequence[float | None]) -> float | None:
    """Sample standard deviation across seeds; None below two values."""
    kept = [v for v in values if v is not None]
    if len(kept) < 2:
        return None
    mean = sum(kept) / len(kept)
    return round((sum((v - mean) ** 2 for v in kept) / (len(kept) - 1)) ** 0.5, 6)
