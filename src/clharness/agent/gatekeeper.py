"""The Gatekeeper: deterministic accept, trim or reject of a proposal. No model, no repair.

Rules, in order, each a function in `RULES`:

    catalog           every technique is in the catalog, with known parameters inside their ranges
    runnable          (when a check is given, as in real mode) the bench can build the spec:
                      `techniques.build_hooks` and `config_from_spec` succeed and no refused
                      combination is used; the ValueError text is the reason
    shape             seeds are the PREREG seeds of the phase; a screen spec has at most one
                      treatment; a spec called a control holds only control techniques
    matched_compute   every spec uses the frozen bench and stream and no optimizer override, so
                      all arms spend the same gradient tokens. One exception: an exploratory
                      claim test (`notes: "explore:<claim id>"`) may set `optim.lr` inside the
                      declared range; its notes label the run
    explore           an exploratory spec tests an accepted claim card: its claim id is known and
                      its configuration is that card's test or control arm
    duplicate         a configuration that already has a settled row (numbers compared by value,
                      so 16 and 16.0 are one), is in flight, used up its retries, or appears twice
    budget            estimated cost must fit the compute budget; controls are kept first, and
                      treatments may spend only the phase's share of what is left. Exploratory
                      specs spend only the explore share, reserved up front

A rule returns the specs it drops. A spec dropped by an earlier rule is not shown to a later
one. If nothing is left the verdict is REJECT.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .catalog_real import explore_claim_id, is_explore, lr_in_range
from .contracts import (
    WHOLE_PROPOSAL,
    Dropped,
    Phase,
    Proposal,
    ProposedSpec,
    RunSpec,
    Verdict,
    canonical_json,
    config_hash,
    retryable,
    spec_hash,
    technique_map,
    technique_names,
)
from .roles import GateRequest

Rule = Callable[[list[ProposedSpec], GateRequest], list[Dropped]]
BenchCheck = Callable[[RunSpec], list[str]]  # why the bench cannot run a spec; empty = it can

# Share of the compute left that one proposal's treatments may spend. Keeps a model planner from
# spending the whole budget in the screen and leaving nothing to replicate or combine.
PHASE_SHARE = {Phase.SCREEN: 0.5, Phase.REPLICATE: 0.7, Phase.COMBINE: 1.0}


def _drop(item: ProposedSpec, rule: str, why: str) -> Dropped:
    return Dropped(spec_hash=spec_hash(item.spec), rule=rule, reason=why)


def catalog(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
    out = []
    for item in specs:
        problems = request.catalog.problems(technique_map(item.spec))
        if problems:
            out.append(_drop(item, "catalog", "; ".join(problems)[:300]))
    return out


def make_runnable(check: BenchCheck) -> Rule:
    """A rule that drops every spec `check` finds a problem with (real mode:
    `catalog_real.bench_problems`). Not in `RULES`: the simulator's names are not plugins."""
    def runnable(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
        out = []
        for item in specs:
            problems = check(item.spec)
            if problems:
                out.append(_drop(item, "runnable", "; ".join(problems)[:300]))
        return out
    return runnable


def shape(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
    seeds = (request.prereg.screen_seeds if request.phase == Phase.SCREEN
             else request.prereg.replicate_seeds)
    controls = {e.name for e in request.catalog.controls()}
    out = []
    for item in specs:
        names = set(technique_names(item.spec))
        if item.spec.seed not in seeds:
            why = f"seed {item.spec.seed} is not a PREREG {request.phase.value} seed {list(seeds)}"
        elif item.role == "control" and not names <= controls:
            why = f"called a control but holds treatments {sorted(names - controls)}"
        elif request.phase == Phase.SCREEN and len(names - controls) > 1:
            why = "the screen tests one technique at a time"
        else:
            continue
        out.append(_drop(item, "shape", why))
    return out


def matched_compute(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
    out = []
    for item in specs:
        if item.spec.bench != request.bench:
            why = f"bench {item.spec.bench!r} is not the frozen bench {request.bench!r}"
        elif list(item.spec.stream) != list(request.stream):
            why = "stream differs from the frozen bench, so gradient tokens would not match"
        elif item.spec.optim and not _explore_lr_ok(item.spec, request):
            if is_explore(item.spec) and set(item.spec.optim) == {"lr"} and request.lr_range:
                low, high = request.lr_range
                why = f"optim.lr={item.spec.optim['lr']!r} is outside the explore range {low:g} to {high:g}"
            else:
                why = (f"optimizer override {sorted(item.spec.optim)} is not allowed on the frozen "
                       "bench (only optim.lr, only on an exploratory claim test)")
        else:
            continue
        out.append(_drop(item, "matched_compute", why))
    return out


def _explore_lr_ok(spec: RunSpec, request: GateRequest) -> bool:
    return (is_explore(spec) and set(spec.optim) == {"lr"} and request.lr_range is not None
            and lr_in_range(spec.optim["lr"], request.lr_range))


def explore(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
    """An exploratory spec must come from an accepted claim card, arm for arm."""
    out = []
    for item in specs:
        if not is_explore(item.spec):
            continue
        claim = explore_claim_id(item.spec)
        arms = request.claim_arms.get(claim or "")
        if arms is None:
            why = f"exploratory spec names no accepted claim card ({item.spec.notes[:80]!r})"
        elif config_hash(item.spec) not in arms:
            why = f"spec is neither the test nor the control arm of claim {claim}"
        else:
            continue
        out.append(_drop(item, "explore", why))
    return out


def _by_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(f"{float(value):.6g}")
    if isinstance(value, dict):
        return {k: _by_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_by_value(v) for v in value]
    return value


def config_key(spec: RunSpec) -> str:
    """`spec_hash` with numbers compared by value: `rank=16` and `rank=16.0` give one key."""
    payload = spec.model_dump(mode="json", exclude={"name", "notes", "parent_spec_hash"})
    return canonical_json(_by_value(payload))


def duplicate(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
    """Drop what was already run or launched. An infra or timeout failure may be run again
    until its retries are used up."""
    settled = {config_key(r.spec) for r in request.rows if not retryable(r)}
    exhausted = set(request.retry_exhausted)
    pending = set(request.pending_spec_hashes)
    seen: set[str] = set()
    out = []
    for item in specs:
        key = config_key(item.spec)
        if key in settled:
            why = "this spec already has a result row"
        elif spec_hash(item.spec) in exhausted:
            why = "this spec failed on infrastructure or timeout on every allowed attempt"
        elif spec_hash(item.spec) in pending:
            why = "this spec is already launched"
        elif key in seen:
            why = "this spec appears twice in the proposal"
        else:
            seen.add(key)
            continue
        out.append(_drop(item, "duplicate", why))
    return out


def budget(specs: list[ProposedSpec], request: GateRequest) -> list[Dropped]:
    """Keep controls first, then the rest in proposal order, until the money runs out."""
    left = request.budget.compute_usd_left
    treatments_left = left * PHASE_SHARE.get(request.phase, 1.0)
    explore_left = request.explore_usd_left
    out = []
    ordered = sorted(specs, key=lambda item: item.role != "control")  # stable: controls first
    for item in ordered:
        key = spec_hash(item.spec)
        cost = request.estimated_usd.get(key)
        control = item.role == "control"
        if cost is None:
            out.append(Dropped(spec_hash=key, rule="budget", reason="no cost estimate"))
        elif is_explore(item.spec):
            if explore_left is None:
                why = "no explore budget is reserved"
            elif cost > explore_left + 1e-9:
                why = (f"costs ${cost:.2f}, only ${max(explore_left, 0):.2f} of the explore "
                       "share is left")
            else:
                explore_left -= cost
                continue
            out.append(Dropped(spec_hash=key, rule="budget", reason=why))
        elif cost > left + 1e-9:
            if control:
                why = f"a control costs ${cost:.2f} and only ${max(left, 0):.2f} is left"
                return [Dropped(spec_hash=WHOLE_PROPOSAL, rule="budget", reason=why)]
            why = f"costs ${cost:.2f}, only ${max(left, 0):.2f} left after the specs kept so far"
            out.append(Dropped(spec_hash=key, rule="budget", reason=why))
        elif not control and cost > treatments_left + 1e-9:
            why = (f"treatments in {request.phase.value} may spend "
                   f"{PHASE_SHARE.get(request.phase, 1.0):.0%} of the budget left")
            out.append(Dropped(spec_hash=key, rule="budget", reason=why))
        else:
            left -= cost
            if not control:
                treatments_left -= cost
    return out


RULES: dict[str, Rule] = {
    "catalog": catalog,
    "shape": shape,
    "matched_compute": matched_compute,
    "explore": explore,
    "duplicate": duplicate,
    "budget": budget,
}


class Gatekeeper:
    """`runnable`: a bench check (real mode: `catalog_real.bench_problems`), run right after the
    catalog rule. Leave it out on the simulator, whose technique names are not plugins."""

    def __init__(self, rules: dict[str, Rule] | None = None,
                 runnable: BenchCheck | None = None) -> None:
        base = dict(RULES if rules is None else rules)
        if runnable is not None:
            ordered: dict[str, Rule] = {}
            for name, rule in base.items():
                ordered[name] = rule
                if name == "catalog":
                    ordered["runnable"] = make_runnable(runnable)
            ordered.setdefault("runnable", make_runnable(runnable))
            base = ordered
        self.rules = base

    def review(self, proposal: Proposal, request: GateRequest) -> Verdict:
        alive = list(proposal.specs)
        dropped: list[Dropped] = []
        for rule in self.rules.values():
            found = rule(alive, request)
            dropped += found
            if any(d.spec_hash == WHOLE_PROPOSAL for d in found):
                return Verdict(token="REJECT", dropped=tuple(dropped))
            alive = _without(alive, found)
        kept = tuple(dict.fromkeys(spec_hash(item.spec) for item in alive))
        if not kept:
            reasons = dropped or [
                Dropped(spec_hash=WHOLE_PROPOSAL, rule="empty", reason="the proposal has no specs")
            ]
            return Verdict(token="REJECT", dropped=tuple(reasons))
        if dropped:
            return Verdict(token="TRIM", kept=kept, dropped=tuple(dropped))
        return Verdict(token="ACCEPT", kept=kept)


def _without(specs: list[ProposedSpec], dropped: list[Dropped]) -> list[ProposedSpec]:
    """Remove one occurrence per drop, last occurrence first, so a repeat keeps its first copy."""
    alive = list(specs)
    for drop in dropped:
        for index in range(len(alive) - 1, -1, -1):
            if spec_hash(alive[index].spec) == drop.spec_hash:
                del alive[index]
                break
    return alive
