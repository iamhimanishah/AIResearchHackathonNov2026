"""The real-mode technique catalog and the checks that a spec can run on the frozen bench.

The simulator has its own catalog (`sim.default_catalog`, names such as `layer_freeze`). A loop on
real GPUs must plan, gate and scout against the plugins that exist, so this catalog is built from
`clharness.techniques.available()` (names and parameter schemas) narrowed by the technique
owner's ranges (docs/research-agent.md, Comments, Member B). Defaults are the frozen design
parameters (`clharness.analysis.design.PARAMS`) where the design fixes one, else the plugin's.

`bench_problems(spec)` is the gate's "can the bench run this" check: it calls the same code a run
calls before any GPU is used (`techniques.build_hooks` and `executor.run.config_from_spec`) and
adds the combinations the technique owner asked the gate to refuse. Both imports are lazy, so the
agent package still imports without the harness config machinery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..analysis import design
from . import explore as _explore
from .contracts import (
    Catalog,
    CatalogEntry,
    ClaimCard,
    ParamSpec,
    RunSpec,
    StreamDomain,
    arm_label,
    build_spec,
    canonical_json,
    config_hash,
)

# What the agent may vary: (low, high) inclusive for numbers, a tuple of choices otherwise. From
# docs/research-agent.md, Comments / Member B (peter, 14 November). One table for the whole
# agent: it lives in `explore.py` and is imported here.
RANGES: dict[str, dict[str, tuple[Any, ...]]] = _explore.RANGES
# The run learning rate a claim test may set (`optim.lr`), and only on an exploratory spec.
# Spans the PREREG's lower-LR controls (A13) up to the bench rate.
LR_RANGE: tuple[float, float] = _explore.LR_RANGE
EXPLORE_NOTE = "explore:"  # prefix of `RunSpec.notes` on every exploratory claim-test spec
EXPLORE_NAME = "explore-"  # prefix of `RunSpec.name` on every exploratory run, whoever made it

DESCRIPTIONS = {
    "replay": "Mix a share of earlier or general text into every batch; can select by a signal.",
    "lr_rewarm": "Re-warm and re-decay the learning rate at the start of each domain.",
    "wavg": "Average the trained weights with earlier weights (boundary, final or EMA).",
    "freeze": "Freeze the bottom and/or top transformer layers and optionally the embeddings.",
    "lora": "Train low-rank adapters instead of the full weights (sets the run's learning rate).",
    "distill": "Add a distillation loss against the base model's output distribution.",
}


def stream() -> tuple[StreamDomain, ...]:
    """The frozen fast-bench stream (`configs/bench/fast.yaml`)."""
    return tuple(StreamDomain(name=n, tokens=t) for n, t in design.STREAM)


def _param(name: str, key: str, schema: Mapping[str, Any]) -> ParamSpec | None:
    """A ParamSpec from the plugin schema and the owner's range; None when the agent may not set
    it (no range given, or the plugin default is null, which a ParamSpec cannot hold)."""
    allowed = RANGES.get(name, {}).get(key)
    default = design.PARAMS.get(name, {}).get(key, schema.get("default"))
    if allowed is None or default is None:
        return None
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in allowed)
    if numeric and len(allowed) == 2:
        return ParamSpec(default=default, min=float(allowed[0]), max=float(allowed[1]))
    return ParamSpec(default=default, choices=tuple(allowed))


def real_catalog() -> Catalog:
    """One entry per registered plugin, exactly the names of `techniques.available()`.

    No-mitigation is a spec with no technique, not an entry. Cost multipliers are 1.0: the bench
    measured 18 to 23 minutes per run whatever the technique, and `launch_design` prices runs flat.
    """
    from .. import techniques

    entries = []
    for name, schema in techniques.available().items():
        props = schema.get("properties", {})
        params = {k: p for k, s in props.items() if (p := _param(name, k, s)) is not None}
        entries.append(CatalogEntry(
            name=name, description=DESCRIPTIONS.get(name, schema.get("title", name)),
            params=params, cost_multiplier=1.0, is_control=False, role="other",
        ))
    return Catalog(entries=tuple(entries))


LORA_LR_CAP = 1e-3  # the highest LoRA learning rate that has run (3e-3 never has)
LORA_PLUGIN_LR = 1e-3  # the lora plugin's default `lr`, used when a spec does not set one


def lora_lr_limit(rank: float) -> float:
    """The highest learning rate LoRA may use at `rank`: 1e-3 x sqrt(8 / rank), never above 1e-3.

    The technique owner's joint bound (peter, agentchat, 14 November): with alpha = 2 x rank the
    per-step weight change grows roughly with lr x sqrt(rank). Rank 8 at 1e-3 is stable on three
    seeds; rank 256 at 1e-3 diverged on three. 1e-3 at rank 8, 3.5e-4 at 64, 1.8e-4 at 256."""
    return min(LORA_LR_CAP, 1e-3 * (8.0 / float(rank)) ** 0.5)


def lora_problem(params: Mapping[str, Any], run_lr: float | None = None) -> str | None:
    """Why a lora configuration is refused, or None. The learning rate checked is the one the
    run would use: the plugin's `lr` (default 1e-3, it overrides the run's), or, when `lr` is
    set to None, the run's own (`run_lr`, else the bench rate)."""
    rank = params.get("rank", design.PARAMS.get("lora", {}).get("rank", 8))
    lr = params.get("lr", LORA_PLUGIN_LR)
    if lr is None:
        lr = run_lr if run_lr is not None else design.BENCH_LR
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (rank, lr))
    if not numbers or rank <= 0:
        return None  # not a number: the plugin's own validation reports it
    limit = lora_lr_limit(rank)
    if lr > limit * (1 + 1e-9):
        return (f"lora lr {lr:g} at rank {rank:g} is above the limit {limit:.3g} "
                "(lr <= 1e-3 x sqrt(8 / rank), and never above 1e-3)")
    return None


def combination_problems(techniques: Mapping[str, Mapping[str, Any]],
                         lr: float | None = None) -> list[str]:
    """Combinations the technique owner asked the gate to refuse (Member B, "Combinations to
    refuse", and the LoRA rank and learning-rate bound). Parameters not given take the design
    default. `lr` is the run's `optim.lr`, if the spec sets one."""
    def p(name: str, key: str, fallback: Any = None) -> Any:
        return techniques[name].get(key, design.PARAMS.get(name, {}).get(key, fallback))

    out = []
    if "distill" in techniques and p("distill", "rows", "all") == "replay" \
            and "replay" not in techniques:
        out.append("distill rows=replay needs replay in the same run")
    if "freeze" in techniques and p("freeze", "bottom_layers", 13) + p("freeze", "top_layers", 0) > 25:
        out.append("freeze bottom_layers + top_layers must be at most 25 (26 layers)")
    if "replay" in techniques and p("replay", "select", "random") != "random" \
            and p("replay", "share", 0.05) * p("replay", "pool", 4) > 1:
        out.append("replay share x pool is above 1: forward-only tokens would pass gradient tokens")
    if "wavg" in techniques and p("wavg", "alpha", 0.25) >= 1 and p("wavg", "mode", "boundary") == "boundary":
        out.append("wavg alpha=1 in boundary mode resets to the base model after every domain")
    if "lora" in techniques and (why := lora_problem(techniques["lora"], lr)):
        out.append(why)
    return out


def bench_problems(spec: RunSpec) -> list[str]:
    """Why the frozen bench cannot run `spec`; empty when it can. Never touches a GPU."""
    from ..executor.run import config_from_spec
    from ..techniques import build_hooks

    techniques = {t.name: dict(t.params) for t in spec.techniques}
    try:
        build_hooks(spec.techniques)
        config_from_spec(spec)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError too
        text = " ".join(line.strip() for line in str(exc).splitlines()
                        if line.strip() and "errors.pydantic.dev" not in line)
        return [text[:300]]
    return combination_problems(techniques, spec.optim.get("lr"))


def full_params(techniques: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every parameter written out (plugin defaults filled in), so two ways of writing one arm
    compare equal. A name that is not a plugin (the simulator's) is kept as given."""
    from .. import techniques as registry

    registry._load_plugins()
    out = {}
    for name, params in techniques.items():
        cls = registry.REGISTRY.get(name)
        if cls is None:
            out[name] = dict(params)
            continue
        try:
            out[name] = cls.Params.model_validate(dict(params)).model_dump(mode="json")
        except ValueError:
            out[name] = dict(params)
    return out


def arm_key(techniques: Mapping[str, Mapping[str, Any]], lr: float | None = None) -> str:
    """One string per arm configuration: full parameters, numbers by value, and the run LR."""
    full = full_params(techniques)
    norm = {n: {k: (float(f"{float(v):.6g}") if isinstance(v, (int, float))
                    and not isinstance(v, bool) else v) for k, v in ps.items()}
            for n, ps in full.items()}
    rate = None if lr is None or abs(lr - design.BENCH_LR) < 1e-12 else float(f"{lr:.6g}")
    return canonical_json({"techniques": norm, "lr": rate})


def design_claim_arms() -> dict[str, int]:
    """arm_key -> PREREG claim id, for every arm a claim of `design.CLAIMS` tests (its first
    element). Comparators are left out: no-mitigation is the comparator of many claims and
    repeating it is not repeating a claim."""
    out: dict[str, int] = {}
    for claim in design.CLAIMS:
        for test in claim["tests"]:
            arm = design.claim_spec(test[0], 0)
            techniques = {t.name: dict(t.params) for t in arm.techniques}
            out.setdefault(arm_key(techniques, arm.optim.get("lr")), claim["id"])
    return out


def design_exploratory_configs() -> frozenset[str]:
    """Config hashes (seed left out) of the published-claims arms that are not screen arms.

    `launch_design` runs them under their design names (`replay-25`, `lora-r256`, `freeze-top`)
    with no explore marker. PREREG 6.5 calls them exploratory, so FREEZE must never pick one. An
    arm that is also a screen arm (`replay-5` is the replay-only half) stays confirmatory."""
    screen = {config_hash(s) for s in design.screen()}
    claims = {config_hash(design.claim_spec(name, 0)) for name in design.CLAIM_ARMS}
    return frozenset(claims - screen)


def claim_arm_spec(techniques: Mapping[str, Mapping[str, Any]], lr: float | None, seed: int,
                   stream_: Sequence[StreamDomain], notes: str) -> RunSpec:
    """The spec of one arm of a claim card at one seed. A run LR other than the bench's goes in
    `optim.lr`, which the Gatekeeper allows only on an exploratory spec."""
    optim = {"lr": lr} if lr is not None and abs(lr - design.BENCH_LR) > 1e-12 else {}
    label = arm_label(techniques) + (f"-lr{lr:g}" if optim else "")
    return build_spec(f"explore-{label}-s{seed}", seed, stream_, techniques, optim=optim,
                      notes=notes)


def claim_specs(card: ClaimCard, seeds: Sequence[int],
                stream_: Sequence[StreamDomain]) -> list[tuple[RunSpec, str]]:
    """(spec, "test" | "control") for a card at each seed: the test arm paired with its control.
    Notes are `explore:<card id>` and `explore:<card id>:control`, the label every row keeps."""
    out = []
    for seed in seeds:
        out.append((claim_arm_spec(card.test_arm, card.test_lr, seed, stream_,
                                   f"{EXPLORE_NOTE}{card.id}"), "test"))
        out.append((claim_arm_spec(card.control_arm, card.control_lr, seed, stream_,
                                   f"{EXPLORE_NOTE}{card.id}:control"), "control"))
    return out


def is_explore(spec: RunSpec) -> bool:
    """Exploratory: a claim test (`notes: "explore:<claim id>"`) or any run named `explore-*`
    (the team's naming rule for hand-launched exploratory runs). FREEZE never picks one."""
    return spec.notes.startswith(EXPLORE_NOTE) or spec.name.startswith(EXPLORE_NAME)


def explore_claim_id(spec: RunSpec) -> str | None:
    """The claim id an exploratory spec tests (`notes: "explore:<claim id>[:control]"`)."""
    if not spec.notes.startswith(EXPLORE_NOTE):
        return None
    return spec.notes[len(EXPLORE_NOTE):].split(":")[0] or None


def lr_in_range(value: Any, bounds: Sequence[float] = LR_RANGE) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and bounds[0] <= float(value) <= bounds[1]
