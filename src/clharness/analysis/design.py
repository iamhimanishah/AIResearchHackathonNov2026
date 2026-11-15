"""The pre-registered screening design (analysis/PREREG.md, section 6.2) as code.

Three blocks, all at seed 0 and all on the frozen fast bench:

  FACTORIAL   2^(6-2) resolution IV over the six techniques, 16 runs. Generators
              distill = replay*lr_rewarm*lora, wavg = replay*lr_rewarm*freeze. Chosen because both
              controls (no-mitigation, the reference arm) are runs of the design.
  HALVES      replay alone and lr_rewarm alone. With no-mitigation and the reference arm they
              complete the 2x2 of the reference recipe, so replay x lr_rewarm is estimable even
              though the factorial aliases it with lora x distill and freeze x wavg.
  SELECTION   the reference arm with replay's `select` signal switched from random (the control:
              plain replay) to loss or act_nov, keeping the high or the low end. Selection is a
              parameter of replay (docs/TECHNIQUES.md, decision 2), not a seventh factor.

Parameters are the technique defaults used by workstream 3's specs (configs/bench/specs/). They
are frozen with the PREREG; changing one is an amendment.

Imports only the spec module, so the agent can build the grid from it without torch.
"""

from __future__ import annotations

import itertools
from typing import Any

from clharness.spec import RunSpec, StreamDomain, TechniqueSpec

FACTORS = ("replay", "lr_rewarm", "lora", "freeze", "distill", "wavg")
GENERATORS = {"distill": ("replay", "lr_rewarm", "lora"), "wavg": ("replay", "lr_rewarm", "freeze")}

# Frozen parameters, matching configs/bench/specs/reference-s0.json and sanity/*.json, except
# distill top_k (256, not the sanity spec's 64; PREREG A15). lora keeps its plugin default lr 1e-3:
# a stated exception to the bench LR (PREREG 0.3, A15).
PARAMS: dict[str, dict[str, Any]] = {
    "replay": {"share": 0.05, "source": "general"},
    "lr_rewarm": {"rewarm_steps": 8, "min_lr_ratio": 0.1, "schedule": "cosine"},
    "lora": {"rank": 8},
    "freeze": {"bottom_layers": 13},
    "distill": {"weight": 1.0, "top_k": 256},  # PREREG A15: 64 under-represents distillation (lit scan)
    "wavg": {"alpha": 0.25},
}
SELECTION_PARAMS: dict[str, dict[str, Any]] = {
    "loss": {"select": "loss", "pool": 4},
    "act_nov": {"select": "act_nov", "pool": 4, "layer": 13},
}
STREAM = (("code", 10_000_000), ("legal", 10_000_000))  # configs/bench/fast.yaml, frozen 2026-11-14
NO_MITIGATION: frozenset[str] = frozenset()
REFERENCE = frozenset({"replay", "lr_rewarm"})
BENCH_LR = 1e-4  # configs/bench/fast.yaml
# PREREG A13: no-mitigation at lower learning rates, the "just lower the LR" controls. Run at the
# replication seeds (0, 1, 2); seed 0 is A5's calibration run at that rate (same spec hash).
LR_CONTROLS = (5e-5, 3e-5)


def factorial() -> list[frozenset[str]]:
    """The 16 technique sets of the 2^(6-2) design, no-mitigation first."""
    base = FACTORS[:4]
    runs = []
    for levels in itertools.product((False, True), repeat=4):
        on = {f for f, x in zip(base, levels) if x}
        for derived, parents in GENERATORS.items():
            # +1/-1 product of the parents: on when an even number of parents are off
            if sum(p not in on for p in parents) % 2 == 0:
                on.add(derived)
        runs.append(frozenset(on))
    return sorted(runs, key=lambda s: (len(s), sorted(s)))


HALVES = [frozenset({"replay"}), frozenset({"lr_rewarm"})]
SELECTION = [(signal, keep) for signal in ("loss", "act_nov") for keep in ("high", "low")]


def label(techniques: frozenset[str], selection: tuple[str, str] | None = None) -> str:
    name = "+".join(sorted(techniques)) or "no-mitigation"
    return f"{name}+select-{selection[0]}-{selection[1]}" if selection else name


def spec(techniques: frozenset[str], seed: int, selection: tuple[str, str] | None = None,
         stream: tuple[tuple[str, int], ...] = STREAM, lr: float | None = None,
         params: dict[str, dict[str, Any]] | None = None, name: str | None = None) -> RunSpec:
    """One run spec. `selection=(signal, keep)` sets replay's selection (needs replay on).
    `lr` overrides the bench learning rate; it is only for the lower-LR controls (A13)."""
    if selection and "replay" not in techniques:
        raise ValueError("selection is a parameter of replay; switch replay on")
    if lr is not None and techniques:
        raise ValueError("a learning-rate override is only for the no-mitigation LR controls")
    techs = []
    for tech in sorted(techniques):
        p = {**PARAMS[tech], **(params or {}).get(tech, {})}  # `params`: published-claims arms (6.5)
        if tech == "replay" and selection:
            p.update(SELECTION_PARAMS[selection[0]], keep=selection[1])
        techs.append(TechniqueSpec(name=tech, params=p))
    if name is None:
        name = f"{label(techniques, selection)}-lr{lr:g}" if lr is not None else label(techniques, selection)
    return RunSpec(
        name=f"{name}-s{seed}", seed=seed,
        stream=[StreamDomain(name=n, tokens=t) for n, t in stream],
        optim={"lr": lr} if lr is not None else {},
        techniques=techs, notes="screen (analysis/PREREG.md 6.2)" if lr is None else "LR control (PREREG A13)",
    )


def lr_controls(seeds: range = range(3)) -> list[RunSpec]:
    """No-mitigation at each lower learning rate, at the replication seeds (PREREG A13)."""
    return [spec(NO_MITIGATION, s, lr=lr) for lr in LR_CONTROLS for s in seeds]


def lr_of(row_spec: RunSpec) -> float | None:
    """The overridden learning rate of a lower-LR control, else None."""
    lr = row_spec.optim.get("lr")
    return float(lr) if lr is not None and not row_spec.techniques and float(lr) != BENCH_LR else None


def screen(seed: int = 0) -> list[RunSpec]:
    """Every screening run: factorial (16) + halves (2) + selection (4) = 22 specs."""
    out = [spec(t, seed) for t in factorial()]
    out += [spec(t, seed) for t in HALVES]
    out += [spec(REFERENCE, seed, sel) for sel in SELECTION]
    return out


def selection_of(row_spec: RunSpec) -> tuple[str, str] | None:
    """(signal, keep) when the spec's replay selects by a signal, else None (random = control)."""
    for t in row_spec.techniques:
        if t.name == "replay" and t.params.get("select", "random") != "random":
            return t.params["select"], t.params.get("keep", "high")
    return None


def is_design_spec(row_spec: RunSpec) -> bool:
    """True when every technique uses the frozen design parameters (selection aside) and the
    stream is the bench's. Off-design rows (parameter variants, sanity runs) stay out of the
    effects fit and of promotion; they are still counted and flagged."""
    if tuple((d.name, d.tokens) for d in row_spec.stream) != STREAM:
        return False
    if row_spec.optim and not (lr_of(row_spec) in LR_CONTROLS and set(row_spec.optim) == {"lr"}):
        return False
    for t in row_spec.techniques:
        params = {k: v for k, v in t.params.items() if not (t.name == "replay" and k in ("select", "pool", "keep", "layer"))}
        if t.name not in PARAMS or params != PARAMS[t.name]:
            return False
    return True


def alias_chains() -> list[list[tuple[str, str]]]:
    """Pairs of factors whose interaction columns coincide in the factorial (PREREG 6.2)."""
    runs = factorial()
    chains: dict[tuple[bool, ...], list[tuple[str, str]]] = {}
    for a, b in itertools.combinations(sorted(FACTORS), 2):
        col = tuple(((a in r) == (b in r)) for r in runs)  # +1 when both on or both off
        chains.setdefault(col, []).append((a, b))
    return sorted(chains.values())


# ---------------------------------------------------------------------------------------------
# Phase 0 (PREREG section 0): the controls, run before the freeze and before any technique
# comparison. sigma_noise and the acquisition margins come from these runs.
# ---------------------------------------------------------------------------------------------

PHASE0_LIST = "configs/bench/phase0.txt"  # the spec files, one path per line, for `clh run --modal`
# Phase-0 runs that already ran (before the current executor). They are reused after
# re-measuring them from their saved checkpoints (pinned item split, own-boundary acquisition):
# same spec hash, no retraining. Spec file -> the committed rows file holding the original row.
PHASE0_REUSE = {
    **{f"configs/bench/specs/no-mitigation-s{s}.json": "results/rows.noise.jsonl" for s in range(5)},
    "configs/bench/specs/calibration-code-legal-lr5e-5-s0.json": "results/rows.calibration.jsonl",
    "configs/bench/specs/calibration-code-legal-lr3e-5-s0.json": "results/rows.calibration.jsonl",
}


def phase0() -> list[RunSpec]:
    """Every phase-0 run, reused or new: no-mitigation at the bench LR (seeds 0-4), the
    reference arm (seeds 0-2), and no-mitigation at each lower LR (seeds 0-2)."""
    out = [spec(NO_MITIGATION, s) for s in range(5)]
    out += [spec(REFERENCE, s) for s in range(3)]
    out += lr_controls(range(3))
    return out


def phase0_new() -> list[RunSpec]:
    """The phase-0 runs that still need a GPU: the reference arm (seeds 0-2) and the lower-LR
    controls at seeds 1-2. The noise floor and the seed-0 LR controls already ran (PHASE0_REUSE)."""
    reused_seed0_lr = {(lr, 0) for lr in LR_CONTROLS}
    return [s for s in phase0()
            if frozenset(s.technique_names()) == REFERENCE
            or (lr_of(s) is not None and (lr_of(s), s.seed) not in reused_seed0_lr)]


# ---------------------------------------------------------------------------------------------
# Published-claims family (PREREG section 6.5): exploratory, 3 seeds, run with the screen.
# Each claim from docs/published-claims.md is a test of one arm against one comparator.
# ---------------------------------------------------------------------------------------------

# arm name -> (techniques, parameter overrides on top of PARAMS)
CLAIM_ARMS: dict[str, tuple[frozenset[str], dict[str, dict[str, Any]]]] = {
    "replay-5": (frozenset({"replay"}), {}),  # = the reference half replay-only
    "replay-25": (frozenset({"replay"}), {"replay": {"share": 0.25}}),
    "replay+distill-on-replay": (frozenset({"replay", "distill"}), {"distill": {"rows": "replay"}}),
    "lora-r8": (frozenset({"lora"}), {}),
    "lora-r256": (frozenset({"lora"}), {"lora": {"rank": 256}}),
    "freeze-bottom": (frozenset({"freeze"}), {}),
    "freeze-top": (frozenset({"freeze"}), {"freeze": {"bottom_layers": 0, "top_layers": 13}}),
}
CLAIM_SEEDS = range(3)
# comparators that are not claim arms: name -> spec builder for a seed
_COMPARATORS = {
    "no-mitigation": lambda s: spec(NO_MITIGATION, s),
    "reference": lambda s: spec(REFERENCE, s),
    "lr-3e-05": lambda s: spec(NO_MITIGATION, s, lr=3e-5),
    "rewarm-only": lambda s: spec(frozenset({"lr_rewarm"}), s),
}

# outcome: "retention" (primary, report half) or "acq_<domain>" (log own-boundary perplexity
# ratio, so "learns less" is positive). sign: the published direction of (arm - comparator).
# kind: "directional" passes or fails; "magnitude" is reported next to the published number.
CLAIMS: list[dict[str, Any]] = [
    {"id": 1, "claim": "Replay plus LR re-warm beats no mitigation", "source": "Ibrahim et al. 2024",
     "tests": [("reference", "no-mitigation", "retention", +1, "directional")]},
    {"id": 2, "claim": "Peak LR is the main knob; warmup barely matters", "source": "Ibrahim et al. 2024",
     "tests": [("lr-3e-05", "no-mitigation", "retention", +1, "directional"),
               ("rewarm-only", "no-mitigation", "retention", 0, "magnitude")]},
    {"id": 3, "claim": "Replay is worth about 1 to 3 points at 1B", "source": "Abbes et al. 2026",
     "tests": [("replay-5", "no-mitigation", "retention", +1, "directional"),
               ("replay-25", "no-mitigation", "retention", +1, "directional"),
               ("replay-25", "replay-5", "retention", +1, "directional")]},
    {"id": 5, "claim": "LoRA forgets less and learns less; higher rank learns more", "source": "Biderman et al. 2024",
     "tests": [("lora-r8", "no-mitigation", "retention", +1, "directional"),
               ("lora-r8", "no-mitigation", "acq_legal", +1, "directional"),
               ("lora-r256", "lora-r8", "acq_legal", -1, "directional")]},
    {"id": 6, "claim": "Distilling on replay tokens beats plain replay", "source": "Beyond One-Hot Replay, 2026",
     "tests": [("replay+distill-on-replay", "replay-5", "retention", +1, "directional")]},
    {"id": 7, "claim": "Freezing the top half beats freezing the bottom half", "source": "Freeze Deep, Train Shallow, 2026",
     "tests": [("freeze-top", "freeze-bottom", "retention", +1, "directional")]},
]
# Covered elsewhere or not testable here: 4 (Reptile averaging: no plugin), 8 (selection vs
# random: the selection family, 6.3b), 9 (averaging vs 10% replay: the wavg and replay main
# effects of the screen, exploratory).
CLAIMS_ELSEWHERE = {4: "not tested: no Reptile plugin", 8: "selection family (PREREG 6.3b)",
                    9: "screen main effects of wavg and replay (exploratory)"}


def claim_spec(name: str, seed: int) -> RunSpec:
    """The spec of a claim arm or comparator at a seed."""
    if name in CLAIM_ARMS:
        techs, params = CLAIM_ARMS[name]
        return spec(techs, seed, params=params, name=name)
    return _COMPARATORS[name](seed)


def claims_specs() -> list[RunSpec]:
    """Every claim-arm run at seeds 0-2. Comparators come from phase 0 or the screen."""
    return [claim_spec(n, s) for n in CLAIM_ARMS for s in CLAIM_SEEDS]


def claims_new() -> list[RunSpec]:
    """The claim-arm runs not already scheduled elsewhere (replay-5 at seed 0 is the screen's
    replay-only half)."""
    elsewhere = {x.spec_hash() for x in screen() + phase0()}
    return [x for x in claims_specs() if x.spec_hash() not in elsewhere]
