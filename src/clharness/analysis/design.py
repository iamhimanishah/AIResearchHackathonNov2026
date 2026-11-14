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

# Frozen parameters, matching configs/bench/specs/reference-s0.json and sanity/*.json.
PARAMS: dict[str, dict[str, Any]] = {
    "replay": {"share": 0.05, "source": "general"},
    "lr_rewarm": {"rewarm_steps": 8, "min_lr_ratio": 0.1, "schedule": "cosine"},
    "lora": {"rank": 8},
    "freeze": {"bottom_layers": 13},
    "distill": {"weight": 1.0, "top_k": 64},
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
         stream: tuple[tuple[str, int], ...] = STREAM, lr: float | None = None) -> RunSpec:
    """One run spec. `selection=(signal, keep)` sets replay's selection (needs replay on).
    `lr` overrides the bench learning rate; it is only for the lower-LR controls (A13)."""
    if selection and "replay" not in techniques:
        raise ValueError("selection is a parameter of replay; switch replay on")
    if lr is not None and techniques:
        raise ValueError("a learning-rate override is only for the no-mitigation LR controls")
    techs = []
    for name in sorted(techniques):
        params = dict(PARAMS[name])
        if name == "replay" and selection:
            params.update(SELECTION_PARAMS[selection[0]], keep=selection[1])
        techs.append(TechniqueSpec(name=name, params=params))
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
