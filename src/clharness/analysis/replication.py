"""The replication set (analysis/PREREG.md 6.3 and 6.3b), as run specs.

    from clharness.analysis.replication import replication_specs
    plan = replication_specs(rows)
    plan["specs"]       the runs to launch: each promoted configuration at seeds 1 and 2
    plan["ranking"]     the factorial ranking the choice came from, for the decision log
    plan["selection"]   the selection-variant choice, likewise

The choice is not made here. It is read from `attribution.analyze`, the same code that writes the
analysis report, so the set that is replicated is by construction the set the report calls
promoted: ranked on the selection half, on the seed-0 screen rows only, one row per spec hash,
among configurations that pass the acquisition gate. Nothing reads the report half.

A promoted configuration's seed-1 and seed-2 specs are its seed-0 screen spec with the seed and
the name changed, so every parameter is carried over exactly. The controls (no-mitigation, the
reference arm, the lower-LR arms) already ran at seeds 0 to 2 in phase 0 and are never returned.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from clharness.analysis import attribution as A
from clharness.spec import ResultRow, RunSpec

REPLICATION_SEEDS = (1, 2)


def _at_seed(spec: RunSpec, seed: int) -> RunSpec:
    base = spec.name.removesuffix(f"-s{spec.seed}")
    return spec.model_copy(update={
        "seed": seed, "name": f"{base}-s{seed}", "parent_spec_hash": spec.spec_hash(),
        "notes": f"Replication of {spec.name} (PREREG 6.3): same configuration, seed {seed}.",
    })


def replication_specs(rows: Iterable[ResultRow], seeds: Sequence[int] = REPLICATION_SEEDS,
                      allow_simulated: bool = False) -> dict[str, Any]:
    """The runs still to launch for replication, with the rankings they follow from.

    `rows`: every result row at hand (screen, controls, anything already replicated). Rows of
    specs that are already finished are left out of `specs`, so calling this again after some
    replication runs have landed returns only what is missing, and the ranking does not move.
    """
    rows, _ = A.dedupe_rows(list(rows))  # real-backend rows win ties (A._row_rank, red-team RT-08)
    report = A.analyze(rows, confirmatory=False, n_boot=200, allow_simulated=allow_simulated)
    if report["screen_reference"] is None:
        # without the seed-0 reference arm the gate cannot be applied, and the ranking would
        # pick configurations that fail it: refuse rather than launch paid runs (red-team RT-01)
        raise ValueError("no seed-0 reference-arm row (replay+lr_rewarm) in these rows; include "
                         "results/rows.reference.remeasured.jsonl before planning replication")
    usable = [r for r in rows if r.status == "finished"
              and (allow_simulated or r.backend in A.REAL_BACKENDS)]
    screen = A.group([r for r in usable if r.spec.seed == A.SCREEN_SEED])
    finished = {r.spec_hash for r in usable}

    promoted = [d for d in report["promotion"] if d["promoted"]]
    chosen_selection = [d for d in report["selection_screen"] if d.get("promoted")]
    specs: list[RunSpec] = []
    for d in [*promoted, *chosen_selection]:
        seed0 = screen[d["key"]].rows[0].spec
        specs += [_at_seed(seed0, s) for s in seeds]
    planned = len(specs)
    specs = [s for s in specs if s.spec_hash() not in finished]
    return {
        "specs": specs,
        "planned": planned,
        "already_finished": planned - len(specs),
        "ranking": report["promotion"],
        "selection": report["selection_screen"],
        "promoted": [d["config"] for d in promoted],
        "promoted_selection": [d["config"] for d in chosen_selection],
        "rule": "PREREG 6.3: top 3 factorial configurations on the selection half among those "
                "passing the acquisition gate, from seed-0 rows; 6.3b: per selection signal, the "
                "better keep direction if it passes the gate. Each at seeds 1 and 2.",
    }
