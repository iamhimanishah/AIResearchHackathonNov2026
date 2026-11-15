# Pending work: agent workstream (brendan), night of 14 November

Working file for the items in the fix plan (two designs per problem, merged) and
`docs/reviews/red-team-57fa138.md`. One section per item, written before the code: goal,
acceptance criteria, design, tests, out of scope; then the commit and the test result.
Nothing here is pushed until brendan approves.

## Item 1: real mode follows the pre-registered design

**Goal.** In real mode the replicate phase proposes exactly
`clharness.analysis.design.replication_specs(rows)["specs"]` (PREREG 6.3: "the agent applies this
rule mechanically, and its decision log shows the ranking it used"), and real mode reads its
thresholds from a draft `configs/agent/thresholds.yaml` transcribed from PREREG.md.

**Acceptance criteria.**
- On a copy of `results/agent/rows.jsonl`, the replicate proposal's spec hashes equal
  `replication_specs(rows)["specs"]`'s, the ranking is in the decision, 0 specs are submitted,
  the rows file is byte-identical, `report.json` is absent and no `FreezeToken` is minted.
- The rows the analysis code sees have the report half blinded (NaN), and the ranking is the
  same as on the unblinded rows.
- When every replication run already has a row, the gate drops them as duplicates and the loop
  moves on to combine ("nothing new to run"), never a repeat or a halt. When they are launched
  but have no row yet (justin: 8 runs in flight), read-only mode proposes them and does not
  repeat the phase.
- The simulator keeps its planner.
- `configs/agent/thresholds.yaml` loads through `contracts.load_prereg`, each value cites its
  PREREG section, the header says "DRAFT: himani to confirm", and `loop.py --executor real`
  uses it when `--prereg` is not given.
- (justin's note) The Conductor writes nothing under the explore directory except the inbox:
  `build_real` refuses an `--out` inside the inbox's directory.

**Design.**
- New `src/clharness/agent/prereg_plan.py`: `blind(a1_row)`, `blinded_rows(path)`,
  `replicate_plan(rows_path) -> (proposed specs, log data)`. Reuses `design.replication_specs`
  (no copy of the rule) and `read_jsonl`. If `specs` is empty because all are finished, it
  proposes the full planned set from the seed-0 rows so the gate's duplicate rule moves on.
- `Conductor(replicate_plan=None)`: when set, `_plan(REPLICATE)` goes to FALLBACK_PLAN (no model
  call) and `_fallback_plan(REPLICATE)` calls `_prereg_replicate`, which logs `ranking`,
  `selection_ranking`, `promoted`, `rule` and `proposed_spec_hashes` in the decision data. The
  gate still runs (catalog, bench, budget).
- `loop.build_real` passes `replicate_plan`; `loop.main` defaults `--prereg` to
  `configs/agent/thresholds.yaml`.

**Tests** (`tests/agent/test_prereg_plan.py`).
- `test_real_mode_replicate_proposes_exactly_the_prereg_set`: the acceptance run above.
- `test_blinding_hides_the_report_half_and_keeps_the_ranking`.
- `test_replicate_with_every_run_finished_moves_on_without_repeating`.
- `test_the_simulator_keeps_its_planner`.
- `test_thresholds_draft_transcribes_prereg_and_is_the_default`.
- `test_out_inside_the_explore_directory_is_refused`.
- Changed: `test_real_mode_cli_needs_thresholds_and_has_no_launch_flag` checked that no
  `--prereg` meant exit 2; there is now a default, so it checks that a missing thresholds file
  still means exit 2.

**Out of scope.** The screen phase in real mode still uses the fallback planner (it proposes
catalog-default single-technique specs that are not `design.screen()`); see "For brendan".
Budget trimming of PREREG cells (the gate still applies the cap; see "For brendan").
