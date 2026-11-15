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

**Result.** Commit `904a019`. `make test`: 520 passed, 2 skipped, 1 deselected. On a copy of
`results/agent/rows.jsonl` with the `$80` cap of `configs/agent.yaml`, the proposal is the 10
pre-registered specs and the gate keeps 4 (budget: replicate treatments may spend 70% of what
is left); with a `$700` cap (the test) all 10 are proposed and logged.

## Item 2: RT-16, unpriced rows are charged the estimate

**Goal.** `Conductor._spent` charges the estimate for every row without a positive, finite
price, infra failures included, so a lost Modal call cannot make the budget look unspent.

**Acceptance criteria.**
- An unpriced infra row lowers `compute_usd_left` by the spec's estimate (today: by 0).
- A NaN, infinite, zero or negative price is charged the estimate.
- The explore share counts an unpriced exploratory infra row the same way.
- Existing budget tests pass.

**Design.** One expression in `conductor._spent`: `usd if math.isfinite(usd) and usd > 0 else
estimate`. Reuses `_estimate`. The plan's optional `cost=` on `ModalExecutor._failed` is left
out: it changes what the launcher writes on the grid's money path tonight.

**Tests** (`tests/agent/test_redteam_fixes.py`).
- `test_rt16_an_unpriced_infra_row_is_charged_the_estimate`.
- `test_rt16_a_nan_or_negative_price_is_charged_the_estimate`.
- `test_rt16_the_explore_share_counts_an_unpriced_infra_row`.

**Out of scope.** Keeping the measured cost on the launcher's wrong-spec and git-sha refusals
(modal_executor; after the confirmatory rows).

**Result.** Commit `b220349`. `make test`: 526 passed, 2 skipped, 1 deselected.

## Item 3: RT-30, a timed-out run is re-queued once

**Goal.** The Operator re-queues a `timeout` failure once, as it does an `infra` failure
(PREREG 8: "re-queued once by the executor"), and every place that treats a failure as
retryable reads one shared constant.

**Acceptance criteria.**
- A timeout is re-submitted once and its spec settles on attempt 2.
- Two timeouts settle the spec (no third run), and the gate then drops it as exhausted.
- An infra failure then a timeout share the same cap of 2 attempts.
- A re-queue submit that raises `ExecutorUnreachable` does not kill the wait; it is retried on
  the next poll.
- One constant `contracts.RETRYABLE_KINDS` (and `retryable(row)`), used by the Operator,
  `Conductor._retry_exhausted`, `gatekeeper.duplicate`, `ModalExecutor.submit`, the simulator
  and the stub operator in roles.py. A static test finds no other `("infra", "timeout")` literal.

**Design.** `contracts.RETRYABLE_KINDS = frozenset({"infra", "timeout"})`, `retryable(row)`.
`operator._settled` uses it with `MAX_INFRA_ATTEMPTS` (kept as the one cap). The re-queue
`submit` in `Operator.wait` is wrapped in `except ExecutorUnreachable` and is not marked
requeued, so the next tick tries again. The gate's reason reads "infrastructure or timeout".

**Tests** (`tests/agent/test_redteam_fixes.py`).
- `test_rt30_a_timeout_is_retried_once_and_settles_on_attempt_two`.
- `test_rt30_two_timeouts_settle_and_the_gate_drops_the_spec`.
- `test_rt30_infra_then_timeout_share_the_cap`.
- `test_rt30_a_requeue_survives_an_unreachable_executor`.
- `test_rt30_one_constant_for_retryable_failures` (static).

**Out of scope.** Charging a timeout the 50-minute cost (RT-16 already charges the estimate);
the ARCHITECTURE "unreachable 3 times -> HALTED" rule.
