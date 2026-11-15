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

**Result.** Commit `ef29554`. `make test`: 531 passed, 2 skipped, 1 deselected.

## Item 4: RT-15 + RT-17 (lock part), one writer per `--out`

**Goal.** Two processes can never write the same `--out` at once: a single-writer lock (stdlib
`fcntl.flock`) that the loop takes, and that justin's `launch_design` and `explore` take with
one line each (justin said yes to both call sites).

**Acceptance criteria.**
- While another process holds `<out>/.agent.lock`, `loop.main` prints `refused:` and returns
  2, and writes no state, decision log or model ledger.
- The lock is released when the holder exits, even by SIGKILL (flock is dropped by the kernel).
- Nested use in one process is a no-op (the kill tests build several Conductors in one
  process); different `--out` directories do not block each other.
- `launch_design.main --go` and `explore.main` take the same lock on their `--out` with one
  added line each, and exit 2 when it is held.
- The lock file is never unlinked and is git-ignored.

**Design.** New `src/clharness/agent/lock.py`: `AlreadyRunning(ContractError)`,
`single_writer(out_dir)` (context manager), `lock_or_exit(out_dir)` (holds the lock for the
life of the process, prints `refused:` and raises `SystemExit(2)` when held). A per-process
registry of held paths makes nesting a no-op. The holder's pid is written into the file for a
person to read. `loop.main` wraps everything after argument parsing in `single_writer`.
`.gitignore` gets `results/**/.agent.lock` (not a bare `*.lock`, which would hide `uv.lock`).

**Tests** (`tests/agent/test_redteam_fixes.py`).
- `test_rt15_a_held_lock_refuses_the_loop_and_writes_nothing` (holder in a subprocess).
- `test_rt15_the_lock_is_released_when_the_holder_is_killed`.
- `test_rt15_the_lock_is_reentrant_in_one_process_and_per_directory`.
- `test_rt15_launch_design_and_explore_take_the_lock` (a held lock makes each main exit 2).

**Out of scope.** `--resume` and the `~r{n}` decision-key suffix; the rows-file lock inside
`launch()` (the `--out` lock in `launch_design.main` covers the same rows file).

**Result.** Commit `ee152d3`. `make test`: 536 passed, 2 skipped, 1 deselected. Touched justin's
`launch_design.py` and `explore.py`: one `lock_or_exit(args.out)` line and its import in each.

## Item 5: RT-32, one model-spend ledger for the loop and the Scout CLI

**Goal.** `loop.py` and `scout.py` build their `BudgetedLLM` on one ledger,
`cfg.path("llm_calls")`, and the guard re-reads that ledger before every call, so two entry
points cannot each spend the whole model cap. (justin: explore keeps its own $10 cap and its own
log, `results/agent/explore/llm_calls.jsonl`, tonight; it is not routed through this ledger.)

**Acceptance criteria.**
- Two guards on one ledger share the cap: after one spends, a guard built earlier is refused
  when its call would pass the cap.
- `llm.budgeted_llm(cfg, agents)` writes to `cfg.path("llm_calls")`; a config whose `root` is
  relative resolves against the repository, not the working directory.
- `loop.main --llm openai` and `scout.main` build their guard through `budgeted_llm` (no
  `log_path=args.out/...`); a static test checks neither passes `log_path=`.
- The existing planner-budget tests pass.

**Design.** `llm.budgeted_llm(cfg, agents, inner=None)` builds `OpenAIClient` (or the injected
client) inside `BudgetedLLM` on `ledger_path(cfg)`. `BudgetedLLM._check_cap` refreshes
`spent_usd` from the ledger before the estimate. `loop._main_locked` and `scout.main` call the
factory. Explore is untouched.

**Tests** (`tests/agent/test_redteam_fixes.py`).
- `test_rt32_two_guards_on_one_ledger_share_the_cap`.
- `test_rt32_the_factory_writes_to_the_config_ledger`.
- `test_rt32_loop_and_scout_build_their_guard_on_the_shared_ledger`.
- `test_rt32_no_entry_point_but_explore_passes_its_own_ledger` (static).

**Out of scope.** A lock and reservation lines around check-and-append (two calls at the same
instant can still each pass the check once); merging the old per-directory ledgers.

**Result.** Commit `0f9686e`. `make test`: 540 passed, 2 skipped, 1 deselected.

## Item 6: Scout S1-S3, a timeout, kept search errors, per-item validation

**Goal.** The Scout's edges stop costing the loop: an Exa call cannot hang for 10 minutes, a
partial search failure is recorded instead of dropped, and one bad or extra model item rejects
only itself instead of throwing away the whole paid-for shortlist.

**Acceptance criteria.**
- S1: `ExaSearch(timeout=30.0)` passes the timeout to `exa.search` (and to the `Exa` client);
  a timeout from the SDK becomes a `SearchError` naming it.
- S2: `ScoutResult.search_errors` and `ScoutResult.searches` are set whenever the Scout runs
  (also when it skips), and the SCOUT decision data carries `searches` and `search_errors`.
- S3: a draft with 7 claims of which 1 is invalid keeps the first 5 valid ones; the invalid one
  and the one past the limit are in `rejected` with a reason; the same for candidates.
  `ScoutDraft.model_json_schema()` is byte-identical to before (sha256 pinned in the test).

**Design.**
- `search.ExaSearch.__init__(timeout=DEFAULT_TIMEOUT_S=30.0)`; `search(..., timeout=)`.
- `roles.ScoutResult`: `search_errors: tuple[str, ...] = ()`, `searches: int = 0` (agent-internal,
  written to the candidates file; not a shared format).
- `agents.ScoutDraft`: a `model_validator(mode="wrap")` that validates each item alone with
  `ClaimDraft`/`CandidateDraft`, keeps the first `MAX_SCOUT_ITEMS` valid ones and records the
  rest in a private attribute (`PrivateAttr`, so no schema change). `LLMScout.scout` puts those
  first in `rejected`.
- `Conductor._scout` adds `searches` and `search_errors` to the decision data. The Scout CLI
  prints the search errors.

**Tests** (`tests/agent/test_scout_fixes.py`).
- `test_s1_exa_gets_a_timeout_and_a_timeout_is_a_search_error`.
- `test_s2_partial_search_errors_are_kept_and_logged`.
- `test_s2_a_skip_still_records_the_search_errors`.
- `test_s3_one_bad_item_rejects_only_itself_and_the_list_is_cut_to_five`.
- `test_s3_the_model_facing_schema_is_unchanged`.

**Out of scope.** S4 (resume without paying twice), S5 to S8 (checks that match the explore gate,
the hand-off by configuration, CLI parity, sources evidence).

**Result.** Commit `3573e8f`. `make test`: 545 passed, 2 skipped, 1 deselected.

## Item 7: past.dev as a second reader in the Scout (S9)

**Goal.** The Scout reads what the other agents recorded on past.dev (claims already found,
hypotheses already tested or refused) before it drafts, without ever seeing a run result, and
writes its own claim cards there so the "scout" label is really written by the Scout.

**Acceptance criteria.**
- `LLMScout(memory=None)` takes an optional `PastMemory`. Before the prompt it calls
  `memory.recall(query, "scout")`.
- Firewall (brendan + justin): a statement is kept only if its writers are non-empty and all
  in {scout, explorer}; any statement with "analyst" among its writers is dropped, and so is
  any statement whose content contains "Measured effect" (a resolved verdict with a
  selection-half effect size). Both filters are tested.
- The kept statements go into a new `shared_memory` block, wrapped with
  `context.wrap_untrusted(..., "memory")`; a recalled `<<<END DATA` stays quoted.
- `shared_memory` is in `CONTEXT_MAP["scout"].reads` and in `cards/scout.md`;
  `card_problems` is empty for the real card; the firewall test lists the new block and still
  checks that the Scout sees no rows.
- A missing key or a service that is down gives no block and never raises.
- After the Conductor writes the explore inbox, the Scout writes each of its cards with
  `PastMemory.write_claim_card(card_id, hyp)` on the line read back by `explore.read_inbox`,
  so the id (`remember-claim-<card_id>`) and the text are exactly explore's, and explore's
  identical write replaces it instead of duplicating it.
- The SCOUT decision logs `memory_kept`, `memory_dropped` and the memory events.
- `--memory pastdev` on `loop.py` and `scout.py`; without `PAST_API_KEY` it prints a notice and
  runs with an empty memory.
- ARCHITECTURE's Scout rows are updated in the same commit.

**Design.** `agents.scout_memory_filter(recalled)` (pure, tested alone) and
`LLMScout._recall()`; `context.build_scout_prompt(shared_memory=...)`; `ScoutResult` gains
`memory_kept` / `memory_dropped`; `LLMScout.remember_cards(result, inbox)` reuses
`explore.read_inbox` and `pastdev.write_claim_card` (imported, not edited);
`Conductor._scout` calls it after `_write_claim_cards` and logs `memory.events`.

**Tests** (`tests/agent/test_scout_memory.py`; `tests/agent/test_kernel_context.py` updated).
- `test_s9_the_filter_keeps_scout_and_explorer_and_drops_analyst_and_verdicts`.
- `test_s9_recalled_statements_reach_the_prompt_quoted_and_filtered`.
- `test_s9_memory_down_or_unconfigured_gives_no_block_and_never_raises`.
- `test_s9_the_scout_writes_its_cards_with_explores_id_and_text` (loop in real mode with an
  inbox; the stored text equals `write_claim_card` on explore's read-back).
- `test_s9_the_card_and_map_allow_shared_memory`.
- `test_s9_memory_flag_on_both_entry_points`.

**Out of scope.** The Scout CLI writes no inbox (S6), so it recalls but does not write cards.
