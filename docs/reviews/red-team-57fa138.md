# Red-team review of main at 57fa138

Five reviewers read commit 57fa138 on `main` without changing it. The five areas were bench core and executor, techniques, analysis and pre-registration, the agent system and page, and security, hygiene and reproducibility. An independent skeptic then checked every finding and removed the ones that did not hold. **47 findings survived: 0 critical, 4 high, 13 medium and 30 low.** 39 are confirmed and 8 are plausible. 10 were already known from an earlier review. Three items were refuted and dropped; they are listed at the end.

Fix these first:

1. **RT-01:** the acquisition gate passes every configuration when the seed-0 reference row is missing. Promotion can then pick the wrong configurations for paid replication runs.
2. **RT-03:** `agent/loop.py` ignores the $80 compute cap in `configs/agent.yaml` and spends against a $700 default.
3. **RT-04:** near the end of its budget the Conductor rejects its own fallback plan and halts. It never reaches FREEZE and never writes a report.
4. **RT-02:** FREEZE can pick an exploratory published-claims arm as the confirmatory winner.
5. **RT-05:** the bootstrap method, and so the reported confidence intervals, change silently with the working directory.

RT-08 (de-duplication runs before the backend filter) is worth fixing together with RT-01, because it is one way the seed-0 reference row goes missing.

## Summary table

| id | severity | verdict | area | title | file:line |
|---|---|---|---|---|---|
| RT-01 | high | confirmed | Analysis (himani) | Acquisition gate passes everything when the seed-0 reference row is missing | src/clharness/analysis/attribution.py:139 |
| RT-02 | high | confirmed | Agent (brendan) | FREEZE can pick an exploratory claims arm as the confirmatory winner | src/clharness/agent/conductor.py:615 |
| RT-03 | high | confirmed | Agent (brendan) | loop.py ignores the $80 config cap and uses a $700 default | src/clharness/agent/loop.py:107 |
| RT-04 | high | confirmed | Agent (brendan) | Near the end of the budget the loop halts with no FREEZE or report | src/clharness/agent/conductor.py:420 |
| RT-05 | medium | confirmed | Security/repro | Bootstrap mode and CIs depend on the working directory | src/clharness/analysis/attribution.py:316 |
| RT-06 | medium | confirmed | Bench (justin) | Re-running an existing spec hash on Modal deletes the original checkpoints and row | src/clharness/executor/run.py:112 |
| RT-07 | medium | confirmed | Bench (justin) | items.py never labels an item 'lost' on 3-checkpoint runs, and hard-codes the domain order | src/clharness/items.py:56 |
| RT-08 | medium | confirmed | Analysis (himani) | De-duplication runs before the backend filter, so a sim row can replace a real row | src/clharness/analysis/attribution.py:781 |
| RT-09 | medium | confirmed | Analysis (himani) | PREREG 8 validity rules (tokens within 1%, anomalous grad steps) are not enforced | src/clharness/analysis/attribution.py:782 |
| RT-10 | medium | confirmed | Analysis (himani) | Claim arms enter the pre-registered effects fit | src/clharness/analysis/attribution.py:790 |
| RT-11 | medium | confirmed | Analysis (himani) | distill×lora is aliased with freeze×wavg but not labelled as aliased | src/clharness/analysis/attribution.py:275 |
| RT-12 | medium | confirmed | Agent (brendan) | Explore gate accepts freeze specs the bench cannot run; hypotheses stay 'running' | src/clharness/agent/explore.py:372 |
| RT-13 | medium | confirmed | Agent (brendan) | Scout CLI crashes after paying for search and the model, and uses the simulator catalog | src/clharness/agent/scout.py:55 |
| RT-14 | medium | confirmed | Agent (brendan) | veto.jsonl is never read; `autonomy: veto` does nothing | src/clharness/agent/conductor.py:161 |
| RT-15 | medium | confirmed | Agent (brendan) | No single-instance lock, and LoopState.resume has no caller | src/clharness/agent/conductor.py:132 |
| RT-16 | medium | confirmed | Agent (brendan) | Infra rows are charged $0 in the Conductor's spend | src/clharness/agent/conductor.py:217 |
| RT-17 | medium | plausible | Agent (brendan) | `<rows>.modal.json` is overwritten whole, so a shared rows file loses call ids and double-bills | src/clharness/agent/modal_executor.py:184 |
| RT-18 | low | confirmed | Bench (justin) | `clh run --modal --detach` never writes rows to the local file | src/clharness/executor/modal_app.py:217 |
| RT-19 | low | confirmed | Bench (justin) | fetch-items treats any single items file as a complete download | src/clharness/cli.py:521 |
| RT-20 | low | confirmed | Bench (justin) | A domain's token budget can end early with no error | src/clharness/train/loop.py:316 |
| RT-21 | low | confirmed | Bench (justin) | Eval items and model weights are not pinned in the run path | src/clharness/evals/lm_eval_runner.py:194 |
| RT-22 | low | confirmed | Techniques (peter) | With train.compile on, the distill teacher shares the student's forward, so KL is 0 | src/clharness/techniques/distill.py:150 |
| RT-23 | low | confirmed | Techniques (peter) | Return values of on_domain_end are dropped | src/clharness/train/loop.py:358 |
| RT-24 | low | confirmed | Analysis (himani) | Margins and the rule-4 control are recomputed from seed 0 only | src/clharness/analysis/attribution.py:801 |
| RT-25 | low | confirmed | Analysis (himani) | PREREG 8 collapse sensitivity, majority rate and some secondary outcomes are missing | src/clharness/analysis/attribution.py:815 |
| RT-26 | low | confirmed | Analysis (himani) | Cost chart uses GPU-hours, and its dot and bar come from different estimators | src/clharness/analysis/figures.py:413 |
| RT-27 | low | confirmed | Analysis (himani) | to_markdown raises KeyError in confirmatory mode without no-mitigation rows | src/clharness/analysis/attribution.py:985 |
| RT-28 | low | confirmed | Agent (brendan) | The explore cap applies per invocation, with no total cap | src/clharness/agent/explore.py:483 |
| RT-29 | low | confirmed | Agent (brendan) | Resumed launch_design/explore runs drop their launch decisions from the log | src/clharness/agent/launch_design.py:181 |
| RT-30 | low | confirmed | Agent (brendan) | The Operator never re-runs a timed-out run | src/clharness/agent/operator.py:32 |
| RT-31 | low | confirmed | Agent (brendan) | The explore prompt does not escape web text inside `<source>` tags | src/clharness/agent/explore.py:553 |
| RT-32 | low | confirmed | Agent (brendan) | The model budget is tracked per log file, so each entry point gets the full cap | src/clharness/agent/llm.py:309 |
| RT-33 | low | confirmed | Agent (brendan) | The fallback planner replicates the top 2 treatments, not 3 | src/clharness/agent/roles.py:383 |
| RT-34 | low | confirmed | Security/repro | A rerun reuses the original item directory, so reproduced rows pair with the wrong item files | src/clharness/executor/run.py:110 |
| RT-35 | low | confirmed | Security/repro | Seed-0 lower-LR control rows were produced from a dirty tree | results/rows.phase0.remeasured.jsonl:6 |
| RT-36 | low | confirmed | Security/repro | Stale-deployment check ignores uv.lock and pyproject.toml | src/clharness/executor/modal_app.py:54 |
| RT-37 | low | confirmed | Security/repro | openai is not a declared dependency; OPENAI_API_KEY is missing from .env.example | src/clharness/agent/llm.py:229 |
| RT-38 | low | confirmed | Security/repro | README hard-codes a personal Modal profile name | README.md:95 |
| RT-39 | low | confirmed | Security/repro | README says the loop cannot do LoRA, loss terms or per-domain re-warm | README.md:129 |
| RT-40 | low | confirmed | Security/repro | Local agent folders and CLAUDE.local.md are not gitignored | .gitignore:37 |
| RT-41 | low | plausible | Bench (justin) | A hung run may be billed up to 3 × 75 minutes if retries cover timeouts | src/clharness/executor/modal_app.py:141 |
| RT-42 | low | plausible | Bench (justin) | Remeasured rows cannot be told apart from their originals | src/clharness/executor/run.py:149 |
| RT-43 | low | plausible | Bench (justin) | NaN items are dropped silently but n_items reports the full half | src/clharness/executor/rows.py:62 |
| RT-44 | low | plausible | Analysis (himani) | 0/1 coding lets omitted interactions bias the kept effects | src/clharness/analysis/attribution.py:259 |
| RT-45 | low | plausible | Analysis (himani) | Confirmatory interval is a percentile bootstrap over 3 seeds (liberal) | src/clharness/analysis/attribution.py:359 |
| RT-46 | low | plausible | Agent (brendan) | Two different "exploratory" markers; a claim card can be paid for twice | src/clharness/agent/explore.py:56 |
| RT-47 | low | plausible | Agent (brendan) | Source links put web URLs into href without a scheme check | src/clharness/page/render.py:201 |

---

## High

### RT-01: Acquisition gate passes everything when the seed-0 reference row is missing
- **Severity / verdict:** high, confirmed
- **What is wrong:** `acquisition_gate()` returns `{"passes": True, "domains": {}}` when `ref is None`. `analyze()` builds `screen_ref` from seed-0 rows only, and `promote()`, `promote_selection()` and `choose_lr_control()` all use it. Nothing warns, because `out["reference"]` still names the reference arm from the all-seed set.
- **Evidence:** src/clharness/analysis/attribution.py:139. Callers are at about lines 467, 487 and 507, and `screen_ref` is built at about lines 799-800.
- **Failure scenario:** the screen's F04 run has the same spec hash as reference-s0. It lives only in `results/rows.reference.remeasured.jsonl`, not in the screen rows file. If someone runs `clh plan-replication` or `launch_design --only replication --have <files>` without that file, `screen_ref` is None. A test on `results/fake/rows.jsonl` with the reference seed-0 row removed changed the promoted set from {distill+freeze+replay, distill+freeze+lr_rewarm, distill+lr_rewarm+wavg} to {all-six, lora+replay+wavg, distill+lora+lr_rewarm+replay}. All three of the new configurations fail the gate when the reference is present. `replication_specs` would then launch 6 paid Modal runs of the wrong configurations, and those runs set the confirmatory family.
- **Smallest fix:** fail closed. Return `passes=False` when `ref is None`. Make `analyze` and `replication_specs` raise, or promote nothing, with a clear message when `screen_ref` or `screen_base` is None.
- **Owner area:** analysis and pre-registration (himani)
- **Known from earlier review:** no

### RT-02: FREEZE can pick an exploratory published-claims arm as the confirmatory winner
- **Severity / verdict:** high, confirmed
- **What is wrong:** `_freeze` keeps every row where `usable(row) and not is_explore(row.spec)`. `is_explore` only matches notes that start with `"explore:"`. Claim arms from `design.claims_new()` are built by `design.spec` with `notes="screen (analysis/PREREG.md 6.2)"`, so they are never treated as exploratory. FREEZE also never checks that an arm passed this loop's gate.
- **Evidence:** src/clharness/agent/conductor.py:615 (lines 614-616); catalog_real.py:202; design.py:99; the loop.py docstring at lines 17-20.
- **Failure scenario:** `launch_design --go` (all) writes the claim arms at seeds 0-2 into `results/agent/rows.jsonl`. The Conductor runs with `--rows` on that file and replicate seeds 0-2. replay-25 or freeze-top has 3 usable rows, clears the acquisition floor and becomes the frozen "best replicated arm". `report.json` then reports as the winner an arm that PREREG 6.5 calls exploratory and that the Planner and Gatekeeper never chose.
- **Smallest fix:** limit FREEZE candidates to spec hashes this Conductor accepted at a GATE (keep the union of `verdict.kept` in state). Or give claim specs a note that `is_explore` recognises.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no
- **Note:** the skeptic downgraded this from critical. `report.json` is the Conductor's report, not the PREREG analysis pipeline, and the scenario needs replicate seeds 0-2 in the `--prereg` file.

### RT-03: loop.py ignores the config's $80 compute cap and uses a $700 CLI default
- **Severity / verdict:** high, confirmed
- **What is wrong:** `--compute-cap-usd` defaults to 700.0 and goes straight to `build_real`. From the config, only models, prices, `model_budget_usd` and `llm_max_output_tokens` are read. `compute_budget_usd`, `compute_reserve_fraction`, `deadline`, `phase_timeout_minutes` and `executor.concurrency` are never read.
- **Evidence:** src/clharness/agent/loop.py:107; build_real is called at line 167; configs/agent.yaml:16 sets `compute_budget_usd: 80.0`; DIRECTION.md:177 gives $700 as the whole team's Modal cap.
- **Failure scenario:** the documented overnight command (`--executor modal --config configs/agent.yaml ... --go`) runs with a $700 cap minus a 30% reserve. That is about 6× the agreed $80, and there is no deadline.
- **Smallest fix:** when `--config` is given, take the cap, reserve, deadline, phase timeout and concurrency from it. Make `--compute-cap-usd` required for `--executor modal`.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no
- **Note:** downgraded from critical, because the dry run prints "cap $700".

### RT-04: Near the end of the budget the loop halts with no FREEZE or report
- **Severity / verdict:** high, confirmed
- **What is wrong:** the gatekeeper's budget rule rejects the whole proposal when any one control costs more than what is left. `_out_of_room` only checks that one baseline run still fits. When the fallback plan is rejected for a reason other than duplicates, `_gate` goes to HALTED with "the deterministic plan was rejected: this is a bug".
- **Evidence:** src/clharness/agent/conductor.py:420; `_out_of_room` at conductor.py:269-275; gatekeeper.py:212-215.
- **Failure scenario:** reproduced with `build_sim(infra_fail_rate=0)`. Caps of $10, $28, $40, $50 and $61 all end halted, with no `report.json`. Caps of $70, $100, $200 and $700 reach done. At a $40 cap, $7.50 is left and one baseline run costs $2.50, but the replicate plan's controls do not all fit. Finished screen and replicate rows are never frozen.
- **Smallest fix:** in `_gate`, treat a REJECT with any budget drop as out of room and go to FREEZE. Or have `_out_of_room` compare against the cheapest full plan, not one run.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

---

## Medium

### RT-05: Bootstrap mode and CIs depend on the working directory
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `checkpoint_items` globs `Path(row.items_path)`. That path is repo-relative (`results/items/<hash>/checkpoints`), so it resolves against the current directory. When nothing matches, `paired_bootstrap` silently falls back to `mode="parametric"`, and `grid_analysis.md` never prints the mode. `cmd_fetch_items` resolves against REPO_ROOT, so fetching and reading disagree.
- **Evidence:** src/clharness/analysis/attribution.py:316 (fallback at lines 346-357); executor/run.py:31; cli.py:520; tests/test_fetch_items.py does not cover reading a relative path back.
- **Failure scenario:** `clh grid-analysis` on the three REPORTED_ROWS files, run from outside the repo root, switched all 6 intervals from "items" to "parametric". The lr_rewarm vs_base CI moved from [-0.055, -0.014] to [-0.077, +0.005]. Another CI moved from [+0.004, +0.046] to [-0.017, +0.066]. Both now include zero. The same happens when the Conductor or the explore loop calls `analyze()` from another directory.
- **Smallest fix:** resolve a relative `items_path` against REPO_ROOT. Print the bootstrap mode in the markdown. Warn, or fail under `--confirmatory`, when a real-backend row's item files are missing. Add a read-back test.
- **Owner area:** analysis (himani); found by the security and reproducibility reviewer
- **Known from earlier review:** no
- **Note:** `make reproduce` runs from the repo root, so the committed numbers are correct.

### RT-06: Re-running an existing spec hash on Modal deletes the original checkpoints and overwrites its row
- **Severity / verdict:** medium, confirmed
- **What is wrong:** the run directory is `spec_hash[:16]`. On a rerun, `shutil.rmtree(run_dir)` deletes it, and `modal_app.py` overwrites `/vol/rows/<spec_hash>.json`. That includes overwriting with a failed row. `make reproduce-runs` reruns phase0.txt plus every `# REUSE` spec, which are the exact hashes behind the committed rows.
- **Evidence:** src/clharness/executor/run.py:112 (lines 111-113, run_name at lines 35-37); modal_app.py:156-161; Makefile:59-62.
- **Failure scenario:** `make reproduce-runs` deletes the original checkpoints on the volume. A later `remeasure` reads the reproduced run's checkpoints but keeps the original row's cost, started_at and git_sha. The result is a hybrid row. If the rerun fails part-way, the finished volume row is replaced by a failed one. The same happens on an agent respawn, or on a second `clh run --modal` of a finished spec.
- **Smallest fix:** put an attempt id in the run directory and the row file name, or have `reproduce-runs` write to `/vol/runs-repro`. Never delete a directory that holds a finished row.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no
- **Note:** downgraded from high. The committed rows and item files are safe in git. RT-34 covers the local item-file side of the same mechanism.

### RT-07: items.py never labels an item 'lost' on 3-checkpoint runs, and hard-codes the domain order
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `not v[-persist:].any()` with `persist=3` includes the base checkpoint on a 3-checkpoint fast-bench run, so `classify([1,0,0]) == classify([1,1,0]) == 'unstable'`. `DOMAIN_NAMES = ["code","biomedical","legal","math"]` is indexed by stream position, so legal in a code→legal stream is labelled "biomedical".
- **Evidence:** src/clharness/items.py:56; lines 31 and 120.
- **Failure scenario:** `clh items <hash16>` on a local fast-bench run reports 0 lost questions. findings.py, report.py and demo.py then say nothing was persistently lost. On other stream orders, losses are credited to the wrong domain.
- **Smallest fix:** `p = min(persist, len(v)-1)`. Read domain names from `config.resolved.json` `data.domains`.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no
- **Note:** this applies to local runs only. Modal runs fetched into `results/items` are not affected.

### RT-08: De-duplication runs before the backend filter, so a sim row can replace a real row
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `dedupe_rows` runs before the `REAL_BACKENDS` filter in `analyze()` and in replication.py:46-49. `_row_rank` ignores the backend, and `>=` keeps the later row.
- **Evidence:** src/clharness/analysis/attribution.py:781.
- **Failure scenario:** reproduced by appending local-backend copies of reference-s0 and calibration-code-legal-lr3e-5-s0. `analyze()` dropped both real rows. The rule-4 control changed from lr 3e-5 to lr 5e-5. The sanity check fell to 2 pairs, with an interval of -0.50 to +5.05 that no longer excludes 0. Promotion also lost its screen reference (see RT-01). In `replication_specs`, a shadowed real row is planned and paid for again.
- **Smallest fix:** filter by backend before `dedupe_rows`, or make `r.backend in REAL_BACKENDS` the first element of `_row_rank`.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no
- **Note:** this is reachable only if someone points SimExecutor at `results/agent`. Its default is its own output directory.

### RT-09: PREREG 8 validity rules are enforced nowhere
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `analyze()` keeps every row with `status == "finished"`. Nothing compares `cost.gradient_tokens` with `spec.gradient_tokens()`. `anomalous_grad_steps` is logged by the loop, but it is not a ResultRow field.
- **Evidence:** src/clharness/analysis/attribution.py:782; executor/rows.py:147-148; run.py:92.
- **Failure scenario:** a run that under-counts gradient tokens by 5% (PREREG A10 names this risk), or one with 3% anomalous steps, enters promotion and the paired tests. It is never marked invalid. Today's committed rows are 0.27% over, which is within tolerance.
- **Smallest fix:** treat a finished row as invalid when its tokens are off by more than 1%, and list it in `not_finished` with the reason. Add the anomalous-step fraction to the row.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no

### RT-10: Claim arms enter the pre-registered effects fit
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `design_rows` accepts any spec where `is_design_spec` is true. That includes the claim arms lora-r8, freeze-bottom and replay-5, because they use the default parameters. PREREG 7 limits the fit to the factorial, the halves and the replications.
- **Evidence:** src/clharness/analysis/attribution.py:790; design.py:130.
- **Failure scenario:** once the phase-1 claim arms land, 6 or more lora-alone and freeze-alone rows join the selection-half fit. The lora and freeze main effects then mostly measure those single-technique contrasts, with nothing flagging it. The fake rows already put lora-r8-s0 and freeze-bottom-s0 into the fit.
- **Smallest fix:** limit `design_rows` to technique sets in `factorial()` plus HALVES, or exclude `_claim_keys()`.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no

### RT-11: distill×lora is aliased with freeze×wavg but not labelled as aliased
- **Severity / verdict:** medium, confirmed
- **What is wrong:** the main-fit matrix has rank 9 of 10, so freeze×wavg is dropped. Each rw-chain pair is passed as its own singleton chain, and the twin detection uses `np.array_equal` on raw 0/1 vectors, which never matches. The kept distill×lora coefficient therefore shows `aliased_with=[]`. PREREG Amendment 3 says the halves make all three interactions estimable, but that is only true for replay×lr_rewarm.
- **Evidence:** src/clharness/analysis/attribution.py:275.
- **Failure scenario:** a true freeze×wavg effect is shown in `effects.svg` and the markdown as a clean distill×lora effect.
- **Smallest fix:** pass `[('distill','lora'),('freeze','wavg')]` as one chain. Correct the Amendment 3 wording.
- **Owner area:** analysis (himani)
- **Known from earlier review:** yes

### RT-12: Explore gate accepts freeze specs the bench cannot run, and their hypotheses stay 'running'
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `gate_one` checks each range alone (bottom_layers 0-25, top_layers 0-25) but does not call `catalog_real.combination_problems`, which holds the freeze-sum rule. The freeze plugin raises only in `configure_model`, inside the GPU container. `resolve_one` sees finished rows only.
- **Evidence:** src/clharness/agent/explore.py:372; freeze.py:50.
- **Failure scenario:** the model proposes freeze bottom 13 + top 13. Two H100 runs start, fail after the model loads, are billed and are never retried. The hypothesis stays "running" in every later cycle.
- **Smallest fix:** call `bench_problems` (or at least `combination_problems`) in `gate_one`. Mark a hypothesis inconclusive once its rows are final but not finished.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-13: Scout CLI crashes after paying for search and the model, and checks against the simulator catalog
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `if result.candidate:` refers to a field that does not exist. ScoutResult has `claims` and `candidates`, so the line raises AttributeError. Line 53 builds the request with `sim.default_catalog()`, uses no runnable check, the default seeds (0,) and an empty stream.
- **Evidence:** src/clharness/agent/scout.py:55; roles.py:188.
- **Failure scenario:** every live Scout run through this CLI, which SCOUT-HANDOFF.md:204 documents, makes three Exa searches and one model call, writes the file, then exits with a traceback. Cards that name real plugins are rejected as "unknown technique", and cards for fake techniques pass.
- **Smallest fix:** print `result.claims` and `result.candidates`. Use `real_catalog()`, `stream()` and `runnable=bench_problems`.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-14: veto.jsonl is never read; `autonomy: veto` does nothing
- **Severity / verdict:** medium, confirmed
- **What is wrong:** only `candidate_queue` reads veto.jsonl, and only for approve and reject. `halt`, `freeze_now`, `block_technique`, `block_spec` and `skip_scout` are never applied. The contracts.py:820 docstring says the Conductor reads the file at every transition.
- **Evidence:** src/clharness/agent/conductor.py:161.
- **Failure scenario:** a person appends `{kind: halt}` to stop a runaway loop, and it keeps launching.
- **Smallest fix:** read veto.jsonl at the top of `step()` and act on each kind.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** yes

### RT-15: No single-instance lock, and LoopState.resume has no caller
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `Conductor.__init__` loads state.json with no lock. There is no flock or lockfile anywhere in `agent/`. `LoopState.resume` (states.py:298) is never called.
- **Evidence:** src/clharness/agent/conductor.py:132.
- **Failure scenario:** two `loop --go` processes on one `--out` both launch runs and both write state.json. A halted loop cannot be resumed without editing the file by hand.
- **Smallest fix:** hold an flock on `<out>/.lock` for the life of the process. Add a `--resume` flag that calls `LoopState.resume`.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** yes

### RT-16: Infra rows are charged $0 in the Conductor's spend
- **Severity / verdict:** medium, confirmed
- **What is wrong:** `r.cost.usd if r.cost.usd > 0 or r.failure_kind == "infra" else self._estimate(r.spec)` charges an infra row $0 when its usd is 0. A machine lost mid-run was still billed.
- **Evidence:** src/clharness/agent/conductor.py:217 (lines 217-218).
- **Failure scenario:** repeated infra losses make the budget look unspent.
- **Smallest fix:** when usd is 0, charge infra rows the estimate, or measured wall time times the rate.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** yes

### RT-17: `<rows>.modal.json` is overwritten whole by each process
- **Severity / verdict:** medium, plausible
- **What is wrong:** ModalExecutor loads `_calls` and `launches` once in `__init__`, and `_save()` rewrites the whole file from memory. `submit()` checks only its own in-memory calls. The loop.py docstring suggests pointing `--rows` at the file that launch_design is filling. `append_jsonl`'s `_repair_tail` can truncate another writer's half-written line.
- **Evidence:** src/clharness/agent/modal_executor.py:184 (lines 110-114); contracts.py:960-972.
- **Failure scenario:** launch_design saves the call for spec X. The Conductor's executor then saves only spec Y, so X's call id is gone from disk. If the Conductor plans X, it spawns X again, which bills it twice. Both processes can append rows for one spec, and two infra rows count as 2 attempts.
- **Smallest fix:** re-read and merge the state file under fcntl.flock, and lock appends to rows.jsonl. Or document that two executors must never share one rows file.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no
- **Note:** this needs both processes running at once on overlapping specs. The skeptic could not confirm that the Conductor's spec hashes and design.screen's coincide.

---

## Low

### RT-18: `clh run --modal --detach` never writes rows to the local file
- **Severity / verdict:** low, confirmed
- **What is wrong:** rows are appended to `out` only inside the local entrypoint loop. That loop dies with the client. The remote side saves only `/vol/rows/<spec_hash>.json`.
- **Evidence:** src/clharness/executor/modal_app.py:217 (lines 217-228, 160); cli.py:491, 594.
- **Failure scenario:** 14 phase-0 specs are launched with `--detach` and the laptop is closed. The runs are billed (about $18), but `results/rows.phase0.jsonl` gets no rows. The runs look lost and are likely relaunched, which then triggers RT-06.
- **Smallest fix:** add `clh pull-rows`, which runs `modal volume get clh-work rows/` and appends the missing rows. Or drop `--detach` from `clh run`.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no
- **Note:** the data is not lost. remeasure reads `/vol/rows`, and `modal volume get` works.

### RT-19: fetch-items treats any single items file as a complete download
- **Severity / verdict:** low, confirmed
- **What is wrong:** the download is skipped (line 521) and accepted (line 526) as soon as any `ckpt_*/items_*.npy` exists.
- **Evidence:** src/clharness/cli.py:521.
- **Failure scenario:** an interrupted `modal volume get` leaves only one file. Later fetches skip the run, and remeasure or item analysis fails or reads partial data.
- **Smallest fix:** before skipping, require every task file at both the base and the final boundary checkpoint.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** yes

### RT-20: A domain's token budget can end early with no error
- **Severity / verdict:** low, confirmed
- **What is wrong:** `if step_tokens == 0: break` and the StopIteration path end the domain loop. `measure_run` never compares final gradient tokens with the spec's budget.
- **Evidence:** src/clharness/train/loop.py:316 (and line 302).
- **Failure scenario:** a future `select_batch` hook that returns None for a whole step stops a domain below its budget. The row is still "finished", so token-matched comparisons silently compare under-trained arms. No current hook triggers this.
- **Smallest fix:** mark the run invalid when per-domain gradient tokens fall short of the budget.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no

### RT-21: Eval items and model weights are not pinned in the run path
- **Severity / verdict:** low, confirmed
- **What is wrong:** `verify_eval_manifest` (the docs_sha256 check) runs only from `clh verify` (cli.py:100). model.py:63 loads `google/gemma-3-1b-pt` with no revision. Only casehold.yaml pins a dataset revision.
- **Evidence:** src/clharness/evals/lm_eval_runner.py:194.
- **Failure scenario:** an upstream revision of an HF eval dataset or of the model changes which documents the frozen indices select, or changes the base weights. Rows from before and after the change are then pooled as one frozen bench.
- **Smallest fix:** check docs_sha256 once per run in `EvalHook.on_run_start`. Pin a model `revision` in configs/model/gemma3-1b-pt.yaml.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no

### RT-22: With train.compile on, the distill teacher shares the student's forward, so KL is 0
- **Severity / verdict:** low, confirmed
- **What is wrong:** loop.py:157 sets `self.model.forward = torch.compile(...)` as an instance attribute. `copy.deepcopy` treats functions as atomic, so the teacher created at distill.py:150 keeps the same wrapper, still bound to the student.
- **Evidence:** src/clharness/techniques/distill.py:150; loop.py:157; distill.py:181. Reproduced with a compiled Linear: the deep copy returned the student's output.
- **Failure scenario:** any run with `train.compile: true` and distill gets KL of about 0 at every step. Distill becomes a silent no-op while `extra_forward_tokens` is still charged. Spec runs are not affected today, because every bench config has compile set to false.
- **Smallest fix:** add `teacher.__dict__.pop('forward', None)` after the deepcopy, or refuse distill when compile is on. Add a test.
- **Owner area:** techniques (peter)
- **Known from earlier review:** no

### RT-23: Return values of on_domain_end are dropped
- **Severity / verdict:** low, confirmed
- **What is wrong:** each hook iteration rebinds `logs := {}`, and the dict is never passed to `self.logger.log`. hooks.py:31 says hook return values are logged. on_domain_start's logs are logged at line 288.
- **Evidence:** src/clharness/train/loop.py:358.
- **Failure scenario:** a new technique that follows the documented contract logs nothing from on_domain_end. wavg works around this today through `_pending`.
- **Smallest fix:** build one dict across all hooks and log it before the boundary checkpoint.
- **Owner area:** techniques (peter); the change goes in the shared loop.py
- **Known from earlier review:** no

### RT-24: Margins and the rule-4 control are recomputed from seed 0 only
- **Severity / verdict:** low, confirmed
- **What is wrong:** `acquisition_margins(screen_base)` gets a single row, so it always returns ACQ_GATE, and the 5-seed noise term can never apply. `choose_lr_control` uses one seed per LR (-1.6 vs -1.9 points), while PREREG 0.4 froze L = 3e-5 from 3-seed means (-1.47 vs -2.43). A test locks this behaviour in.
- **Evidence:** src/clharness/analysis/attribution.py:801 (and line 132).
- **Failure scenario:** both methods agree today. With more noise, the code's control could differ from the frozen one, because the 0.3-point difference is below sigma_selection (0.59).
- **Smallest fix:** compute margins from the phase-0 seeds 0-4 and use the frozen L, or pin both as constants.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no
- **Note:** PREREG line 272 freezes both margins at 1.05, so the margin value the code uses is correct today.

### RT-25: PREREG 8 collapse sensitivity, majority rate and some secondary outcomes are missing
- **Severity / verdict:** low, confirmed
- **What is wrong:** `analyze()` records only `collapse_flags`. It does not recompute the outcome without the collapsed task and does not print the 0.58 majority rate. The PREREG 4 secondary outcomes (end-of-stream acquisition, general_ppl_ratio, persistent losses) are absent.
- **Evidence:** src/clharness/analysis/attribution.py:815.
- **Failure scenario:** a promoted configuration whose PubMedQA collapses to one answer gets no sensitivity row, although PREREG 8 requires one.
- **Smallest fix:** add the non-collapsed mean for flagged rows, print the majority rate, and add the secondary outcomes.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no

### RT-26: Cost chart uses GPU-hours, and its dot and bar come from different estimators
- **Severity / verdict:** low, confirmed
- **What is wrong:** x is `extra_gpu_hours`, which Amendment 4 replaced with forward tokens. y is an unpaired mean difference against all 5 no-mitigation seeds, while the bar is the paired vs_base interval.
- **Evidence:** src/clharness/analysis/figures.py:413; attribution.py:716.
- **Failure scenario:** for a seed-0-only configuration, the dot can sit outside its own 95% bar (by up to about 1.15 points).
- **Smallest fix:** plot `vs_base['delta']` against `extra_forward_tokens`.
- **Owner area:** analysis (himani)
- **Known from earlier review:** yes

### RT-27: to_markdown raises KeyError in confirmatory mode without no-mitigation rows
- **Severity / verdict:** low, confirmed
- **What is wrong:** `analyze()` writes `sanity_reference_vs_no_mitigation` only when both arms exist (line 873). `to_markdown` reads it unconditionally in confirmatory mode.
- **Evidence:** src/clharness/analysis/attribution.py:985.
- **Failure scenario:** reproduced on fake rows with no-mitigation removed. The JSON is written, then `to_markdown` crashes and `grid_analysis.md` is missing.
- **Smallest fix:** use `report.get(...)` and skip the line when the key is absent.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no

### RT-28: The explore cap applies per invocation, with no total cap
- **Severity / verdict:** low, confirmed
- **What is wrong:** `cycle()` starts `budget_left = cap_usd` ($30) on every call. `launch()` starts `spent_estimate = 0.0` on every call (launch_design.py:155).
- **Evidence:** src/clharness/agent/explore.py:483.
- **Failure scenario:** N runs of `explore --go` spend up to N × $30. None of it counts against the Conductor's cap or the $80 agreed in the config.
- **Smallest fix:** sum the spend already in `<out>/rows.jsonl` plus in-flight calls, and refuse when that total plus the new cost passes `cfg.compute_budget_usd`.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-29: Resumed launch_design/explore runs drop their launch decisions from the log
- **Severity / verdict:** low, confirmed
- **What is wrong:** the keys `{stage}/launch#{wave}` (where wave restarts at 0), `{stage}/plan` and `{stage}/done` repeat across runs. `DecisionLog.append` returns the existing entry when a key repeats (contracts.py:1162-1164).
- **Evidence:** src/clharness/agent/launch_design.py:181.
- **Failure scenario:** a rerun after an interruption launches new specs, which costs real money, but the audit log shows only the first run's plan, launches and cost estimate.
- **Smallest fix:** make the keys unique per invocation, for example by using the entry count.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-30: The Operator never re-runs a timed-out run
- **Severity / verdict:** low, confirmed
- **What is wrong:** `_settled` re-queues only `failure_kind == "infra"`. The gate's duplicate rule (gatekeeper.py:166-167), `ModalExecutor.submit` and the executor docstring all treat timeouts as retryable.
- **Evidence:** src/clharness/agent/operator.py:32.
- **Failure scenario:** the gate accepts a re-proposed timed-out spec and charges it as pending, but the Operator skips it. If every spec in a phase times out, the loop halts.
- **Smallest fix:** re-queue `failure_kind in ("infra", "timeout")` while attempts are below the maximum.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-31: The explore prompt does not escape web text inside `<source>` tags
- **Severity / verdict:** low, confirmed
- **What is wrong:** `json.dumps` does not escape `<` or `>`, so a snippet that contains `</source>` ends the quoted block early. The Conductor's path uses `wrap_untrusted` instead.
- **Evidence:** src/clharness/agent/explore.py:553.
- **Failure scenario:** a crawled page's text lands outside the quoted block and steers which in-range runs get bought. The code gates still limit what can be bought.
- **Smallest fix:** build the block with `context.wrap_untrusted(render_sources(found), 'search')`.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-32: The model budget is tracked per log file, so each entry point gets the full cap
- **Severity / verdict:** low, confirmed
- **What is wrong:** BudgetedLLM rebuilds its spend from its own `log_path` only. loop, explore and scout each use a different `llm_calls.jsonl`.
- **Evidence:** src/clharness/agent/llm.py:309.
- **Failure scenario:** with `model_budget_usd: 10`, the three entry points together can spend $30. A new `--out` directory resets the spend to zero.
- **Smallest fix:** use one shared ledger path (`cfg.path('llm_calls')`) for every entry point.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-33: The fallback planner replicates the top 2 treatments, not 3
- **Severity / verdict:** low, confirmed
- **What is wrong:** the code uses `StubPlanner(top_k=2)`, and `_top_arms` ranks arms by their best single row. PREREG 6.3 promotes the top 3.
- **Evidence:** src/clharness/agent/roles.py:383.
- **Failure scenario:** the deterministic path replicates fewer arms than PREREG asks for. The PREREG replication itself runs separately, so only the Conductor's fallback path is affected.
- **Smallest fix:** set `top_k=3` and rank by arm mean.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** yes

### RT-34: A rerun reuses the original item directory, so reproduced rows pair with the wrong item files
- **Severity / verdict:** low, confirmed
- **What is wrong:** `run_name` and `items_path` are both derived from the spec hash, so a rerun maps to the same paths. fetch-items skips the download when local files exist (cli.py:521). The rerun also deletes the volume directory and overwrites the volume row (see RT-06).
- **Evidence:** src/clharness/executor/run.py:110.
- **Failure scenario:** after `make reproduce-runs`, each row in `rows.reproduced.jsonl` points at the original run's committed item files. An items-mode bootstrap would then resample the old run's per-item results around the new run's delta. `clh remeasure` can no longer recover the original from the volume.
- **Smallest fix:** put an attempt id in the run directory and in `items_path`. Or refuse to delete a directory that holds a finished row without `--force`, and have fetch-items check a hash recorded in the row.
- **Owner area:** bench core and executor (justin); found by the security and reproducibility reviewer
- **Known from earlier review:** no
- **Note:** this happens only if someone runs grid-analysis on the reproduced rows, which no Makefile target does.

### RT-35: Seed-0 lower-LR control rows were produced from a dirty tree
- **Severity / verdict:** low, confirmed
- **What is wrong:** calibration-code-legal-lr5e-5-s0 and lr3e-5-s0 have `git_sha 424fd4a...-dirty`. Seeds 1 and 2 of the same arms were run at the clean commit 41fc2308. These seed-0 rows are the rule-4 screen-seed controls, and they are in REPORTED_ROWS.
- **Evidence:** results/rows.phase0.remeasured.jsonl:6; PREREG.md:87-89.
- **Failure scenario:** the code that produced these rows cannot be recovered. The uncommitted diff is unknown. Harm to the results is speculative.
- **Smallest fix:** rerun or remeasure the two rows from a clean commit, or record the hash of the dirty diff. Warn on `-dirty` rows in validate-row or reproduce-figures.
- **Owner area:** security, hygiene and reproducibility (results)
- **Known from earlier review:** no

### RT-36: Stale-deployment check ignores uv.lock and pyproject.toml
- **Severity / verdict:** low, confirmed
- **What is wrong:** `_local_code_version` runs `git status` on `src configs manifests` only, while the image is built with `uv_sync` from the working-tree uv.lock.
- **Evidence:** src/clharness/executor/modal_app.py:54.
- **Failure scenario:** a local, uncommitted dependency bump is deployed under a clean sha. The rows then claim an environment they did not run on.
- **Smallest fix:** add uv.lock and pyproject.toml to the paths that `git status` checks.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no

### RT-37: openai is not a declared dependency; OPENAI_API_KEY is missing from .env.example
- **Severity / verdict:** low, confirmed
- **What is wrong:** llm.py imports openai lazily. openai arrives only transitively through exa-py, and `make setup` installs only `--extra dev`. `.env.example` omits OPENAI_API_KEY, although it is the default key and scout.py's docstring says it is needed.
- **Evidence:** src/clharness/agent/llm.py:229.
- **Failure scenario:** a clean `make setup` followed by `--llm openai` raises ModelNotConfigured. Without a direct pin, the openai version can drift.
- **Smallest fix:** add an `llm` extra with a bounded openai pin, and add `OPENAI_API_KEY=` to .env.example.
- **Owner area:** agent system (brendan); found by the security and hygiene reviewer
- **Known from earlier review:** no
- **Note:** the error message says openai is not a dependency on purpose, so this is a setup and documentation gap.

### RT-38: README hard-codes a personal Modal profile name
- **Severity / verdict:** low, confirmed
- **What is wrong:** the README tells everyone to run `modal profile activate` with a specific profile name. That name is the local part of a committer's personal email. It was added in cb97e65.
- **Evidence:** README.md:95 and 110 (the skeptic saw lines 94 and 109).
- **Failure scenario:** other users get "profile not found", or activate the wrong workspace if a profile with that name exists in their config. It also puts a personal identifier in a public repo, although the full email is already in the git history.
- **Smallest fix:** replace the name with `uv run modal profile activate <your-profile>`.
- **Owner area:** security and repo hygiene
- **Known from earlier review:** no

### RT-39: README says the loop cannot do LoRA, loss terms or per-domain re-warm
- **Severity / verdict:** low, confirmed
- **What is wrong:** the README lists "Three things the loop cannot do yet": change the model before the optimizer is built, add a loss term, and re-warm the LR per domain. But `configure_model` and `extra_loss` exist (hooks.py:104-107), and lora.py, distill.py and lr_rewarm.py ship.
- **Evidence:** README.md:129 (the skeptic saw lines 121-126).
- **Failure scenario:** a judge concludes that the LoRA, distill and re-warm results cannot come from this harness. That hurts the interpretability score.
- **Smallest fix:** rewrite "Adding a technique" to describe `configure_model`, `extra_loss` and the per-domain schedule.
- **Owner area:** security and repo hygiene (docs)
- **Known from earlier review:** no

### RT-40: Local agent folders and CLAUDE.local.md are not gitignored
- **Severity / verdict:** low, confirmed
- **What is wrong:** .gitignore lists only `.claude/settings.local.json`. CLAUDE.local.md is ignored only through one clone's `.git/info/exclude`. `.worktrees/` and `.agents/` show as untracked.
- **Evidence:** .gitignore:37.
- **Failure scenario:** `git add -A` on a teammate's clone commits CLAUDE.local.md, which holds private details, into the public repo, or adds a nested worktree as a gitlink.
- **Smallest fix:** add `/.worktrees/`, `/.agents/` and `CLAUDE.local.md` to .gitignore.
- **Owner area:** security and repo hygiene
- **Known from earlier review:** no

### RT-41: A hung run may be billed up to 3 × 75 minutes if retries cover timeouts
- **Severity / verdict:** low, plausible
- **What is wrong:** `timeout=RUN_TIMEOUT_S` (75 minutes) is set together with `Retries(max_retries=2)`. The code comment says retries cover machine failures only, but a timeout cannot be caught inside `run_spec`. Whether Modal retries timeouts was not checked against its docs.
- **Evidence:** src/clharness/executor/modal_app.py:141 (lines 138-142).
- **Failure scenario:** a hang in lm-eval or the HF Hub costs about 225 H100-minutes (about $15) per spec, with no row written. A normal run takes 18-20 minutes.
- **Smallest fix:** check Modal's behaviour. Then set retries to 0, or lower the timeout to about 40 minutes and record timeouts as failed rows.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** no

### RT-42: Remeasured rows cannot be told apart from their originals
- **Severity / verdict:** low, plausible
- **What is wrong:** `remeasure_row` keeps spec_hash, git_sha and started_at. `rows.reference.jsonl` and `rows.reference.remeasured.jsonl` share all three, and neither has `remeasured_at`. Their numbers differ: rewarm-only-s0 selection is -0.046 vs -0.034, and report is -0.037 vs -0.049.
- **Evidence:** src/clharness/executor/run.py:149.
- **Failure scenario:** anyone who loads both files gets two finished rows for one hash and no way to tell which is current. The remeasured file predates the `remeasured_at` field (c41e103 vs 99a41c1). REPORTED_ROWS excludes the original file.
- **Smallest fix:** set git_sha to the measuring commit and keep the training sha in a new field. Prefer rows with `remeasured_at` in dedup. Backfill the field in the committed files.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** yes

### RT-43: NaN items are dropped silently but n_items reports the full half
- **Severity / verdict:** low, plausible
- **What is wrong:** the code uses `np.nanmean` with `n_items=len(idx)`. lm_eval_runner.py:217 fills missing docs with NaN. The invalid check catches only an all-NaN mean.
- **Evidence:** src/clharness/executor/rows.py:62 (and line 147).
- **Failure scenario:** if some docs drop at one checkpoint, base and final are averaged over different items. The row is still "finished" with the full count. With explicit `samples=` indices this is not a realistic path today.
- **Smallest fix:** mark the row invalid on any NaN, or mask the same positions in both and report the true count.
- **Owner area:** bench core and executor (justin)
- **Known from earlier review:** yes

### RT-44: 0/1 coding lets omitted interactions bias the kept effects
- **Severity / verdict:** low, plausible
- **What is wrong:** `fit_effects` uses 0/1 product columns on an unbalanced design, so omitted 2-factor interactions are not orthogonal to the kept columns. A planted unit distill×freeze interaction moved lora by -0.19, distill×lora by +0.27 and lr_rewarm×replay by -0.16.
- **Evidence:** src/clharness/analysis/attribution.py:259.
- **Failure scenario:** the reported replay×lr_rewarm and the lora and wavg main effects carry bias from other chains, and they are shown as clean estimates.
- **Smallest fix:** use ±1 coding, or fit on the balanced factorial plus a separate 2×2 contrast. At least document that the coefficients are average effects.
- **Owner area:** analysis (himani)
- **Known from earlier review:** no
- **Note:** PREREG 7 prescribes 0/1 indicators for this exploratory fit, so this is a methodology caveat, not a code defect.

### RT-45: Confirmatory interval is a percentile bootstrap over 3 seeds (liberal)
- **Severity / verdict:** low, plausible
- **What is wrong:** `paired_bootstrap` resamples k=3 seeds plus items and takes percentiles. Amendment 2 fixed this only for the claims acquisition tests. On committed rows, the items-mode sanity interval is +0.73 to +4.73 points, against the parametric -0.14 to +5.64 in PREREG 0.4. The skeptic did not re-check these numbers.
- **Evidence:** src/clharness/analysis/attribution.py:359.
- **Failure scenario:** between-seed variance is under-estimated, so rule 1 can pass in noise-only cases more often than the nominal 5%.
- **Smallest fix:** use a t-based seed-level interval (df = n-1), or record an amendment that acknowledges the coverage.
- **Owner area:** analysis (himani)
- **Known from earlier review:** yes
- **Note:** PREREG 7 prescribes the percentile interval, so this is a design choice, not an implementation bug.

### RT-46: Two different "exploratory" markers; a claim card can be paid for twice
- **Severity / verdict:** low, plausible
- **What is wrong:** explore.py uses `notes="exploratory: ..."`, but `is_explore` checks for `"explore:"`. `spec_hash` excludes notes. The Conductor writes plannable cards to the explore inbox and also plans them in combine at seeds (0,1), which are the same as EXPLORE_SEEDS.
- **Evidence:** src/clharness/agent/explore.py:56.
- **Failure scenario:** (a) explore rows that reach a file the Conductor reads are treated as confirmatory. (b) If explore runs a card first, the Conductor runs the same 4 runs again. The skeptic notes that (a) needs a non-default `--out`, and (b) was not fully traced.
- **Smallest fix:** use one marker, and give claim cards one owner.
- **Owner area:** agent system (brendan)
- **Known from earlier review:** no

### RT-47: Source links put web URLs into href without a scheme check
- **Severity / verdict:** low, plausible
- **What is wrong:** `<a href="{escape(s['url'])}">` stops attribute breakout, but it still allows `javascript:` and `data:` URLs.
- **Evidence:** src/clharness/page/render.py:201.
- **Failure scenario:** a search result with a `javascript:` URL runs script on the local results page when someone clicks it. This is unlikely.
- **Smallest fix:** link only `http://` and `https://` URLs, and show anything else as plain text.
- **Owner area:** agent system and page (brendan)
- **Known from earlier review:** no

---

## Checked and found correct

**Bench core and executor (justin)**
- `measure_run` refuses incomplete runs: the boundary domains must equal the stream, and the last checkpoint must be a boundary (rows.py:94-96). The earlier "last checkpoint used as final" issue is fixed.
- `config_from_spec` raises when a technique override would replace an optim key the spec set (run.py:62-65). The earlier silent-overwrite issue is fixed.
- Task files outside the protected set go to report_only and are kept out of the halves and mean_delta. The earlier "every items file is a task" issue is mostly fixed.
- The pinned split manifest has disjoint selection and report positions covering 0..499 for all 4 tasks. The extra manifest matches it on indices and doc hashes.
- Per-item vectors are written in manifest order, which matches the split positions.
- The heldout, probe and stream splits are disjoint. Sources are pinned by revision, and their hashes are checked before training.
- Every technique gets the same token budget (153 steps at 65,536 tokens). The over-budget pre-check runs before the base eval.
- The metric keys `measure_run` reads match what EvalHook writes.
- NaN rows survive a write and read-back unchanged.
- No two committed spec files share a spec hash.
- The stale-deployment check builds the image and the sha from the same tree, and the agent refuses dirty trees.
- The executor fast tests pass (68).

**Techniques (peter)**
- Gradient-token matching holds. Replay swaps rows in place. Extra forward passes are counted separately in `extra_forward_tokens`.
- The replay share uses a running total. Replay is off while the source domain trains. `chunk_ids` and `replay_mask` are set on every batch. Replay seeding is distinct from the training streams and is deterministic.
- act_nov statistics are updated only from non-replayed rows after their forward pass, with a random fallback until the statistics are ready.
- Hooks run in name order, so freeze configures before lora. The distill teacher is copied after LoRA injection, while B=0.
- The distill coverage scaling and the top-k bucket KL are correct.
- wavg averages only trainable float parameters and handles LoRA scaling. It averages before the boundary checkpoint.
- lr_rewarm's overrides and per-domain schedule are correct. Conflicting overrides are detected.
- No state leaks between runs.
- Freeze and LoRA enable input gradients under gradient checkpointing.
- The technique fast tests pass (101).

**Analysis and pre-registration (himani)**
- Holm-Bonferroni is correct, and the families are separated as PREREG 7 says.
- `decide()` rules 1-4 match PREREG 5, including Amendment 5.
- Promotion and replication use only seed-0 rows and the selection half. The report half is read only in confirmatory mode.
- `spec_key` is consistent with `spec_hash`. The LR controls pool correctly.
- The factorial is resolution IV, with F01 = no-mitigation and F04 = reference. The alias chains match PREREG 6.2.
- The item-level paired bootstrap pairs indices correctly, and item mode runs on committed rows. The parametric fallback and `paired_seed_t` are correct.
- SIGMA_NOISE = 0.0115 reproduces from the committed rows. The PREREG 0.4 table matches.
- The claims table and verdict logic match PREREG 6.5. Units are consistent.

**Agent system and page (brendan)**
- The selection/report firewall types hold. FreezeToken cannot be forged, and only frozen hashes are opened.
- The page keeps report values sealed and escapes untrusted text. The server binds 127.0.0.1.
- The Exa key is never exposed to the model or logged. Sources enter Conductor prompts wrapped and neutralised.
- `check_claim` refuses bad sources, invalid arms, out-of-range LRs and duplicates.
- The Gatekeeper's explore rule is correct, and the explore budget accounting adds up.
- FREEZE and `_valid_treatments` skip `explore:` rows.
- The decision log is idempotent and tamper-checked. BudgetedLLM checks worst-case cost before each call.
- The duplicate rule compares numbers by value. The Conductor's transitions are all in TRANSITIONS.

**Security, hygiene, tests and reproducibility**
- No secrets in git history (all refs scanned). `.env.example` holds placeholders only. Keys are read from the environment and never written to rows or logs. The Modal image uploads only src/, configs/ and manifests/.
- No personal filesystem paths in tracked results, configs, manifests or docs.
- `uv lock --check` passes, and the key dependencies have bounded pins.
- `make test`: 457 passed, 2 skipped (expected), 1 deselected. No weakened tests found.
- reproduce-figures from the repo root reproduces `grid_analysis.json` to within float rounding. The five figures are byte-identical.
- The page renders without `results/agent`.
- Simulated backends are excluded unless `allow_simulated` is set.
- Every REPORTED_ROWS row has its item files committed.
- MBPP code execution is documented.

## Dropped during verification

- Bench: "spec_hash is not canonical (calibration lr1e-4 is the same experiment as no-mitigation-s0)". Refuted: the separate hash is documented and deliberate in PREREG section 0.
- Techniques: "Some invalid technique params pass pre-launch validation and fail only on the GPU". Refuted: the gatekeeper's catalog rule checks the real ranges (contracts.py:586).
- Techniques: "No-op parameter settings are accepted as distinct arms". Refuted: most of the examples are blocked by the real catalog ranges, which both the gatekeeper and the explore loop enforce.
