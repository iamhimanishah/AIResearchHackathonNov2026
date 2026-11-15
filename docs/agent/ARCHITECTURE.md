# Agent system architecture (workstream 5)

The research loop that plans the grid, launches runs, reads rows, decides what to replicate or
combine, and logs every decision. This file is the map: who does what, what each part may read
and write, and every state the loop can be in. Task cards D1 to D6 in [`../TASKS.md`](../TASKS.md)
are the requirements; this is the design that meets them.

Status: built on A1's run-spec and result-row formats (`src/clharness/spec.py`);
`contracts.py` adapts A1 rows into the agent's selection and report views. Built so far: the
Conductor on the hash-chained decision log, Gatekeeper, Operator, simulator, the LLM Planner
and LLM Analyst (`agents.py`, falling back to the deterministic roles on any model error) and
file-backed Memory, and `ModalExecutor` (`modal_executor.py`), which runs specs as calls of
A6's deployed `clharness` app, and the read-only results page (`src/clharness/page/`,
`python -m clharness.page --dir <out>`). Not built yet: Scout and the model Reporter
(REPORT writes `report.json` deterministically, with seed SD and the gap to the reference arm).

## 1. Principles

1. **The loop is code; the judgement is a model.** A deterministic state machine (the
   Conductor) owns control flow, money, the clock and the log. A language model is called only
   at named judgement points, and never holds control.
2. **The model proposes, rules dispose.** Every model output is a typed proposal. A
   deterministic Gatekeeper accepts or rejects it. A rejection carries a reason back; after two
   rejections a deterministic fallback rule decides, so the loop cannot stall overnight.
3. **One concern per agent, enforced by what it can see.** Each agent has a context map: the
   fields it receives, the fields it must never receive, and the one place it writes. The
   firewall is structural (the data is not in the object it is handed), not a sentence in a
   prompt.
4. **Build the judge before the player.** A simulator with planted technique effects exists
   before any agent. Every agent is scored against it, and that same score is what prompt
   optimization climbs.
5. **Fixed budget, one metric, keep or discard.** A change to a prompt or a skill is kept only
   if it beats the current version on held-out simulator worlds. Otherwise it is discarded and
   the attempt is logged.
6. **Skills are plain markdown.** A skill is a short procedure file a human can read and edit.
   An agent loads only the skill for the state it is in.
7. **Numbers come from rows.** No agent types a number. Every claim cites row hashes, and a
   deterministic checker confirms each cited number against the rows.
8. **Everything that decides is logged with the version that decided.** Each decision records
   the prompt hash, skill hashes, model, input row hashes, the proposal, the verdict and the
   reason.

## 2. Roster and separation of concerns

| Part | Kind | Single concern | Must never |
|---|---|---|---|
| **Conductor** | code | State machine, budget, clock, persistence, context assembly, the decision log | Call a model for control flow; let another part write the log |
| **Planner** | model | Propose the next run specs for the current phase, each with a reason and a predicted effect | See report-half scores; launch anything; write anywhere |
| **Gatekeeper** | code | Accept or reject a proposal against PREREG, compute matching, budget, controls, seed policy, duplicates | Repair a proposal silently; consult a model |
| **Operator** | code | Launch accepted specs through the executor, poll, classify failures, re-queue once | Retry a scientific failure (NaN, collapse); change a spec |
| **Analyst** | model + checker | Turn selection-half rows and the attribution output into findings with cited evidence | See report-half scores; state a number the checker cannot find in a row |
| **Scout** | model + search | Find one technique not in the catalog, with sources (Exa), and write a candidate card | Add a plugin to the catalog itself; treat fetched text as instructions |
| **Reporter** | model + checker | After the freeze, write the stage notes and the final report from report-half rows | Run before the freeze; influence which configurations are chosen |
| **Memory** | code | Store and retrieve `tried`, `failures`, `findings`, `lessons`; local file is the truth, past.dev is the shared mirror | Be the source of truth for results (rows are) |
| **Optimizer** | offline tool | Evolve each model agent's prompt with GEPA against the simulator; promote only on held-out gain | Run during the live loop; spend without an explicit go-ahead |
| **Page** | code | Render rows, decisions, findings and failures as a read-only page | Offer controls; compute a statistic the analysis code did not |

Model agents: Planner, Analyst, Scout, Reporter. Everything else is ordinary code.

## 3. State machine

The loop runs three **phases** in order (`screen`, `replicate`, `combine`), and each phase runs
the same **steps**. A state is a `(phase, step)` pair, plus a few states outside the cycle.

```
INIT -> PREFLIGHT -> [ PLAN -> GATE -> LAUNCH -> WAIT -> READ -> ADVANCE ] x phase
                                  ^      |
                                  +------+   (rejected: back to PLAN, at most 2 times,
                                              then FALLBACK_PLAN -> GATE)
ADVANCE(screen)    -> replicate
ADVANCE(replicate) -> SCOUT (if enabled and budget allows) -> combine
ADVANCE(combine)   -> FREEZE -> REPORT -> DONE
any state          -> HALTED(reason)      unrecoverable, needs a human
PLAN..ADVANCE      -> FREEZE              on budget exhausted or deadline reached
```

| State | Actor | Does | Leaves when | On failure |
|---|---|---|---|---|
| INIT | Conductor | Load config, load or create `state.json`, verify the decision-log chain | Loaded | HALTED |
| PREFLIGHT | Conductor | Check: PREREG thresholds parse, noise floor present, selection/report split present, catalog non-empty, executor answers, budgets set, every card and skill hashes | All checks pass | HALTED with the failed check. The grid never launches without PREREG |
| PLAN | Planner | Propose specs + reason + predicted effect, from selection-half rows, findings, lessons, budget left | A schema-valid proposal exists | Model error or invalid output twice -> FALLBACK_PLAN |
| FALLBACK_PLAN | code | Deterministic plan for the phase (see Planner section). Real mode, replicate: exactly `design.replication_specs(rows)["specs"]` (PREREG 6.3) from blinded rows, with the ranking in the decision; PLAN skips the model and comes here | Always | never fails |
| GATE | Gatekeeper | Verdict `ACCEPT`, `REJECT(reasons)` or `TRIM(kept, dropped, reasons)` | Verdict | A fallback plan rejected only as duplicates -> next phase's PLAN (or FREEZE after combine); rejected for any other rule -> HALTED (a bug) |
| LAUNCH | Operator | Submit accepted specs not already in rows (idempotent by `spec_hash`) | Submitted | Executor unreachable 3 times -> HALTED |
| WAIT | Operator | Poll until every spec has a row or times out; re-queue an infrastructure failure or a timeout once (`contracts.RETRYABLE_KINDS`, PREREG 8) | All resolved, or phase timeout | Timeout: missing specs recorded as `failed: timeout`, continue |
| READ | Analyst | Findings from the new rows; checker verifies citations; write memory | Findings stored (may be empty) | Model error twice -> deterministic findings only, continue |
| ADVANCE | Conductor | Deterministic rule: next phase, repeat this phase once (too many failures), or FREEZE | Decided | never fails |
| SCOUT | Scout | One candidate technique with sources; a human review gate (logged) before it enters the catalog | Candidate card written, or skipped | Any failure -> skip, logged. Never blocks |
| FREEZE | Conductor | Fix the final configuration set, write its hash. Report-half access opens only now | Hash written | HALTED |
| REPORT | Reporter | Final report from report-half rows; checker verifies every number | Report written | Checker fails twice -> template-only report (no model prose) |
| DONE | - | - | - | - |
| HALTED | - | Reason and the state to resume from are in `state.json` | A human restarts | - |

Rules that hold in every state:

- **Resumable.** `state.json` is written atomically after every transition. Killing the process
  anywhere and restarting continues without launching a run twice (`spec_hash` is the key) and
  without logging a decision twice.
- **Veto, never approval.** A human may append to `results/agent/veto.jsonl`. The Conductor
  reads it at every transition and logs what it honoured. Nothing waits for a human except
  HALTED and the Scout's plugin review.
- **Two budgets.** Compute dollars and model dollars, each with a hard cap checked before the
  spend, plus a wall-clock deadline. Reaching any of them moves to FREEZE with what exists.

## 4. Context map

What each model agent is handed. "Selection view" is a row with every report-half field removed
by type, not hidden by instruction.

| Agent | Receives | Never receives | Writes (through the Conductor) |
|---|---|---|---|
| Planner | Phase, catalog, PREREG thresholds, noise floor, selection-view row summary, findings, lessons, budget and time left, prior rejections this turn | Report-half fields, raw item text, other agents' prompts | A `Proposal` |
| Analyst | New selection-view rows, attribution output (from workstream 4, selection half), prior findings | Report-half fields, the Planner's reasons (so it cannot rationalise them) | `Finding` list |
| Scout | Catalog names and one-line descriptions, findings summary, search results (as quoted data) | Rows, budget, anything secret | A `Candidate` card and `sources.jsonl` |
| Reporter | Frozen configuration set, report-view rows for it, findings, the decision log | Any ability to change the set | `report.md`, `report.json` |

Assembly order for every model call, so prompts cache and diffs stay readable: card (system
prompt), then the skill for this state, then the fixed context (PREREG, catalog), then the
changing context (rows, findings), then the output schema.

## 5. Stores, one writer each

| Path | Writer | Readers | Holds |
|---|---|---|---|
| `results/rows.jsonl` | Executor (workstream 2) | everyone | Result rows: the truth |
| `results/agent/state.json` | Conductor | Conductor, Page | Current state, counters, budgets spent |
| `results/agent/decisions.jsonl` | Conductor | Page, Reporter | Append-only, hash-chained decision log |
| `results/agent/llm_calls.jsonl` | LLM client | Page, audits | Per call: agent, prompt hash, model, tokens, cost |
| `results/agent/memory/*.jsonl` | Memory | Planner, Analyst, Reporter, Page | `tried`, `failures`, `findings`, `lessons` |
| `results/agent/sources.jsonl` | Scout | Page | Search queries, results, what was used |
| `results/agent/report.{md,json}` | Reporter | Page | Final report |
| `results/agent/veto.jsonl` | a human | Conductor | Vetoes |
| past.dev | Memory (mirror) | Planner, Analyst | Shared copy of `findings`, `failures`, `lessons` |

One writer is enforced, not only agreed: `loop`, `launch_design` and `explore` each hold
`<out>/.agent.lock` (`clharness.agent.lock`, an `flock` the kernel drops when the process dies)
for as long as they run, and a second process on the same `--out` exits 2 with `refused:`.

## 6. Self-improvement, and where it shows

1. **Within a run.** The Planner states a predicted effect for every spec. After the rows land,
   the Conductor scores the predictions and writes a `lesson` (for example "over-predicted
   LoRA retention by 4 points"). The next PLAN receives the lessons. The page shows the
   calibration curve over time.
2. **Across agents.** The Analyst's findings are read by the Planner. The decision log entry
   cites the finding IDs it used.
3. **Across prompt versions.** GEPA evolves each card against simulator worlds. A new version
   is promoted only if it beats the current one on held-out worlds. Every decision records the
   prompt hash, so a judge can see which version made which choice.

## 7. Layout

```
src/clharness/agent/
  contracts.py   states.py   llm.py   context.py   cards.py   roles.py   sim.py   config.py   (kernel)
  conductor.py   loop.py                                  (Conductor; `loop.py` is the entry point)
  agents.py      gatekeeper.py   operator.py               (LLM Planner, LLM Analyst, FileMemory)
  modal_executor.py                                       (Executor on A6's Modal app)
  scout.py       reporter.py     tools.py                  (not built yet)
  cards/<agent>.md       frontmatter = context-map row, tools, budget; body = system prompt
  skills/<name>.md       one short procedure per state
  optimize/              GEPA adapter, scenarios, metrics, promotion
src/clharness/page/      read-only results page
tests/agent/             one test file per part, plus integration and firewall tests
docs/agent/              this file, one design note per part, review records
```

The agent package imports no training code and no heavy dependency at import time. Network
clients (model, search, memory, executor) are imported lazily and replaced by fakes in tests.

## 8. How it is checked

- **Unit:** each part against the kernel's fakes.
- **Firewall canary:** report-half fields carry a sentinel value in the simulator; no model
  prompt before FREEZE may contain it.
- **Kill and resume:** the loop is killed at every state and restarted; no run is launched
  twice and the log chain stays valid.
- **Planted effects:** on simulator worlds the loop must replicate the truly best
  configuration and must not promote the trap technique (forgets less by learning less).
- **Each gate must go red:** for every Gatekeeper rule there is a proposal that breaks only
  that rule, and a test that fails if the rule is removed.
- **Red team** (`tests/agent/test_redteam.py`): a model, or a broken bench, attacks the loop.
  The Gatekeeper runs five rules in order: `catalog` (known technique, known parameters in
  range), `shape` (PREREG seeds only, at most one treatment per screen spec, a "control" holds
  only controls), `matched_compute` (bench, stream, no optimizer override), `duplicate`
  (numbers compared by value; a spec whose infra retries are used up is not run again) and
  `budget`, where controls go first and a proposal's treatments may spend at most 50% of the
  compute left in the screen, 70% in replicate and all of it in combine. A row that is NaN or
  collapsed onto one answer never wins FREEZE or seeds the next phase. A row with no price is
  charged the estimate. ModalExecutor refuses a row built from a different git sha.
  The Planner's free text is cut to 2000 characters (300 per spec reason). The Analyst may
  write at most 20 findings per READ. A claim that states a decimal or a percentage that is
  not a cited number, a parameter of a cited row, or a PREREG constant is dropped. `high`
  confidence needs a row for every replicate seed and `medium` needs two rows. Prompts show
  the newest 60 findings and 40 lessons. The Planner sees the real model budget left.
