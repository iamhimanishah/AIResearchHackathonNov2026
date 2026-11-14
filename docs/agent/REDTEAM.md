# Red-team pass on the agent harness (patch 0006)

Method: a model Planner or Analyst, and a broken bench, were treated as adversaries. Each attack
is a test in `tests/agent/test_redteam.py`. 24 attacks; 19 broke the harness before the fix,
5 were already held. After patch 0006 all 24 pass; `make test`: 228 passed, 3 skipped.

## Confirmed breaks (fixed)

| # | Attack | What went wrong | Fix |
|---|--------|-----------------|-----|
| 1 | Planner proposes an unknown technique, an unknown parameter, or `replay.rate=5.0` | Gate accepted it; the run would launch | New `catalog` rule (uses `Catalog.problems`) |
| 2 | Planner sets `bench="full"` | Gate accepted; compute no longer matched | `matched_compute` checks the bench |
| 3 | Planner picks seeds 7, 8, 9 (seed shopping) | Accepted in screen and replicate | New `shape` rule: PREREG seeds of the phase only |
| 4 | Treatment labelled `role="control"` | Accepted and jumped the budget queue | `shape`: a control holds only control techniques |
| 5 | Two treatments in one screen spec | Accepted; the screen stops attributing effects | `shape`: one treatment per screen spec |
| 6 | `rank=16` re-proposed as `rank=16.0` | Different hash, so the same config ran twice | `duplicate` compares numbers by value |
| 7 | Re-propose a spec whose infra retries are used up | Re-launched forever (one per plan) | `duplicate` drops specs with exhausted retries |
| 8 | Screen proposal of 54 cheap treatments | Spent the whole budget in the screen | `budget`: treatments get 50% (screen), 70% (replicate) of what is left |
| 9 | 5000 findings in Memory | Planner prompt grew without bound (cost and context overflow) | Prompts show the newest 60 findings, 40 lessons |
| 10 | Planner budget view | `model_usd_left` was always 0.0 | Reads `BudgetedLLM.remaining_usd` |
| 11 | 200 kB rationale | Copied into the decision log (405 kB after 3 steps) | Rationale cut to 2000 chars, reasons to 300 |
| 12 | Analyst cites one real number, then claims "cuts forgetting by 0.42" / "85%" | Claim stored as a finding the Planner later trusts | Decimals and percentages in a claim must match a cited value, a cited row's parameter, or a PREREG constant |
| 13 | Analyst calls one seed `high` confidence | Stored as high | `high` needs a row per replicate seed, `medium` two rows |
| 14 | Analyst returns 500 findings | All stored | At most 20 per READ |
| 15 | A replicated arm that answers one letter to 99% of items | Won FREEZE | `contracts.usable`: collapsed or NaN rows never rank (FREEZE and fallback) |
| 16 | Bench row with `est_usd = null` | Counted as $0; budget looked unspent | Charged the executor estimate |
| 17 | Fallback plan finds everything already run in screen | Jumped straight to FREEZE, skipping replicate and combine | Moves to the next phase's PLAN (new GATE edge in `states.py`) |
| 18 | Modal returns a row built from another git sha | Accepted as a result | Recorded as an infra failure |
| 19 | Modal cost estimate | Same for every technique (distill costs more) | Uses catalog cost multipliers |

## Attacks that already held

- A NaN retention or NaN acquisition row did not win FREEZE (held by luck of ordering; now
  explicit via `usable`).
- A Planner that only proposes runs that fail could not halt the loop.
- Failure text from a row that tries to close the untrusted-data block and inject `SYSTEM:`
  lines is neutralised.
- Proposals round-trip through JSON unchanged.
- From earlier patches, still covered: report-half canary never reaches a prompt before
  FREEZE, edited or truncated decision log is refused on restart, FreezeToken cannot be minted
  without a logged FREEZE, page escapes file text and refuses POST.

## Open (not fixed, need a decision or another team)

- A1 has no failure-kind field; retryable failures are recognised by the `infrastructure:` /
  `timeout:` prefix of the reason. A6 must keep writing those prefixes.
- The `shape` rule assumes screen specs test one technique; if B-team wants screen combos,
  this rule needs a flag.
- The 50% / 70% budget shares are my choice; change `PHASE_SHARE` in `gatekeeper.py` if you
  want different numbers.
- No real OpenAI or Modal run has been done; model behaviour on real prompts is still untested.
