# Pre-registration: forgetting-reduction study

**Status: DRAFT.** It becomes binding when it is merged to `main` before the screening grid
launches (sync point 2). The agent records the commit SHA of this file in its first decision-log
entry. After that, changes follow the amendment protocol in section 11.

- **Owner:** Member C (workstream 4).
- **Task card:** C1 in [`docs/TASKS.md`](../docs/TASKS.md).
- **Decisions:** each one is marked **decided** or **provisional**. A provisional item has a
  default that applies unless someone changes it before the freeze. The full record of options
  considered is in [Appendix A](#appendix-a-decision-record).

Units follow `src/clharness/spec.py`: scores and deltas are fractions (a 5-point drop is
−0.05), and perplexity ratios are trained / base.

---

## 1. Question

At matched gradient tokens, which forgetting-reduction techniques, alone or in combination,
retain more of Gemma 3 1B's protected capabilities than the reference arm (replay plus
per-domain LR re-warm, Ibrahim et al. 2024)? And do they do it without learning the new domains
less?

**Confirmatory claims** are tested under the rules below. They are limited to:
1. The three configurations promoted from screening (section 6.3), each against the reference
   arm.
2. The reference arm against no-mitigation. This is a sanity check that the reference arm
   works on this bench.

Everything else is **exploratory**: the screening effects, the interactions, the second task
sequence and the per-task breakdowns. These are reported with intervals, but they are not
claimed as wins.

## 2. Bench and arms

- **Bench:** the frozen fast bench (`configs/bench/fast.yaml`, task A5):
  - code then biomedical, about 10M gradient tokens per domain, LR 1e-4;
  - the bench config's SHA is recorded in every result row.

  If A5 freezes different numbers, this section is updated before the freeze. That is not an
  amendment.
- **No-mitigation:** all techniques off.
- **Reference arm:** `replay` (5% of general text) plus `lr_rewarm` (per-domain warmup and
  decay). Its parameters are fixed by B1 before the freeze. They are not tuned on the report
  half.
- **Techniques screened** (factor names match the result-row technique names): `replay`,
  `lr_rewarm`, `lora`, `freeze`, `distill`, `wavg` and `select`. Each one's parameters are
  fixed at one level for the screen, recorded in `TECHNIQUES.md` by B, and frozen with this
  file.

## 3. Item split (decided)

- Each multiple-choice task's 500 protected items are split 250/250 into a **selection half**
  and a **report half**.
- **Method:** stratified by whether the base model answers the item correctly.
  - Source: checkpoint 0 of `pilot-chosen-200M`.
  - Within each stratum, items are shuffled with seed `20261114` and dealt alternately to the
    two halves.
- **Output:** [`manifests/eval-protected-v1-split.json`](../manifests/eval-protected-v1-split.json),
  built by `scripts/make_item_split.py`. Check it with `--check`. The frozen
  `eval-protected-v1.json` is not modified.
- The executor reads this file when it computes the two halves of a result row
  (`half_indices` in `src/clharness/executor/rows.py`). `tests/test_item_split.py` checks that
  it does.

Base accuracy per half:

| Task | Selection | Report |
|---|---|---|
| ARC-Easy | 0.712 | 0.712 |
| HellaSwag | 0.636 | 0.636 |
| PIQA | 0.736 | 0.736 |
| PubMedQA | 0.648 | 0.644 |

**Who sees which half:**
- **Selection half:** used for every decision. That covers the screening ranking, promotion
  to replication, hyperparameter choices and the agent's choices.
- **Report half:** used only for the numbers we report. The agent's validator rejects any
  plan whose inputs include report-half fields. The report half is read for the confirmatory
  tests only after all replication runs are finished (target: Sunday about 10:00).

## 4. Outcomes

| Role | Outcome | Row field | Direction |
|---|---|---|---|
| **Primary** (decided) | End-of-stream mean change across ARC-Easy, HellaSwag, PIQA and PubMedQA, on the report half | `retention.report.mean_delta` | higher is better |
| **Co-primary gate** (decided) | Acquisition, measured at each domain's own boundary: the geometric mean over trained domains of held-out perplexity right after that domain's training, divided by base. Rows written before this field existed fall back to `mean_ppl_ratio`, and the analysis reports how many rows did | `acquisition.mean_own_boundary_ppl_ratio` | lower is better |
| Secondary | Acquisition kept at the end of the stream: perplexity on the trained domains at the end, divided by base. This mixes learning with forgetting | `acquisition.mean_ppl_ratio` | lower |
| Secondary | Per-task end-of-stream change, report half | `retention.report.per_task[*].delta` | higher |
| Secondary | MBPP pass@1 change at the end of the stream (provisional: secondary only) | per-item file | higher |
| Secondary | Persistent losses: items right at base and wrong at each of the last 3 checkpoints | per-item file | lower |
| Secondary | General-web perplexity ratio (untrained slice) | `acquisition.general_ppl_ratio` | lower |
| Secondary | Cost: gradient tokens, extra forward tokens, wall-clock, update bytes, est. USD | `cost.*` | report |
| Diagnostic | Answer share per task | `answer_share` | flag above 0.90 |

## 5. Decision rule: "beats the reference arm" (decided)

A configuration X beats the reference arm R only if **all three** of these hold, on the report
half, over the paired replication seeds:

1. **Significant:** the 95% paired bootstrap interval (section 7) for
   Δ = primary(X) − primary(R) excludes 0, after Holm correction across the confirmatory set.
2. **Large enough to matter:** the point estimate Δ is above **2 × σ_noise**. σ_noise is the SD
   of the primary outcome across the 5 no-mitigation noise-floor seeds (section 6.1).
   **σ_noise = TBD** (filled in from C3 before the freeze).
3. **Learns enough:** acquisition(X) ≤ **1.05 × acquisition(R)**, using the own-boundary
   measure from section 4.

The retention–acquisition **Pareto frontier** is always shown next to these results. A
configuration that is on the frontier but fails rule 3 is reported as a trade-off, not a win.

If no configuration meets all three, we report the one closest to doing so, with the reason it
fell short. That is a pre-registered outcome, not a failure.

## 6. Design

### 6.1 Noise floor (decided)
- No-mitigation at **5 seeds (0–4)**. The seed changes only the data order: the stream seed is
  `train.seed + domain_index`, and Gemma has no dropout in this setting.
- Specs: `configs/bench/specs/no-mitigation-s0.json` to `-s4.json`. They are identical except
  for `seed` (code then biomedical, 10M tokens each, no techniques).
- σ_noise = the SD of `retention.report.mean_delta` over these 5 runs. The selection-half SD is
  recorded too.
- These runs finish before the grid launches.

### 6.2 Screening (decided): 2⁷⁻³ fractional factorial, 16 runs, seed 0

- **Generators:** distill = replay·lr_rewarm·lora, wavg = lr_rewarm·lora·freeze,
  select = replay·lora·freeze.
- **Resolution IV:** no main effect is aliased with a pairwise interaction.
- **Plus one control,** the reference arm. It is not one of the 16 runs, so the total is
  **17 runs**. Run S01 (all off) is no-mitigation at seed 0, shared with the noise floor.

| Run | replay | lr_rewarm | lora | freeze | distill | wavg | select |
|---|---|---|---|---|---|---|---|
| S01 | · | · | · | · | · | · | · |
| S02 | on | on | · | on | · | · | · |
| S03 | · | · | on | on | on | · | · |
| S04 | on | · | on | · | · | on | · |
| S05 | · | on | · | · | on | on | · |
| S06 | · | on | on | · | · | · | on |
| S07 | on | · | · | · | on | · | on |
| S08 | · | · | · | on | · | on | on |
| S09 | on | on | on | · | on | · | · |
| S10 | · | on | on | on | · | on | · |
| S11 | on | · | · | on | on | on | · |
| S12 | on | · | on | on | · | · | on |
| S13 | · | on | · | on | on | · | on |
| S14 | on | on | · | · | · | on | on |
| S15 | · | · | on | · | on | on | on |
| S16 | on | on | on | on | on | on | on |
| REF | on | on | · | · | · | · | · |

**Pairwise interactions are aliased in groups of three.** The screen cannot separate the
interactions within a group:

| Alias group |
|---|
| replay×lr_rewarm = lora×distill = wavg×select |
| replay×lora = lr_rewarm×distill = freeze×select |
| replay×freeze = lora×select = distill×wavg |
| replay×distill = lr_rewarm×lora = freeze×wavg |
| replay×wavg = lr_rewarm×select = freeze×distill |
| replay×select = lr_rewarm×wavg = lora×freeze |
| lr_rewarm×freeze = lora×wavg = distill×select |

The reference arm's own interaction (replay × lr_rewarm) shares a group with lora × distill and
wavg × select. Any interaction claim needs a follow-up run that separates its group (6.4).

**Known couplings:**
- `lora` with `freeze`: LoRA adapters only on the unfrozen layers.
- `wavg` with `lora`: averaging the adapter toward zero.
- `select` acts on new-domain examples.

These are documented in `TECHNIQUES.md`. The factorial treats them as on/off factors anyway.

### 6.3 Replication (decided)
- Promote the **top 3 configurations**, ranked on the *selection-half* primary outcome
  among the screening runs that pass rule 3 on the selection half.
- Run these 3, the reference arm and no-mitigation at **3 seeds each (0, 1, 2)**, paired by
  seed. Seed 0 reuses the screening and noise-floor runs. That makes **10 new runs**.
- The agent applies this rule mechanically, and its decision log shows the ranking it used.

### 6.4 Follow-ups (exploratory)
If time remains, the agent may run one fold-over or targeted pair to separate one alias group
(section 6.2). Results are exploratory.

## 7. Statistics

- **Paired hierarchical bootstrap** for Δ between two arms, with 10,000 resamples:
  1. Resample the paired seeds with replacement.
  2. Within each, resample the report-half items per task with replacement, using the same
     items for both arms, so the pairing holds at the item level too.
  3. Recompute the per-task means and then the 4-task mean.

  Report the percentile 95% interval.
- **Multiplicity (decided):** apply Holm–Bonferroni across the confirmatory family, which is
  the 3 promoted configurations vs the reference arm at α = 0.05. The reference-vs-no-mitigation
  sanity check is reported on its own. Screening effects are exploratory: intervals, no
  p-values.
- **Screening attribution (exploratory):** an OLS fit of the selection-half primary outcome on
  the 7 factors (±1 coding). That gives main effects, plus the 7 alias-group contrasts. Intervals
  use σ_noise as the known residual SD, because the screen is unreplicated.
- **Fake-row check:** before the grid launches, the analysis must recover the planted effects in
  `results/fake/planted_effects.json` within their intervals.

## 8. Validity rules (provisional)

A row is **invalid**, and leaves the analysis with its reason logged, if any of these hold:
- NaN or inf loss, or anomalous gradient steps above 1% of all steps (the loop's
  `anomalous_grad_steps`).
- `cost.gradient_tokens` differs from the spec's gradient tokens by more than 1%.

A row is **failed** (with a reason, and re-queued once by the executor) if it was killed by the
budget guard, timed out, or hit an infrastructure error.

**Answer collapse** (decided: flag, don't invalidate). The executor records `answer_share`
and leaves the row `finished`; the analysis flags it.
- If any task's `answer_share` is above 0.90, the row is flagged and stays in the analysis.
  Collapse is itself a form of forgetting.
- For flagged rows, the primary outcome is also reported without the collapsed task, as a
  sensitivity check.
- PubMedQA's majority-class rate (0.58) is printed next to its scores.

## 9. Matched compute (provisional)

- All arms in a comparison use the same **gradient tokens** (`spec.gradient_tokens()`).
- Extra forward-only compute, such as teacher passes for `distill` or scoring passes for
  `select`, is recorded in `cost.extra_forward_tokens`. It does not count against the match.
  It enters the cost-adjusted ranking: retention gained per extra GPU-hour, and per byte
  shipped.
- LoRA's own learning rate differs from full fine-tuning. That is part of the technique's
  definition and is stated in `TECHNIQUES.md`.

## 10. Second task sequence (exploratory)

Tinker LoRA on `Qwen/Qwen3-8B`, SST-2 then MNLI then AG News (task C5). Its rules:
- Arms: naive, replay-200, joint, and the top 2 techniques from 6.3, at 3 seeds.
- Metrics: gap closed = (method − naive) / (joint − naive), and average forgetting.
- Intervals from a bootstrap over seeds and eval items.

The track is **exploratory**. It tests whether the main result generalizes; it is not a
confirmatory claim. If joint − naive < 0.03, gap closed is reported with a warning that its
denominator is unstable.

## 11. Amendment protocol

This file will change, and the rules for how keep the results honest:

- **Before the freeze:** edit freely. Settle the provisional items here.
- **After the freeze:** every change gets a row in the amendment log below, in the same
  commit as the change. Each row gives the time, the change, the reason, and whether
  report-half results had already been seen.
- **Changes made after seeing the report half** (for example tuning a threshold or swapping
  an outcome): results under the new rule are labelled **exploratory** wherever they appear,
  and the pre-registered result is still reported next to them.
- **Bug fixes** that change numbers (for example a scoring bug) are amendments too. Re-run
  the affected rows and keep the old rows marked `invalid` with the reason.

### Amendment log

| # | Time (PT) | Change | Reason | Report half seen? |
|---|---|---|---|---|
| — | — | (none yet) | — | — |

---

## Appendix A: decision record

Every option considered, with its trade-offs, so a choice can be revisited later with the
reasoning intact. Changing one after the freeze is an amendment (section 11).

### A1. Primary outcome: **decided**, end-of-stream MC mean
| Option | Pros | Cons |
|---|---|---|
| **End-of-stream MC mean (chosen)** | Already in the row; matches the September baselines; easiest to pitch | One checkpoint is noisy (about 68 flips between checkpoints, 61% reverse); relies on seeds |
| Mean over boundaries | Rewards protecting throughout; uses the existing `boundaries` field | Rewards merely delaying forgetting; with 2 domains it is nearly the end value |
| Persistent losses | Least noisy; item-level interpretability | Needs per-checkpoint item files and more code; harder to explain in 3 minutes |
| End mean + persistent losses as secondary | Primary stays simple, and robustness is reported | Two outcomes to explain. (Persistent losses are kept as a secondary in section 4 anyway.) |

### A2. Acquisition rule: **decided**, 5% floor plus frontier
| Option | Pros | Cons |
|---|---|---|
| **Floor 5% + frontier (chosen)** | Clear win rule plus full transparency | Two things to explain; 5% is a judgement call |
| Within 5% only | Blocks "won by not learning"; LoRA and freezing can still compete | Why 5%? |
| Within 2% only | Strictest | Likely rules out LoRA and freezing by design |
| Pareto frontier only | No arbitrary threshold | No single winner; "improved on results" is harder to claim |

### A2b. What "acquisition" measures: **decided**, perplexity at each domain's own boundary
Decided on 2026-11-14, before the freeze, after the first fast-bench run (`no-mitigation-s0`,
`results/rows.calibration.jsonl`). On that run the end-of-stream ratio barely showed any
learning:
- biomedical was 1.00 (no change);
- code was 1.81, worse than base, because the biomedical domain undid it.

The end-of-stream number mixes learning with forgetting, so a gate on it can't tell "learned
less" from "forgot more". The gate now reads the ratio at each domain's own boundary. The
end-of-stream ratio stays as a secondary outcome.

| Option | Pros | Cons |
|---|---|---|
| **Measure at own boundary (chosen)** | Separates learning from forgetting. No bench change and no reruns: the boundary checkpoints already record it | A small addition to the row format and the executor. Biomedical learning may still be small |
| Lower the LR to 3e-5 | The September 3e-5 run learned more | It forgets far less (−3.4 vs −11.9 points), so effects are smaller with less power, and calibration must be redone |
| Pareto frontier only (drop the gate) | No code or bench change | A technique that learns nothing could "win"; weak for rigour |
| More tokens per domain (e.g. 30M) | More learning | About 3× the cost and time (about 55 minutes per run); may not fit the overnight grid |

### A3. Win rule: **decided**, paired CI and more than 2 × σ_noise
| Option | Pros | Cons |
|---|---|---|
| **Paired CI excludes 0 and Δ > 2σ_noise (chosen)** | Significant and large enough to matter; strongest for rigour | Hardest to pass with 3 seeds |
| Paired CI excludes 0 | Standard, and tight thanks to pairing | A tiny win counts |
| Non-overlapping CIs | Conservative, visual | Ignores pairing, so real wins can fail it |
| Δ > 2σ_noise only | One number, easy to pitch | No formal test |

### A4. Item split: **decided**, stratified by base correctness
| Option | Pros | Cons |
|---|---|---|
| **Stratify by base correctness (chosen)** | Equal forgetting headroom in both halves | Needs base per-item results (they exist) |
| Stratify by gold label | Equal answer mix; protects the collapse check | Doesn't balance difficulty |
| Both strata | Balances difficulty and answer mix | Some cells are tiny; more code |
| Plain seeded random | Simplest | Halves can differ by about 2–3 points by chance |

### A5. Screening design: **decided**, 2⁷⁻³ (16 runs)
| Option | Pros | Cons |
|---|---|---|
| **2⁷⁻³ resolution IV, 16 runs (chosen)** | Every main effect is clean of pairwise interactions; about $40; fits one night | Pairs are aliased in groups of 3; interactions need follow-ups |
| Plackett–Burman, 12 runs | Cheapest | Main effects partly mixed with interactions; weak for an interactions claim |
| One-at-a-time, 8 runs | Simplest to explain | No interactions; loses the novelty angle |
| 2⁷⁻², 32 runs | More interactions separable | 2× the cost; may not fit overnight |

### A6. Seed policy: **decided**, top 3 plus controls at 3 seeds
| Option | Pros | Cons |
|---|---|---|
| **Top 3 + no-mitigation + reference at 3 seeds (chosen)** | Enough for paired CIs on the claims that matter | Other configurations stay single-seed (exploratory) |
| Top 5 at 3 seeds | Wider coverage | About 21 runs; may not finish by Sunday 10:00 |
| Top 2 at 5 seeds | Tightest CIs on the headline | Fewer claims |
| Agent decides adaptively (successive halving) | Strongest autonomy story | Harder to pre-register; the rule must be fixed in advance |

### A7. Noise floor: **decided**, 5 seeds, data order only
| Option | Pros | Cons |
|---|---|---|
| **5 seeds, data order only (chosen)** | Stable SD estimate | 5 × about $2.50; about 35 minutes in parallel |
| 3 seeds | Cheaper, faster | An SD from 3 runs is very uncertain |
| 5 + reference arm at 3 | Checks that noise doesn't depend on the technique | 8 runs before the grid |

### A8. Multiple comparisons: **decided**, Holm on the confirmatory family only
| Option | Pros | Cons |
|---|---|---|
| **Holm on the confirmatory family (chosen)** | Rigorous where it matters | Slightly harder to pass |
| Benjamini–Hochberg FDR | Less conservative across many tests | Controls false discoveries in expectation only; harder to explain |
| No correction | Simplest | A judge may flag it; with 7 techniques, one false win is likely |

### A9. Answer collapse: **decided**, flag above 0.90, don't invalidate
Decided on 2026-11-14, before the freeze. The executor (`executor/rows.py`) previously marked
collapse `invalid`; it now records the share and keeps the row.

| Option | Pros | Cons |
|---|---|---|
| **Flag > 0.90, keep, add a sensitivity check (chosen)** | Collapse is itself a forgetting finding | One more column to explain |
| Invalidate > 0.95 | Keeps degenerate scores out of the means | Throws away real forgetting; can flatter a technique |
| Drop PubMedQA from the mean | Removes the noisiest task | Loses the biomedical capability the stream should stress |

### A10. Invalid-run rules: **provisional**
| Option | Default | Pros | Cons |
|---|---|---|---|
| NaN/inf, or anomalous grad steps above 1% | **in** | Catches backend faults | — |
| Gradient tokens off by more than 1% | **in** | Enforces matched compute | Replay and selection must count tokens exactly |
| General perplexity ratio above 3 | out | Catches a blown-up run | A harsh LR legitimately forgets a lot (1e-4 reached 1.84) |
| Budget-killed or timed out | **in** (as `failed`) | Obvious | Partial rows are lost |

### A11. MBPP: **provisional**, secondary only
| Option | Pros | Cons |
|---|---|---|
| **Secondary only (default)** | Keeps a noisy, low-base task out of the decision | Code ability is part of "what was kept" |
| Not run on the fast bench | Saves eval time | Loses the only generative capability |
| In the primary mean | Covers generation | SE about 2.2 points on a 0.11 base; very noisy |

### A12. Matched compute: **provisional**, equal gradient tokens with extra compute reported
| Option | Pros | Cons |
|---|---|---|
| **Equal gradient tokens; extra reported as cost (default)** | Matches the bench design and the row's `Cost` fields | Distillation gets about 1.3× FLOPs "free" in the primary test |
| Equal FLOPs | Strictest fairness | Needs FLOP accounting per technique; complex |
| Equal wall-clock | What a practitioner pays | Depends on GPU and noise; hard to reproduce |
