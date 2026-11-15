# Pre-registration: forgetting-reduction study

**Status: FROZEN** when the pull request that adds this line merges to `main` (14 November
2026). Every rule and every number is set. σ_noise, the acquisition margins and the rule-4
control come from phase 0 (section 0.4).
- No technique comparison has been read. The screen has not launched (sync point 2).
- The agent records the merge commit's SHA as the frozen version in its first decision-log
  entry.
- From here on, changes follow the amendment protocol in section 11.

- **Owner:** himani (workstream 4).
- **Task card:** C1 in [`docs/TASKS.md`](../docs/TASKS.md).
- **Record:** every option considered, with its trade-offs, is in
  [Appendix A](#appendix-a-decision-record).

Units follow `src/clharness/spec.py`: scores and deltas are fractions (a 5-point drop is
−0.05), and perplexity ratios are trained / base.

---

## 0. Decisions for the freeze, and phase 0

**The technique comparison is on hold until phase 0 is done and this file is frozen.** The
code→legal calibration (A5) is one seed per learning rate, so on its own it fixes no threshold.

| LR | Report-half retention | Legal perplexity ratio (end of stream) |
|---|---|---|
| 1e-4 | −4.9 points | 0.78 |
| 5e-5 | −2.0 points | 0.76 |
| 3e-5 | −1.2 points | 0.75 |

### 0.1 Primary metric and what the acquisition floor uses
- **Primary metric:** the report-half change in the 4-task multiple-choice mean (ARC-Easy,
  HellaSwag, PIQA, PubMedQA) at the end of the stream, against base. Row field:
  `retention.report.mean_delta`.
- **Acquisition floor: both domains, each one separately.**
  - Code is measured right after the code domain, and legal right after the legal domain
    (`acquisition.own_boundary_ppl_ratio`, section 4).
  - A configuration passes only if it passes on **code and on legal**. An average of the two
    would let better legal learning hide worse code learning.
  - **The code check is a damage guard, not acquisition (A15).** On this bench no arm learns
    code: the own-boundary code ratio is above 1 for every phase-0 arm (no-mitigation 1.128,
    reference 1.044). So the code check reads "does not damage code more than the control
    allows". Legal is the one domain that measures learning (ratios of 0.74–0.79).
  - The end-of-stream ratios stay as secondary outcomes. They can't serve as the floor, because
    code's end-of-stream ratio (1.79 at 1e-4) mostly measures forgetting caused by legal.

### 0.2 Acquisition margin, and when a technique is better than its control
- **Margin, per domain d:** m_d = max(1.05, exp(2 × SD_d)), where SD_d is the SD of
  log(own-boundary perplexity ratio) for domain d over the 5 no-mitigation noise-floor seeds.
  - The gate passes if mean ratio_d(X) ≤ m_d × mean ratio_d(control) for both domains.
  - The 5% floor allows, for legal at 0.78, a ratio up to 0.82: losing at most about a fifth of
    the reference arm's legal gain.
  - The noise term makes sure the margin is never inside run-to-run noise. It is fixed from the
    noise floor, never from technique results.
- **Better than its control** (section 5): a promoted configuration X wins only if, on the
  report half over 3 paired seeds, all four rules hold:
  1. X beats the reference arm: the paired bootstrap interval excludes 0 after Holm.
  2. That difference is above 2 × σ_noise.
  3. X passes the acquisition gate against the reference arm on code and legal.
  4. X beats the chosen lower-LR control with the same tests as rules 1 and 2.

### 0.3 Learning-rate controls in the main comparison
- **The bench LR stays 1e-4** (frozen by A5). Every technique runs at 1e-4, **except LoRA**,
  which runs at the plugin's default of 1e-3 (A15). Adapters barely learn at 1e-4, so that is how
  LoRA is actually used.
  - LoRA rank 8 vs rank 256 is matched on LR.
  - LoRA vs no-mitigation differs in LR as well as method. Every LoRA row and claim 5 say so.
- **5e-5 and 3e-5 are both control arms** (no mitigation, 3 seeds each). Lowering the LR alone
  cuts forgetting, so a technique has to beat that, not only the reference arm.
- **Rule 4 compares against one of them,** chosen on the *selection half*: the one with the
  better retention among those passing the acquisition gate against the reference arm. In the
  calibration both learned legal at least as well as 1e-4 (0.76 and 0.75 vs 0.78), so the
  harder one, 3e-5, may well be the control. A technique at 1e-4 then has to forget less than
  no-mitigation at 3e-5 does. That bar is deliberately hard.
- **If neither lower-LR arm passes the gate on code and legal,** rule 4 holds trivially, and the
  report says so.

### 0.4 Seed plan for phase 0, and the rows that are reused
Phase 0 is the controls only. The exact list is `configs/bench/phase0.txt`, which equals
`design.phase0()`; `tests/test_design.py` keeps them in sync.

| Arm | Seeds | Spec files (`configs/bench/specs/`) | State |
|---|---|---|---|
| No-mitigation, LR 1e-4 (the noise floor) | 0–4 | `no-mitigation-s0.json` … `-s4.json` | **Ran** (A, `results/rows.noise.jsonl`): re-measure |
| Reference arm (replay + LR re-warm) | 0–2 | `reference-s0.json`, `reference-s1.json`, `reference-s2.json` | **GPU: 3 runs** |
| No-mitigation, LR 5e-5 | 0 | `calibration-code-legal-lr5e-5-s0.json` | **Ran** (A5 calibration): re-measure |
| No-mitigation, LR 5e-5 | 1–2 | `no-mitigation-lr5e-05-s1.json`, `-s2.json` | **GPU: 2 runs** |
| No-mitigation, LR 3e-5 | 0 | `calibration-code-legal-lr3e-5-s0.json` | **Ran** (A5 calibration): re-measure |
| No-mitigation, LR 3e-5 | 1–2 | `no-mitigation-lr3e-05-s1.json`, `-s2.json` | **GPU: 2 runs** |

**7 GPU runs**: `clh run --modal $(grep -v '^#' configs/bench/phase0.txt) --out results/rows.phase0.jsonl`.

**Reused rows, re-measured.** Seven rows already ran: no-mitigation seeds 0–4, and the two
seed-0 lower-LR calibration runs.
- Each has the same spec hash as its design run.
- They were measured by an older executor (the plain random item split, no own-boundary
  acquisition), so they are **re-measured from their saved checkpoints** with the current
  executor before use. No retraining.
- The original rows stay in their files; the re-measured rows go to
  `results/rows.phase0.remeasured.jsonl`.

**Not reused:**
- `calibration-code-legal-lr1e-4-s0` is the same training configuration as `no-mitigation-s0`,
  but its spec sets `lr` explicitly, so it hashes differently (rule: one spec, one experiment).
  It is kept as a reproducibility check: same seed and configuration, so its scores should match
  `no-mitigation-s0` up to GPU nondeterminism.
- The code→biomedical rows (the first `no-mitigation-s0` and `calibration-b16-s0`) are from the
  old stream and are not used.

**Phase 0 results: the frozen values.** Computed by `clharness.analysis.attribution` from
`results/rows.phase0.remeasured.jsonl`, `rows.phase0.jsonl` and `rows.reference.remeasured.jsonl`.
All 14 phase-0 specs are present, each with own-boundary acquisition and the pinned split.

| Value | Frozen at |
|---|---|
| **σ_noise, report half** (rule 2 uses this) | **1.15 points**, so 2σ = **2.30 points** |
| σ_noise, selection half | 0.59 points |
| **Margin m_code** (damage guard) | **1.05** (2 SD of log ratio = 0.010) |
| **Margin m_legal** | **1.05** (2 SD of log ratio = 0.006) |
| **Rule-4 control L** | **no-mitigation at LR 3e-5**: selection half −1.47 vs −2.43 at 5e-5; passes the gate on code (1.027 ≤ 1.096) and legal (0.754 ≤ 0.782) |
| Sanity check, reference vs no-mitigation (report half, 3 paired seeds) | **+2.83 points** (95% interval −0.14 to +5.64; parametric bootstrap, see below) |

Report-half means over seeds (points):

| Arm | Mean | SD | n |
|---|---|---|---|
| No-mitigation 1e-4 | −6.44 | 1.15 | 5 |
| Reference arm | −3.13 | 0.85 | 3 |
| No-mitigation 5e-5 | −2.80 | 0.46 | 3 |
| No-mitigation 3e-5 | −1.07 | 0.80 | 3 |

Own-boundary legal ratios: 0.787, 0.744, 0.759 and 0.754 respectively.

**What this implies, stated before the screen.**
- No-mitigation at 3e-5 forgets less than the published reference recipe at 1e-4, with about
  the same legal learning.
- To win, a configuration at 1e-4 must beat that by more than 2.30 points, which means ending
  above the base model on the report half. A "no winner" result is likely. It is a
  pre-registered outcome (section 5), and it answers the question "does any technique beat
  simply lowering the learning rate?" with data.

**The item-level bootstrap needs per-item files.**
- The sanity interval above used the parametric fallback (section 7), which is conservative,
  because the per-item checkpoint files are on the Modal volume.
- Before the confirmatory analysis, the base and final `items_<task>.npy` of every analysed run
  must be available locally (they are small), so the bootstrap resamples real items.

### 0.5 Estimated Modal cost
Cost per run comes from A's real code→legal rows: about 19 minutes and $1.21–1.28 each on an
H100, at the executor's $3.95 per GPU-hour.

| Phase | GPU runs | Estimate |
|---|---|---|
| Already spent: noise floor (5) + code→legal calibration (3) | 8 | about $10 |
| **Phase 0, still to run:** 7 runs + re-measuring 7 (CPU, under $0.10) | 7 | **about $9**; **cap $12** with retries |
| Phase 1, screen (22 specs; no-mitigation-s0 and reference-s0 come from phase 0) | 20 | about $25 |
| Phase 1, published-claims arms (section 6.5; 21 specs, `replay-5` at seed 0 is the screen's replay-only half) | 20 | about $25 |
| Phase 2, replication (top 3 × seeds 1–2) + selection family (2 × seeds 1–2) | 10 | about $13 |
| **Still to spend on the whole study** | **57** | **about $72**; under $95 with retries |

That is well inside the $700 Modal cap in DIRECTION.md. Phase 0 runs in parallel in about
20 minutes.

## 1. Question

At matched gradient tokens, which forgetting-reduction techniques, alone or in combination,
retain more of Gemma 3 1B's protected capabilities than the reference arm (replay plus
per-domain LR re-warm, Ibrahim et al. 2024)? And do they do it without learning the new domains
less?

**Confirmatory claims** are tested under the rules below. They are limited to:
1. The three configurations promoted from the factorial screen (section 6.3), each against the
   reference arm and against the best lower-learning-rate no-mitigation arm (rule 4).
2. The two selection signals promoted from the selection screen (section 6.3b), each against
   the reference arm with random selection. This is a separate family with its own Holm
   correction.
3. The reference arm against no-mitigation. This is a sanity check that the reference arm
   works on this bench.

Everything else is **exploratory**, including the published-claims family (section 6.5): the screening effects, the interactions, the second task
sequence and the per-task breakdowns. These are reported with intervals, but they are not
claimed as wins.

## 2. Bench and arms

- **Bench:** the fast bench, frozen by Member A on 2026-11-14 (`configs/bench/fast.yaml`,
  task A5):
  - **code then legal**, 10M gradient tokens per domain, LR 1e-4, 64 sequences per step;
  - the bench config's SHA is recorded in every result row.

  Legal replaced biomedical because legal is learned (held-out perplexity about 22–25% below
  base in the calibration runs) and biomedical was not.
- **No-mitigation:** all techniques off.
- **Reference arm:** `replay` (5% of general text) plus `lr_rewarm` (per-domain warmup and
  decay). Its parameters are fixed by B1 before the freeze. They are not tuned on the report
  half.
- **Techniques screened** (names match workstream 3's plugins): `replay`, `lr_rewarm`, `lora`,
  `freeze`, `distill` and `wavg`.
  - Each technique's parameters are fixed at one level, given in `PARAMS` in
    `src/clharness/analysis/design.py`. They match the specs in `configs/bench/specs/` and are
    frozen with this file.
  - Rows whose parameters differ from these are **off-design**: they are counted and flagged,
    but left out of the fits and of promotion.
- **Selection is a parameter of `replay`,** not a seventh technique (`docs/TECHNIQUES.md`,
  decision 2, confirmed here).
  - It chooses among *replay candidates*: replay draws `pool` = 4 candidates per row it needs,
    scores them, and keeps the ones the signal ranks highest (`keep: high`) or lowest
    (`keep: low`).
  - `select: random` is plain replay and the control.
  - Selection is screened in its own block (6.2) and tested as its own family (6.3b).

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
| **Co-primary gate** (decided) | Acquisition, measured at each domain's own boundary: held-out perplexity right after that domain's training, divided by base, **for code and for legal separately**. Both must pass (section 0.2). Rows written before this field existed fall back to `ppl_ratio`, and the analysis reports how many rows did | `acquisition.own_boundary_ppl_ratio` | lower is better |
| Secondary | Acquisition kept at the end of the stream: perplexity on the trained domains at the end, divided by base. This mixes learning with forgetting | `acquisition.mean_ppl_ratio` | lower |
| Secondary | Per-task end-of-stream change, report half | `retention.report.per_task[*].delta` | higher |
| Not measured | MBPP: the frozen fast bench does not run it (A11) | — | — |
| Secondary | Persistent losses: items right at base and wrong at each of the last 3 checkpoints | per-item file | lower |
| Secondary | General-web perplexity ratio (untrained slice) | `acquisition.general_ppl_ratio` | lower |
| Secondary | Cost: gradient tokens, extra forward tokens, wall-clock, update bytes, est. USD | `cost.*` | report |
| Diagnostic | Answer share per task | `answer_share` | flag above 0.90 |

## 5. Decision rule: "beats the reference arm" (decided)

A configuration X beats the reference arm R only if **all four** of these hold, on the report
half, over the paired replication seeds:

1. **Significant:** the 95% paired bootstrap interval (section 7) for
   Δ = primary(X) − primary(R) excludes 0, after Holm correction across the confirmatory set.
2. **Large enough to matter:** the point estimate Δ is above **2 × σ_noise**. σ_noise is the SD
   of the primary outcome across the 5 no-mitigation noise-floor seeds (section 6.1).
   **σ_noise = 1.15 points** (report half, frozen from phase 0, section 0.4), so the threshold
   is 2.30 points.
3. **Learns enough, and does not damage code:** on **each** trained domain (code and legal; a
   configuration missing either domain fails, amendment 5),
   the mean own-boundary perplexity ratio of X is at most m_d × that of R, with
   m_d = max(1.05, exp(2 × SD_d)) from the noise floor (section 0.2). Both margins are frozen at
   1.05. For legal this is acquisition; for code it is a damage guard (A15).
4. **Beats just lowering the learning rate (A13):** let L be the lower-LR no-mitigation arm
   (5e-5 or 3e-5) with the best *selection-half* primary outcome, among those that pass the
   rule-3 gate against R on both domains.
   - X must beat L with the same tests as rules 1 and 2: the paired bootstrap interval for
     primary(X) − primary(L) excludes 0 after Holm correction across the promoted
     configurations, and that Δ is above 2 × σ_noise.
   - If no lower-LR arm passes the gate, rule 4 holds trivially, and the report says so. If the
     lower-LR rows are **absent** from the analysis, rule 4 **fails**: leaving them out cannot
     waive it (amendment 5).
   - Rule 4 applies to the factorial family only. The selection family compares selection
     signals with random selection, where the learning rate plays no part.

The retention–acquisition **Pareto frontier** is always shown next to these results. A
configuration that is on the frontier but fails rule 3 is reported as a trade-off, not a win.

If no configuration meets all four, we report the one closest to doing so, with the reason it
fell short. That is a pre-registered outcome, not a failure.

## 6. Design

### 6.1 Noise floor (decided)
- No-mitigation at **5 seeds (0–4)**. The seed changes only the data order: the stream seed is
  `train.seed + domain_index`, and Gemma has no dropout in this setting.
- Specs: `configs/bench/specs/no-mitigation-s0.json` to `-s4.json`. They are identical except
  for `seed` (code then legal, 10M tokens each, no techniques). They are the same experiments as
  `design.spec(NO_MITIGATION, seed)` (same spec hash).
- σ_noise = the SD of `retention.report.mean_delta` over these 5 runs. The selection-half SD is
  recorded too.
- These runs finish before the grid launches.

### 6.2 Screening (decided): 2⁶⁻² factorial, plus the reference halves and selection; 22 runs at seed 0

The design is code: `src/clharness/analysis/design.py` (`screen()` builds all 22 specs).
`tests/test_design.py` checks three things: the design is resolution IV, both controls are runs
of it, and every spec is accepted by the real plugins.

**Block 1, factorial (16 runs).**
- A 2⁶⁻² over the six techniques, with generators distill = replay·lr_rewarm·lora and
  wavg = replay·lr_rewarm·freeze.
- **Resolution IV:** no main effect is aliased with a pairwise interaction.
- Among the resolution IV choices, these generators put **both controls inside the design**:
  F01 is no-mitigation (shared with noise-floor seed 0) and F04 is the reference arm.

**Block 2, reference halves (2 runs).** `replay` alone and `lr_rewarm` alone. With F01 and F04
they complete the 2×2 of the reference recipe. That makes replay × lr_rewarm estimable, even
though the factorial aliases it with lora × distill and freeze × wavg.

| Run | replay | lr_rewarm | lora | freeze | distill | wavg | Note |
|---|---|---|---|---|---|---|---|
| F01 | · | · | · | · | · | · | no-mitigation (noise-floor seed 0) |
| F02 | · | · | on | · | on | · |  |
| F03 | · | · | · | on | · | on |  |
| F04 | on | on | · | · | · | · | reference arm |
| F05 | · | on | · | on | on | · |  |
| F06 | on | · | · | on | on | · |  |
| F07 | · | on | · | · | on | on |  |
| F08 | on | · | · | · | on | on |  |
| F09 | · | on | on | on | · | · |  |
| F10 | on | · | on | on | · | · |  |
| F11 | · | on | on | · | · | on |  |
| F12 | on | · | on | · | · | on |  |
| F13 | · | · | on | on | on | on |  |
| F14 | on | on | on | · | on | · |  |
| F15 | on | on | · | on | · | on |  |
| F16 | on | on | on | on | on | on |  |
| H1 | on | · | · | · | · | · | half of the reference recipe |
| H2 | · | on | · | · | · | · | half of the reference recipe |

The interactions are aliased in 7 chains. The factorial alone cannot separate the pairs within
a chain:

| Alias chain |
|---|
| distill×freeze = lora×wavg |
| distill×lora = freeze×wavg = lr_rewarm×replay |
| distill×lr_rewarm = lora×replay |
| distill×replay = lora×lr_rewarm |
| distill×wavg = freeze×lora |
| freeze×lr_rewarm = replay×wavg |
| freeze×replay = lr_rewarm×wavg |

**Block 3, selection on replay (4 runs).** The reference arm with `select` set to `loss` or
`act_nov` (`layer` 13 for act_nov), each with `keep: high` and `keep: low`. The control is F04,
the same arm with random selection.

**Known couplings:**
- `lora` with `freeze`: LoRA adapters only on the unfrozen layers.
- `wavg` with `lora`: averaging the adapter factors toward zero.

These are documented in `docs/TECHNIQUES.md`. The factorial treats them as on/off factors
anyway.

Specs that match the committed `no-mitigation-s0`, `reference-s0`, `replay-only-s0` and
`rewarm-only-s0` have the same spec hash, so those runs are shared and never repeated.

### 6.3 Replication (decided)
- Promote the **top 3 configurations** from the factorial, ranked on the *selection-half*
  primary outcome. Only runs that pass rule 3 (acquisition) are eligible. The controls, the
  reference halves, selection variants and off-design rows are not candidates.
- Run these 3 at **seeds 1 and 2** (6 new runs), paired by seed with their seed-0 screening
  run. The controls they are compared with (no-mitigation, the reference arm and both lower-LR
  arms at seeds 0–2) already ran in phase 0.
- The agent applies this rule mechanically, and its decision log shows the ranking it used.

### 6.3b Selection family (decided)
- For each signal (`loss`, `act_nov`), promote the `keep` direction with the better
  selection-half outcome, if it passes rule 3.
- Run both promoted variants at seeds 1 and 2 (4 new runs). Each is compared with the reference
  arm at the same seeds.
- The two comparisons form their own family, with its own Holm correction. The question is
  whether a model-internal signal chooses better replay examples than chance.

### 6.4 Follow-ups (exploratory)
If time remains, the agent may run one fold-over or targeted pair to separate one alias group
(section 6.2). Results are exploratory.

### 6.5 Published-claims family (decided; exploratory)
This section is Member A's proposal (`docs/published-claims.md`), adopted before the freeze.
Besides ranking techniques, the study tests published forgetting claims on the frozen bench
and reports, for each one, whether it reproduced. Because the bench holds model, stream and
budget constant, a claim that fails here is still informative.
- **The family is exploratory.** It is not part of the win rule (section 5) and has no Holm
  correction. Every test is reported, whatever its result.
- **It is code:** `CLAIMS`, `CLAIM_ARMS` and `claims_specs()` in
  `src/clharness/analysis/design.py`. `clh grid-analysis --confirmatory` writes the claims
  table.

**Claim arms** (3 seeds each, 0–2, run with the screen in phase 1):

| Arm | Techniques and parameters | New runs |
|---|---|---|
| `replay-5` | replay at 5% (the design default) | 2 (seed 0 is the screen's replay-only half) |
| `replay-25` | replay at 25% (the paper's own level) | 3 |
| `replay+distill-on-replay` | replay 5% + `distill` with `rows: replay` | 3 |
| `lora-r8` | LoRA rank 8 (the design default) | 3 |
| `lora-r256` | LoRA rank 256 | 3 |
| `freeze-bottom` | freeze the bottom 13 layers (the design default) | 3 |
| `freeze-top` | freeze the top 13 layers (`bottom_layers: 0, top_layers: 13`) | 3 |
| **Total** | | **20 runs, about $25** |

Comparators come from phase 0 (no-mitigation, the reference arm, LR 3e-5) or from the screen
(rewarm-only).

**Tests** (arm − comparator, report half):

| # | Published claim (source) | Arm vs comparator | Outcome | Published sign |
|---|---|---|---|---|
| 1 | Replay plus LR re-warm beats no mitigation (Ibrahim et al. 2024) | reference vs no-mitigation | retention | + |
| 2 | Peak LR is the main knob (Ibrahim et al. 2024) | LR 3e-5 vs no-mitigation | retention | + |
| 2 | …warmup barely matters (same) | rewarm-only vs no-mitigation | retention | magnitude only (seed 0 only) |
| 3 | Replay is worth about 1 to 3 points at 1B (Abbes et al. 2026) | replay-5 vs no-mitigation; replay-25 vs no-mitigation; replay-25 vs replay-5 | retention | + |
| 5 | LoRA forgets less and learns less; higher rank learns more (Biderman et al. 2024) | lora-r8 vs no-mitigation (retention and legal acquisition); lora-r256 vs lora-r8 (legal acquisition) | retention / acq_legal | +, +, − |
| 6 | Distilling on replay tokens beats plain replay (Beyond One-Hot Replay, 2026) | replay+distill-on-replay vs replay-5 | retention | + |
| 7 | Freezing the top half beats freezing the bottom half (Freeze Deep, Train Shallow, 2026) | freeze-top vs freeze-bottom | retention | + |

**Covered elsewhere or not tested:**
- Claim 4 (Reptile averaging stacks with replay): there is no plugin, so it is not tested
  unless B builds one before the freeze.
- Claim 8 (selection rarely beats random): answered by the selection family (6.3b). It
  reproduces if neither signal wins.
- Claim 9 (averaging roughly matches 10% replay): compared through the screen's `wavg` and
  `replay` main effects, as a magnitude only.

**What "reproduced" means** (fixed before any claims run is read):
- **A directional test reproduces** if the paired 95% interval of (arm − comparator) excludes 0
  in the published direction.
  - Retention uses the paired hierarchical bootstrap (section 7).
  - Acquisition uses the log own-boundary perplexity ratio of the named domain, with a
    **paired t interval** over seeds (df = n − 1). Amendment 2: this was a bootstrap over paired
    seeds, which at 3 seeds excludes 0 in about 25% of no-effect cases.
- **Other verdicts:** "opposite" means the interval excludes 0 the other way. "Not reproduced
  (interval includes 0)" otherwise. "Too few seeds" means fewer than 2 paired seeds.
- **A magnitude test** reports our effect next to the published number, with no pass or fail.
- **The lower-LR caution:** every retention row also reports the arm against the chosen
  lower-LR control (rule 4's L). So a claim that reproduces against 1e-4 but loses to just
  lowering the learning rate says so.
- **Different parameters:** a claim tested with parameters other than the paper's says so in
  its row. For example, our replay share is 5% and 25%, while Abbes et al. also used 50%.
- **Cautions** for the write-up: our learning rate is harsh and our budget tiny (20M tokens
  against billions). A failure to reproduce may reflect that, not a fault in the paper.

## 7. Statistics

- **Paired hierarchical bootstrap** for Δ between two arms, with 10,000 resamples:
  1. Resample the paired seeds with replacement.
  2. Within each, resample the report-half items per task with replacement, using the same
     items for both arms, so the pairing holds at the item level too.
  3. Recompute the per-task means and then the 4-task mean.

  Report the percentile 95% interval.

  The item level needs each run's base and final `items_<task>.npy`. Without them the code
  falls back to parametric item noise, which is wider and conservative, and the report says
  which mode each interval used.
- **Multiplicity (decided):** apply Holm–Bonferroni within each confirmatory family at α = 0.05:
  - the 3 promoted configurations vs the reference arm;
  - the 2 selection signals vs the reference arm.

  The reference-vs-no-mitigation sanity check is reported on its own. Screening effects are
  exploratory: intervals, no p-values.
- **Screening attribution (exploratory):** least squares of the selection-half primary outcome
  on 0/1 technique indicators, over the design rows (factorial, halves and replications; not
  selection variants or off-design rows):
  - **Main fit:** the 6 main effects plus replay × lr_rewarm, and its two alias partners in the
    factorial (lora × distill, freeze × wavg). The halves separate replay × lr_rewarm from the
    two partners; the partners stay aliased with each other, so they are one column, reported
    as lora × distill = freeze × wavg (the sum of both). (Corrected wording, red-team RT-11: the
    text first said all three were separately estimable.)
    A coefficient reads as "what switching this on changes". (Amendment 3: without the partner
    columns, their effects leaked into replay × lr_rewarm.) Published-claims arms whose
    technique set is not in the factorial or the halves (lora alone, freeze alone) are not
    design rows.
  - **Chain fit:** on the factorial runs only, the 6 main effects plus one column per alias
    chain. Each chain estimate is the sum of the pairs in its chain, and it is reported with
    all of their names.
  - Intervals use the residual SD when there are spare degrees of freedom, otherwise σ_noise.
- **Selection screen:** each selection variant's selection-half outcome minus the reference
  arm's.
- **Fake-row check:** before the grid launches, the analysis must recover the planted effects in
  `results/fake/planted_effects.json` within their intervals.

## 8. Validity rules (decided)

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

## 9. Matched compute (decided)

- All arms in a comparison use the same **gradient tokens** (`spec.gradient_tokens()`).
- Extra forward-only compute, such as teacher passes for `distill` or scoring passes for
  `select`, is recorded in `cost.extra_forward_tokens`. It does not count against the match.
  It enters the cost-adjusted ranking: retention gained per extra forward token, and per byte
  shipped. Amendment 4: this was per extra GPU-hour, but wall-clock differences between runs are
  mostly noise.
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

**Settings** (decided by himani on 2026-11-14 at 19:25 PT, before any C5 data; amendment 6):
- **Model and training:** LoRA rank 32 on `Qwen/Qwen3-8B`. Learning rate 3e-4, constant. Batch
  32 examples, max length 256 tokens, 2 epochs per stage. The loss is on the label words only.
- **LR check:** 3e-4 is within the LoRA limit in the agent catalog, 1e-3 × √(8/32) = 5e-4.
- **Data:** frozen in `manifests/tinker-c5.json`: 5,000 train and 1,000 eval per task.
  - SST-2's eval set is its 872 validation examples plus 128 training examples outside the
    5,000-example training subsample.
  - MNLI uses `validation_matched`; AG News uses `test`.
- **Scoring:** label-word log-likelihood (the sum over its tokens), argmax over the labels.
- **Arms:**
  - replay-200: a fixed buffer of 200 examples per earlier task, mixed in by replacement, so
    each stage keeps the same number of examples as naive (tokens match only approximately,
    and the realized share is recorded);
  - joint: the same total number of examples as naive.
- **Rows:** smoke-test rows (50/50 per task) are pipeline checks only and never enter a metric.
- **Code:** `src/clharness/tinker/` (PR #13); `clh tinker-run`, `clh tinker-analyze`.

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
| 1 | 2026-11-14 17:46 | Section 6.2: the reference arm is F04 in the text, as in the table (the text said F02) | Text error found by the independent review. No rule changes | No: no technique-comparison or claim-arm rows existed |
| 2 | 2026-11-14 17:46 | Section 6.5: claim acquisition tests use a paired t interval over seeds (df = n − 1) instead of a bootstrap over paired seeds | Review simulation: a 3-seed percentile bootstrap excludes 0 in about 25% of no-effect cases, which would decide claim 5's two acquisition tests | No: no claim-arm rows existed |
| 3 | 2026-11-14 17:46 | Section 7: the main fit adds the alias partners of replay × lr_rewarm (lora × distill, freeze × wavg) | Review simulation: planting only lora × distill showed up as replay × lr_rewarm. The halves make the partners estimable | No |
| 4 | 2026-11-14 17:46 | Section 9: the cost ranking divides by extra forward tokens, not extra GPU-hours | Wall-clock differences between runs are noise; for example, the lower-LR controls scored about 300 points per "extra GPU-hour" | No |
| 5 | 2026-11-14 17:46 | Sections 5 and 0.2, made explicit: rule 4 fails when the lower-LR rows are absent (it still holds trivially when they are present and fail the gate); a configuration missing an acquisition domain fails rule 3 | The code waived rule 4 when its rows were left out, and passed a configuration with a domain missing | No |
| 6 | 2026-11-14 19:25 | Section 10: C5 settings fixed (LoRA rank 32, LR 3e-4 constant, batch 32, length 256, 2 epochs, the data manifest and SST-2 eval mix, scoring, arm definitions) | Section 10 named the model and arms but not the training settings; fixed here before any C5 run | No: no C5 rows existed |
| — | 2026-11-14 | Implementation fixes, with no rule change (PR #9, independent review): screen-time decisions (promotion, selection screen, lower-LR control, margins) use seed-0 rows only; rows are de-duplicated by spec hash; σ_noise is pinned at the frozen 1.15 points; finished rows from non-real backends are excluded unless explicitly allowed | Make the code do what sections 0, 3, 5 and 6.3 already say | No |
| — | 2026-11-14 | Implementation fixes, with no rule change (red-team review `docs/reviews/red-team-57fa138.md`, RT-01, 05, 08, 09, 10, 11, 24, 25, 26, 27): the acquisition gate fails closed without a seed-0 reference row, and replication planning refuses; relative item paths are read from the repo root and the bootstrap mode is printed; a real-backend row always wins de-duplication over a local copy; finished rows off their gradient-token budget by more than 1% are invalid (section 8); claims arms outside the factorial and halves stay out of the effects fit (section 7); lora × distill is labelled as aliased with freeze × wavg (section 7 wording corrected); the margins (1.05 on both domains) and the rule-4 control L (no-mitigation at 3e-5) are the values frozen in section 0.4, not re-derived from seed 0 (this supersedes the PR #9 row for those two); flagged rows get the without-collapsed-task sensitivity and the 0.58 majority rate (section 8), and configurations report end-of-stream acquisition and the general-web ratio (section 4); the cost chart plots the paired gain against extra forward tokens (Amendment 4) | Make the code do what sections 0.4, 4, 5, 7, 8 and 9 already say | No: run before any `--confirmatory` analysis |

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

The bench then moved to code then legal (A5), and the reason still holds there. In the
code→legal calibration at 1e-4, legal ends at 0.78, so it is learned. Code, though, ends at
1.79, because the legal domain undid it.

The end-of-stream number mixes learning with forgetting, so a gate on it can't tell "learned
less" from "forgot more". The gate now reads the ratio at each domain's own boundary. The
end-of-stream ratio stays as a secondary outcome.

| Option | Pros | Cons |
|---|---|---|
| **Measure at own boundary (chosen)** | Separates learning from forgetting. No bench change and no reruns: the boundary checkpoints already record it | A small addition to the row format and the executor. Biomedical learning may still be small |
| Lower the LR to 3e-5 | The September 3e-5 run learned more | It forgets far less (−3.4 vs −11.9 points), so effects are smaller with less power, and calibration must be redone |
| Pareto frontier only (drop the gate) | No code or bench change | A technique that learns nothing could "win"; weak for rigour |
| More tokens per domain (e.g. 30M) | More learning | About 3× the cost and time (about 55 minutes per run); may not fit the overnight grid |

### A2c. Acquisition gate per domain, with a noise-based margin: **decided**
Decided on 2026-11-14, before the freeze. Each domain (code, legal) passes separately, with
margin m_d = max(1.05, exp(2 × SD_d)) from the noise floor.

| Option | Pros | Cons |
|---|---|---|
| **Each domain separately, margin max(5%, 2 SD) (chosen)** | Better legal learning can't hide worse code learning. The margin never sits inside run-to-run noise, and it's set from the noise floor, not from technique results | Stricter: a run must pass twice |
| Geometric mean of the two domains, 5% | One number | One domain can mask the other |
| Legal only | Legal is the domain that is clearly learned on this bench | Ignores code learning; a technique could block code learning for free |
| Code only | — | Code ends worse than base at every LR, so it's the noisier domain to gate on alone |

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

### A5. Screening design: **decided**, 2⁶⁻² factorial + reference halves + selection block (22 runs)
Revised on 2026-11-14, before the freeze. Workstream 3 built selection as a parameter of
`replay` (`docs/TECHNIQUES.md`, decision 2), not as a technique named `select`. In the first
design (2⁷⁻³ over 7 factors), several runs switched selection on without replay, which cannot
run.

| Option | Pros | Cons |
|---|---|---|
| **2⁶⁻² + halves + selection block, 22 runs (chosen)** | Resolution IV over the real techniques. Both controls are runs of the design. The halves make the reference interaction estimable. Selection is tested where it exists, on replay. About $26 at $1.20 per run | 7 alias chains (pairs within a chain need follow-ups); two families to correct |
| 2⁷⁻³ with `select` as a factor, 17 runs (the first design) | One design for everything | Runs with selection but no replay can't exist; the reference interaction was aliased in a chain of 3 |
| 2⁶⁻¹ resolution VI, 32 runs | Every pair separable | 2× the runs; still needs the selection block |
| One-at-a-time + selection, 11 runs | Simplest | No interactions at all; loses the novelty angle |

**Replay candidates vs new-domain candidates** (confirming decision 2): replay candidates were
chosen. Selection then composes with the reference arm, random selection is exactly plain
replay (a free control), and scoring costs about 20% extra forward tokens. New-domain selection
would need a change to the shared loop's accumulation.

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

### A10. Invalid-run rules: **decided** (the defaults below)
| Option | Default | Pros | Cons |
|---|---|---|---|
| NaN/inf, or anomalous grad steps above 1% | **in** | Catches backend faults | — |
| Gradient tokens off by more than 1% | **in** | Enforces matched compute | Replay and selection must count tokens exactly |
| General perplexity ratio above 3 | out | Catches a blown-up run | A harsh LR legitimately forgets a lot (1e-4 reached 1.84) |
| Budget-killed or timed out | **in** (as `failed`) | Obvious | Partial rows are lost |

### A11. MBPP: **decided**, not measured on the fast bench
The frozen fast bench (A5) leaves MBPP out of its eval list to save about 170 seconds per
checkpoint, so it isn't an outcome of this study.

| Option | Pros | Cons |
|---|---|---|
| **Secondary only (default)** | Keeps a noisy, low-base task out of the decision | Code ability is part of "what was kept" |
| Not run on the fast bench | Saves eval time | Loses the only generative capability |
| In the primary mean | Covers generation | SE about 2.2 points on a 0.11 base; very noisy |

### A12. Matched compute: **decided**, equal gradient tokens with extra compute reported
| Option | Pros | Cons |
|---|---|---|
| **Equal gradient tokens; extra reported as cost (default)** | Matches the bench design and the row's `Cost` fields | Distillation gets about 1.3× FLOPs "free" in the primary test |
| Equal FLOPs | Strictest fairness | Needs FLOP accounting per technique; complex |
| Equal wall-clock | What a practitioner pays | Depends on GPU and noise; hard to reproduce |

### A14. Published-claims family: **decided**, exploratory, 3 seeds, 20 new runs
Proposed by Member A (`docs/published-claims.md`), adopted on 2026-11-14 before the freeze.

| Option | Pros | Cons |
|---|---|---|
| **Exploratory family, 3 seeds, all four testable claims (chosen)** | Anchors the results in published work; novelty (one bench, many papers' claims); no change to the win rule | About $25; 20 more runs in phase 1 |
| Confirmatory, with Holm across claims | Stronger claims | Many tests, so it is hard to pass; these are not the study's main question |
| Only claims the screen already covers (no new arms) | Free | Leaves out the 4 claims that need parameter changes |
| Single seed per arm | Cheaper (about $9) | At 1 seed the noise (SD about 1.1 points between two arms) swamps most published effects |

### A15. Three settings decided at the freeze (2026-11-14)
Raised by Members A and B after phase 0.

| Question | Chosen | Alternatives considered |
|---|---|---|
| The code acquisition check, given that no arm learns code (own-boundary code ratio 1.03–1.13, above 1 everywhere) | **Keep it, relabelled as a code damage guard.** It still blocks a technique that harms code more than the control allows | Legal only (simpler, but code damage could pass); keep the wording (misleading) |
| LoRA's learning rate: the plugin defaults to 1e-3, the bench is 1e-4 | **Keep 1e-3 and state the exception** (0.3). LoRA is tested the way it is used; r8 vs r256 is matched | Force 1e-4 (`lr: null`): LoRA would barely learn, so "forgets less" becomes trivial and claim 5 uninformative |
| Distillation `top_k` | **256**, from the literature scan: under ~25 hurts, ~300 is close to the full vocabulary. Needs a 4-step sanity run for memory | 64 (already ran on a GPU, but may under-represent distillation) |

### A13. Lower-learning-rate controls: **decided**, added to the win rule as rule 4
Decided on 2026-11-14, before the freeze.
Raised by Member A's code→legal calibration (TASKS A5; one seed each). Lowering the learning
rate cuts forgetting with the same legal acquisition:

| LR | Change in selection-half mean |
|---|---|
| 1e-4 | −7.5 points |
| 5e-5 | −3.1 points |
| 3e-5 | −0.7 points |

So "just lower the LR" is itself a strong mitigation, and a technique should be compared with
it as well as with the reference arm.

| Option | Pros | Cons |
|---|---|---|
| **Add to the win rule (chosen):** X must also beat the best lower-LR no-mitigation arm whose acquisition is within the 5% gate | Answers the obvious judge question ("why not lower the LR?"). One extra comparison per promoted configuration | 2 lower-LR arms × 3 seeds (6 runs, about $7); one more condition to pass |
| Report lower-LR arms as an exploratory comparison only | No change to the decision rule | Leaves the strongest simple baseline out of the claim |
| Add LR as a 7th factor in the screen | Gives the LR × technique interactions | Changes the frozen bench's LR per run; doubles the screen or loses resolution |
