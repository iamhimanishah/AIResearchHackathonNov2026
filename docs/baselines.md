# Baselines

The numbers a forgetting-reduction technique has to beat. All of it was measured in September
2026, before the hackathon. Tables are computed from `results/baselines/` and the
`analysis.json` of each run under `results/runs/`.

## Setup

- **Model:** `google/gemma-3-1b-pt`, full fine-tuning, AdamW, constant learning rate, no replay
  and no other mitigation.
- **Stream:** code → biomedical → legal → math, one pass, in that order.
- **Protected set (frozen 2026-09-05):** ARC-Easy, HellaSwag, PIQA and PubMedQA at 500 items
  each, scored at every checkpoint; MBPP at 200 items (3-shot, generated code executed), scored
  at domain boundaries. Item indices are in `manifests/eval-protected-v1.json`.
- **Also measured:** held-out perplexity on each domain plus a general-web slice.

## Base model

From `results/baselines/evals_protected-v1_cuda.md`.

| Task | Metric | Score | ± se | Chance | Majority class |
|---|---|---|---|---|---|
| ARC-Easy | acc_norm | 0.706 | 0.020 | 0.250 | 0.276 |
| HellaSwag | acc_norm | 0.634 | 0.022 | 0.250 | 0.272 |
| PIQA | acc_norm | 0.738 | 0.020 | 0.500 | 0.506 |
| PubMedQA | acc | 0.642 | 0.021 | 0.333 | 0.580 |
| MBPP | pass@1 | 0.110 | 0.022 | 0 | n/a |

Checkpoint 0 of each training run re-scores the base model under the run's own precision
setting and gives slightly different values (multiple-choice mean 0.6825). Run tables below use
checkpoint 0 as their base.

## No-mitigation runs

End of run against base, in points.

| Run | LR | Tokens/domain | MC mean | ARC-Easy | HellaSwag | PIQA | PubMedQA | MBPP |
|---|---|---|---|---|---|---|---|---|
| `pilot-lr1e-5-50M` | 1e-5 | 50M | −1.0 | −3.0 | +0.2 | −1.4 | +0.2 | −1.5 |
| `pilot-lr3e-5-50M` | 3e-5 | 50M | −3.4 | −5.6 | −3.8 | −2.6 | −1.6 | −5.5 |
| `pilot-lr1e-4-50M` | 1e-4 | 50M | −11.9 | −16.0 | −16.6 | −8.2 | −6.6 | −7.0 |
| `pilot-chosen-200M` | 3e-5 | 195M | −7.4 | −14.2 | −9.0 | −5.8 | −0.4 | −4.5 |

Multiple-choice mean after each domain.

| Run | Base | After code | After biomedical | After legal | After math |
|---|---|---|---|---|---|
| `pilot-lr1e-5-50M` | 0.683 | 0.690 | 0.668 | 0.666 | 0.673 |
| `pilot-lr3e-5-50M` | 0.683 | 0.671 | 0.642 | 0.641 | 0.649 |
| `pilot-lr1e-4-50M` | 0.683 | 0.561 | 0.574 | 0.543 | 0.564 |
| `pilot-chosen-200M` | 0.683 | 0.647 | 0.592 | 0.577 | 0.609 |

Held-out per-document perplexity, end of run divided by base. Above 1 is forgetting, below 1 is
acquisition.

| Run | General | Code | Biomedical | Legal | Math |
|---|---|---|---|---|---|
| `pilot-lr3e-5-50M` | 1.17 | 1.12 | 0.99 | 0.77 | 0.87 |
| `pilot-lr1e-4-50M` | 1.84 | 1.41 | 1.30 | 0.95 | 0.99 |
| `pilot-chosen-200M` | 1.43 | 1.24 | 1.09 | 0.79 | 0.84 |

Per-document perplexity was not recorded for the 1e-5 run.

## What these runs showed

Findings from the September analysis that matter when designing a comparison.

- **Forgetting is graded and silent.** No run had a loss spike or an anomalous gradient step.
  The damage is only visible in the protected-set scores.
- **A learning rate of 1e-4 forgets fast.** About 80% of its total loss happens inside the
  first 50M tokens, in the code domain. It is stable, so it works as a stress setting for
  short runs. At that rate even the trained domains end near or worse than base on perplexity.
- **Single-checkpoint differences are noisy.** In the 195M run a median of about 68 of the
  2,200 protected items changed between consecutive checkpoints, and about 61% of flips were
  later reversed. Standard error on a 500-item task is about 2 points. Compare techniques over
  seeds, or on persistent losses, not on one checkpoint.
- **Persistent losses in the 195M run:** ARC-Easy 59 items, HellaSwag 49, PIQA 35, PubMedQA 54,
  MBPP 13, counting an item as lost if it is wrong at each of the last three checkpoints. Most
  of them began during the code domain.
- **An aggregate score can hide a collapse.** PubMedQA sat at exactly its majority-class
  accuracy (0.580) through the legal domain because the model gave one answer to every
  question. The share of items given the most common answer went 0.62 → 0.85 → 0.98 → 1.00 →
  0.86 across the boundaries, and the score after code looked better than base. Track the
  answer distribution, not only accuracy.
- **Little of the loss comes back with a brief re-warm.** A 2M-token pass over general text at
  each boundary recovered at most 30% of the multiple-choice drop.
- **Throughput:** 12k to 15.5k training tokens per second on one H100 (the 195M run and a
  short probe, respectively), so 20M tokens is roughly 22 to 27 minutes before evaluation time.

## Caveats

- One seed per configuration.
- In the three 50M sweep runs, the per-item files at domain-boundary checkpoints hold the state
  after the re-warm probe, because of a bookkeeping bug fixed on 2026-09-09. Scores in
  `metrics.json` and `analysis.json` are unaffected. The 195M run's boundary files were
  regenerated from saved weights.
- `analysis.json` files were produced by `clh analyze` with the full run outputs present.
  Model weights, per-checkpoint SAE feature files and the per-example signal log are not in
  this repository, so feature-level analysis cannot be regenerated from what is here.
