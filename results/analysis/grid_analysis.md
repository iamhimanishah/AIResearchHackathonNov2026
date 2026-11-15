# Grid analysis

Halves read: selection, report

Rows: {'total': 62, 'finished': 62, 'failed': 0, 'invalid': 0, 'duplicates_dropped': 0, 'excluded_backend': 0, 'invalid_tokens': 0}

Acquisition margins (frozen in PREREG 0.4): code 1.050, legal 1.050

Noise floor: `no-mitigation` over 5 seeds, σ_noise = 1.15 points (frozen), measured σ(selection) = 0.59 points, measured σ(report) = 1.15 points

## Retention effects (selection half, points)

| Effect | Estimate | 95% CI |
|---|---|---|
| (baseline) | -4.76 | -5.66 to -3.86 |
| distill | +3.01 | +1.24 to +4.79 |
| freeze | +1.04 | +0.04 to +2.03 |
| lora | +2.99 | +1.21 to +4.77 |
| lr_rewarm | +0.42 | -1.39 to +2.23 |
| replay | -0.53 | -2.04 to +0.99 |
| wavg | +1.61 | +0.61 to +2.61 |
| lr_rewarm×replay | +1.13 | -1.78 to +4.05 |
| distill×lora (= freeze×wavg) | -3.81 | -7.20 to -0.41 |

### Interaction chains (exploratory; each estimate is the sum of its chain)

| Chain | Estimate | 95% CI |
|---|---|---|
| distill×freeze = lora×wavg | -1.80 | -2.94 to -0.67 |
| distill×lora = freeze×wavg = lr_rewarm×replay | -2.54 | -3.67 to -1.40 |
| distill×lr_rewarm = lora×replay | -0.74 | -1.87 to +0.40 |
| distill×replay = lora×lr_rewarm | -1.09 | -2.22 to +0.05 |
| distill×wavg = freeze×lora | -2.85 | -3.99 to -1.72 |
| freeze×lr_rewarm = replay×wavg | -0.74 | -1.81 to +0.33 |
| freeze×replay = lr_rewarm×wavg | -0.69 | -1.76 to +0.38 |

## Promotion (selection half)

| Rank | Config | Selection primary | Acq. ratio | Gate | Promoted |
|---|---|---|---|---|---|
| 1 | freeze+lr_rewarm+replay+wavg | -0.80 | 0.882 | pass | yes |
| 2 | freeze+wavg | -1.40 | 0.893 | pass | yes |
| 3 | distill+lora | +0.40 | 0.911 | fail |  |
| 4 | lora+lr_rewarm+wavg | +0.10 | 0.896 | fail |  |
| 5 | distill+lr_rewarm+wavg | +0.00 | 0.895 | fail |  |
| 6 | lora+replay+wavg | +0.00 | 0.894 | fail |  |
| 7 | distill+freeze+lr_rewarm | -0.30 | 0.894 | fail |  |
| 8 | distill+freeze+replay | -0.70 | 0.917 | fail |  |
| 9 | freeze+lora+lr_rewarm | -0.80 | 0.911 | fail |  |
| 10 | distill+lora+lr_rewarm+replay | -0.80 | 0.916 | fail |  |
| 11 | distill+replay+wavg | -0.90 | 0.923 | fail |  |
| 12 | distill+freeze+lora+lr_rewarm+replay+wavg | -1.20 | 0.936 | fail |  |
| 13 | freeze+lora+replay | -1.30 | 0.908 | fail |  |
| 14 | distill+freeze+lora+wavg | -1.30 | 0.930 | fail |  |

## Selection on replay (selection half, against the reference arm)

| Config | Δ vs reference | Seeds | Gate | Promoted |
|---|---|---|---|---|
| lr_rewarm+replay+select-act_nov-high | +0.60 | 1 | pass | yes |
| lr_rewarm+replay+select-act_nov-low | +0.30 | 1 | pass |  |
| lr_rewarm+replay+select-loss-high | +0.90 | 1 | pass | yes |
| lr_rewarm+replay+select-loss-low | +0.00 | 1 | pass |  |

## Confirmatory tests vs the reference arm (report half)

Lower-LR control (A13): `no-mitigation-lr3e-05` (frozen at LR 3e-05; the rule applied to these rows picks `no-mitigation-lr3e-05`)

| Config | Seeds | Δ | 95% CI | p (Holm) | Bootstrap | > 2σ | Acq. gate | Δ vs LR control | Beats LR control | Wins |
|---|---|---|---|---|---|---|---|---|---|---|
| freeze+lr_rewarm+replay+wavg | 3 | +2.13 | +0.80 to +3.43 | 0.006 | items | no | pass | +0.07 (-1.07 to +1.23) | no | **no** |
| freeze+wavg | 3 | +1.43 | -0.73 to +3.80 | 0.234 | items | no | pass | -0.63 (-2.83 to +1.67) | no | **no** |

### Selection family (Holm across signals)

| Config | Seeds | Δ | 95% CI | p (Holm) | > 2σ | Beats reference |
|---|---|---|---|---|---|---|
| lr_rewarm+replay+select-act_nov-high | 3 | +0.13 | -1.50 to +1.87 | 1.000 | no | **no** |
| lr_rewarm+replay+select-loss-high | 3 | -0.43 | -1.77 to +0.83 | 1.000 | no | **no** |

Sanity, reference vs no-mitigation: Δ = +2.83 (+0.67 to +4.73), items bootstrap over 3 seeds.

## Published claims (exploratory, report half)

| # | Claim | Arm vs comparator | Outcome | Δ (95% CI) | Seeds | Verdict | vs lower-LR control |
|---|---|---|---|---|---|---|---|
| 1 | Replay plus LR re-warm beats no mitigation | reference vs no-mitigation | retention | +2.83 (+0.67 to +4.73) | 3 | reproduced | -2.07 (-3.23 to -0.83) |
| 2 | Peak LR is the main knob; warmup barely matters | lr-3e-05 vs no-mitigation | retention | +4.90 (+2.83 to +6.87) | 3 | reproduced |  |
| 2 | Peak LR is the main knob; warmup barely matters | rewarm-only vs no-mitigation | retention | +1.80 (-0.40 to +3.90) | 1 | reported | -4.60 (-6.40 to -2.80) |
| 3 | Replay is worth about 1 to 3 points at 1B | replay-5 vs no-mitigation | retention | -0.03 (-1.53 to +1.50) | 3 | not reproduced (interval includes 0) | -4.93 (-6.40 to -3.47) |
| 3 | Replay is worth about 1 to 3 points at 1B | replay-25 vs no-mitigation | retention | -1.10 (-2.73 to +0.50) | 3 | not reproduced (interval includes 0) | -6.00 (-8.27 to -3.47) |
| 3 | Replay is worth about 1 to 3 points at 1B | replay-25 vs replay-5 | retention | -1.07 (-2.87 to +0.90) | 3 | not reproduced (interval includes 0) | -6.00 (-8.27 to -3.47) |
| 5 | LoRA forgets less and learns less; higher rank learns more | lora-r8 vs no-mitigation | retention | +4.67 (+2.37 to +6.93) | 3 | reproduced | -0.23 (-1.33 to +0.90) |
| 5 | LoRA forgets less and learns less; higher rank learns more | lora-r8 vs no-mitigation | acq_legal | +0.014 (+0.011 to +0.017) | 3 | reproduced |  |
| 5 | LoRA forgets less and learns less; higher rank learns more | lora-r256 vs lora-r8 | acq_legal | +0.170 (+0.148 to +0.192) | 3 | opposite |  |
| 6 | Distilling on replay tokens beats plain replay | replay+distill-on-replay vs replay-5 | retention | -0.10 (-1.07 to +0.87) | 3 | not reproduced (interval includes 0) | -5.03 (-6.47 to -3.60) |
| 7 | Freezing the top half beats freezing the bottom half | freeze-top vs freeze-bottom | retention | -0.30 (-2.20 to +1.80) | 3 | not reproduced (interval includes 0) | -2.80 (-4.37 to -1.27) |

Claim 4: not tested: no Reptile plugin

Claim 8: selection family (PREREG 6.3b)

Claim 9: screen main effects of wavg and replay (exploratory)

Pareto frontier (selection): distill+lora, lora+lr_rewarm+wavg, distill+lr_rewarm+wavg, lora+replay+wavg, freeze+lr_rewarm+replay+wavg, no-mitigation-lr3e-05, lr_rewarm

## Answer collapse (PREREG 8: flagged, kept; sensitivity without the collapsed task)

Majority-class rate: pubmedqa_pqal 0.58

| Run | Collapsed | selection primary | selection without | report primary | report without |
|---|---|---|---|---|---|
| freeze-top-s0 | pubmedqa_pqal | -4.10 | -4.53 | -4.00 | -4.80 |
| freeze-top-s1 | pubmedqa_pqal | -3.20 | -3.73 | -3.80 | -4.00 |
