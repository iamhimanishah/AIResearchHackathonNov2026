# Grid analysis

Halves read: selection

Rows: {'total': 16, 'finished': 16, 'failed': 0, 'invalid': 0, 'duplicates_dropped': 0, 'excluded_backend': 0}

Noise floor: `no-mitigation` over 5 seeds, σ_noise = 1.15 points (frozen), measured σ(selection) = 0.59 points

## Retention effects (selection half, points)

| Effect | Estimate | 95% CI |
|---|---|---|
| (baseline) | -5.72 | -6.19 to -5.25 |
| lr_rewarm | +2.32 | +1.18 to +3.46 |
| replay | +0.02 | -1.12 to +1.16 |
| lr_rewarm×replay | +0.01 | -1.64 to +1.67 |

Not estimable from these rows: distill×lora, freeze×wavg


## Promotion (selection half)

| Rank | Config | Selection primary | Acq. ratio | Gate | Promoted |
|---|---|---|---|---|---|

Pareto frontier (selection): no-mitigation-lr3e-05, lr_rewarm
