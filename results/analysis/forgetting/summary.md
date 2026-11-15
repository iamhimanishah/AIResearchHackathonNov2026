# Forgetting structure (exploratory, selection half, stream code:10000000+legal:10000000)

Built from: `results/agent/explore-justin/rows.jsonl`, `results/agent/rows.jsonl`, `results/rows.calibration.jsonl`, `results/rows.explore.jsonl`, `results/rows.noise.jsonl`, `results/rows.phase0.jsonl`, `results/rows.phase0.remeasured.jsonl`, `results/rows.plumbing.jsonl`, `results/rows.reference.jsonl`, `results/rows.reference.remeasured.jsonl`, `results/rows.sanity.jsonl` (39 arms with item files). Rebuild: `clh forget-sets results/agent/explore-justin/rows.jsonl results/agent/rows.jsonl results/rows.calibration.jsonl results/rows.explore.jsonl results/rows.noise.jsonl results/rows.phase0.jsonl results/rows.phase0.remeasured.jsonl results/rows.plumbing.jsonl results/rows.reference.jsonl results/rows.reference.remeasured.jsonl results/rows.sanity.jsonl --out results/analysis/forgetting`

## Repeatability: do seeds lose the same questions?
AUC of 'lost by k other seeds' for a held-out seed's losses; 0.5 = random.

arm | seeds | known | loss rate | AUC | lost by every seed (if random)
---|---|---|---|---|---
distill+replay | 3 | 683 | 16% | 0.78 | 42 (3.1)
freeze+lr_rewarm+replay+wavg | 3 | 683 | 5% | 0.88 | 14 (0.1)
freeze+wavg | 3 | 683 | 6% | 0.84 | 16 (0.2)
freeze[freeze.bottom_layers=0,freeze.top_layers=13] | 3 | 683 | 13% | 0.87 | 48 (1.4)
freeze[freeze.bottom_layers=13] | 3 | 683 | 9% | 0.77 | 14 (0.5)
lora[lora.lr=0.0001,lora.rank=256] | 3 | 683 | 5% | 0.84 | 13 (0.1)
lora[lora.rank=256] | 3 | 683 | 34% | 0.79 | 113 (26.9)
lora[lora.rank=8] | 3 | 683 | 6% | 0.84 | 21 (0.1)
lr_rewarm+replay | 3 | 683 | 11% | 0.79 | 19 (0.8)
lr_rewarm+replay[replay.keep=high,replay.layer=13,replay.pool=4,replay.select=act_nov] | 3 | 683 | 11% | 0.79 | 22 (0.8)
lr_rewarm+replay[replay.keep=high,replay.pool=4,replay.select=loss] | 3 | 683 | 10% | 0.78 | 18 (0.6)
lr_rewarm[lr_rewarm.schedule=constant] | 3 | 683 | 17% | 0.81 | 47 (3.3)
no-mitigation | 5 | 683 | 16% | 0.87 | 32 (0.1)
no-mitigation@lr3e-05 | 3 | 683 | 7% | 0.85 | 21 (0.3)
no-mitigation@lr5e-05 | 3 | 683 | 9% | 0.82 | 24 (0.6)
no-mitigation@lr6e-05 | 3 | 683 | 11% | 0.82 | 28 (1.0)
replay[replay.share=0.05] | 3 | 683 | 16% | 0.76 | 34 (2.8)
replay[replay.share=0.25] | 3 | 683 | 17% | 0.81 | 45 (3.6)

## Dose or target: does an arm protect the fragile questions specifically?
Slope of the arm's losses on each question's fragility in the other arms (logit), against the slope expected if every arm only set a dose (simulated null, 95% band). Below the band = protects fragile questions beyond its dose; above = concentrates the damage on them. If the relative slope tracks the loss rate across the plain learning-rate arms, the additive model does not hold; then compare arms only at similar loss rates.
**The reading depends on the pool.** A question's fragility is its loss rate in the OTHER arms listed here (39 arms), so adding or removing arms can move a slope across the band (rebuilt with more arms, no-mitigation and the reference arm moved from targeted to dose only). Quote a reading together with its pool, never as a property of the arm alone.

arm | seeds | loss rate | slope | if dose only | relative | reading
---|---|---|---|---|---|---
lora[lora.rank=256] | 3 | 34% | 0.61 | 1.16 (1.08 to 1.30) | 0.53 | targeted
distill+lora | 1 | 4% | 0.88 | 1.26 (0.94 to 1.59) | 0.70 | targeted
freeze[freeze.bottom_layers=13] | 3 | 9% | 0.93 | 1.24 (1.10 to 1.42) | 0.75 | targeted
replay[replay.share=0.25] | 3 | 17% | 0.97 | 1.23 (1.10 to 1.35) | 0.79 | targeted
freeze+lora+replay | 1 | 5% | 1.00 | 1.25 (1.01 to 1.63) | 0.80 | targeted
distill+freeze+lora+wavg | 1 | 3% | 1.03 | 1.24 (0.91 to 1.59) | 0.83 | dose only
freeze[freeze.bottom_layers=0,freeze.top_layers=13] | 3 | 13% | 1.05 | 1.23 (1.09 to 1.39) | 0.85 | targeted
freeze+wavg | 3 | 6% | 1.08 | 1.23 (1.09 to 1.45) | 0.87 | targeted
freeze+lr_rewarm+replay+wavg | 3 | 5% | 1.11 | 1.22 (1.07 to 1.43) | 0.91 | dose only
replay[replay.share=0.05] | 3 | 16% | 1.13 | 1.22 (1.09 to 1.32) | 0.92 | dose only
distill+replay | 3 | 16% | 1.19 | 1.22 (1.09 to 1.33) | 0.98 | dose only
distill+replay+wavg | 1 | 8% | 1.22 | 1.22 (0.99 to 1.53) | 1.00 | dose only
distill+freeze+lr_rewarm | 1 | 4% | 1.22 | 1.21 (0.94 to 1.76) | 1.00 | dose only
lora[lora.lr=0.0001,lora.rank=8] | 1 | 2% | 1.24 | 1.24 (0.84 to 2.03) | 1.00 | dose only
no-mitigation | 5 | 16% | 1.25 | 1.23 (1.13 to 1.33) | 1.01 | dose only
freeze+lora+lr_rewarm | 1 | 4% | 1.25 | 1.23 (0.97 to 1.62) | 1.02 | dose only
distill+freeze+replay | 1 | 6% | 1.28 | 1.23 (0.93 to 1.51) | 1.04 | dose only
lr_rewarm[lr_rewarm.schedule=constant] | 3 | 17% | 1.29 | 1.21 (1.11 to 1.36) | 1.06 | dose only
distill+freeze+lora+lr_rewarm+replay+wavg | 1 | 2% | 1.32 | 1.25 (0.91 to 1.97) | 1.06 | dose only
lr_rewarm+replay[replay.keep=high,replay.layer=13,replay.pool=4,replay.select=act_nov] | 3 | 11% | 1.34 | 1.22 (1.10 to 1.39) | 1.10 | dose only
lr_rewarm+replay | 3 | 11% | 1.35 | 1.23 (1.09 to 1.39) | 1.10 | dose only
lr_rewarm+replay[replay.keep=high,replay.pool=4,replay.select=loss] | 3 | 10% | 1.38 | 1.23 (1.10 to 1.41) | 1.12 | dose only
lr_rewarm+replay[replay.keep=low,replay.layer=13,replay.pool=4,replay.select=act_nov] | 1 | 10% | 1.37 | 1.21 (1.01 to 1.56) | 1.13 | dose only
lr_rewarm+replay[replay.keep=low,replay.pool=4,replay.select=loss] | 1 | 10% | 1.39 | 1.23 (1.00 to 1.49) | 1.13 | dose only
lora[lora.lr=0.0002,lora.rank=256] | 1 | 8% | 1.42 | 1.23 (0.99 to 1.53) | 1.16 | dose only
lora[lora.rank=8] | 3 | 6% | 1.45 | 1.22 (1.07 to 1.43) | 1.19 | concentrating
distill+lora+lr_rewarm+replay | 1 | 3% | 1.53 | 1.20 (0.89 to 1.62) | 1.28 | dose only
lora[lora.lr=0.0001,lora.rank=256] | 3 | 5% | 1.54 | 1.20 (1.05 to 1.43) | 1.28 | concentrating
no-mitigation@lr6e-05 | 3 | 11% | 1.57 | 1.23 (1.12 to 1.35) | 1.28 | concentrating
lora+replay+wavg | 1 | 3% | 1.58 | 1.23 (0.96 to 1.87) | 1.29 | dose only
lr_rewarm[lr_rewarm.schedule=cosine] | 1 | 11% | 1.63 | 1.24 (1.04 to 1.53) | 1.32 | concentrating
distill+lr_rewarm+wavg | 1 | 4% | 1.65 | 1.22 (0.97 to 1.59) | 1.35 | concentrating
no-mitigation@lr5e-05 | 3 | 9% | 1.75 | 1.23 (1.06 to 1.39) | 1.42 | concentrating
lr_rewarm@lr3e-05 | 2 | 5% | 1.81 | 1.24 (1.05 to 1.44) | 1.46 | concentrating
no-mitigation@lr1e-05 | 2 | 3% | 1.88 | 1.23 (0.95 to 1.65) | 1.52 | concentrating
no-mitigation@lr3e-05 | 3 | 7% | 1.90 | 1.23 (1.06 to 1.43) | 1.54 | concentrating
lora+lr_rewarm+wavg | 1 | 3% | 1.98 | 1.22 (0.85 to 1.60) | 1.62 | concentrating
freeze@lr3e-05 | 2 | 6% | 2.24 | 1.25 (1.05 to 1.43) | 1.79 | concentrating
agent_code@lr3e-05 | 2 | 6% | 2.18 | 1.19 (1.00 to 1.41) | 1.83 | concentrating

## When and how questions are lost

arm | lost | already lost after code | lost after code, back at end | same wrong option (chance) | base margin lost / kept
---|---|---|---|---|---
agent_code@lr3e-05 | 88 | 30% | 35% | 92% (49%) | -
distill+freeze+lora+lr_rewarm+replay+wavg | 17 | 29% | 17% | - | -
distill+freeze+lora+wavg | 23 | 30% | 12% | - | -
distill+freeze+lr_rewarm | 28 | 50% | 48% | - | -
distill+freeze+replay | 38 | 47% | 61% | - | -
distill+lora | 28 | 21% | 54% | - | -
distill+lora+lr_rewarm+replay | 20 | 15% | 57% | - | -
distill+lr_rewarm+wavg | 29 | 38% | 56% | - | -
distill+replay | 338 | 42% | 35% | 92% (46%) | -
distill+replay+wavg | 53 | 45% | 47% | - | -
freeze+lora+lr_rewarm | 27 | 30% | 27% | - | -
freeze+lora+replay | 37 | 32% | 29% | - | -
freeze+lr_rewarm+replay+wavg | 95 | 59% | 38% | 98% (52%) | -
freeze+wavg | 125 | 50% | 40% | 97% (47%) | -
freeze@lr3e-05 | 83 | 28% | 28% | 87% (46%) | -
freeze[freeze.bottom_layers=0,freeze.top_layers=13] | 261 | 45% | 25% | 96% (46%) | -
freeze[freeze.bottom_layers=13] | 179 | 43% | 43% | 97% (49%) | -
lora+lr_rewarm+wavg | 21 | 48% | 50% | - | -
lora+replay+wavg | 20 | 45% | 44% | - | -
lora[lora.lr=0.0001,lora.rank=256] | 107 | 33% | 34% | 96% (51%) | -
lora[lora.lr=0.0001,lora.rank=8] | 15 | 40% | 54% | - | -
lora[lora.lr=0.0002,lora.rank=256] | 54 | 30% | 36% | - | -
lora[lora.rank=256] | 697 | 41% | 30% | 88% (46%) | -
lora[lora.rank=8] | 123 | 37% | 39% | 97% (48%) | -
lr_rewarm+replay | 220 | 41% | 43% | 94% (49%) | -
lr_rewarm+replay[replay.keep=high,replay.layer=13,replay.pool=4,replay.select=act_nov] | 218 | 46% | 36% | 92% (49%) | -
lr_rewarm+replay[replay.keep=high,replay.pool=4,replay.select=loss] | 199 | 45% | 46% | 91% (46%) | -
lr_rewarm+replay[replay.keep=low,replay.layer=13,replay.pool=4,replay.select=act_nov] | 68 | 50% | 45% | - | -
lr_rewarm+replay[replay.keep=low,replay.pool=4,replay.select=loss] | 71 | 46% | 46% | - | -
lr_rewarm@lr3e-05 | 72 | 33% | 38% | 100% (52%) | -
lr_rewarm[lr_rewarm.schedule=constant] | 345 | 41% | 36% | 93% (48%) | -
lr_rewarm[lr_rewarm.schedule=cosine] | 77 | 38% | 40% | - | -
no-mitigation | 553 | 39% | 36% | 91% (45%) | -
no-mitigation@lr1e-05 | 45 | 13% | 67% | 100% (50%) | -
no-mitigation@lr3e-05 | 149 | 34% | 26% | 93% (48%) | -
no-mitigation@lr5e-05 | 194 | 29% | 39% | 95% (47%) | -
no-mitigation@lr6e-05 | 235 | 32% | 34% | 94% (47%) | -
replay[replay.share=0.05] | 329 | 41% | 38% | 93% (45%) | -
replay[replay.share=0.25] | 358 | 45% | 37% | 91% (47%) | -
