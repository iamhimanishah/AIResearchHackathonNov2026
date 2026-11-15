# Forgetting structure (exploratory, selection half, stream code:10000000+legal:10000000)

## Repeatability: do seeds lose the same questions?
AUC of 'lost by k other seeds' for a held-out seed's losses; 0.5 = random.

arm | seeds | known | loss rate | AUC | lost by every seed (if random)
---|---|---|---|---|---
lora | 3 | 683 | 5% | 0.84 | 13 (0.1)
lr_rewarm | 3 | 683 | 17% | 0.81 | 47 (3.3)
lr_rewarm+replay | 3 | 683 | 11% | 0.79 | 19 (0.8)
no-mitigation | 5 | 683 | 16% | 0.87 | 32 (0.1)
no-mitigation@lr3e-05 | 3 | 683 | 7% | 0.85 | 21 (0.3)
no-mitigation@lr5e-05 | 3 | 683 | 9% | 0.82 | 24 (0.6)
no-mitigation@lr6e-05 | 3 | 683 | 11% | 0.82 | 28 (1.0)

## Dose or target: does an arm protect the fragile questions specifically?
Slope of the arm's losses on each question's fragility in the other arms (logit), against the slope expected if every arm only set a dose (simulated null, 95% band). Below the band = protects fragile questions beyond its dose; above = concentrates the damage on them. If the relative slope tracks the loss rate across the plain learning-rate arms, the additive model does not hold; then compare arms only at similar loss rates.

arm | seeds | loss rate | slope | if dose only | relative | reading
---|---|---|---|---|---|---
lora'' | 1 | 2% | 0.84 | 1.20 (0.89 to 1.83) | 0.70 | targeted
replay | 1 | 16% | 0.95 | 1.30 (1.11 to 1.49) | 0.73 | targeted
lr_rewarm+replay | 3 | 11% | 1.05 | 1.25 (1.13 to 1.38) | 0.84 | targeted
no-mitigation | 5 | 16% | 1.15 | 1.29 (1.19 to 1.40) | 0.89 | targeted
lr_rewarm | 3 | 17% | 1.23 | 1.29 (1.17 to 1.40) | 0.95 | dose only
lora | 3 | 5% | 1.18 | 1.22 (1.06 to 1.40) | 0.97 | dose only
lora' | 1 | 8% | 1.26 | 1.26 (1.03 to 1.52) | 1.00 | dose only
no-mitigation@lr1e-05 | 2 | 3% | 1.35 | 1.19 (0.98 to 1.53) | 1.14 | dose only
lr_rewarm' | 1 | 11% | 1.44 | 1.26 (1.06 to 1.51) | 1.15 | dose only
no-mitigation@lr6e-05 | 3 | 11% | 1.47 | 1.26 (1.14 to 1.39) | 1.17 | concentrating
no-mitigation@lr5e-05 | 3 | 9% | 1.66 | 1.26 (1.13 to 1.43) | 1.32 | concentrating
lr_rewarm@lr3e-05 | 2 | 5% | 1.64 | 1.24 (1.02 to 1.43) | 1.32 | concentrating
no-mitigation@lr3e-05 | 3 | 7% | 1.77 | 1.26 (1.12 to 1.46) | 1.41 | concentrating
freeze@lr3e-05 | 2 | 6% | 1.77 | 1.22 (1.04 to 1.39) | 1.45 | concentrating
agent_code@lr3e-05 | 2 | 6% | 2.08 | 1.26 (1.06 to 1.44) | 1.65 | concentrating

## When and how questions are lost

arm | lost | already lost after code | lost after code, back at end | same wrong option (chance) | base margin lost / kept
---|---|---|---|---|---
agent_code@lr3e-05 | 88 | 30% | 35% | 92% (49%) | -
freeze@lr3e-05 | 83 | 28% | 28% | 87% (46%) | -
lora | 107 | 33% | 34% | 96% (51%) | -
lora' | 54 | 30% | 36% | - | -
lora'' | 15 | 40% | 54% | - | -
lr_rewarm | 345 | 41% | 36% | 93% (48%) | -
lr_rewarm' | 77 | 38% | 40% | - | -
lr_rewarm+replay | 220 | 41% | 43% | 94% (49%) | -
lr_rewarm@lr3e-05 | 72 | 33% | 38% | 100% (52%) | -
no-mitigation | 553 | 39% | 36% | 91% (45%) | -
no-mitigation@lr1e-05 | 45 | 13% | 67% | 100% (50%) | -
no-mitigation@lr3e-05 | 149 | 34% | 26% | 93% (48%) | -
no-mitigation@lr5e-05 | 194 | 29% | 39% | 95% (47%) | -
no-mitigation@lr6e-05 | 235 | 32% | 34% | 94% (47%) | -
replay | 109 | 50% | 35% | - | -
