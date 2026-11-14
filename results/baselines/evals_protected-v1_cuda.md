# Base-model eval baseline: protected-v1

model: google/gemma-3-1b-pt, device: cuda

| task | tier | shots | metric | score | ±se | chance | majority | margin | n | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| arc_easy | 1 | 0 | acc_norm | 0.706 | 0.020 | 0.250 | 0.276 | +0.456 | 500 | 0.3 min |
| hellaswag | 1 | 0 | acc_norm | 0.634 | 0.022 | 0.250 | 0.272 | +0.384 | 500 | 0.3 min |
| piqa | 1 | 0 | acc_norm | 0.738 | 0.020 | 0.500 | 0.506 | +0.238 | 500 | 0.1 min |
| pubmedqa_pqal | 1 | 0 | acc | 0.642 | 0.021 | 0.333 | 0.580 | +0.309 | 500 | 0.2 min |
| mbpp | 2 | 3 | pass_at_1 | 0.110 | 0.022 | 0.000 | - | +0.110 | 200 | 2.6 min |
