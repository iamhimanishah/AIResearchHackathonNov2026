# Base-model eval baseline: protected-v1

model: google/gemma-3-1b-pt, device: mps

| task | tier | shots | metric | score | ±se | chance | majority | margin | n | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| arc_easy | 1 | 0 | acc_norm | 0.708 | 0.020 | 0.250 | 0.276 | +0.458 | 500 | 1.0 min |
| hellaswag | 1 | 0 | acc_norm | 0.636 | 0.022 | 0.250 | 0.272 | +0.386 | 500 | 2.9 min |
| piqa | 1 | 0 | acc_norm | 0.736 | 0.020 | 0.500 | 0.506 | +0.236 | 500 | 0.8 min |
| pubmedqa_pqal | 1 | 0 | acc | 0.644 | 0.021 | 0.333 | 0.580 | +0.311 | 500 | 3.2 min |
| mbpp | 2 | 3 | pass_at_1 | 0.105 | 0.022 | 0.000 | - | +0.105 | 200 | 6.3 min |
