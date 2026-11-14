# Base-model eval baseline: candidates

model: google/gemma-3-1b-pt, device: mps

| task | tier | shots | metric | score | ±se | chance | majority | margin | n | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| arc_easy | 1 | 0 | acc_norm | 0.708 | 0.020 | 0.250 | 0.276 | +0.458 | 500 | 1.4 min |
| hellaswag | 1 | 0 | acc_norm | 0.634 | 0.022 | 0.250 | 0.272 | +0.384 | 500 | 3.2 min |
| piqa | 1 | 0 | acc_norm | 0.740 | 0.020 | 0.500 | 0.506 | +0.240 | 500 | 1.0 min |
| pubmedqa_pqal | 1 | 0 | acc | 0.644 | 0.021 | 0.333 | 0.580 | +0.311 | 500 | 3.0 min |
| casehold | 1 | 0 | acc_norm | 0.210 | 0.018 | 0.200 | 0.236 | +0.010 | 500 | 15.3 min |
| gsm8k | 2 | 5 | exact_match/strict-match | 0.005 | 0.005 | 0.000 | - | +0.005 | 200 | 13.6 min |
| mbpp | 2 | 3 | pass_at_1 | 0.115 | 0.023 | 0.000 | - | +0.115 | 200 | 10.0 min |
| humaneval | 2 | 0 | pass@1/create_test | 0.061 | 0.019 | 0.000 | - | +0.061 | 164 | 31.3 min |
| mmlu (57 subjects) | 1 | 0 | acc | 0.259 | 0.014 | 0.250 | 0.394 | +0.009 | 1026 | 6.6 min |
