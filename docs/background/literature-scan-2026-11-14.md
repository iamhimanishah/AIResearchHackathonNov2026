# Literature scan, 14 November 2026

Two searches run during the hackathon to guide the grid: one with ordinary web search, one with
the Exa API (23 calls; the queries are listed at the end). An agent read the returned text; a
human has not re-read the papers. Items the agent could not confirm in the text are marked
unverified. Our own calibration numbers are in [`../TASKS.md`](../TASKS.md), card A5.

## What it means for the plan

1. **Peak learning rate is the main knob.** Ibrahim et al. found warmup length made no
   measurable difference while halving the peak forgot less. Our own calibration agrees: from
   1e-4 to 3e-5, forgetting fell from about 6 points to about 1 with the same acquisition. So
   the grid needs no-mitigation arms at lower learning rates as controls.
2. **Compare at matched acquisition.** Report "learned loss" (each domain right after training
   on it) and "retained loss" (all domains at the end), and compare forgetting only among arms
   within a tolerance of the reference arm's learned loss.
3. **Pair seeds with the reference arm** (same seed and data order) and eliminate on the paired
   difference. The irace racing tool runs no elimination before 5 instances by default and
   suggests 2 with a t-test for machine-learning tuning, so two paired seeds is a defensible
   minimum before dropping a configuration.
4. **Screen wide, keep the top third, hold back about 40% of the budget for replication.** This
   is the search agent's recommendation, not a sourced rule.

## Starting values per technique

| Technique | Starting point | Source |
|---|---|---|
| Replay | 5% for a weak shift, 25% for a strong one; 50% slowed adaptation | [Ibrahim et al.](https://arxiv.org/abs/2403.08763), [Revisiting Replay and Gradient Alignment](https://arxiv.org/abs/2508.01908) |
| LR re-warm | 1% warmup, decay floor 0.1 × peak, optimizer state reset at each boundary | [Ibrahim et al.](https://arxiv.org/abs/2403.08763) |
| LoRA | Rank 256, alpha 2r, all attention and MLP modules, LR about 10× full fine-tuning; rank 16 to 64 under-learns | [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) |
| Distillation | Weight 1, temperature 0.5 to 1 | [arXiv 2604.15574](https://arxiv.org/abs/2604.15574), [Beyond One-Hot Replay](https://aclanthology.org/2026.findings-emnlp.1299.pdf) |
| Top-k distillation | Fewer than about 25 logits was worse than no distillation; about 300 approaches the full distribution | [Sparse Logit Sampling](https://doi.org/10.18653/v1/2025.acl-long.885) |
| Weight averaging | 0.5 toward base every 25% of steps roughly matched 10% replay on one model, not on another | [Soup to go](https://arxiv.org/abs/2501.05559) |
| Layer freezing | Freezing the top half beat full training on Qwen3-0.6B (29.4 vs 26.6); freezing the bottom half was worse. On Gemma 3 4B, freezing the middle layers kept comprehension | [Freeze Deep, Train Shallow](https://arxiv.org/html/2605.11416), [arXiv 2606.00284](https://arxiv.org/abs/2606.00284) |

## Combinations

- **Replay plus distillation stack when the KL term is applied to the replay tokens** (next-token
  loss on domain text, KL to the frozen base on replay text): +3 to +5 points over plain replay
  on two models, about 20% more time per token.
  [Beyond One-Hot Replay](https://aclanthology.org/2026.findings-emnlp.1299.pdf)
- Matching the base model's last hidden state helped far more than logit KL in one study.
  [GeRe](https://arxiv.org/pdf/2508.04676)
- Replay plus Reptile-style averaging had the lowest forgetting at every model size.
  [Revisiting Replay and Gradient Alignment](https://arxiv.org/abs/2508.01908)
- Weight averaging and replay look like substitutes; nobody was found testing them together.
- **Not found anywhere:** replaying general pretraining-like text against replaying earlier
  domains, head to head. A cheap cell for our grid.

## Selection by a model-internal signal

The evidence leans against it beating random at our scale.

- Positive: hidden-state selection beat random by about 2 points, selecting from pools of
  millions. [arXiv 2503.01807](https://arxiv.org/abs/2503.01807)
- Negative: perplexity and gradient selection are "almost never compute-optimal".
  [arXiv 2410.16208](https://arxiv.org/abs/2410.16208)
- For replay: loss-based selection roughly matched random.
  [Replay on Demand](https://arxiv.org/html/2609.40089)
- Sparse-autoencoder features: only competitive with other selectors.
  [Follow the Features](https://arxiv.org/html/2611.04085)

## Candidate techniques not implemented

Each is cheap as a training-loop hook.

| Technique | Mechanism | Source |
|---|---|---|
| High-perplexity token masking | Drop the loss on tokens whose perplexity under the model exceeds 2.5 (about 20 to 24% of tokens). May suppress acquisition | [arXiv 2501.14315](https://arxiv.org/abs/2501.14315) |
| Reptile-style averaging | Every 500 batches, interpolate the weights toward their values 500 batches earlier (coefficient 0.1; check the direction) | [arXiv 2508.01908](https://arxiv.org/abs/2508.01908) |
| Periodic averaging toward the base | Every 25% of steps, reset to 0.5 × base + 0.5 × current | [Soup to go](https://arxiv.org/abs/2501.05559) |
| Loss-adaptive learning rate (FINCH) | Lower the rate on high-loss batches | [arXiv 2605.20005](https://arxiv.org/html/2605.20005v1) |
| Train-then-revert | At a domain boundary, reset the middle layers to base weights | [arXiv 2606.00284](https://arxiv.org/abs/2606.00284) |
| Distil on replay tokens | See "Combinations" | [Beyond One-Hot Replay](https://aclanthology.org/2026.findings-emnlp.1299.pdf) |

## Unverified

- That irace gives every configuration the same (instance, seed) pairs.
- Details of "Lazy Paired Hyper-Parameter Tuning" (IJCAI 2013).
- A "freeze top, train middle" recommendation seen only in search snippets.
- No primary source was opened for successive halving's replicate rule.

## Exa queries sent

1. ablation combining experience replay and knowledge distillation in continual pretraining of large language models
2. continual pretraining LLM replay source: general pretraining data versus previous domain data in the sequence, comparison of replay buffer composition
3. data selection for LLM fine-tuning using gradients or sparse autoencoder features or activations compared against random selection and loss/perplexity-based selection at matched compute
4. random selection is a strong baseline: data selection methods for language model training fail to beat random when accounting for selection cost
5. which layers to freeze during continual pretraining of a language model to reduce catastrophic forgetting: bottom layers, top layers, embeddings
6. top-k logit knowledge distillation for language models with large vocabulary: sparse or truncated teacher logits versus full distribution
7. irace package iterated racing user guide firstTest number of instances before first elimination test Friedman paired t-test
8. successive halving Hyperband under noisy evaluations: number of seeds or replicates per rung before elimination, paired comparison common random numbers
9. sequential domain continual pre-training of LLM: replaying previous domains data versus general pretraining corpus, which mitigates forgetting better
10. simple new method to mitigate catastrophic forgetting in LLM continued pretraining or fine-tuning 2026: regularization, weight interpolation, token-level loss reweighting
11. experience replay of earlier tasks versus replay of generic pretraining data in LLM continual fine-tuning ablation: mixing both sources
