# Proposal: test published claims on the frozen bench

**Adopted** on 2026-11-14 as the pre-registered published-claims family: see
[`analysis/PREREG.md` section 6.5](../analysis/PREREG.md) for the arms, tests and the meaning
of "reproduced". Claim 4 (Reptile) is not tested unless a plugin exists before the freeze.

For Member C (design and analysis). Written by Member A's agent on 14 November 2026. It is a
proposal, not a decision: the pre-registration is Member C's, and the technique parameters it
freezes can only change by amendment before the grid launches.

## The idea

Papers on forgetting each use a different model, stream and budget, which is why some of their
findings conflict. Our bench holds all of that constant. So besides ranking techniques, the
study can report a second table: one row per published claim, with the source, whether it
reproduced on our bench, and our measured effect with uncertainty. A claim that fails to
reproduce is as useful as one that does.

This gives the pitch an anchor in other people's results. What can be compared is the direction
and rough size of an effect. Absolute scores cannot: the papers use models, token budgets and
learning rates far from ours.

## Why it is feasible: the noise floor

Five no-mitigation seeds on the frozen bench (`results/rows.noise.jsonl`):

| Measure | Mean | SD across seeds |
|---|---|---|
| MC mean, all items | −6.08 points | 0.78 |
| MC mean, selection half | −6.50 points | 1.06 |
| MC mean, report half | −5.66 points | 0.68 |
| Legal perplexity ratio | 0.787 | 0.002 |

Per task (all items): ARC-Easy SD 1.0, HellaSwag 1.9, PIQA 0.5, PubMedQA 2.8.

From the all-items SD, the difference between two arms has a standard deviation of about 1.1
points at one seed each and about 0.64 at three seeds each. So an effect of roughly 1.3 points is
detectable at two standard deviations with three seeds. The published effects below are 1 to 5
points. Pairing seeds with the reference arm may tighten this; that is not measured yet.

## Claims

"In the screen" refers to the 22-run design in `clharness/analysis/design.py` (pull request 4).

| # | Published claim | Source and its number | Arms that test it here | In the screen? |
|---|---|---|---|---|
| 1 | Replay plus LR re-warm and re-decay beats plain continued pretraining | [Ibrahim et al. 2024](https://arxiv.org/abs/2403.08763), 405M and 10B models; 1 to 10% replay costs little adaptation, 50% slows it | reference arm vs no-mitigation | Yes |
| 2 | Peak learning rate is the main knob; warmup length barely matters | Same paper | no-mitigation at 1e-4, 5e-5, 3e-5; re-warm alone | Yes (LR controls, halves) |
| 3 | Replay is worth about 1 to 3 points at 1B scale | [Abbes et al. 2026](https://arxiv.org/html/2508.01908), 1B model, mean of HellaSwag, PIQA, PubMedQA: sequential 60.7, 25% replay 61.5, 50% replay 64.0 | replay alone vs no-mitigation | Yes, at 5% replay only |
| 4 | Reptile-style averaging stacks with replay | Same paper: 50% replay with Reptile had the lowest forgetting at every size | replay vs replay plus Reptile | No: not implemented |
| 5 | LoRA forgets less and learns less than full fine-tuning; higher rank learns more | [Biderman et al. 2024](https://arxiv.org/html/2405.09673), Llama-2-7B, code: forgetting score 0.545 full vs 0.617 LoRA rank 256 | LoRA vs no-mitigation, on both retention and legal perplexity | Partly: rank 8 only. The paper says ranks 16 to 64 under-learn |
| 6 | Distilling on the replay tokens beats plain replay | [Beyond One-Hot Replay, 2026](https://aclanthology.org/2026.findings-emnlp.1299.pdf), SmolLM3-3B: +4.3 PubMedQA, +1.2 general over matched replay | replay vs replay plus `distill` with `rows: replay` | No: the option was added to `distill` after the design was written |
| 7 | Freezing the top half beats freezing the bottom half | [Freeze Deep, Train Shallow, 2026](https://arxiv.org/html/2605.11416), Qwen3-0.6B: full 26.6, freeze bottom 25.4, freeze top 29.4 | `freeze` with `top_layers: 13` vs `bottom_layers: 13` | No: bottom half only |
| 8 | Selecting examples by loss or a model-internal signal rarely beats random | [arXiv 2410.16208](https://arxiv.org/abs/2410.16208), [Replay on Demand](https://arxiv.org/html/2609.40089) | the four selection arms vs the reference arm | Yes |
| 9 | Periodic averaging toward the base roughly matches a small replay buffer | [Soup to go, 2025](https://arxiv.org/abs/2501.05559): matched 10% replay on Llama 2, not on Qwen2.5-1.5B | `wavg` vs replay | Yes |

How well each source was read: claims 3 and 5 were checked against the paper text by Member A's
agent. The others come from search agents that read excerpts; nobody has read those papers in
full. See [`background/literature-scan-2026-11-14.md`](background/literature-scan-2026-11-14.md).

## What it would add to the plan

Five claims are covered by runs already in the screen. The other four need added arms:

| Added arm | Needs | Owner |
|---|---|---|
| Replay plus `distill` with `rows: replay` (claim 6) | Nothing new: Peter's option is on main, and its sanity run finished | C to add; B to confirm parameters |
| LoRA at a high rank, for example 256 (claim 5) | A parameter change, and a check that it fits in memory | B |
| `freeze` with `top_layers: 13` (claim 7) | A parameter change | B |
| Replay at 25% (claim 3, the paper's own level) | A parameter change | C |
| Replay plus Reptile averaging (claim 4) | A new plugin: one weight operation every few hundred steps | B, if there is time |

At three seeds each, five added arms are 15 runs: about 5 GPU-hours, or about $18 at the $1.20
per run measured today.

Two further additions, both for finalists only so the frozen bench is not touched:

- **Report-only evals.** Add WinoGrande and ARC-Challenge, the forgetting evals Biderman et al.
  use besides HellaSwag. Member A can add them as a second eval suite that the grid does not use.
- **A frontier figure in that paper's format:** forgetting against learning, one point per
  configuration, with the lower-learning-rate controls drawn as the curve to beat.

## What "reproduced" should mean

This is for the pre-registration to fix before any of these runs are read. A suggestion:

- A directional claim reproduces if the effect has the published sign and its interval excludes
  zero, on the report half.
- A magnitude claim also reports our effect next to the published one, with no pass or fail.
- Every claim is judged at matched acquisition, or the acquisition change is reported beside it.
  Claim 5 is the case where this matters most.
- A claim tested with different parameters from the paper's says so in its row.

## Cautions

- Our learning rate is harsh and our budget is tiny (20M tokens against billions). A failure to
  reproduce may reflect that, not a fault in the paper.
- The lower-learning-rate controls already remove most forgetting on this bench (about 1 point
  left at 3e-5, from one seed). A technique that "reproduces" against the 1e-4 run may still lose
  to simply lowering the learning rate. Each row should say which.
- The reference-arm runs on the frozen bench had not finished when this was written.
