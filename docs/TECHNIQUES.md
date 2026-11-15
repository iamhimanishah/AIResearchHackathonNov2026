# Techniques: design and status

Workstream 3 (owner: peter). How a forgetting-reduction technique plugs into the harness, how it
meets the other workstreams, and what is still undecided. Task cards are in
[`TASKS.md`](TASKS.md) under "Member B".

## The path of one run

```
RunSpec (JSON)                       src/clharness/spec.py            workstream 1
  name, seed, bench, stream, optim, techniques: [{name, params}]
        |
        v  config_from_spec          src/clharness/executor/run.py    workstream 2
RunConfig = frozen bench config + the spec's overrides
        |
        v  build_hooks(techniques)   src/clharness/techniques/        workstream 3
Trainer(cfg, measurement hooks + technique hooks)
        |
        v  measure_run               src/clharness/executor/rows.py   workstream 2
ResultRow -> analysis (4), agent and page (5)
```

The training loop never names a technique. The executor, the analysis and the agent know
techniques only as `{name, params}` entries in a spec. Everything a technique does happens
inside its own module, through hook points the loop already has.

## What a technique is

One module in `src/clharness/techniques/`, holding one `BaseHook` subclass with:

- `name`: the string a spec uses to switch it on.
- `Params`: a pydantic model of its parameters. A wrong type, an out-of-range value or a
  misspelled key fails before the model loads, with a message that names the technique.
- `@register`: adds it to the registry. Every module in the package is imported on first
  lookup, so a new technique is one new file and no edits elsewhere.

`build_hooks(techniques)` validates and builds the hooks, always in name order, so the order
of a spec's list carries no meaning. A technique may also define `config_overrides(params)`, a
small dict merged into the run config by `apply_config_overrides(cfg)` before training. That is
how a technique that is really a setting, such as `lr_rewarm`, is still switched on by name.
`available()` returns each technique's name and the JSON Schema of its params, so the planning
agent can read what it may set without importing torch code by hand.

## Which hook point each technique uses

| Technique | Module | Hook point | Extra cost to report |
|---|---|---|---|
| `replay` | `replay.py` | `select_batch`: swap a share of rows for replay rows | none (rows are replaced, not added) |
| `replay` with `select: loss` or `act_nov` | `replay.py`, `selection.py` | as above, plus a forward-only pass that scores a pool of candidates | forward-only tokens: share x pool of the training tokens (20% at share 0.05, pool 4) |
| `wavg` | `wavg.py` | `on_domain_end`: pull weights toward base; EMA in `on_step_end` | none |
| `lora` | `lora.py` | `configure_model`: add adapters in place; `config_overrides`: learning rate | fewer trainable parameters, smaller update |
| `freeze` | `freeze.py` | `configure_model`: set `requires_grad=False` | fewer trainable parameters |
| `distill` | `distill.py` | `extra_loss`: KL to a frozen base copy, on all rows or (`rows: new | replay`) on some | forward-only tokens for the teacher, on the chosen rows only |
| `lr_rewarm` | `lr_rewarm.py` | `config_overrides`: switches on A4's per-domain schedule in `optim` | none |

The reference arm is `replay` plus `lr_rewarm`: `configs/bench/specs/reference-s0.json`, with
`replay-only-s0.json` and `rewarm-only-s0.json` to split the recipe into its two halves.

**Selection** is a parameter of `replay`, not a technique of its own. When replay needs k rows it
draws k x `pool` candidates from the source, scores them under the current model, and keeps k:

- `select: random` is plain replay and the control. No pool, no scoring, no extra cost.
- `select: loss` keeps by mean token loss.
- `select: act_nov` keeps by activation novelty at `layer`: how far the candidate's mean
  residual is from the running statistics of the sequences the model has been training on.
- `keep: high | low` chooses which end of the ranking is kept, so the grid can test both.

**Distilling on replayed rows** is `distill` with `rows: replay`, together with `replay`: the
model is free to move on new-domain text and is held to the base model's predictions on the
replayed text. The teacher runs only on those rows, so at a 5% share it costs 5% of a full
teacher pass. The term is weighted per token like the language-model loss, so a distilled token
is pulled as hard as under `rows: all` at the same `weight`, whatever the share. Sanity spec:
`configs/bench/specs/sanity/replay-distill-on-replay.json`.

The SAE-feature signal (`sae_rare`) is not built: it needs SAE weights on the GPU, and
activation-based selection is fourth in the cut order.

## Rules every technique follows

- **Gradient tokens match the no-mitigation run.** A technique may change which sequences are
  trained on, never how many tokens a step or a run spends.
- **What ran is recorded in full.** `apply_config_overrides` writes every parameter of every
  technique into the run config, defaults included, so `config.resolved.json` says exactly
  what ran even if a default changes later.
- **Extra compute is counted.** Forward-only passes (a scoring pass, a teacher pass) add their
  tokens to `ctx.state["extra_forward_tokens"]`; the executor copies that into
  `cost.extra_forward_tokens` (see decision 3).
- **Replayed or reordered rows stay labelled.** A hook that returns a new batch tensor sets
  `.chunk_ids` itself (-1 for rows that are not from the current domain's stream) and
  `.replay_mask`. Tensors drop Python attributes when sliced or cloned.
- **Anything that should be measured happens before the checkpoint.** `on_domain_end` runs
  before the boundary checkpoint; `on_run_end` runs after the last one and is never evaluated.
- **Seeded.** Any randomness comes from `cfg.train.seed`, so a spec plus a seed is one result.
- **Proven on the tiny model.** Each technique has a CPU test on the random tiny Gemma that
  checks the behaviour itself (the share replayed, the weights restored), not only that it ran.

## Status

| Piece | State |
|---|---|
| Registry, `RunConfig.techniques`, config overrides, `clh train` wiring | done, with tests |
| `replay` (B2), `wavg` (B6), `lr_rewarm` | done, with tiny-model tests and a GPU sanity run each |
| Selection (B3): `random`, `loss`, `act_nov` on `replay` | done, with tiny-model tests and GPU sanity runs; `sae_rare` not built |
| `lora`, `freeze` (B4), `distill` (B5) | done, with tiny-model tests and GPU sanity runs, including LoRA at rank 256 and freeze top half |
| Reference arm (B1) | run at seeds 0 to 2 on the frozen bench; see Results so far |
| Combinations | every run of the screening design trains on the tiny model (`tests/test_screen_combos.py`); on a GPU, the all-six-on arm and replay plus distil-on-replay have run for four steps |
| Sanity runs on a GPU | every row in `results/rows.sanity.jsonl` finished on a Modal H100; see Results so far (four steps per domain, so only status and cost mean anything) |

Sanity results will be added to this file from result rows only, never typed in.

## Results so far

A snapshot, written by a script that reads the row files and typed by nobody. Sources:
`results/rows.plumbing.jsonl`, `results/rows.sanity.jsonl`, `results/rows.phase0.remeasured.jsonl`,
`results/rows.phase0.jsonl`, `results/rows.reference.remeasured.jsonl`, `results/rows.explore.jsonl`
and the LoRA arms of `results/agent/rows.jsonl` and `results/agent/explore-peter/rows.jsonl`, as of
15 November.
The analysis behind `make reproduce-figures` is the authority; this table is for orientation.

### Sanity runs: does each technique run on the real model?

Four optimizer steps per domain on a Modal H100. Only status and cost mean anything here.

| Spec | Techniques | Status | Wall clock (s) | Est. cost ($) | Gradient tokens | Forward-only tokens | Trainable parameters |
|---|---|---|---|---|---|---|---|
| `plumbing-extra-evals` | none | finished | 413 | 0.45 | 524,288 | 0 | 999,885,952 |
| `sanity-distill` | distill | finished | 342 | 0.38 | 524,288 | 524,288 | 999,885,952 |
| `sanity-freeze` | freeze | finished | 309 | 0.34 | 524,288 | 0 | 348,948,608 |
| `sanity-freeze-top` | freeze | finished | 358 | 0.39 | 524,288 | 0 | 348,948,608 |
| `sanity-lora` | lora | finished | 390 | 0.43 | 524,288 | 0 | 6,522,880 |
| `sanity-lora-r256` | lora | finished | 464 | 0.51 | 524,288 | 0 | 208,732,160 |
| `sanity-lr-rewarm` | lr_rewarm | finished | 438 | 0.48 | 524,288 | 0 | 999,885,952 |
| `sanity-replay` | replay | finished | 323 | 0.35 | 524,288 | 0 | 999,885,952 |
| `sanity-replay-select-actnov` | replay | finished | 343 | 0.38 | 524,288 | 102,400 | 999,885,952 |
| `sanity-replay-select-loss` | replay | finished | 341 | 0.37 | 524,288 | 102,400 | 999,885,952 |
| `sanity-wavg` | wavg | finished | 297 | 0.33 | 524,288 | 0 | 999,885,952 |
| `sanity-replay-distill-on-replay` | distill, replay | finished | 315 | 0.35 | 524,288 | 25,600 | 999,885,952 |
| `sanity-replay-distill-topk256` | distill, replay | finished | 394 | 0.43 | 524,288 | 25,600 | 999,885,952 |
| `sanity-all-on` | distill, freeze, lora, lr_rewarm, replay, wavg | finished | 343 | 0.38 | 524,288 | 524,288 | 3,261,440 |

### First full runs on the frozen bench

Code then legal, 10M tokens each. Retention is the change in the multiple-choice mean against the base model, in points; perplexity ratios are trained / base at the end of the stream. Mean (SD across seeds).

| Arm | Seeds | Retention, report half | Retention, selection half | Legal perplexity ratio | General perplexity ratio |
|---|---|---|---|---|---|
| no mitigation, lr 1e-4 | 5 | -6.44 (1.15) | -5.72 (0.59) | 0.787 | 1.386 |
| replay, lr 1e-4 | 1 | -5.60 | -5.70 | 0.790 | 1.295 |
| lr_rewarm, lr 1e-4 | 1 | -4.90 | -3.40 | 0.738 | 1.183 |
| lr_rewarm+replay, lr 1e-4 | 3 | -3.13 (0.85) | -3.37 (0.38) | 0.744 | 1.120 |
| no mitigation, lr 5e-05 | 3 | -2.80 (0.46) | -2.43 (0.55) | 0.759 | 1.131 |
| no mitigation, lr 3e-05 | 3 | -1.07 (0.80) | -1.47 (0.15) | 0.754 | 1.055 |

What the full runs say so far, with replay alone and re-warm alone at one seed each:

- The reference arm keeps about 3 points more than no mitigation and learns legal slightly
  better, so it did not buy retention by learning less.
- Re-warm alone does most of that, and what it does is lower the average learning rate (see
  Exploratory runs, below). Replay alone is inside the seed noise on the multiple-choice mean,
  though it shows on general-text perplexity.
- Lowering the learning rate to 3e-5 with no technique forgets less than the reference arm at
  1e-4, at nearly the same legal perplexity. Any technique at 1e-4 has to beat that control.

### Exploratory runs

These runs are outside the pre-registered screen and claims families. Their specs are in
`configs/bench/specs/explore/` and carry `explore:` in `notes`. Their rows are in
`results/rows.explore.jsonl`, which the reported-rows list does not read. The LoRA arms at two
seeds were run by the research agent (instance `peter`) and are in
`results/agent/explore-peter/rows.jsonl`. The table sets them all beside the control arms and
the two LoRA arms of the claims family. It is printed by
`scripts/techniques_table.py`. Columns are as in the table above; the code column is code
perplexity at the end of the stream, after legal training, over the base model's.

| Arm | Seeds | Retention, report half | Retention, selection half | Legal perplexity ratio | Code perplexity ratio, end of stream | General perplexity ratio | Rows in |
|---|---|---|---|---|---|---|---|
| no mitigation, lr 0.0001 | 5 | -6.44 (1.15) | -5.72 (0.59) | 0.787 | 1.688 | 1.386 | `rows.phase0.remeasured.jsonl` |
| no mitigation, lr 6e-05 | 3 | -4.63 (1.20) | -3.30 (1.10) | 0.763 | 1.476 | 1.171 | `rows.explore.jsonl` |
| no mitigation, lr 5e-05 | 3 | -2.80 (0.46) | -2.43 (0.55) | 0.759 | 1.335 | 1.131 | `rows.phase0.jsonl`, `rows.phase0.remeasured.jsonl` |
| no mitigation, lr 3e-05 | 3 | -1.07 (0.80) | -1.47 (0.15) | 0.754 | 1.205 | 1.055 | `rows.phase0.jsonl`, `rows.phase0.remeasured.jsonl` |
| reference arm (lr_rewarm + replay), lr 0.0001 | 3 | -3.13 (0.85) | -3.37 (0.38) | 0.744 | 1.482 | 1.120 | `rows.phase0.jsonl`, `rows.reference.remeasured.jsonl` |
| lr_rewarm, warmup only (constant schedule), lr 0.0001 | 3 | -6.53 (1.11) | -6.23 (0.72) | 0.790 | 1.471 | 1.407 | `rows.explore.jsonl` |
| LoRA rank 8, lr 0.001 | 3 | -1.30 (1.05) | -1.00 (0.66) | 0.797 | 1.062 | 1.032 | `agent/rows.jsonl` |
| LoRA rank 8, lr 0.000565685 | 2 | -0.20 (0.57) | +0.35 (0.35) | 0.806 | 1.026 | 1.008 | `agent/explore-peter/rows.jsonl` |
| LoRA rank 8, lr 0.0001 | 3 | +0.70 (0.36) | +0.90 (0.26) | 0.860 | 1.012 | 0.993 | `rows.explore.jsonl` |
| LoRA rank 64, lr 0.0003 | 2 | -0.95 (0.64) | -0.50 (0.42) | 0.772 | 1.041 | 1.028 | `agent/explore-peter/rows.jsonl` |
| LoRA rank 64, lr 0.0001 | 3 | +0.77 (0.38) | +0.60 (0.26) | 0.806 | 1.011 | 1.002 | `agent/explore-peter/rows.jsonl`, `rows.explore.jsonl` |
| LoRA rank 256, lr 0.001 | 3 | -19.00 (1.00) | -17.77 (1.12) | 0.944 | 4.779 | 2.444 | `agent/rows.jsonl` |
| LoRA rank 256, lr 0.0002 | 1 | -1.50 | -2.30 | 0.757 | 1.096 | 1.083 | `rows.explore.jsonl` |
| LoRA rank 256, lr 0.00015 | 2 | -0.90 (0.14) | -0.70 (0.28) | 0.760 | 1.057 | 1.052 | `agent/explore-peter/rows.jsonl` |
| LoRA rank 256, lr 0.0001 | 3 | -0.27 (0.81) | -0.13 (1.04) | 0.767 | 1.032 | 1.028 | `rows.explore.jsonl` |

What they say. All of it is exploratory, and an arm with one seed is a lead, not a finding.

- **The reference arm's gain is mostly a lower average learning rate.** Its per-domain cosine
  schedule averages about 5.5e-5 over a run, against about 9.2e-5 for the bench's constant
  schedule. No mitigation at a constant 6e-5, which has the same average, matches the reference
  arm on the selection half. On the report half it is behind by less than three seeds can
  separate from noise. The reference arm keeps a small edge in legal perplexity.
- **Warmup alone does nothing for retention.** `lr_rewarm` with a constant schedule, which
  keeps the 8 warmup steps and drops the decay, matches no mitigation at 1e-4 on the
  multiple-choice mean and on legal perplexity. It does lose less code perplexity.
- **The LoRA collapse at rank 256 was the learning rate.** At the plugin's default of 1e-3,
  rank 256 diverges on three seeds. At 1e-4 and at 2e-4 it trains normally.
- **LoRA at rank 256 and the bench's own learning rate forgets almost nothing on three seeds and
  still learns legal.** Its legal and code perplexity at each domain's own boundary are inside
  the acquisition gate's margins against the reference arm, by this file's reading of the rule;
  the analysis is the authority. Because it runs at 1e-4, the comparison with no mitigation at
  1e-4 has no learning-rate confound. Against the strongest control, no mitigation at 3e-5, it
  is within seed noise on the report half and learns legal a little less well. It loses much
  less code perplexity and somewhat less general perplexity.
- **At a matched learning rate, the higher rank learns more, on three seeds each.** At 1e-4,
  legal perplexity improves from rank 8 to rank 64 to rank 256, the three seeds of each rank
  agree closely, and none of the three forgets much. The confirmatory analysis
  (`results/analysis/grid_analysis.md`, claim 5) marks "higher rank learns more" as opposite;
  its rank-256 arm ran at the default 1e-3 and diverged, so that verdict is about the learning
  rate.
- **Step size explains part of the rank gap, not all of it.** Rank 64 at 3e-4 comes close to
  rank 256 at 1e-4 on legal perplexity. Rank 8 does not: neither at its default 1e-3 nor at
  the rate that matches rank 256 on `lr` x sqrt(`rank`). These are the research agent's arms,
  two seeds each.
- **At rank 256, forgetting rises with the learning rate well before training fails.** From
  1e-4 through 1.5e-4 to 2e-4, retention falls step by step while legal perplexity improves
  only slightly.
- **"Learns the new domain" here means legal.** For code, perplexity is a damage guard and not
  a measure of learning (`analysis/PREREG.md`, A15): training on code does not improve it in
  these arms.
- None of this can be a pre-registered win. The rows are exploratory, and rule 4 needs a
  2.30-point margin over no mitigation at 3e-5, which would mean scoring above the base model.

## Decisions that cross workstreams

peter agreed to the proposal in each of these on November 14. Decisions 1, 3 and 4 are now in
the code on both sides (executor commit 0f0c85f). Decision 2 is built as proposed; Member C has
not yet confirmed it as the hypothesis for the main grid.

**1. How `lr_rewarm` is switched on (peter, Member A, Member C).** A4 made re-warm three
`optim` fields (`schedule_scope`, `rewarm_steps`, `min_lr_ratio`). The fake rows, the grid and
the attribution math treat it as a technique named `lr_rewarm`, switched on and off like the
others. Proposed: keep it a technique in the spec. A technique class may define
`config_overrides(params)`, a small dict merged into the bench config; `lr_rewarm` is a
technique with overrides and no hook behaviour. `clh train` already calls
`apply_config_overrides(cfg)`; the executor needs the same call (decision 3). The same mechanism lets
`lora` carry its own learning rate. The alternative is that the agent writes `optim` overrides
and the analysis infers "re-warm on" from them, which puts a special case in two workstreams.

**2. What `select` chooses among (peter, Member C).** Two different hypotheses:

- *Replay candidates:* draw a pool of replay rows several times larger than needed and keep
  the ones the signal ranks highest. Tests "which old examples protect best". Scoring cost is
  small: at 5% replay and a 4x pool it is forward-only passes over 20% of the tokens.
- *New-domain candidates:* draw twice the new-domain rows and keep half. Tests "which new
  examples are safest to learn". Scoring cost is a forward pass over twice the training tokens.

Recommended for the main grid: replay candidates, because it is cheap, it composes with the
reference arm, and random selection among replay candidates is exactly plain replay, which
gives the control for free. It would make the signal a parameter of `replay`
(`select: random | loss | act_nov | sae_rare`, `pool: 4`) with the scoring code in `select.py`.
New-domain selection needs a change in the shared loop: a micro-batch that a hook skips
should not use up one of the step's accumulation slots, otherwise the step trains on fewer
tokens than planned.

**3. Small asks of the executor (Member A).**

- Use this registry and drop the one on `a5-a6-fast-bench-modal`. The call
  `build_hooks(spec.techniques)` is unchanged; `register` here takes the class, reads its
  `name`, and requires a `Params` model.
- In `config_from_spec`, copy `spec.techniques` into the config and then call
  `apply_config_overrides(cfg)`, so `config.resolved.json` records what was applied and
  `lr_rewarm` takes effect.
- Read `trainer.ctx.state.get("extra_forward_tokens", 0)` into `cost.extra_forward_tokens`.

**4. Order of techniques in a spec (Member A, Member D).** The spec says techniques apply in
list order, and the order is part of the spec hash. `[replay, select]` and `[select, replay]`
would be two experiments. Proposed: the spec validator sorts techniques by name, so the order in
a spec carries no meaning and cannot create duplicate runs. `build_hooks` already applies
them in name order; `RunSpec` now sorts them too.

## Known limits

- LoRA adds its adapters in place, so module paths stay as they are and activation capture
  still finds the layers. The checkpoint's state dict does gain adapter keys: reloading a saved
  checkpoint into a plain model needs the adapter merged first. The fast bench keeps no weights.
- LoRA sets the learning rate for the whole run (default 1e-3, against the bench's 1e-4). It
  is recorded in the resolved config; compare LoRA runs with that in mind. The default suits
  rank 8 only. With `alpha` at 2 x `rank` the size of a step grows roughly with
  `lr` x sqrt(`rank`), so a higher rank needs a lower rate: keep `lr` at or under
  1e-3 x sqrt(8 / `rank`). Rank 256 at the default diverges (Exploratory runs, above). The
  plugin does not enforce the limit, and its default is unchanged so that the rows already
  measured keep their meaning. The explore gate enforces it (`lora_lr_limit` in
  `agent/catalog_real.py`).
- Distillation keeps a frozen copy of the model in memory and holds the teacher's full logits
  for each micro-batch (about 2 GB in bf16 on the real model), plus what autograd keeps of the
  KL computation until backward. It fits on an H100 in the sanity runs.
- Weight averaging leaves the optimizer's moments untouched. With LoRA it scales only the B
  factors, so `alpha` is exact on the adapter's weight change; the EMA mode averages both
  factors, which is not the same thing.
- Activation-novelty selection scores candidates against everything trained so far in the
  run. Its statistics are not reset at a domain boundary, so code rows still count while
  training legal. That matches the definition in `train/signals.py`.
- The sanity and reference rows above were produced before two fixes of 14 November: replay
  runs no longer pay for an unused per-sequence loss pass (their wall clock was a few percent
  high), and `distill` with `rows: replay` is now weighted per token (it was 4 times stronger
  per replayed token at a micro-batch of 4). Retention numbers of the reference arm are not
  affected by either.
