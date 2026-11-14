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
teacher pass. Sanity spec: `configs/bench/specs/sanity/replay-distill-on-replay.json` (not yet
run on a GPU).

The SAE-feature signal (`sae_rare`) is not built: it needs SAE weights on the GPU, and
activation-based selection is fourth in the cut order.

## Rules every technique follows

- **Gradient tokens match the no-mitigation run.** A technique may change which sequences are
  trained on, never how many tokens a step or a run spends.
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
| `replay` (B2), `wavg` (B6), `lr_rewarm` | done, with tiny-model tests; no real-model run yet |
| Selection (B3): `random`, `loss`, `act_nov` on `replay` | done, with tiny-model tests; `sae_rare` not built |
| `lora`, `freeze` (B4), `distill` (B5) | done, with tiny-model tests; no real-model run yet |
| Reference arm (B1) | specs written and checked through `config_from_spec`; needs a fast-bench run |
| Sanity runs on a GPU | all eight specs in `configs/bench/specs/sanity/` finished on Modal H100s at commit 314150a; rows in `results/rows.sanity.jsonl` (four steps per domain, so only status and cost mean anything) |

Sanity results will be added to this file from result rows only, never typed in.

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
- LoRA sets the learning rate for the whole run (default 1e-3, against the bench's 1e-4),
  because adapters trained at the full fine-tuning rate barely learn. It is recorded in the
  resolved config; compare LoRA runs with that in mind.
- Distillation keeps a frozen copy of the model in memory and holds the teacher's full logits
  for each micro-batch (about 2 GB in bf16 on the real model). Not yet run on a GPU.
- Weight averaging leaves the optimizer's moments untouched, and averaging LoRA factors is not
  the same as averaging the weight change they produce.
