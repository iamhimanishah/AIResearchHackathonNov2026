# Task cards

One card per task in the dependency table in [`DIRECTION.md`](DIRECTION.md#task-dependencies).
Each card has the same parts:

- **Goal**
- **Where** (files to touch, and existing code to reuse)
- **Steps**
- **Done when**
- **Open questions** (decide early, and write the answer into the card)
- **Options to try**

Owners: replace "Member A–D" with names. When you settle an open question, strike it through and
write the decision next to it, so the other workstreams see it.

Facts every card relies on (from `docs/baselines.md` and the code):

- Throughput is 12k–15.5k training tokens per second on one H100. 20M tokens takes about 22–27
  minutes before evals.
- `Trainer.run` (`src/clharness/train/loop.py`) builds the optimizer, *then* calls
  `on_run_start`. `_make_optimizer` already skips parameters with `requires_grad=False`.
- `lr_at` uses one global step and a global schedule. The pilot configs use `constant` with 50
  warmup steps at the start of the stream only.
- `select_batch` sees every micro-batch (`[b, seq_len+1]` token ids) and may replace or drop it.
  Tokens count toward the current domain's budget, whatever their source.
- `signals.py` already logs, per training sequence, the loss, entropy, activation novelty, SAE
  rarity and SAE FVU (fraction of variance unexplained). That gives selection signals with no
  new instrumentation.
- `report.py` already has an SVG `Chart` class and the answer-collapse check
  (`dashboard_said_fine`).
- At LR 1e-4, about 80% of forgetting happens in the first 50M code tokens. The standard error
  on a 500-item task is about 2 points.

---

## Member A: bench core and executor

### A0 · Restore the missing `clharness.data` package (blocker, do first)
- **Goal:** make the harness importable from a clean clone.
- **Problem:** `.gitignore` had `data/`, which also matched `src/clharness/data/`. So the
  `clharness.data` package was never committed. The code imports four modules from it:
  `build`, `loader`, `manifest` and `sources`. Because of that, `clh build-data`, `clh train`
  and the tests that load the trainer fail on a fresh clone.
- **Steps:**
  1. ~~Change the ignore rule to `/data/`~~ Done: the rule is now anchored, so only the
     top-level data folder is ignored.
  2. Commit `src/clharness/data/` (with `__init__.py`, `build.py`, `loader.py`, `manifest.py`
     and `sources.py`) from the machine where the harness was built (Justin's). It should now
     show up in `git status` there.
  3. Run `make test` on a clean clone.
- **Done when:** `git clone && make setup && make test` passes on a machine that never had the
  repo.
- **Open questions:** none left on the ignore rules. A scan of every `clharness.*` import
  found no other missing modules.

### A1 · Run-spec and result-row formats, with fake rows
- **Goal:** let B, C and D build against fixed formats before any real run exists.
- **Where:** new `src/clharness/spec.py` (pydantic or dataclasses). JSON Schema in
  `configs/schema/`. Fake rows in `results/fake/rows.jsonl`.
- **Steps:**
  1. Encode the run spec exactly as in "The contract": name, seed, stream (domains and tokens),
     optimizer overrides, techniques with parameters.
  2. Encode the result row:
     - the full spec and git SHA;
     - retention per task and mean (Δ vs checkpoint 0), on both the selection half and the
       report half;
     - acquisition: held-out perplexity ratio per trained domain;
     - cost: gradient tokens, wall-clock, GPU type, shipped update size in bytes;
     - share of the most common answer per multiple-choice task;
     - status and reason.
  3. Add `spec_hash` (sha256 of the canonical JSON) as the idempotency key.
  4. Write a generator for about 40 plausible fake rows. Give techniques known effects plus
     noise, so C can test that the attribution analysis recovers them.
- **Done when:** B, C and D have confirmed it in the team chat, and `clh validate-row` passes on
  the fake rows.
- **Status:** drafted, waiting for B, C and D to confirm. Code is in `src/clharness/spec.py`
  and `src/clharness/fake.py`. Regenerate with `clh fake-rows` and `clh export-schema`.
- **Open questions:**
  - ~~Retention per checkpoint or only at boundaries?~~ Decided: the row holds the final
    per-task numbers and the mean after each domain (`retention.boundaries`). Per-checkpoint
    data stays in the run's own files.
  - ~~Store per-item outcomes in the row, or point to a file?~~ Decided: a pointer
    (`items_path`).
  - ~~Units for "size of the update that would ship"?~~ Decided: both (`cost.update_bytes` and
    `cost.trainable_params`).
  - Scores and deltas are fractions, not points. Perplexity ratios are trained / base.
  - `spec_hash` leaves out `name`, `notes` and `parent_spec_hash`, so relabelling a spec does
    not make it a new experiment.
  - Per-domain LR re-warm appears in the fake rows as a technique named `lr_rewarm`, so the
    on/off grid treats it like the others. A4 decides whether it stays a technique or becomes
    an `optim` override.
- **Options to try:** ~~add a free-form `notes` field, and a `parent_spec_hash`~~ Both added.

### A2 · Pre-optimizer model hook
- **Goal:** let a technique change the model (LoRA wrap, freeze layers) before AdamW sees the
  parameters.
- **Where:** `hooks.py` (new `configure_model(ctx) -> nn.Module | None`) and `loop.py`. Call it
  in `Trainer.__init__` after `load_model` and **before** `torch.compile`, and update `self.model`
  and `ctx.model`.
- **Steps:**
  1. Add a no-op default to `BaseHook`.
  2. Call the hooks in order, each one receiving the previous one's model.
  3. Log the trainable parameter count.
  4. Test with the tiny model in `tests/test_loop_tiny.py`: freeze everything except one layer,
     then check that the other weights are unchanged after a step.
- **Done when:** a test proves frozen weights stay identical, and a LoRA-wrapped model trains.
- **Status:** done. `configure_model` runs in `Trainer.__init__` before `torch.compile` and the
  optimizer. The counts are in `ctx.state["trainable_params"]` and `["total_params"]`.
  `decoder_layers` unwraps a PEFT model, so activation capture still finds the layers. Eval
  hooks see the wrapped model (`ctx.model`), not a merged one.
- **Open questions:**
  - Does the activation capture (`ResidualCapture`) still find the decoder layers after a PEFT
    wrap? The module paths change to `base_model.model...`.
  - Should eval and SAE hooks see the merged model or the wrapped one?
- **Options to try:** expose `ctx.state["trainable_params"]` so the result row's cost field
  can read it.

### A3 · Loss-term hook
- **Goal:** let distillation and weight penalties (EWC/L2-SP) add to the loss.
- **Where:** `hooks.py` (new `extra_loss(ctx, batch, logits, labels) -> Tensor | None`) and
  `loop.py:_loss_and_count`. Sum the returned terms into `loss` before `backward`, and log each
  term separately.
- **Steps:**
  1. Pass the logits *with* grad (not the detached ones used for `on_micro_batch`).
  2. Keep the term in fp32.
  3. Test that a zero term gives bit-identical training, and that a constant-gradient term
     changes it.
- **Done when:** the tiny-model test passes, and the per-term logs appear in `metrics.json`.
- **Status:** done. `train/loss` stays the language-model loss; each term is logged as
  `train/extra_loss/<hook name>`. The term is cast to fp32. Chunking a KL over the 262k
  vocabulary is left to the technique. Extra terms do not count as gradient tokens.
- **Open questions:**
  - Memory: the vocabulary is 262k, so a full teacher-logit tensor at micro-batch 4×1024 is
    about 2 GB in bf16. Is a chunked KL needed? Probably yes; reuse the chunking pattern from
    `chunked_cross_entropy`.
  - Should the extra loss count toward "gradient tokens"? Recommended: no, but report the extra
    FLOPs as cost.

### A4 · Per-domain LR re-warm
- **Goal:** the reference arm (Ibrahim et al. 2024) needs a warmup and decay per domain, not
  one global schedule.
- **Where:** `config.py` (new `optim.schedule_scope: global | per_domain`,
  `optim.rewarm_steps`, `optim.min_lr_ratio`) and `loop.py:lr_at`. Pass
  `(step - domain_start_step, domain_steps)` when the scope is per-domain.
- **Steps:**
  1. Track `domain_start_step` in `run()`.
  2. Unit-test the LR curve shape for a 2-domain stream.
  3. Plot it once to confirm.
- **Done when:** the logged `train/lr` shows a re-warm at each boundary, and the global-scope
  default is unchanged (the existing tests pass).
- **Status:** done. Set `optim.schedule_scope: per_domain`, `optim.rewarm_steps` and
  `optim.min_lr_ratio`, with `optim.schedule: cosine` (or `linear`) for the decay. A run spec
  can set all of these under `optim`. The values to use are still the open questions below.
- **Open questions:**
  - Re-warm length: 1% or 5% of the domain's steps?
  - Peak LR: the same 1e-4, or lower?
  - Decay shape and floor: cosine to 10% of peak (as in the paper), or linear?
  - The no-mitigation arm uses a constant LR. Does a cosine per-domain schedule change the
    total "learning budget" enough to confound the comparison? Decide this with C.
- **Options to try:** re-warm without re-decay, and decay only, as cheap ablations that show
  which half of the recipe matters.

### A5 · Frozen fast bench config
- **Goal:** one run in about 30 minutes that still forgets clearly.
- **Where:** new `configs/bench/fast.yaml`, including the model and eval configs (as
  `configs/pilot/base.yaml` does).
- **Steps:**
  1. Stream: code → biomedical, about 8–10M tokens each at LR 1e-4 (about 20M total, about
     25 minutes of training).
  2. Turn off the expensive instruments: `recover_tokens: 0`, SAEs off or one layer, CKA and
     mknn off. Keep `signal_log` only if B3 needs it.
  3. Score the multiple-choice tasks at boundaries and at the end only. Run MBPP at the end
     only, or drop it.
  4. Run once, measure the wall-clock split between training and evals, then freeze it.
- **Done when:** one run finishes in ≤ 35 minutes on an H100, and its MC-mean drop is at least
  4 points (2× the per-task standard error).
- **Status:** met by one run each, seed 0, on a Modal H100 (rows in
  `results/rows.calibration.jsonl`). Batch size is settled at 64 sequences per step. The
  learning rate stays open until Member C answers the acquisition problem below; after that
  the bench is frozen.

  | Spec | Seqs/step | Wall | MC mean, selection half | MC mean, report half | Code ppl ratio | Biomedical ppl ratio | General ppl ratio |
  |---|---|---|---|---|---|---|---|
  | `no-mitigation-s0` | 64 | 18.1 min | −10.7 | −6.2 | 1.81 | 1.00 | 1.45 |
  | `calibration-b16-s0` | 16 | 19.2 min | −16.2 | −12.9 | 2.02 | 1.16 | 2.79 |

  - Decided: 64 sequences per step (the config as committed). It forgets clearly and damages
    the model less than 16.
  - Problem for the design (Member C): acquisition is close to zero. Held-out biomedical
    perplexity is unchanged after 10M biomedical tokens, and code perplexity is worse than the
    base model's after the biomedical domain. A technique cannot visibly "learn less" on a
    bench that learns almost nothing, so the acquisition floor in the pre-registration needs
    thought, or the bench needs a lower learning rate.
  - The two halves of the protected set differ by 4.5 points on the same run. The noise floor
    (C3) has to be measured on each half.
  - A run costs about $1.20, not the $2.50 planned for.
- **Open questions:**
  - Is 10M tokens per domain enough to forget at 1e-4? The 50M run lost 11.9 points, most of
    it early, so probably yes. Check it.
  - Keep MBPP? It is slow and noisy (200 items, base 0.11).
  - `micro_batch_size`: 4 OOMed at 8 on an 80 GB card with the recover probe. Without the
    probe, 8 may fit and run faster.
  - Use `compile: true` for speed? It must come after A2's hook.
- **Options to try:** an H100 vs A100 throughput and cost comparison; seq_len 512 vs 1024.

### A6 · Modal executor
- **Goal:** `clh run spec.json --modal` returns a result row. `clh run-many specs/*.json`
  fans out.
- **Where:** new `src/clharness/executor/modal_app.py`.
- **Steps:**
  1. Build the image from `uv.lock`.
  2. Create a Modal Volume for the 6.7 GB `data/` and the HF cache, built once with
     `clh build-data`, then `clh verify`.
  3. Add `HF_TOKEN` as a Modal secret. The Gemma licence must be accepted on that account.
  4. Run on one GPU per run. Write the result row to the volume and append it to
     `results/rows.jsonl`.
  5. Retry twice on infrastructure errors only, never on a NaN or a bad result. A row that
     fails sanity is marked `invalid`, not retried.
  6. Budget guard: estimated $ = GPU-hours × price. Refuse to launch past $700 (70% of the
     team's Modal credit).
- **Done when:** sync point 1. One spec goes to a row in ≤ 35 minutes, 4 run in parallel, and
  a killed run comes back as `failed` with a reason.
- **Open questions:**
  - Modal H100 price (about $4/hr, unverified) and the GPU concurrency limit per workspace.
  - Where does the agent read rows from: the Modal volume, a Modal dict, or git?
  - Timeout per run (45 minutes?) and what happens to its partial row.
- **Options to try:** run on A100-80GB if H100 capacity is short. Add a `--dry-run` that
  prints the cost estimate.

### A7 · Tinker backend
- **Goal:** the second task sequence (C5) produces the same result-row format.
- **Where:** `src/clharness/executor/tinker_backend.py`. Devin can scaffold it.
- **Steps:**
  1. Map the run spec (task stream, LoRA rank, technique) onto Tinker's training and sampling
     calls.
  2. Map accuracy per task onto "retention", and the new task's accuracy onto "acquisition".
- **Done when:** a 200-example smoke run on SST-2 produces a valid row.
- **Open questions:**
  - ~~Tinker's smallest model and its price per token~~ Decided: `Qwen/Qwen3-8B` at $0.44/M
    train (see C5). No Llama models are listed on Tinker now, and Qwen3-8B-Base was retired on
    June 12, 2026.
  - ~~Does the SDK expose logprobs, custom losses, or LoRA weight export?~~ Yes to all three.
    Checked in `tinker` 0.39.0 (SDK methods only, no API call yet):
    - **Logprobs:** `SamplingClient.compute_logprobs`, for verbalizer scoring.
    - **Custom loss:** `TrainingClient.forward_backward_custom` and
      `ServiceClient.create_custom_loss`, for EWC and distillation.
    - **Weight export:** `TrainingClient.save_weights_external` and
      `RestClient.get_external_weights_urls`, to download adapters for merging on Modal.
    - `ServiceClient.get_server_capabilities()` is deprecated. Use
      `tinker.RestClient().get_server_capabilities().result()`.
  - ~~**Blocker:** the team's Tinker org returned `402` (billing)~~ Resolved on 2026-11-14
    once the hackathon credit was applied.
    - The account lists 38 models, including `Qwen/Qwen3-8B`.
    - A rank-8 LoRA training client for Qwen3-8B was created successfully, and its tokenizer
      is `Qwen2Tokenizer`.
    - No training tokens have been spent yet.
    - If the 402 comes back, check the org's credit balance in the Tinker console.
  - Tinker's concurrency limit.
- **Options to try:** if Tinker can't export weights, skip merging techniques on this track.

### A8 · `make reproduce`
- **Goal:** one command that rebuilds every reported number and figure.
- **Where:** `Makefile`, `scripts/reproduce.py`, README section.
- **Steps:**
  1. `make reproduce-figures`: from `results/rows.jsonl` to all figures and tables, no GPU,
     under 2 minutes.
  2. `make reproduce-runs`: re-launches the report-half specs through A6.
  3. Test both from a clean clone.
- **Done when:** a teammate on a fresh clone gets byte-identical figures.
- **Open questions:** commit the rows to git, or download them from the volume? Recommended:
  commit them; they are small.

---

## Member B: techniques

Every technique is a hook plugin under `src/clharness/techniques/`. Each one needs:

- a `params` schema;
- a sanity run on `configs/smoke.yaml`, plus one fast-bench run;
- an entry in a `TECHNIQUES.md` table with: what it does, its parameters, its cost, and its
  sanity result.

Gradient tokens must match the no-mitigation run. When a technique spends extra compute
(scoring passes, a teacher forward), report it under cost.

### B1 · Reference arm: replay plus per-domain LR re-warm
- **Goal:** the arm that every "improvement" must beat by more than the noise.
- **Where:** a combination of B2 and A4, plus `configs/bench/reference.yaml`.
- **Steps:**
  1. Set the paper's defaults: 5% replay of general (pretraining-like) text, then re-warm and
     re-decay per domain.
  2. Do one fast-bench run, and compare with no-mitigation on retention *and* acquisition.
- **Done when:** it retains more than no-mitigation by more than the noise floor, without
  losing acquisition. If it doesn't, report that: it is a finding.
- **Open questions:**
  - Replay source: the `general` slice already in the manifest (pretraining-like), or old
    domains (code, while training biomedical)? The paper replays pretraining data.
  - Replay share: 5% (paper default) or tuned? Tuning it on the selection half is fair, but
    say so.
- **Options to try:** 1%, 5% and 25% replay. Re-warm only, and replay only, to split the
  recipe into its two halves.

### B2 · Replay
- **Goal:** mix replayed sequences into each batch at matched gradient tokens.
- **Where:** `techniques/replay.py`, using `select_batch`. Swap `round(p × b)` rows of the
  micro-batch for sequences from a replay `PackedStream` (`data/loader.py`).
- **Steps:**
  1. Build the replay stream in `on_run_start` from the manifest's `general` split, or from
     the earlier domains.
  2. Swap the rows in `select_batch`, with a seeded RNG.
  3. Log the replay share actually used.
- **Done when:** the logged share equals `p`, gradient tokens are unchanged, and a smoke run
  passes.
- **Open questions:**
  - Swap rows (replace mode, the new domain gets fewer tokens) or add rows (more compute)?
    Recommended: replace for the main grid, add as one cost-reported variant.
  - Are buffer sequences drawn fresh, or from a fixed buffer of N sequences? A fixed buffer is
    closer to the literature.
- **Options to try:** p ∈ {0.01, 0.05, 0.25}; a fixed buffer of 1k vs 10k sequences.

### B3 · Example selection: controls, then the activation signal
- **Goal:** test whether choosing *which* replay or new examples to train on, using a
  model-internal signal, beats random and loss-based selection.
- **Where:** `techniques/select.py`. Read the per-sequence scores that `signals.py` already
  computes (`loss`, `entropy`, `act_nov_L`, `sae_rare_L`, `sae_fvu_L`).
- **Steps:**
  1. **Random** control: keep k of b with a seeded RNG.
  2. **Loss-based** control: keep the top-k by current loss (SuRe/Rho-1 family).
  3. **Activation-based:** keep the top-k by `act_nov` or `sae_rare` at layer 13.
  4. In all three, k/b is the same and the token budget is the same, with oversampling to
     compensate.
- **Done when:** all three run on the fast bench, with the scoring cost reported.
- **Open questions:**
  - Select among *new-domain* examples (learn efficiently) or *replay* candidates (protect
    efficiently)? These are different hypotheses; pick one for the main grid.
  - High novelty or low novelty? It's unclear which protects: high novelty means the example
    is more disruptive, so the selector could keep it or drop it. Screen both signs at one
    seed.
  - Scoring cost: an extra forward pass over the candidates. Oversample 2× and keep half?
- **Options to try:** reducible loss (current loss − base-model loss); token-level masking
  (Rho-1) vs sequence-level selection; SAE layer 7 vs 13 vs 22.

### B4 · LoRA adapters, then layer freezing
- **Goal:** limit where updates can go.
- **Where:** `techniques/lora.py` and `techniques/freeze.py`, both using A2's
  `configure_model`. LoRA via `peft`.
- **Steps:**
  1. LoRA: rank r on the attention and MLP projections, with a separate LR (LoRA usually needs
     about 10× the full-fine-tuning LR).
  2. Freezing: freeze the embeddings plus the bottom n layers, or every other layer.
  3. Sanity-check both on smoke, then the fast bench.
- **Done when:** the trainable parameter count is logged, both runs finish, and the shipped
  update size is in the row.
- **Open questions:**
  - Which LR for LoRA (1e-4 is too low for LoRA; try 1e-3)? That makes "matched compute"
    differ in more than one way, so record it.
  - One adapter for the whole stream, or a new adapter per domain (merged or stacked)?
  - Freeze embeddings only, the bottom half, or the top half? The earlier analysis found the
    loss starts in code. Check which layers drift most (`metrics/cka.py`).
- **Options to try:** LoRA r ∈ {8, 64}; freeze the bottom 13 of 26 layers vs the top 13.

### B5 · Distillation from the base model
- **Goal:** keep the output distribution close to the base model's on the training batch.
- **Where:** `techniques/distill.py`, using A3's `extra_loss`. Keep a frozen bf16 copy of the
  base model in `on_run_start`.
- **Steps:**
  1. Run the teacher forward on the same batch.
  2. Compute a chunked KL(student‖teacher) with weight λ.
  3. Log the KL term.
- **Done when:** the KL term falls as λ rises, and the run fits in memory on an H100.
- **Open questions:**
  - Distil on the new-domain batch, on replay data, or on both?
  - Forward or reverse KL, and at what temperature?
  - Full vocabulary, or top-k logits only (much cheaper with a 262k vocabulary)?
  - Cost: about 1.3× FLOPs per step. Report it.
- **Options to try:** λ ∈ {0.1, 1.0}; top-k=64 logits vs the full vocabulary.

### B6 · Weight averaging
- **Goal:** average the trained weights with the base weights (or across checkpoints) to give
  back some forgetting.
- **Where:** `techniques/wavg.py`. Average in `on_domain_end` or `on_run_end`; no hook change
  needed.
- **Steps:**
  1. Keep the base weights on CPU.
  2. At the end (or at each boundary), set θ ← α·θ_base + (1−α)·θ.
  3. Run evals after averaging.
- **Done when:** retention and acquisition curves over α ∈ {0.25, 0.5} exist.
- **Open questions:**
  - Average at each boundary (running) or once at the end? At each boundary is closer to
    "continual".
  - An EMA of the weights during training instead (zero extra evals)?
- **Options to try:** an EMA with decay 0.999; averaging only at the end.

---

## Member C: design and analysis

### C1 · `analysis/PREREG.md`
- **Goal:** commit to the decision rules *before* seeing results, for rigour and for Q&A.
- **Where:** `analysis/PREREG.md`, committed with a timestamp before sync point 2.
- **Steps:** write down each of the following.
  1. **The split:** which items are in the selection half and which in the report half. Make
     it a seeded split of each task's 500 items into 250/250, stored in `manifests/`
     *as a new file* (do not edit the frozen ones).
  2. **Primary metric:** the change in MC mean on the report half at the end of the stream.
     Acquisition is a co-primary metric with a floor: a technique "wins" only if acquisition
     stays at least as good as no-mitigation minus δ.
  3. **Decision rule:** beat the reference arm by more than max(2 × noise-floor SD, a paired
     bootstrap CI that excludes 0).
  4. **Seed policy:** screen at 1 seed, then 3+ seeds for the top configurations and the
     controls.
  5. **What counts as invalid:** answer collapse (most-common-answer share above 0.9),
     NaN/anomalous steps, a run killed by the budget.
- **Done when:** merged before the grid launches, and D's agent reads its thresholds from it.
- **Open questions:**
  - δ for acquisition: 5% of the perplexity ratio?
  - Does MBPP enter the primary metric? Recommended: no, report it separately.
  - 250 items per task half gives a standard error of about 3 points per task. Is the MC mean
    over 4 tasks tight enough? Check it against the noise floor (C3).

### C2 · Attribution math, bootstrap and cost ranking (on fake rows first)
- **Goal:** each technique's own effect, its pairwise interactions, uncertainty on all of it,
  and a ranking per unit of cost.
- **Where:** `src/clharness/analysis/attribution.py`. Reuse `analyze.py` and `findings.py`
  where possible.
- **Steps:**
  1. Fit retention ~ techniques + pairwise interactions (OLS over the on/off grid), with
     cluster-bootstrap CIs over seeds and items.
  2. Paired comparison against the reference arm on matching seeds.
  3. Cost-adjusted ranking: retention gained per extra GPU-hour, and per byte shipped.
  4. Check that the analysis recovers the effects planted in A1's fake rows.
- **Done when:** on the fake rows, the planted effects are recovered inside the CIs.
- **Open questions:**
  - Grid design for 6 techniques: a full 2⁶ factorial is 64 runs (too many). Options:
    - a fractional factorial (2⁶⁻² = 16 runs, resolution IV: main effects clean, pairs
      aliased in groups);
    - Plackett–Burman (12 runs, main effects only);
    - one-at-a-time plus all pairs of the top 3.
    Pick one with D, because the agent executes it.
  - Bootstrap over items, seeds, or both (hierarchical)? Recommended: both.
- **Options to try:** a Shapley-style attribution for the best combination; analysing
  persistent losses (items wrong at the last 3 checkpoints) as a less noisy outcome.

### C3 · Noise floor
- **Goal:** know how big a difference has to be before it means anything.
- **Where:** specs in `configs/bench/noise/`, launched through A6.
- **Steps:**
  1. Run no-mitigation on the fast bench at 5 seeds. The seed changes the data order: the
     stream seed is `train.seed + domain_index`.
  2. Report the SD of the MC-mean change, per task and averaged.
  3. Also run the reference arm at 3 seeds.
- **Done when:** the SDs are in PREREG.md, and the agent uses them as its threshold.
- **Open questions:**
  - Does the seed also need to change the model's dropout or initialization? Gemma has no
    dropout in pretraining mode, so the data order is the main source of noise. Check this.
  - 3 or 5 seeds? Five is better, but costs 5 × about $2.50.
- **Options to try:** check whether the spread between items or between seeds dominates.
  That decides which bootstrap matters.

### C4 · Analysis on real rows, and the pitch figures
- **Goal:** the numbers and figures that the pitch and the page show.
- **Where:** `analysis/figures.py`, writing to `results/figures/`. Reuse `report.py:Chart` for
  SVG.
- **Figures:**
  1. Retention vs acquisition scatter, one point per configuration with CIs, the reference
     arm marked, and a dominance frontier.
  2. Main-effect and interaction heatmap with CIs.
  3. Cost-adjusted ranking bar chart.
  4. Per-task change bars, with the answer-collapse flag.
  5. The second-sequence check (from C5).
- **Done when:** the figures are generated by `make reproduce-figures` from the committed
  rows.
- **Open questions:** which one figure opens the pitch? Recommended: (1). It shows "forgets
  less without learning less" in a single image.

### C5 · Second task sequence on Tinker (time-boxed, first to cut)
- **Goal:** check that the top result generalizes to a published sequence and another model.
- **Where:** A7's backend, with specs in `configs/tinker/`.
- **Steps:**
  1. **Model: `Qwen/Qwen3-8B` (decided).** LoRA, SST-2 → MNLI → AG News, 5k train / 1k eval
     per task.
  2. Score classification by verbalizer log-likelihood.
  3. Arms: naive, replay-200, joint and the top 2 techniques, at 3 seeds.
  4. Report gap closed = (method − naive) / (joint − naive), and the average forgetting.
- **Done when:** a figure exists with CIs, or the track is cut with the reason logged.
- **Budget:** Tinker list price for Qwen3-8B (32K context, from
  thinkingmachines.ai/tinker/pricing, dated Oct 19, 2026), per million tokens:

  | Train | Sample | Prefill | Cached prefill |
  |---|---|---|---|
  | $0.44 | $0.60 | $0.195 | $0.039 |

  - A 3-task run (about 2.5M training tokens over 2 epochs, plus about 0.75M eval tokens)
    costs about **$1.25**.
  - The 15 planned runs cost about **$19**, against a cap of $70 (70% of one member's Tinker
    credit).
- **Open questions:**
  - ~~Tinker price, model and features (see A7)~~ Model and price settled (above). Features
    (logprobs, custom loss, weight export) are still to be confirmed in the SDK (see A7).
  - Qwen3-8B is an *instruct* model, so zero-shot scores on SST-2 and AG News start high and
    leave less room for gains. Does that weaken the comparison with the main grid, which uses
    the pretrained Gemma 3 1B? Mitigation: run naive and replay at 1 seed on
    `Qwen/Qwen3.5-9B-Base` (the only small base model, $1.463/M train, about $8 in total).
  - Does each top technique even translate? LR re-warm and replay do. SAE-based selection
    needs the SAEs for that model, which may not exist, so substitute loss-based selection.
  - 1 or 2 epochs per task (2 doubles the cost).
- **Options to try:**
  - Two task orders (forward and reversed) if the budget allows.
  - Fallback models, in order:
    1. `Qwen3.5-4B` ($0.737/M train): the smallest dense model.
    2. `GPT-OSS-20B` ($0.396/M train, MoE with about 3.6B active): the cheapest, but its chat
       format is awkward for plain text-to-text tasks.
    3. Llama-3.2-1B trained with PEFT on Modal: off Tinker, uses spare Modal credit.

---

## Member D: agent and page

### D1 · Agent v1: plan, launch, read, decide, log
- **Goal:** the research loop runs without a human, and every decision is explained (the
  autonomy criterion carries 20% of the score).
- **Where:** `src/clharness/agent/`:
  - `loop.py`;
  - `tools.py`: `list_techniques`, `plan_grid`, `launch(specs)`, `poll()`, `read_rows()`,
    `log_decision()`.
  The decision log goes to `results/agent/decisions.jsonl`.
- **Steps:**
  1. **Deterministic core:** a state machine with the stages screen → replicate → combine →
     report.
  2. **LLM step** (OpenAI) at each decision point. It gets the rows and PREREG thresholds as
     input, and outputs the next specs plus a written reason.
  3. **Validator:** rejects specs that break compute matching or the budget, or that look at
     the report half.
  4. Each log entry records: timestamp, stage, inputs (row hashes), choice, reason, and
     whether it was accepted or rejected.
- **Done when:** on fake rows, the agent runs the full screen → replicate cycle and writes a
  readable decision log.
- **Open questions:**
  - How much freedom does the LLM get? Recommended: the LLM proposes, and deterministic rules
    check, so there are no runaway costs.
  - Seed allocation: a fixed rule (top-k by effect/SE gets 3 seeds), or a bandit or successive
    halving?
  - Which OpenAI model and what cost per decision? Keep the total under $70.
  - Human-in-the-loop? The judges reward autonomy. Allow a human *veto*, logged, but never
    require approval.
- **Options to try:** have the agent write a short "what I learned" note after each stage
  (good for past.dev and the pitch).

### D2 · Agent launches the screening grid
- **Goal:** sync point 2. The overnight grid runs, launched by the agent, not by hand.
- **Steps:**
  1. Check that A6, C1, C3, B1 and at least 4 of B2–B6 are done.
  2. The agent plans the grid (the design comes from C2) and launches it.
  3. Stay 30 minutes to watch the first rows land.
- **Done when:** at least 90% of the screening runs are `finished` by 07:00. Failures carry a
  reason and are re-queued once.
- **Open questions:** how many runs fit overnight? The 9-hour window, 30 minutes per run and
  the concurrency limit set the number. Choose the grid size from that, not the other way
  round.

### D3 · Live results page
- **Goal:** what judges see in the finals live demo. It is a viewer, with no controls.
- **Where:** `src/clharness/page/`. Reuse `report.py:Chart` for SVG.
- **Steps:**
  1. Read `results/rows.jsonl` and `decisions.jsonl`.
  2. Render four parts:
     - a results table with CIs;
     - figure C4-1;
     - the agent's decision timeline;
     - a "failed and invalid runs" panel.
  3. Auto-refresh every minute.
- **Done when:** it shows live rows during the grid, and works on a phone-sized screen.
- **Open questions:**
  - Hosting options:
    - a Modal web endpoint (live, but must stay up);
    - a static page rebuilt by a Modal cron;
    - a GitHub Pages snapshot (safest for judging).
    Recommended: the Modal endpoint, with a static fallback.
- **Options to try:** a per-item "what was lost" gallery (`report.py:gallery`) for the
  interpretability criterion.

### D4 · past.dev shared memory
- **Goal:** the past.dev challenge: more than one agent writes, more than one reads, and one
  uses what another wrote.
- **Steps:**
  1. Two agents: the **planner** (D1) and an **analyst** (C2's analysis as an agent). The
     analyst writes findings ("distillation λ=1 collapses PubMedQA answers"). The planner reads
     them before planning the next stage.
  2. Store three memories, documented in the README: `tried` (spec hash → outcome), `failures`
     (reason, spec), `findings` (claim, evidence rows).
- **Done when:** the decision log shows a planner decision that cites an analyst finding.
- **Open questions:**
  - The past.dev API and auth: what does it actually store and query?
  - Which memory is the source of truth, past.dev or `rows.jsonl`? Recommended: rows in git
    are the truth, and past.dev holds the agents' notes.

### D5 · Submission
- **Goal:** sync point 4, before Sunday 14:00.
- **Steps:**
  1. Record a 2-minute demo video: page → agent log → headline figure.
  2. Write a project description that credits the earlier harness work (see the "Open issue:
     prior work" section).
  3. Choose the challenges entered (Modal, past.dev, and Exa if D6 is done).
  4. Add the repo link.
- **Done when:** submitted by 13:45, with a 15-minute buffer.
- **Open questions:** Originator upload shares full session transcripts. The team decides this
  before anyone uploads.

### D6 · Exa in the main loop (priority 4, only after 1–3 work)
- **Goal:** the agent finds a technique it hasn't tried, the team wraps it, and the agent
  tests it. The Exa challenge needs the call in the main loop.
- **Steps:**
  1. At the "combine" stage, the agent queries Exa ("continual pretraining forgetting
     mitigation 2025 2026"), filtered to arXiv.
  2. The agent summarizes the candidates, picks one that isn't in the catalog, and logs the
     sources.
  3. B (or a coding agent) implements it as a plugin. The agent screens it.
- **Done when:** the decision log links the Exa sources to a run row. The README points to the
  call site, and the demo shows the sources.
- **Open questions:**
  - Can the agent write the plugin itself (more autonomy, more risk), or does a human wrap it?
    Recommended: the agent drafts it, a human reviews it in under 15 minutes, and the review is
    logged.
- **Options to try:** candidates to expect include O-LoRA, InfLoRA, sharpness-aware updates,
  and model merging (TIES/DARE). Have one cheap fallback ready.
