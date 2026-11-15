# Handoff to Devin: C5, the Tinker backend and the second task sequence

Owner: **himani** (workstream 4). Devin **builds and tests**; **himani reviews and runs every
Tinker job**. Devin must not call the Tinker API. There is no Tinker key in Devin's `.env`; test
with a mocked client.

Deadline: the core arms run on **Saturday night**. Hacking ends **Sunday 15 November 14:00 PT**.
Open **one PR** first (core backend plus the three core arms). Open a second, small PR on
Sunday morning only if himani asks for the technique arms.

## What C5 is
The pre-registered second task sequence (`analysis/PREREG.md` section 10, exploratory):
- **Model:** `Qwen/Qwen3-8B`, trained with LoRA on Tinker.
- **Stream:** three classification tasks in order: **SST-2 → MNLI → AG News**.
- **Data:** each task subsampled to **5,000 train / 1,000 eval** examples, deterministically.
- **Arms:** naive (sequential fine-tuning), **replay-200**, and **joint** (multitask, the
  ceiling), each at **3 seeds**. The top 2 techniques from the main screen are added on Sunday
  if they carry over.
- **Metrics:**
  - **gap closed** = (method − naive) / (joint − naive), on the final average accuracy;
  - **average forgetting** = mean over the earlier tasks of (best accuracy − final accuracy).
  - Intervals come from a bootstrap over seeds and eval items. If joint − naive < 0.03, gap
    closed carries an "unstable denominator" warning.

Read first: `CLAUDE.md`; `docs/DIRECTION.md` ("Second task sequence"); `docs/TASKS.md` cards A7
and C5; `analysis/PREREG.md` sections 10 and 11. Read the tinker SDK (`tinker` 0.39.0 on PyPI)
and the Tinker docs for `ServiceClient.create_lora_training_client`, `forward_backward`,
`optim_step` / `AdamParams`, `save_weights_and_get_sampling_client`, and
`SamplingClient.compute_logprobs`.

## Build (in workstream 4's area, not `executor/`)
New package `src/clharness/tinker/`:

1. **`data.py`: tasks and prompts.**
   - Load SST-2 (`glue`/`sst2`), MNLI (`glue`/`mnli`, `validation_matched` for eval) and
     AG News (`ag_news`) from Hugging Face, at pinned revisions.
   - Subsample 5k train and 1k eval per task with a fixed seed (20261114). Write the chosen
     indices plus a sha256 per task to `manifests/tinker-c5.json`. That is a **new** file; never
     edit the existing manifests.
   - One text-to-text prompt template per task, with a fixed **verbalizer** per label
     (SST-2: negative/positive; MNLI: entailment/neutral/contradiction; AG News:
     World/Sports/Business/Sci/Tech). Keep the templates in code, so their hash can be recorded.
2. **`arms.py`: build each arm's training schedule.** Each arm is a list of training batches per
   stage.
   - **naive:** stage t trains only on task t's 5k examples.
   - **replay-200:** at each stage, a fixed buffer of 200 examples per earlier task (seeded) is
     mixed in **by replacement**. Each replayed example replaces a new-task example, so tokens
     per stage match naive exactly. Record the realized replay share.
   - **joint:** a single stage on the shuffled union of all three tasks. Its total examples
     equal the naive run's total across stages (token-matched).
   - **Epochs:** **2 per stage** by default, with `--epochs` as an option.
3. **`runner.py`: train and evaluate one arm at one seed.**
   - Create a LoRA training client on `Qwen/Qwen3-8B`. Loss is on the verbalizer tokens only:
     mask the prompt.
   - After each stage, save the weights for sampling and score **every seen task's 1k eval
     items** by verbalizer log-likelihood (`compute_logprobs`, argmax over the label
     verbalizers). That builds the accuracy matrix R[stage, task].
   - Save per-item correctness for every stage, so the bootstrap can resample items.
   - Count prefill, sample and train tokens, and estimate USD from the price table in TASKS C5
     ($0.44/M train, $0.60/M sample, $0.195/M prefill).
   - **Budget guard:** refuse to start if the estimated total for the requested runs would push
     C5 past **$70**, counting existing rows in `results/tinker/rows.jsonl`.
   - **Proposed hyperparameters** (himani confirms before any run, and they're logged in
     PREREG section 11 before data):
     - LoRA rank 32, learning rate 3e-4, batch 32 examples, max length 256 tokens;
     - constant LR (re-warm is a technique arm for Sunday).
     - If the Tinker docs or cookbook recommend different LoRA settings for an 8B model, say so
       in the PR and propose them instead.
4. **`rows.py`: the C5 row format.** A pydantic `TinkerRow`, not the shared `ResultRow`, because
   these tasks aren't the protected evals. Fields:
   - arm, seed, model, LoRA config, epochs, hyperparameters;
   - the template hash and manifest sha256;
   - the accuracy matrix R;
   - a path to the per-item files;
   - tokens and est_usd;
   - status and reason;
   - git_sha and timestamps.

   Append rows to `results/tinker/rows.jsonl`, and the per-item `.npy` files under
   `results/tinker/items/`.
5. **`analysis.py`: `clh tinker-analyze`.**
   - Final average accuracy, gap closed and average forgetting per arm.
   - Intervals from a paired bootstrap over seeds and eval items (`numpy`, 10k resamples), and
     the unstable-denominator warning.
   - Writes `results/tinker/analysis.json` and `.md`.
   - Add one SVG, the R[stage, task] heatmap for naive vs replay-200 vs joint, to the
     `clh figures` output directory, or as `clh tinker-figures`.
6. **CLI:**
   - `clh tinker-run --arm {naive,replay-200,joint} --seed N [--epochs 2] [--smoke] [--dry-run]`.
     `--smoke` uses 50 train / 50 eval per task, so himani can check the pipeline for cents.
     `--dry-run` prints the token and USD estimate and exits.
   - `clh tinker-analyze`.

## Tests (no network, no Tinker calls)
- Data: deterministic indices and sha256; the verbalizers cover every label.
- Arms: replay-200 is token-matched to naive per stage; joint is token-matched to the naive
  total; the buffer is 200 per earlier task.
- Runner: with a **fake training and sampling client** (a stub with the same method names), a
  full 3-stage run yields a valid R matrix and per-item files; prompt tokens are masked; the
  budget guard refuses over-cap requests.
- Analysis: gap closed and forgetting on a hand-computed example; the warning fires when
  joint − naive < 0.03.
- `make test` passes.

## Rules
- **Branch and PR:** branch from the latest `origin/main` as `devin/c5-tinker`. Push to remote
  `fork`, then open the PR with:
  ```
  gh pr create -R justinmilner1/AIResearchHackathonNov2026 --base main --head iamhimanishah:devin/c5-tinker
  ```
  Never push to `main`, and never force-push.
- **Don't call the Tinker API**, and don't spend credits. Don't launch Modal runs.
- **Don't edit:** `analysis/PREREG.md`, the existing `manifests/*`, `configs/bench/fast.yaml`,
  any committed `results/rows*.jsonl`, or `.env`.
- **Dependencies:** `tinker` goes in as an **optional** extra (`[project.optional-dependencies]
  tinker`) and is imported lazily. The default tests must pass without it.
- **The PR description** gives the exact commands himani runs: smoke first, then the 9 core
  runs, with their dry-run cost. It ends with `REAL:` / `STUBBED:` / `HARDCODED:`.
- **Ask himani in the PR** when unsure.
