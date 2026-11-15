# clharness: harness reference

*Imported from the pre-hackathon project (September 2026). The repository overview is in the top-level [README](../README.md).*

A continual-learning harness for small open LLMs. It presents an ordered stream of domains to
a model, fine-tunes with no mitigation (or, later, with a selection/replay policy), and at every
checkpoint measures (a) a fixed protected capability set, (b) held-out perplexity per domain,
(c) Gemma Scope 2 SAE feature statistics on a fixed probe set, and (d) drift-immune
representation similarity (linear CKA) against the base model.

## Setup

Requirements: Python 3.12 (uv fetches it), [uv](https://docs.astral.sh/uv/), a Hugging Face
token with access to `google/gemma-3-1b-pt`, and a GPU (CUDA), Apple silicon (MPS), or patience.

```bash
cp .env.example .env      # put HF_TOKEN in it; WANDB_PROJECT optional (offline otherwise)
make setup                # uv sync --extra dev
make env                  # prints device + library versions
```

## Smoke test

```bash
make smoke
```

Builds a tiny data set (5 sources, ~200 docs each, written to `data/smoke/`), freezes an
8-item eval suite, verifies both manifests, records the SAE drift baseline on the base model,
and runs the full training loop: 4 domains × 12k gradient tokens, checkpoints every 3 steps and
at every domain boundary, tier-1 evals + probe statistics at each checkpoint, tier-2 at every
2nd. Takes ~40 min on an M4 (MPS forward is ~1.5–5 ms/token; ~2 min per checkpoint plus a recoverability probe at each boundary) or a few minutes on a datacenter GPU, after the one-time ~4 GB download. Output lands in
`results/runs/smoke/`:

```
checkpoints/ckpt_0000/   meta.json  metrics.json  features_layer{7,13,22}.npz
metrics.jsonl            every logged scalar, one line per step
config.resolved.json     the fully-resolved config that produced the run
base_probe_acts.pt       base-model probe activations used for CKA
wandb/                   offline wandb run (sync with `wandb sync`)
```

`make test` runs the fast unit tests (manifest integrity, metric properties, packing, config
strictness, and the full trainer loop on a tiny random model). `make test-all` adds the end-to-end smoke test.

## Real data

```bash
uv run clh build-data --set real --seq-len 1024      # ~6 GB download + tokenize, ~6 min; data/real/ is gitignored
uv run clh verify configs/pilot/base.yaml            # hashes every file against manifests/real.json
```

Five domains from pinned Hub revisions (see `REAL_SOURCES` in `data/sources.py`): FineWeb
(general), StarCoder Python (code), PubMed abstracts (biomedical), Pile of Law court opinions
(legal), OpenWebMath (math). Per domain: a 200M-token training stream (50M for general), a 2M-token
held-out perplexity slice, a 1M-token probe set. Rebuilding reproduces the manifest byte-for-byte.

## How a run is specified

Everything is YAML under `configs/`; a run config `include`s a model fragment and an eval
fragment and adds data / optim / train / wandb sections. Every result-affecting field is
required (pydantic, `extra="forbid"`): a typo or omission is an error, not a silent default.

```
configs/model/gemma3-1b-pt.yaml   model id, dtype, capture layers, SAE release/ids
configs/eval/<suite>.yaml          lm-eval tasks (+ n_items, fewshot, tier), perplexity slices
configs/<run>.yaml                 domains in order with token budgets, LR/schedule, checkpoints
configs/lm_eval_tasks/             custom lm-eval task yamls (pubmedqa_pqal, casehold on parquet mirrors)
```

Reproducibility contract: `manifests/<set>.json` records every data file (path, bytes, sha256),
the source spec + revision, seeds and doc ids (or their hash); `manifests/eval-<suite>.json` records the exact item indices
and a hash of the selected docs for every lm-eval task. `clh verify CONFIG` checks both;
the trainer refuses to start on a mismatch.

## CLI

```
clh env
clh build-data --set smoke
clh build-eval-manifest configs/smoke.yaml
clh verify configs/smoke.yaml
clh sae-baseline configs/smoke.yaml     # -> results/baselines/sae_drift_base_smoke.json
clh train configs/smoke.yaml
clh baseline-evals configs/baseline-candidates.yaml [--limit-items N]   # deliverable 2 table
clh train configs/pilot/cuda-check.yaml --name cuda-check-b
clh compare-runs results/runs/cuda-check-a results/runs/cuda-check-b   # bitwise repeatability diff
clh item-profile results/rows.*.jsonl          # per arm: fragile core vs other questions forgotten
clh forget-sets results/rows.*.jsonl --out results/analysis/forgetting   # why: see below
```

Fresh GPU box: `bash scripts/remote_setup.sh` (uv, env, data rebuild + verify).

## Layout

```
src/clharness/
  config.py        strict YAML -> pydantic models, include: merging
  env.py           .env loading, device pick, seeding (determinism caveats documented there)
  model.py         Gemma 3 loading; decoder_layers() accessor
  activations.py   forward-hook residual capture (== output_hidden_states[L+1], tested)
  sae.py           Gemma Scope 2 via SAELens; reconstruction metrics (mse, fvu, l0, cos)
  metrics/cka.py   streaming linear CKA (feature-space form, float64)
  metrics/sae_stats.py  per-feature freq / mean magnitude accumulators (compact)
  data/sources.py  deterministic document sources (HF streaming / raw files)
  data/build.py    split -> tokenize -> write jsonl + npy + manifest
  data/loader.py   packed training stream (seeded), fixed held-out / probe chunks
  data/manifest.py sha256 manifests, verify_manifest
  evals/lm_eval_runner.py  in-process lm-eval with frozen item lists (samples=)
  evals/perplexity.py      held-out NLL / PPL on fixed chunks
  train/loop.py    the driver: domain order, token budgets, gradient-token counter, checkpoints
  train/hooks.py   Hook protocol (on_checkpoint, select_batch, ...) — policies plug in here
  train/measure.py EvalHook, ProbeHook, ItemDriftHook
  analysis/forgetting.py  which questions are forgotten, when and how (exploratory)
  wandb_log.py     wandb + metrics.jsonl
```

## Mechanism probes (exploratory)

Per-item files in each checkpoint folder, copied by `clh fetch-items` with the rest:

| File | What | Written by |
|---|---|---|
| `items_<task>.npy` | per question: 1 right, 0 wrong | every run |
| `choices_<task>.npy`, `gold_<task>.npy` | the option picked, the right option | every run |
| `loglik_<task>.npz` | per question and option: `raw` log-likelihood and the `score` the pick is the argmax of (length-normalised for acc_norm); NaN-padded | every run since 2026-11-15 |
| `drift_<task>.npz` | per question and layer: `cos` (cosine of the last prompt token's residual to the base model's) and `norm_ratio` | bench `fast-mech` only |

`configs/bench/fast-mech.yaml` is `fast.yaml` with `eval.item_drift_layers: [7, 13, 22]`.
Training and evals are identical (tested). Its rows have their own spec hash, so a test arm on
it needs its control on it too.

`clh forget-sets` reads those files, selection half only, and reports per arm:
- **Repeatability:** a held-out seed's losses predicted by how many other seeds lost the
  question (AUC; 0.5 = random).
- **Dose or target:** the slope of the arm's losses on each question's fragility in the other
  arms, against a simulated dose-only null.
- **Timing:** losses already there after the first domain, and recoveries.
- **Answer drift:** whether seeds that lose a question pick the same wrong option.
- **Margins and state drift:** base margins and cosine drift of lost vs kept questions, when the
  new files exist.

`analysis.forgetting.agent_summary` turns the result into a few lines for an agent prompt.

## Notes that affect interpretation

- **BOS is excluded from SAE and CKA statistics** (`exclude_bos_from_stats`). Gemma's BOS
  residual has ~10× the norm of other tokens and dominates variance; with it included the SAEs
  appear to explain >99% of variance, without it 78–95% depending on layer.
- **Generation evals execute model output.** MBPP and HumanEval are scored by running the
  generated Python (HF `code_eval`, subprocess + timeout + reliability guard). The runner sets
  `HF_ALLOW_CODE_EVAL=1` for you; run evals on a machine you would be comfortable re-imaging.
- **Precision.** `train.precision` is required. `bf16-pure` is for plumbing only: at small
  learning rates Adam's updates fall below bf16's resolution and are partly rounded away. Real
  runs use `bf16-mixed` (fp32 master weights, bf16 autocast for every forward, including evals).
- **Gradient tokens** are counted from the label mask, so a token-masking policy is counted
  correctly without touching the loop.
- **Determinism:** seeds cover python / numpy / torch; `torch.use_deterministic_algorithms`
  is on with `warn_only`. MPS makes no determinism guarantees. See `env.py`.
