# Modern LLM pretraining: a foundation for the continual-learning work

*Written 2026-09-10 as background for the project. Compiled from four literature sweeps (data,
architecture, training recipe, continual and prospective methods); primary sources are listed
at the end. Tags: **[standard]** = used by most frontier and open labs; **[emerging]** = several
labs or strong evidence, not universal; **[prospective]** = research-stage. Companion to
`docs/overview.md` and `docs/literature-review.md`.*

## 1. The picture in one paragraph

Pretraining is next-token prediction on a very large, mostly web-derived text corpus. "They
shuffle the data" is true, but only for the bulk phase: a 2026 run is a *pipeline* of a curated
and mixed corpus, a decoder-only Transformer with a fairly settled recipe, an optimizer and
learning-rate schedule chosen so the run can be extended or branched, and two or three
*stages* whose data mix and context length change on purpose. Almost every choice in that
pipeline exists to make the model learn a fixed mixture as efficiently and stably as possible.
Continual pretraining, our setting, breaks the fixed-mixture assumption, and most of what goes
wrong (and most of the known fixes) can be read directly off the standard recipe.

## 2. Data: the part that has changed most

**Sourcing and scale [standard].** Nearly all open corpora start from Common Crawl, re-extracted
from raw HTML, then filtered: FineWeb (15T tokens), FineWeb-Edu (1.3T), DCLM (~4T), Dolma 3
(~9T), Nemotron-CC (6.3T, of which 1.9T synthetic). Code, math, PDFs and books are added on top.
Frontier token budgets are 15 to 36T (Llama 3 15.6T, DeepSeek-V3 14.8T, Qwen3 36T); Gemma 3 1B,
our base, saw 2T. The Chinchilla rule of ~20 tokens per parameter is now only a compute-optimal
reference; small models are deliberately over-trained by 100x or more because inference cost
dominates.

**Filtering [standard].** Rule-based filters (Gopher and C4 heuristics, language ID, URL
blocklists) followed by a model-based quality classifier. FineWeb-Edu scored documents for
"educational value" with an LLM and distilled that into a small classifier; DCLM used a fastText
classifier trained on instruction-style positives. Classifier filtering beats perplexity
filtering clearly. Toxicity and PII are mostly masked rather than dropped.

**Deduplication [standard].** Exact and MinHash near-duplicate removal. Dedup cuts verbatim
memorization roughly tenfold and improves loss per step, but *more* dedup is not monotonically
better: FineWeb found global dedup across all snapshots left mostly boilerplate, so pipelines
dedup per snapshot or per shard. Semantic dedup (SemDeDup, D4) is **[emerging]**.

**Mixing [standard practice, emerging methods].** Domain weights (web, code, math, multilingual)
are set by small-scale ablations; Llama 3's final mix was about half general web, a quarter
math and reasoning, a sixth code. Automated methods fit a proxy: DoReMi (a small model reweights
domains), RegMix (hundreds of tiny runs, regress performance on mixture weights), and mixture
scaling laws. Every serious lab runs some version of this; no algorithm is canonical.

**Stages and curriculum [standard].** This is the main exception to "shuffle everything":

1. *Bulk* pretraining on the broad mix at short context (Qwen3: 30T tokens at 4k).
2. *Mid-training / annealing*: the learning rate decays to near zero while the mix shifts to
   high-quality math, code, instruction and reasoning data (Llama 3 annealing; OLMo's
   "Dolmino" mix took a 7B model's GSM8K from 24 to 68). The MiniCPM finding that started this:
   high-quality data pays off mainly during the decay.
3. *Long-context extension* on tens to hundreds of billions of tokens (Llama 3: 8k to 128k in
   six steps).

Within each stage, data is shuffled. Fine-grained ordering (easy-to-hard curricula) gives
single-digit gains at small scale and is not used by frontier labs; one 2025 result matters for
us: under a decaying schedule, what the model retains depends on the learning rate at exposure
time, so data seen late at low LR is learned weakly. Ordering and schedule are entangled.

**Synthetic and rephrased data [standard for rephrasing].** Rewriting web documents with a
model (WRAP, Nemotron-CC) is routine at NVIDIA, Moonshot, Alibaba and Hugging Face; from-scratch
generation (Phi, Cosmopedia) mostly serves small models. Kimi K2 reports that ten rephrasings
seen once beat the raw text seen ten times. Model collapse is real when synthetic data
*replaces* real data across generations and absent when it *accumulates*; labs keep synthetic
data a minority of the mix.

**Repetition [standard].** Up to about four epochs of a corpus costs almost nothing relative to
fresh data (Muennighoff et al.); only the high-quality slice is normally repeated. Distillation
from a larger teacher's logits (Gemma 2/3, Llama 3.2 1B, Minitron) is the largest known lever for
small models and was used to train our base.

## 3. Architecture: settled, with one big fork

**The default dense decoder [standard].** Pre-norm RMSNorm, SwiGLU feed-forward, rotary position
embeddings (RoPE), grouped-query attention, no bias terms, QK-norm for stability, and
interleaved local and global attention layers (Gemma 3 uses five sliding-window layers per global
layer). Small models tie input and output embeddings and go deep and thin (Gemma 3 1B has 26
layers). Gemma 3 1B is a textbook instance of this recipe.

**Mixture-of-experts [standard at frontier].** Every open frontier release since DeepSeek-V3 is
MoE: many small experts, one shared always-on expert, auxiliary-loss-free load balancing, and
activation ratios now between 1/18 and 1/56 of total parameters (DeepSeek-V3 671B total, 37B
active). MoE is not worth it at 1B scale.

**Attention efficiency [emerging, moving fast].** Multi-head latent attention (DeepSeek), trained
sparse attention (NSA, DeepSeek Sparse Attention), and above all *hybrids* that replace most
attention layers with linear or state-space layers (Gated DeltaNet in Qwen3-Next, Mamba-2 in
Nemotron 3 and Granite 4). The recurring finding is that one full-attention layer per four to
ten linear layers keeps retrieval ability while making state size constant in sequence length.

**Notably new in 2025-26.** Multi-token prediction as an auxiliary objective (DeepSeek-V3
onward); gated attention and learned attention sinks, which remove the "massive activation"
dimensions we track in T3 of the research notes; looped or recurrent-depth models at 1 to 3B;
diffusion language models as a fast-inference niche, not a replacement for autoregressive
pretraining.

## 4. The training recipe

**Objective [standard].** Cross-entropy on the next token; fill-in-the-middle for code models;
multi-token prediction increasingly common. Reinforcement-learning-flavoured pretraining
objectives (RPT, RLP) exist as a phase on top of existing bases and are **[prospective]**.

**Optimizer [standard: AdamW].** Betas 0.9 and 0.95, weight decay 0.1, gradient clipping at 1,
no weight decay on embeddings. Muon (orthogonalized momentum) is the first real challenger:
Kimi K2, GLM-5 and DeepSeek-V4 use it. The careful 2025 benchmark (Wen et al., Marin) found
matrix optimizers beat tuned AdamW by 1.1 to 1.4x, with the gap shrinking as models grow; many
2x claims came from under-tuned baselines. Weight decay, which sets the effective angular
learning rate, is now argued to matter more than the optimizer choice.

**Learning-rate schedule [standard: warmup-stable-decay].** Warmup then cosine to 10% of peak was
the 2023-24 norm. WSD (constant LR after warmup, then a short decay of 10 to 20% of the run)
matches cosine and lets you branch a cooldown from any checkpoint, which is why DeepSeek, Kimi
and OLMo use it. The "river valley" picture explains it: a high constant LR moves fast along a
flat river while bouncing across steep valley walls; the decay settles into the floor. Checkpoint
averaging (EMA, LAWA, merging stable-phase checkpoints) can replace the decay **[emerging]**.

**Hyperparameter transfer [emerging].** muP makes the optimal LR width-invariant so you tune on
a tiny model; CompleteP extends this to depth. In practice labs fit empirical power laws for LR
and batch size on a ladder of small runs (DeepSeek, Step Law) and check one mid-size run.

**Batch, precision, stability [standard].** Batch size and sequence length are ramped up during
the run because the critical batch size grows with data seen. bf16 with fp32 master weights is
standard; DeepSeek-V3 trained in FP8 and NVIDIA has trained at FP4. Loss spikes are handled by
QK-norm, z-loss, clipping, and skipping bad batches from a checkpoint.

**Scaling laws [standard].** Kaplan, then Chinchilla (scale parameters and tokens together),
then inference-aware over-training. Loss on a held-out mix predicts downstream accuracy only
noisily, especially below 1B parameters, where discrete accuracies sit near chance and
continuous likelihood metrics are far more predictive (DataDecide, 2025).

**Evaluation during a run [standard].** Train loss, per-source held-out loss, and a small
early-signal suite at fixed checkpoints, with the last few checkpoints averaged to denoise.
That is the shape of our harness.

## 5. Why shuffling is the default, and what breaks when we stop

An i.i.d. shuffle makes each minibatch gradient an unbiased estimate of the gradient on the
whole mixture, so the optimizer approximates joint minimization and every checkpoint has seen
the intended mix. Sequential data biases the gradient toward the current domain for the whole
block; with one shared parameter vector that bias overwrites features earlier domains needed,
and if the block is long relative to the LR, the old minimum is simply no longer a stationary
point. The whole continual-pretraining toolkit is a set of partial repairs of that
unbiasedness:

- **Learning rate is the dominant forgetting knob.** Higher LR learns the new domain faster and
  forgets the old one more, monotonically (Gupta 2023; Ibrahim 2024). Our sweep saw the same:
  the 3e-5 setting was chosen as the graded middle between 1e-5 and 1e-4.
- **Re-warming causes a forgetting spike.** A fully decayed checkpoint sits on the valley floor,
  so re-warming climbs out and perturbs old behaviour. Fixes: re-warm then re-decay with replay
  (Ibrahim), never re-warm and just keep decaying (NVIDIA's "Reuse, Don't Retrain"), or
  "infinite" schedules that hold a constant LR and branch short anneals per phase. Our pilot's
  constant LR after a 50-step warmup is the infinite-schedule stable branch without the anneals.
- **Replay** of 1 to 5% of the old mixture nearly closes the gap to joint training for mild
  shifts; 25% was used for a strong shift like English to German. Our code, biomedical, legal,
  math sequence is a strong shift, and our runs use zero replay, so we are at the high end of
  expected forgetting by design.
- **Weight averaging** across checkpoints or between base and adapted weights (model soups,
  WiSE-FT, "Soup to go") is a cheap forgetting baseline we have not yet run.
- **Position in the stream matters** under any decaying schedule; constant LR is what makes our
  four domains comparable (the comment in the pilot config says exactly this).
- **Plasticity loss** is the other failure mode: under long continual training, effective rank
  falls and units die, and 2026 work shows this in GPT-style models at every size.

Data selection methods that read the model's state exist (Rho-1 selective tokens, MATES
learned influence, ADO online reweighting), but they select for learnability of the target, not
retention of the past. Retention-aware selection is essentially the replay literature plus
2025-26 work on surprise-prioritized and self-generated replay. That is the gap the project's
Goal B sits in.

## 6. Prospective directions worth knowing

- **Memory layers and test-time training [emerging to prospective].** Meta's memory layers
  (128B key-value parameters, beats dense at twice the compute on factual tasks); Titans and
  Google's Nested Learning / HOPE (surprise-gated memory updated at inference); sparse memory
  finetuning, which updates only memory slots the new data activates and cuts forgetting from
  89% to 11% on a held-out QA measure at equal new-knowledge gain. The last one is the closest
  published cousin of feature-level protection.
- **Modular training [emerging].** Branch-Train-Merge and Branch-Train-MiX train per-domain
  experts from a seed and fuse them into an MoE; this is the most credible no-interference
  answer to a four-domain setup, paid for in parameters.
- **On-policy and self-distillation** for less forgetful updates, contested in 2026 results.
- **RL as a pretraining objective**, diffusion LMs, byte-level tokenizer-free models (BLT,
  H-Net): real research, none shipped as a frontier pretraining objective.
- **The industry framing.** Continual learning is openly called a major unsolved problem
  (Patel, Sutskever, Karpathy, Amodei in 2025-26 interviews); optimism centres on memory-layer
  and modular routes, and no frontier lab has published a weight-updating continual system.

## 7. How this maps onto our setup

| Standard practice | Our pilot | Implication |
|---|---|---|
| Shuffled fixed mixture | Four domains in sequence, 200M tokens each | Forgetting is the object of study, not a bug |
| WSD or cosine schedule | Constant 3e-5 after 50-step warmup, no decay | Domains comparable; no valley-floor settling, so evals are noisier |
| 1 to 25% replay | Zero replay | Upper bound on forgetting; replay is the obvious Phase 3 baseline |
| Checkpoint averaging | Not run | Cheap baseline to add |
| Held-out per-source loss + continuous MC metrics | Per-domain perplexity + protected MC set | Prefer continuous likelihood over accuracy at 1B |
| Teacher distillation for small models | Base was distilled | Base features may be unusually compact; relevant to fragility (T4) |

## Key sources

Data: FineWeb (2406.17557), DCLM (2406.11794), Nemotron-CC (2412.02595), OLMo 2 (2501.00656),
Llama 3 (2407.21783), Qwen3 (2505.09388), Gemma 3 (2503.19786), Kimi K2 (2507.20534), RegMix
(2407.01492), DoReMi (2305.10429), MiniCPM/WSD (2404.06395), Muennighoff data-constrained
scaling (2305.16264), WRAP (2401.16380), DataDecide (2504.11393), "How LR Decay Wastes Your
Best Data" (2511.18903).
Architecture: DeepSeek-V3 (2412.19437), DeepSeek-V2/MLA (2405.04434), Gated DeltaNet
(2412.06464), Nemotron 3 (2512.20848), Gated Attention (2505.06708), MobileLLM (2402.14905),
Raschka, "The Big LLM Architecture Comparison" (2025).
Recipe: Wen et al., "Fantastic Pretraining Optimizers" (2509.02046), Muon is Scalable
(2502.16982), Hägele et al. WSD scaling (2405.18392), river-valley (2410.05192), muTransfer
(2203.03466), CompleteP (2505.01618), Step Law (2503.04715), Chinchilla (2203.15556), Beyond
Chinchilla-Optimal (2401.00448), Wortsman small-scale instabilities (2309.14322).
Continual: Gupta et al. re-warming (2308.04014), Ibrahim et al. (2403.08763), Parmar et al.
"Reuse, Don't Retrain" (2407.07263), infinite LR schedules (2503.02844), replay and gradient
alignment (2508.01908), forgetting scaling laws with data injection (2502.06042), Dohare et al.
loss of plasticity (Nature 2024), sparse memory finetuning (2510.15103), Branch-Train-MiX
(2403.07816), Titans (2501.00663), Memory Layers at Scale (2412.09764), RPT (2506.08007).

*Caveats: a few figures were read from paper bodies by the research agents rather than
checked against tables (the NVIDIA and infinite-schedule numbers); Western closed labs publish
no optimizer or schedule details, so "standard" means standard among open-weight reports.*
