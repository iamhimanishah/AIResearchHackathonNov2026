# Literature review — Phase 0 gate (2026-09-06)

Purpose: the plan (§3, Phase 0 gates) requires confirming that the feature-level forgetting
characterization does not already exist and positioning the project against existing work
before Phase 1. This is the first thorough pass; entries are grouped by the question they bear
on. Each ends with **Implication** for our design. A design-review memo with proposed changes
follows in `docs/design-review-2026-09-06.md`.

Method: web search + abstract/full-text reads on 2026-09-06. Coverage is broad but not
exhaustive; gaps are listed at the end. Where a paper's claims look unreliable this is said.

---

## A. Feature-level analysis of fine-tuning and forgetting (closest to Phase 1 / the demo)

**A1. Chopra, "A Mechanistic Investigation of Supervised Fine Tuning" (arXiv 2605.11426, May 2026).**
Gemma 3 1B **IT** + Gemma Scope 2 SAEs at **layers 7, 13, 22** (16k/65k/262k), SFT on four
2,000-example tasks (MultiNLI, GSM8K, WildJailbreak, tool calling), eval every 400 samples.
Tracks raw-activation cosine (stays > 0.96), SAE-latent cosine (drops, e.g. layer 22 0.74 → 0.56),
PCA of the activation delta, and "feature flips" along principal drift directions, then labels
flipped features with an LLM judge (persona/structure/code/reasoning/safety/…). Findings: early
layers change low-rank, deep layers diffusely; code features are disrupted even by non-code SFT;
safety SFT rewrites shallow layers. **No benchmark performance measured; no SAE reconstruction
validity check on the fine-tuned model** (acknowledged). Single independent author.
*Implication:* the mechanics of "track Gemma Scope features across fine-tuning checkpoints" are
published, at our exact layers. What is **not** done there and is our Phase 1: PT model,
continual **pre**-training on domain streams at 10^8 tokens rather than 2k-example SFT,
capability ground truth (protected set + perplexity) alongside the features, SAE validity
tracked at every checkpoint, a drift-immune measure, and the leading-indicator test. Cite and
position explicitly; keep our layers (comparability).

**A2. Masip, van de Ven, Ferrando, Tuytelaars, "Putting a Face to Forgetting: Continual Learning
meets Mechanistic Interpretability" (CoLLAs 2026, arXiv 2601.22012).** Toy model + ViT on split
CIFAR-10 with crosscoders. Formalizes two feature-level forgetting mechanisms: **capacity
degradation** (a feature's direction fades or overlaps others; unrecoverable downstream) vs
**readout misalignment** (feature intact, downstream stops reading it; recoverable). Finds
fading + misalignment dominate, overlap does not; early layers *gain* capacity; depth amplifies
forgetting. States future work: "tracking our metrics on more fine-grained timescales, e.g.
around task transitions" and SAE-based real-time tracking.
*Implication:* adopt their decomposition. Our per-feature frequency/magnitude ≈ fading; readout
must be measured separately (protected-set evals, or linear probes on the probe set). This
turns Phase 1 Q1 ("concentrated or diffuse") into a sharper question: fading vs readout loss,
per layer. Our study is the LLM-scale version of their stated future work.

**A3. Minder, Dumas, …, Nanda, "Narrow Finetuning Leaves Clearly Readable Traces in Activation
Differences" (ICLR 2026, arXiv 2510.13900).** Base-vs-finetune activation differences on the
first few tokens of random text reveal the fine-tuning objective; Gemma/Llama/Qwen 1B–32B.
*Implication:* activation-difference model diffing is an established, cheap readout of "what
changed"; a mean activation delta on the probe set is a near-free extra signal for Phase 1/2.

**A4. Kissane et al., "SAEs (usually) transfer between base and chat models" (2024); Gemma Scope
2 technical paper (2026).** Base SAEs spliced into chat models give similar CE loss for
Mistral-7B and Qwen-0.5B; Gemma v1 2B was the exception (unusually large weight delta). Gemma
Scope 2 **initialises IT SAEs from PT SAEs** and notes PT→IT transfer "typically does not
require resampling a large fraction of latents". A separate replication (AF post) found severe
degradation on Gemma-2b after Python fine-tuning (R² 0.97 → negative).
*Implication:* instrument validity under fine-tuning is model- and magnitude-dependent, and can
collapse. Our per-checkpoint FVU/L0 tracking is necessary, not optional. Gemma Scope 2 metrics
were computed on 2,048 × 1,024-token sequences **with special tokens masked** — consistent with
our BOS exclusion and seq_len 1024.

**A5. "Real-time detection of spurious forgetting" (arXiv 2512.20634, Dec 2025)** monitors
"alignment depth" over the first output tokens to flag shallow (spurious) forgetting during
training. Different signal; overlapping ambition (a monitor).

---

## B. What forgetting is, in LLMs (definitions and prior characterizations)

**B1. Zheng et al., "Spurious Forgetting in Continual Learning of Language Models" (ICLR 2025).**
Large drops on prior tasks often reflect lost **task alignment**, not lost knowledge: after
Task-0 accuracy fell to ~10%, one epoch on half of Task 0 recovered ~100%. Weight-space analysis:
later steps update bottom layers (incl. embeddings) nearly orthogonally to Task 0. Freezing
bottom layers helps. They do not quantify the spurious fraction in realistic settings.
*Implication:* (i) our protected set is log-likelihood MC + perplexity on a **PT** model, which
is less exposed to instruction-alignment artifacts than task-accuracy on IT models — state this
as a design strength; (ii) add a **recoverability probe** to Phase 1: at domain boundaries, take
the checkpoint, briefly re-warm on a small general slice, re-eval; the recoverable share is
"spurious", the rest is genuine. Cheap and it makes the characterization far more credible.

**B2. Harmon, Hochlehnert, Bethge, Prabhu, "Mapping Post-Training Forgetting in Language Models
at Scale" (arXiv 2510.17776, Oct 2025).** Sample-wise 1→0 / 0→1 transitions with chance
adjustment (F_true = max(F − F_chance, 0)). Domain-continual pretraining (Qwen2.5-Coder/-Math,
3B–32B, web-scale domain corpora): **forgetting 6.4–12.9% of items, backward transfer 5.1–6.3%**,
larger models marginally less; culture/knowledge categories most fragile; instruction-following
fidelity weakens (answers embedded in code, unprompted multilingual output).
*Implication:* (i) our frozen item lists make **per-item flip metrics** free — report 1→0 and
0→1, chance-adjusted, alongside accuracy; (ii) **effect-size calibration**: at 3B+ on far more
tokens than ours, forgetting is single-digit to low-double-digit percent of items. At 1B with
100–200M tokens/domain expect the same order; the pilot must produce ≥ ~10% item flips to give
Phase 3 room, and a "2-point retention gain" minimum effect is plausible but not generous.

**B3. Yildiz et al., "Investigating Continual Pretraining in LLMs" (2024).** Domain-sequence
continual pretraining, GPT-2-class and < 1.5B: smaller models show the largest learning *and*
forgetting rates; semantically similar order aids specialization, random order aids transfer.
*Implication:* 1B is a sensitive regime (good for detecting effects); domain order matters —
keep the fixed order plus the reversed-order robustness run.

**B4. Imanov, "Mechanistic Analysis of Catastrophic Forgetting in LLMs During Continual
Fine-tuning" (arXiv 2601.18699, Jan 2026).** Claims CKA drops, attention-head disruption,
gradient-interference localization, Hessian flattening preceding accuracy drops by 1–2 epochs,
and first-epoch gradient alignment predicting final forgetting (r = −0.79). **Reliability
concern:** the stated setup includes full fine-tuning with Hessian analysis of GPT-5.1, Claude
Opus 4.5 and Gemini 2.5 Pro, which is not possible for an outside author. Treat as unverified;
do not build on it. Noted because its headline claims coincide with our leading-indicator
hypothesis and a reviewer will ask.

**B5. Jiang et al., "Instruction Vector" (2024)** and related "superficial vs essential
forgetting" work: fine-tuning "mostly adds specialized reasoning patterns instead of erasing
previous skills". Same message as B1.

**B6. Rofin, Varre, Flammarion, "(How) Learning Rates Regulate Catastrophic Overtraining" (COLM
2026).** Pretraining LR decay increases sharpness, which exacerbates forgetting during SFT.
*Implication:* forgetting depends on the base checkpoint's sharpness; results at 1B PT may not
transfer across model families. Report it as a limitation; the pilot LR sweep stays.

---

## C. Selection signals: loss/gradient-based (the baselines we must beat)

**C1. SuRe (Hazard et al., arXiv 2511.22367, Nov 2025).** Replay priority = per-sequence
mean-token NLL under the current model; buffer 2% of data with equal per-task quota; fast/slow
LoRA (rank 8 on Q/V) merged by EMA (β ≈ 0.995). Main results on **T5-Large** (Llama 3.1 8B in
an appendix): 78.1% standard CL, 75.1% on a 15-task sequence, up to +5 points over prior SOTA;
sequence-level surprise beats random replay, with the gain growing as the replay budget shrinks. *Implication:* the scalar-surprise baseline is exactly this; and Phase 4's
two-timescale system already exists in a simple form — Phase 4 must differ (the consolidation
*signal*), or be dropped.

**C2. Titans (Behrouz et al., 2025).** Surprise = gradient of the memory's associative loss,
momentary + momentum, with a forgetting gate. Confirms: the field's importance signal is a
loss-gradient scalar.

**C3. LESS (Xia et al., 2024): LoRA gradients, random-projected, Adam-aware; influence =
cosine with validation gradients over warmup checkpoints; 5% selected data beats the full set;
a small model's gradient store transfers to larger models. GrADS / "Learn More, Forget Less" (arXiv 2511.08620, Nov 2025);
GradAlign, LearnAlign, gradient-orthogonality selection (2025–26).** Select training examples
by gradient similarity to a target/validation set, or by gradient magnitude/distribution;
GrADS reports forgetting mitigation as a side effect at 5–50% of the data. Some work notes LESS
is "blind to safety": maximizing target similarity picks examples that conflict with general
capabilities.
*Implication:* Phase 3's loss-based baselines are incomplete without a **gradient-alignment
selection baseline** (LESS-style: select by alignment with the protected-set / new-domain
gradients). Since Phase 2's labels *are* gradient alignment, this baseline is the "oracle-ish"
loss-side competitor; a representation signal that beats it is a real result.

**C4. MATES (Yu, Das, Xiong, NeurIPS 2024).** A small **text-only** influence model predicts
oracle influence (locally probed) and steers pretraining data selection; halves FLOPs to a
given performance at 410M–1B. *Implication:* the cleanest framing of our learned head is
"influence prediction from model internals vs from text": MATES is the text-only baseline.

**C5. AC-ODM (2025) / ODM.** Online domain mixing with loss or **gradient-alignment reward**.
*Implication:* domain-level selection by gradient alignment is established; ours is
example-level and protection-aware.

**C5b. "Diversity-driven Data Selection for LM Tuning through Sparse Autoencoder" (arXiv
2502.14050, 2025).** Selects SFT data by diversity of activated SAE features. *Implication:*
SAE-feature-based *selection* has a precedent (diversity criterion, SFT). Ours differs in the
criterion (novelty / weakness / protection) and the setting, but the "first SAE-based data
selection" claim is off the table.
**C6. MIR (Aljundi et al., 2019), GSS (2019), gradient-coreset replay (2022).** Loss-increase-
under-virtual-update and gradient-diversity criteria for replay in classical CL.
*Implication:* "what to replay" by interference is old; state the lineage.

---

## D. Representation-derived importance and protection (Phase 3's protection variant)

**D1. Ning, Xue, Lou, Guo, "From Weights to Features: SAE-Guided Activation Regularization for
LLM Continual Learning" (arXiv 2606.26629, Jun 2026).** Gemma-2 9B-it + LoRA, **public Gemma
Scope 16k SAEs at layers 9/20/31**. Importance = mean |activation| per feature on task data,
propagated over decoder similarity; protect-loss on low-relevance features, guide-loss on
relevant ones. Beats EWC/MAS/SI/ELLA and 10% replay on TRACE and a biomedical sequence; loses to
O-LoRA. Storage 0.4 MB per task vs GBs for weight anchors. Limitation: SAEs capture only 85–95%
of variance; output-format interference not addressed.
**D2. SAE-FD (arXiv 2605.25525, May 2026).** Own gated SAEs on last-layer MLP of 7B chat models;
distill direction + magnitude of previously-active features; best on TRACE among regularizers.
**D3. SAE-FT for CLIP (arXiv 2605.15961, May 2026).** Penalize adding/removing SAE features
during fine-tuning.
**D4. SEEKR (He et al., EMNLP 2024).** Head importance = task sensitivity (‖∂L/∂A‖) ×
forgettability (attention-map change across tasks); attention distillation on top heads; 1%
replay matches 10%. **PIECE (2025):** Fisher / second-order importance to update 0.1% of
parameters. **MAS (2018):** importance from output sensitivity to parameters.
*Implication (D1–D4):* "protect representation directions identified as load-bearing" is now
published for LLMs with the very SAEs we use, on task-incremental SFT. The plan already marks
gradient-projection protection as secondary; demote it further to "reuse D1's method as a
baseline if protection is tested at all". Our distinct axis is **selection** (what to train on /
replay) in **continual pretraining** with compute matching, and the *comparison of signals*, not
protection machinery.

---

## E. Instrument reliability

**E1. Sun et al., "Massive Activations in LLMs" (2024)** and follow-ups (2025–26): a few
dimensions on the first token reach ~10^5× median magnitude, act as fixed biases/attention
sinks, and are architecturally robust.
**E2. Davari et al., "Reliability of CKA as a Similarity Measure" (ICLR 2022).** CKA is driven by
outliers; a single far-away point or a translated subset can move it arbitrarily without
functional change.
**E3. Paulo & Belrose, "SAEs Trained on the Same Data Learn Different Features" (arXiv
2501.16615, 2025).** Only ~30% of features are shared across seeds for a 131k-latent SAE on
Llama 3 8B; TopK worse than ReLU/L1. Follow-ups (2025–26) on SAE benchmark reliability and
feature stability say the same: an SAE's dictionary is a useful decomposition, not the model's
canonical feature list.
*Implication:* any feature-level claim ("these 40 features decayed") is SAE-specific. Gemma
Scope 2 ships a second seed for the 262k/medium SAEs and several widths per layer; replicate the
headline feature-level findings on a second SAE at the same layer and report agreement.
Aggregate statistics (fraction of features decaying, decay concentration) are more robust than
identities.
*Implication:* our observed CKA drop to 0.44 at layer 13 after 49k tokens is expected behaviour
of an outlier-sensitive measure on outlier-heavy activations, not evidence of representation
change. Fix before Phase 1: (i) exclude BOS (done); (ii) report CKA on per-dimension-standardised
activations and/or with the massive-activation dimensions removed; (iii) add an outlier-robust
alternative (e.g. mutual-kNN overlap or Procrustes on standardised activations). Keep raw CKA
only as a secondary number.

---

## F. The vision side: self-directed curricula and consolidation

**F1. Nested Learning / Hope (Google, NeurIPS 2025)**: continuum memory system with modules at
different update frequencies; self-modifying memory. **F2. SEAL (2025)**: model generates its
own fine-tuning data/edits, RL on downstream improvement. **F3. Behrouz, Hashemi, Javanmard,
Mirrokni (the Titans authors, Google), "Language Models Need Sleep" (arXiv 2606.03979, Jun
2026)**: short-term in-context memory consolidated into weights by "knowledge seeding" (upward
distillation from a smaller self) plus a "dreaming" stage where RL generates a rehearsal
curriculum. The abstract does not say what decides *which* memories are consolidated. Also:
sleep-inspired consolidation for proactive interference (2026), semi-parametric memory
consolidation (2025), brain-inspired replay (van de Ven et al.).
*Implication:* the two-timescale, sleep/consolidation vision is being built by several groups,
including the Titans team with far more compute.
Phase 4 as written is not a differentiator. What none of these do is choose *what* to
consolidate by a content-aware, representation-derived importance signal validated against
loss-based signals — which is Phases 2–3. The vision narrative in the pitch should point at
that gap, not at "we will build a sleeping model".

---

## G. Positioning statement (draft)

Published: tracking Gemma Scope features across fine-tuning checkpoints on Gemma 3 1B (A1);
feature-level forgetting mechanisms (A2, vision); SAE-based protection for LLM CL (D1, D2);
loss/gradient-based selection and replay (C1–C6); domain-continual-pretraining forgetting
maps at the benchmark level (B2, B3); spurious-vs-genuine forgetting (B1).

Not published, as far as this pass finds: (1) a feature-level characterization of forgetting
during **continual pretraining** of an open LLM with capability ground truth, instrument-
validity tracking and a drift-immune control, including the leading-indicator test and the
fading-vs-readout decomposition; (2) a **compute-matched, seeded comparison of selection
signals** — loss, gradient alignment, raw-activation novelty, SAE-feature novelty/weakness, and
a learned head reading internals vs a text-only influence model — for continual pretraining,
scored on retention at matched acquisition. (1) is Phase 1; (2) is Phases 2–3. The essay-level
claim ("representation-derived importance beats loss-derived") is untested in this setting;
the closest results (D1, C3, C4) each test one side of it.

---

## H. Gaps in this pass

- Full-text reads pending for B2, B3, C1 (surprise details), C3 (LESS/GrADS setups).
- Covered in the second pass: Harmon et al. numbers (B2), SuRe details (C1), LESS details (C3),
  Ibrahim et al. 2024 (405M/10B; LR re-warm + re-decay + replay matches retraining; replay
  ratio numbers still to extract from the full text), SAE consistency (E3), SAE-based data
  selection (C5b), influence at scale (TRAK / EK-FAC / DataInf / LoRIF exist and are usable for
  Phase 2 labels on a subsample).
- Still open: TRACE/SuperNI conventions for comparability; the exact replay percentages in
  Ibrahim et al.; whether any 2026 work uses SAE novelty/weakness (not diversity) for selection.

---

## I. Harnesses and benchmarks for continual learning of LLMs (added 2026-09-06)

What exists, and what our harness is relative to it.

**I1. TiC-LM (Li et al., Apple, 2025; code apple/ml-tic-lm).** Time-continual pretraining over
114 Common Crawl dumps with time-stratified held-out loss, plus Wikipedia / StackExchange / code
docs evals; compares autoregressive meta-schedules + fixed-ratio replay; matches retrain-from-
scratch at 2.6× less compute; replay essential for generic data, less for specialised domains.
*Implication:* the most relevant public codebase for a temporal stream (the plan's deferred
Phase 4 stream). Reuse it rather than building a Wikipedia-diff stream if that ever comes back.

**I2. Abbes et al., "Revisiting Replay and Gradient Alignment for Continual Pre-Training of
LLMs" (2025).** Llama-family, 100B tokens per language; an efficient meta-experience-replay
implementation gives gradient-alignment benefits at negligible overhead. Finding: small replay
rates beat spending the same compute on model size, but scaling the model beats *high* replay
rates. *Implication:* the generic-replay baseline should be small (5%) by default; gradient
alignment inside replay is a published, cheap upgrade that belongs in the baseline set.

**I3. Ibrahim et al. 2024 (EleutherAI GPT-NeoX code + HF checkpoints).** LR re-warm + re-decay +
5% (weak shift) / 25% (strong shift) replay ≈ retraining. Public continual-pretraining code and
checkpoints exist for 405M/10B. *Implication:* our LR schedule should re-warm/re-decay per
domain rather than stay constant if we want to match practitioner practice; the pilot's
"constant LR" is a deliberate simplification that must be stated.

**I4. TRACE (2023), SuperNI-CL, MedCL; MemoryBench (2025); SkillLearnBench, Continual Learning
Bench (2026).** Task-incremental SFT sequences with AP / FWT / BWT metrics on 7B chat models;
newer ones test agent skills and memory systems. **FOREVER (ACL 2026)**: replay scheduled by a
forgetting-curve in "model time" (optimizer update magnitude) with intensity-aware regularisation,
0.6B–13B. **TFGN (2026)**: read/write parameter decomposition, replay-free, six domains × 1B
tokens at 0.4–9B (single author; large claims; unreviewed).
*Implication:* the standard CL-for-LLM protocol is task-incremental SFT with AP/FWT/BWT. Our
setting (domain continual pretraining, PT model, protected set) is closer to I1–I3 than to
TRACE. Report BWT-style numbers anyway for comparability, and consider a TRACE run only as a
side experiment (the plan already says this for SuRe's sequence).

**No public harness found** that combines: per-checkpoint fixed evals + held-out perplexity +
SAE feature statistics + instrument validity + representation similarity, with a hook API for
selection policies, on an open model with public SAEs. That is the harness's claim; it should be
stated that narrowly.

---

## J. Selective backpropagation and token-level training (added 2026-09-06)

The mechanism side of "what to train on": which tokens/examples receive gradient.

**J1. Selective-Backprop (Jiang et al. 2019)** keeps high-loss examples with probability from
the loss CDF; up to 3.5× faster on CIFAR. **Balles, Archambeau, Zappella (2023), "A Negative
Result on Gradient Matching for Selective Backprop"**: neither loss-based nor gradient-matching
subset selection consistently beats a *random* subset baseline that prior work omitted.
*Implication:* per-example loss-based selection is not reliably better than random even for
efficiency; a random-selection baseline at matched gradient tokens is mandatory in Phase 3 (the
plan has "random replay"; make sure random *selection* is there too).

**J2. Rho-1 / Selective Language Modeling (Lin et al. 2024).** Reference-model excess loss
scores tokens; training on the top fraction gives +16–30% on math tasks at 5–10× fewer tokens.
**ESLM (Bal, Cevher, Muehlebach, 2025)**: reference-free online token selection by per-token
loss/entropy with value-at-risk thresholding; keeps FLOPs down at equal or better perplexity on
GPT-2-scale pretraining. **"80/20" high-entropy-token findings** (2025–26): a small fraction of
high-entropy tokens carries most of the learning signal.
*Implication:* the token-level loss baseline family has a reference-free member (ESLM); include
it. Our harness already counts gradient tokens from the label mask, so token-level policies are
compute-matched automatically.

**J3. Wu et al., "Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning"
(2025).** Masking *high*-perplexity tokens (under the current model) from the SFT loss preserves
non-target-task performance about as well as training on LLM-generated data; Gemma 2 2B IT,
Llama 3 8B. *Implication:* the direct precedent for "token-level selection reduces forgetting",
and note the sign: it drops the *surprising* tokens — the opposite of Rho-1/SuRe's "train on the
surprising ones". Which sign helps depends on whether the goal is acquisition or retention. This
is exactly the select-vs-protect split of memo item 7, at token level; a two-sided token policy
(train on high-excess-loss tokens *unless* they conflict with the protected gradient) is a
natural Phase 3 candidate.

**J4. Gradient routing (Cloud et al. 2024) and SGTM (2025).** Data-dependent gradient masks
localise capabilities to parameter subsets; used for capability removal / unlearning.
*Implication:* the parameter-space cousin of "protect load-bearing directions"; mostly an
unlearning tool. Not a Phase 3 candidate, but the right citation if protection is discussed.

**J5. Token-level unlearning (2026, several).** Entropy/attribution-based token weights for
unlearning. Adjacent only.
