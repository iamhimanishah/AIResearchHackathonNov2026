# Continual learning: a field overview

*Written 2026-09-10 as a foundation for following this project. Deliberately project-agnostic. Sources are listed at the end; items that were only secondhand are marked.*

---

## 1. The problem

Continual learning (CL, also "lifelong learning") asks how a model can keep learning from a stream of new data without retraining from scratch and without keeping all the old data. The obstacle is **catastrophic forgetting**: train a network with ordinary gradient descent on task B after task A, and its performance on A collapses. The field's framing of the goal is the **stability–plasticity trade-off**. Too much stability and the model cannot learn anything new; too much plasticity and it forgets what it knew.

A second, younger problem sits next to forgetting. Dohare, Sutton and colleagues showed in *Nature* (2024) that networks trained on very long streams also **lose plasticity**: after thousands of tasks, standard SGD and Adam learn new tasks no better than a shallow network, and a large fraction of units go dead. So the modern statement of the problem is two-sided: keep what you knew, and keep the ability to learn.

Three standard scenarios (van de Ven et al., *Nature Machine Intelligence* 2022) define how hard a setting is:

| Scenario | What changes | Task identity at test time | Difficulty |
|---|---|---|---|
| Task-incremental | new tasks, separate output heads | given | easiest |
| Domain-incremental | input distribution shifts, labels fixed | not given | medium |
| Class-incremental | new classes keep arriving, single head over all | not given | hardest |

Orthogonal axes: **offline vs online** (each sample seen once), **task-free** (no announced boundaries), and, increasingly, **continual pre-training** of a foundation model on a stream of corpora. The domain-incremental and continual-pre-training framings are the ones that matter most for language models.

## 2. The classic toolkit

Nearly every method is one of four ideas, usually combined.

- **Regularization**: penalize changes to weights that mattered for old tasks, or distill the old model's outputs. Canonical: EWC (Kirkpatrick et al. 2017, Fisher-weighted penalty), SI (Zenke et al. 2017), LwF (Li and Hoiem 2016).
- **Replay / rehearsal**: keep a small buffer of old examples and interleave them with new ones, or regenerate them with a generative model. Canonical: iCaRL (2017), GEM and A-GEM (2017, 2019), experience replay. Replay is the strongest simple baseline and is essentially required in the class-incremental setting.
- **Parameter isolation / architecture**: give each task its own sub-network by growing, pruning, or masking. Canonical: Progressive Networks (2016), PackNet (2018), HAT (2018). Zero forgetting, but capacity grows or fragments.
- **Sparse or low-rank adaptation on a frozen backbone**: keep the pretrained model fixed and learn small task-specific modules. Prompt pools for vision transformers (L2P, DualPrompt, CODA-Prompt, 2022 to 2023); orthogonal LoRA subspaces for LLMs (O-LoRA 2023, InfLoRA 2024); model merging and task arithmetic (add or subtract weight deltas per task).

Newer additions from the LLM era are covered in section 4: on-policy training, memory layers with sparse updates, and architectures that learn at test time.

## 3. How progress is measured, and why the numbers are distrusted

The standard metrics come from the GEM paper (Lopez-Paz and Ranzato 2017): **average accuracy** over all tasks at the end, **backward transfer** (how much learning task *t* hurt earlier tasks; negative means forgetting), and **forward transfer** (how much earlier learning helped later tasks). The **forgetting measure** (peak past accuracy minus final accuracy) is the most quoted single number. Standard benchmarks are split versions of MNIST, CIFAR, and ImageNet, plus CORe50 and CLEAR for realistic streams, and TRACE for continual instruction tuning of LLMs.

The community has repeatedly caught itself measuring the wrong thing. GDumb (2020) showed that "store a balanced buffer and retrain from scratch" beat most published methods. Prabhu et al. (2023) showed that under a fixed compute budget, rather than a fixed memory budget, plain replay wins again. Janson et al. (2022) showed a frozen pretrained ViT with a nearest-mean classifier beats most prompt methods, and Thede et al. (CoLLAs 2024) found prompt-selection mechanisms work "despite, not because of" their design. De Lange et al. (2023) named the **stability gap**: end-of-task metrics hide large transient forgetting right after each task switch, visible only if you evaluate every few steps. The lesson for any new CL study is to include trivial baselines, report compute, and evaluate at fine time granularity.

## 4. The last five years: five shifts

**1. From scratch-trained CNNs to frozen pretrained backbones (2021 to 2023).** Ramasesh et al. (2022) and Mehta et al. (2023) showed pretrained models forget far less, and that this improves with scale. Rehearsal-free prompt methods on ViTs became the default vision setting, and split-CIFAR-from-scratch stopped being the headline benchmark.

**2. The center of gravity moved to language models (2023 to 2025).** The relevant settings became continual pre-training (keep training a base model on new domains, languages, or time slices), continual instruction tuning, and continual alignment. The key practical result is from Ibrahim et al. (TMLR 2024): re-warming the learning rate is the main cause of forgetting when resuming pre-training, and re-warm plus re-decay plus a small replay fraction (about 5% for a mild shift, about 25% for a language change) matches retraining from scratch on the union of data. Guo et al. (ACL 2025) showed the stability gap appears in LLM continual pre-training too: a sharp early drop in general ability, then slow recovery. Scaling laws for continual pre-training loss (Wang et al., ICML 2025) and for how much old data must be mixed in (Bethune et al. 2025) followed.

**3. Plasticity became a first-class problem (2023 to 2025).** The Nature paper's continual backpropagation (re-initialize a small fraction of low-utility units every step) and several CoLLAs 2025 papers reframed CL as sustaining learnability, not only preventing forgetting. No LLM-scale replication of loss of plasticity was found.

**4. Forgetting turned out to be partly superficial.** Kotha et al. (ICLR 2024) showed fine-tuned models often still hold the old capability and can be prompted back into it. Zheng et al. (ICLR 2025, "spurious forgetting") showed that early steps on a new task destroy task alignment rather than knowledge, and that a brief re-exposure restores near-full accuracy. A function-vector analysis (ICLR 2025) concluded forgetting mostly reflects biased activation of task functions rather than overwriting. Unlearning work found the same reversibility. The open question is what fraction of forgetting in realistic settings is recoverable versus genuine, and how to tell them apart from the inside of the model.

**5. The field widened its own definition (2025 to 2026).** Knowledge editing (ROME, MEMIT, WISE) is now understood as single-fact CL and inherits the same interference problem: sequential edits degrade the model. Model merging, unlearning, test-time adaptation, and in-context learning are all inside the CoLLAs 2026 call for papers. Retrieval and product "memory" features are argued to be CL without weight updates.

## 5. The last year (September 2025 to September 2026)

Continual learning went from a subfield to a frontier-lab talking point. Four threads dominated.

- **Learn-at-test-time architectures.** Google's Titans (NeurIPS 2025) gives a model a neural long-term memory that updates its own weights at inference from a "surprise" signal. Nested Learning and the HOPE architecture (Google, November 2025) recast a model as nested optimizers running at different update frequencies and pitched it explicitly as a route past forgetting. Stanford, NVIDIA and collaborators (December 2025) showed that treating long context as continual next-token training, with a meta-learned initialization, scales with context length where state-space models do not. Critics called the results toy-scale; no independent replication was found.
- **On-policy training forgets less.** "RL's Razor" (MIT, September 2025) and "Retaining by Doing" (Princeton, October 2025) showed that forgetting tracks the KL shift from the base policy, that on-policy RL is implicitly KL-minimal, and that it is the on-policy data rather than the KL penalty doing the work. Self-distillation (ICML 2026) gets the same benefit from demonstrations. A July 2026 counterpoint ("RL Forgets!") shows drift still accumulates over long continual post-training.
- **Sparse updates by design.** Meta's sparse memory finetuning (October 2025) updates only the memory-layer slots most activated by new data. On a fact-learning task, full fine-tuning lost 89% of a held-out QA score, LoRA 71%, sparse memory finetuning 11%, at equal new-knowledge acquisition. This is the cleanest demonstration that localized updates cut interference.
- **Agent benchmarks were sobering.** A wave of harnesses (SkillLearnBench, Continual Learning Bench, ContinualSkillBench, EvoHarnessBench, April to September 2026) found that agents mostly adapt through context, that dedicated memory systems often underperform plain in-context learning, and that no CL method wins across tasks.

Around this, the "is CL the bottleneck to AGI?" debate became mainstream. Dwarkesh Patel (June 2025, August 2026), Sutton (September 2025, who then founded Oak Lab in July 2026), Karpathy (October 2025) and Sutskever (November 2025) argued it is. Nathan Lambert (August 2025) and Amodei (February 2026) argued long context, retrieval and memory features will make systems indistinguishable from continual learners without weight updates. Anthropic's Sholto Douglas predicted CL "gets solved" in 2026 (secondhand). Startups raised on the premise (Trajectory, $15M seed, June 2026; Monte, YC 2026). No frontier lab has shipped a product that updates weights per user; ChatGPT and Claude memory are context injection.

## 6. Map of active sub-areas today

| Sub-area | Core question | Momentum |
|---|---|---|
| Continual pre-training of LLMs | LR schedules, replay ratio, data mixing, scaling laws for forgetting | steady, practical |
| Continual fine-tuning / alignment | sequential instruction tuning, safety forgetting, PEFT-based CL, merging | steady |
| On-policy post-training as CL | why RL forgets less; self-distillation | rising fast |
| Memory architectures / test-time learning | Titans, Nested Learning, TTT layers, memory layers, sparse memory finetuning | strongest |
| Agent memory and skill libraries | external memory, meta-learned memory designs | highest paper volume, weak measured gains |
| Mechanistic and representational views of forgetting | fading features vs readout misalignment, spurious forgetting, layer-wise drift | young, unconverged |
| Knowledge editing and unlearning | lifelong edits, reversibility, the reliability/generalization/locality triangle | converging with CL |
| Plasticity and continual RL | loss of plasticity, continual backprop, Oak Lab | steady |
| Robotics and vision-language-action | pretrained robot policies barely forget (March 2026) | rising |
| Federated, on-device, diffusion CL | survey-stage | low visibility |

## 7. What is settled and what is not

**Reasonably established.** Forgetting in continual pre-training scales with learning-rate re-warming and with distribution distance, and a few percent replay is the cheapest fix. In fine-tuning, forgetting grows as a power law in trainable parameters and steps, and LoRA reduces but does not remove it. Sequential weight edits degrade models. Sparse or subspace-localized updates cut interference sharply. Pretrained scale reduces forgetting in the settings tested. Evaluate at fine granularity or you miss the stability gap.

**Contested.** Whether larger models forget more or less (results differ by regime). How much forgetting is overwriting versus suppression or retrieval failure. Whether architectural approaches will displace the replay-plus-schedule recipe at frontier scale. Whether CL needs to happen in weights at all, or whether context and retrieval suffice. What the right benchmarks are for LLM-scale CL.

## 8. Where this project sits

This project studies what happens inside a pretrained language model during continual pre-training on a stream of domains, using sparse autoencoder features and representation-similarity measures as instruments, alongside held-out capability evals. On the map above it sits at the intersection of "continual pre-training" and "mechanistic views of forgetting," and its main design concerns (fine-grained evaluation at domain boundaries, separating fading features from readout misalignment, testing recoverability) are direct responses to the stability gap and the spurious-forgetting findings in sections 3 and 4.

## Short reading list

1. van de Ven, Tuytelaars, Tolias. "Three types of incremental learning." *Nat. Mach. Intell.* 2022. The scenario taxonomy.
2. Wang, Zhang, Su, Zhu. "A Comprehensive Survey of Continual Learning." *TPAMI* 2024. The general reference.
3. Ibrahim et al. "Simple and Scalable Strategies to Continually Pre-train LLMs." *TMLR* 2024. The practical recipe.
4. Shi et al. "Continual Learning of LLMs: A Comprehensive Survey." *ACM CSUR* 2025. The LLM-specific map.
5. Dohare et al. "Loss of plasticity in deep continual learning." *Nature* 2024. The plasticity side.
6. Zheng et al. "Spurious Forgetting in Continual Learning of Language Models." ICLR 2025. Forgetting as lost alignment.
7. Lin et al. "Continual Learning via Sparse Memory Finetuning." Meta, October 2025. Sparsity as the fix.
8. Behrouz and Mirrokni. "Nested Learning." NeurIPS 2025, with the Google Research blog post. The architectural bet.
9. Lambert, "Contra Dwarkesh on Continual Learning" (August 2025) and Patel, "8 Predictions for the Era of Continual Learning" (August 2026). The two sides of the debate.

## Sources

Core framing and methods: nature.com/articles/s42256-022-00568-3; arxiv.org/abs/2302.00487; arxiv.org/abs/2404.16789; nature.com/articles/s41586-024-07711-7; arxiv.org/abs/2210.04428; arxiv.org/abs/2406.09384; arxiv.org/abs/2205.13452; arxiv.org/abs/2303.11165.
LLM continual learning: arxiv.org/abs/2403.08763; arxiv.org/abs/2308.04014; arxiv.org/abs/2406.14833; arxiv.org/abs/2505.07796; arxiv.org/abs/2401.05605; arxiv.org/abs/2310.14152; arxiv.org/abs/2404.00228; arxiv.org/abs/2405.14768; arxiv.org/abs/2309.10105; arxiv.org/abs/2501.13453; arxiv.org/abs/2502.11019.
Last twelve months: research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning; arxiv.org/abs/2501.00663; arxiv.org/abs/2510.15103; arxiv.org/abs/2509.04259; arxiv.org/abs/2510.18874; arxiv.org/abs/2601.19897; arxiv.org/abs/2512.23675; arxiv.org/abs/2606.05661; arxiv.org/abs/2604.20087; arxiv.org/abs/2609.04280; arxiv.org/abs/2603.03818.
Commentary: dwarkesh.com/p/timelines-june-2025; dwarkesh.com/p/era-of-continual-learning; interconnects.ai/p/contra-dwarkesh-on-continual-learning; a16z.com/why-we-need-continual-learning; officechai.com (Oak Lab announcement); thezvi.substack.com (Amodei, Karpathy, Sutton podcast summaries, secondhand).
Venues: proceedings.mlr.press/v330 (CoLLAs 2025); lifelong-ml.cc (CoLLAs 2026, Bucharest, September 14 to 18, 2026); sites.google.com/view/ccfm-neurips2025; cats-icml.github.io.
