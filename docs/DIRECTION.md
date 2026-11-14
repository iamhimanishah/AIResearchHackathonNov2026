# Direction

What we are building this weekend and who builds which part. Read this before starting work.
If something here turns out to be wrong, change this file in the same commit as the work.

## Target

Build a test bench that makes continual-learning experiments fast to run, then use it to
measure which forgetting-reduction techniques work, alone and in combination.

## Outcomes

- **Science:** improve upon existing forgetting-reduction results.
- **Product:** a framework that makes continual-learning experiments easier to run.

## What that means concretely

- **The framework.** A technique plugs in, runs against baselines at matched compute, and is
  scored on three things: what the model kept (retention), what it learned (acquisition), and
  what it cost.
- **The study.** A grid of techniques switched on and off, run on the framework. It yields each
  technique's own effect and the interactions between techniques.
- **The agent.** An agent plans the grid, launches runs, handles failures and decides which
  configurations get more seeds.

"Improve upon existing results" needs a named reference. The reference arm is the published
continual-pretraining recipe of replay plus per-domain learning-rate re-warming (Ibrahim et
al. 2024). A result counts as an improvement only if it beats that arm, not just the
no-mitigation baseline, by more than the measured noise.

## Priorities

In this order. Items 1 and 2 are a complete project; do not start a later item while an
earlier one is broken.

1. The grid running on GPUs with five or six hand-checked techniques.
2. The attribution analysis and a results page, with uncertainty shown.
3. The agent driving runs and choosing which configurations to replicate.
4. The agent finding a new technique in the literature and adding it.

## Workstreams

Each has one owner and one deliverable. Put a name in the owner column when you take one.

| # | Workstream | Deliverable | Owner |
|---|---|---|---|
| 1 | Bench core | The harness changes listed below, a frozen fast bench, and the run-spec and result-row formats | |
| 2 | Executor | One command that runs a spec on a cloud GPU and returns a result row; many in parallel | |
| 3 | Techniques | Each technique as a plugin with a sanity run, including example selection with random and loss-based controls, and the reference arm | |
| 4 | Design and analysis | Grid plan, noise floor, seed policy, attribution math, comparison against the reference arm | |
| 5 | Agent and page | The loop that launches and tracks runs, and the live results page | |

With three people: combine 1 with 2, and 3 with 4. With four: combine 1 with 2.

Candidate techniques: replay, LoRA adapters, layer freezing, weight averaging, per-domain
learning-rate re-warm, distillation from the base model, and example selection by an
activation-based signal.

## The contract between workstreams

Workstreams 2 to 5 depend on each other only through two formats. Both are drafts until the
team confirms them; once confirmed, change them only with everyone's agreement.

**Run spec** (input to one run):

- name and seed
- stream: domains in order, tokens per domain
- optimizer settings that differ from the bench default
- techniques: which are on, with their parameters

**Result row** (output of one run):

- the full run spec and the git commit that produced it
- retention: change in protected-set scores against base, per task and averaged
- acquisition: held-out perplexity on the trained domains against base
- cost: gradient tokens, wall-clock seconds, GPU type, size of the update that would ship
- answer-distribution check per multiple-choice task (see `docs/baselines.md` for why)
- status: finished, failed, or invalid, with a reason

With these fixed, the executor, analysis, agent and page can all be built against fake rows.

## Harness changes needed

The harness in `src/clharness/` runs a domain stream and measures at every checkpoint.
Techniques attach as hooks (`src/clharness/train/hooks.py`). Three things it cannot do yet:

- **Change the model before the optimizer is built.** Needed for LoRA and layer freezing.
- **Add a term to the loss.** Needed for distillation and weight penalties.
- **Re-warm the learning rate per domain.** The schedule is global today.

Also missing: a cloud executor, any comparison across runs, and a short bench config. The
existing configs are multi-hour runs.

## Rules for experiments

These keep results comparable and honest. They apply to every run that is reported.

- **Freeze the bench before the grid starts.** Stream, token budget, learning rate and eval
  items do not change afterward. Proposed default, to confirm with one baseline run: code then
  biomedical, a learning rate that forgets fast (1e-4 did in earlier runs), about 30 minutes
  per run.
- **Measure the noise floor first.** Run the no-mitigation baseline with several seeds. A
  difference smaller than that spread is not a finding.
- **Match compute.** Compare techniques at the same number of gradient tokens, and report cost
  whenever a technique spends more.
- **Always include the controls.** No mitigation, the reference arm, and random selection
  wherever a selection signal is tested.
- **Screen wide, then replicate.** One seed across the grid, then at least three seeds on the
  configurations that matter.
- **Select and report on different items.** The protected set is split in two. Decisions about
  which configurations to pursue use one half; reported numbers use the other.
- **Report retention and acquisition together.** A technique that forgets less by learning
  less has not won.
- **Show uncertainty** on every chart and table.
- **State the limits:** one small model, a short stream, a harsh learning rate.

## Out of scope this weekend

- A general framework. One model, one stream, one plugin interface.
- A new benchmark. The protected set is built from standard public evals.
- A combined training-and-serving runtime.
- Controls in the results page. It is a viewer.

## Stretch

Only after priorities 1 to 3 work.

- The agent searches the literature, writes a new technique as a plugin and adds it to the
  ranking.
- One contrast between supervised and reward-based fine-tuning on the same skill, measuring
  what each forgot. Kept separate from the main grid.
- The top configurations on a second, published task sequence.

## Sync points

1. One baseline run completes end to end on a cloud GPU.
2. Techniques are checked and the grid is launched.
3. Analysis runs on real result rows.

## Where things are

- `README.md`: overview, quick start, what predates the hackathon.
- `docs/baselines.md`: the no-mitigation numbers and what the earlier runs showed.
- `docs/harness.md`: harness reference.
- `docs/background/`: literature review and field overviews.
