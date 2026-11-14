# Direction

What we are building this weekend and who builds which part. Read this before starting work.
If something here turns out to be wrong, change this file in the same commit as the work.

## Target

Build a test bench that makes continual-learning experiments fast to run, then use it to
measure which forgetting-reduction techniques work, alone and in combination.

## Outcomes

- **Science:** improve upon existing forgetting-reduction results.
- **Product:** a framework that makes continual-learning experiments easier to run.

## Hackathon constraints

A summary of the official brief in [`hackathon-brief.md`](hackathon-brief.md). Times are
Pacific, Sunday November 15.

**Track: Open Track (Track 5).** Tracks 1 to 4 each set a fixed problem: mathematics records,
GPU price forecasts, a biology benchmark, and world models. None is continual learning. The
Open Track has the same rules and judging, and the top two teams of each track reach the
finals.

**Deadlines**

| When | What |
|---|---|
| 12:00 | Team list closes on the platform |
| 14:00 | Hacking ends, submissions due. The brief says 14:30 in one place; plan for 14:00 |
| 14:45 | Round 1: 3-minute pitch and 2 minutes of questions |
| 16:45 | Finals, also given as 16:30: 3-minute pitch with a live demo, no questions |

**What to submit:** the track and any challenges entered, the link to this repository, a
2-minute demo video, and a short project description.

**Judging: five criteria, 20% each**

| Criterion | What the brief asks for | What it means for us |
|---|---|---|
| Research rigour | Clear question, sound design, baselines or ablations, claims backed by results | Frozen bench, noise floor, controls and the reference arm |
| Novelty | Beyond existing papers and open-source baselines | Interactions between techniques, cost-adjusted ranking, the selection-signal test |
| Results | A measurable result against a baseline, and a working live demo | Beat the reference arm on the protected set; results page running live |
| Autonomy and self-improvement | How much of the research loop runs without a human, with extra credit if it improves from its own results | The agent plans, launches, reads results and chooses the next runs itself, and later choices use earlier results |
| Interpretability and reproducibility | Decisions can be followed, code runs, failures reported honestly | A log entry for every agent decision, one command to reproduce, failed and invalid runs reported |

Autonomy carries the same weight as rigour. A grid launched by hand scores poorly on it, so the
agent loop is part of the core project, not an extra.

**Sponsor challenges.** Optional, and they stack on top of the track. Which to enter is not
decided.

- **Modal.** Best use of Modal for running many experiments in parallel. A direct fit for the
  executor.
- **Exa.** Needs an Exa call in the project's main loop, a pointer in the repository to where
  it is called, and a demo that shows the sources found. It fits only if the agent's search
  for techniques in the literature sits inside the main loop.
- **past.dev.** A shared memory layer: more than one agent writes, more than one reads, and at
  least one uses what another wrote. It fits if the loop is split into agents that share what
  was tried and what failed. The README must say which memory holds what.
- **Originator.** Agent sessions are uploaded and scored. Uploading shares the full session
  transcripts, so the team decides before anyone does it.

**Rules that bind us**

- Build entirely during the event. No prior commits to the repository.
- Open-source libraries, APIs and pre-trained models are allowed if properly credited.
- The repository is public, and all code and demos are submitted before the deadline.
- All team members are on site.

**Open issue: prior work.** The harness and the September baseline runs in this repository's
first commit were built before the event. Every commit here was made during the event, but
importing earlier work is in tension with "build entirely during the event". This is
unresolved until an organizer answers. Until then:

- Keep the import commit separate and labelled, as it is now.
- Credit the earlier work in the README, the submission description and the pitch.
- Never present the imported harness or the September runs as work done this weekend.
- If the organizers do not allow it, the fallbacks are to depend on the harness as a separately
  published and credited open-source library, if they accept that, or to rebuild the part we
  need during the event.

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

In this order. Items 1 to 3 are the complete project; do not start a later item while an
earlier one is broken.

1. One run goes from a run spec to a result row on a cloud GPU, in about 30 minutes.
2. The agent loop, in its simplest form: it plans the screening grid over five or six
   hand-checked techniques, launches it, reads the results, chooses what to replicate or
   combine next, and logs every decision with its reason.
3. The attribution analysis and a live results page, with uncertainty shown and the
   comparison against the reference arm.
4. The agent searches the literature for a technique it has not tried, adds it as a plugin and
   tests it. This is what the Exa challenge requires, and it must be in the main loop to count.

## Workstreams

Each has one owner and one deliverable. Put a name in the owner column when you take one.

| # | Workstream | Deliverable | Owner |
|---|---|---|---|
| 1 | Bench core | The harness changes listed below, a frozen fast bench, and the run-spec and result-row formats | Member A |
| 2 | Executor | One command that runs a spec on a cloud GPU and returns a result row; many in parallel | Member A |
| 3 | Techniques | Each technique as a plugin with a sanity run, including example selection with random and loss-based controls, and the reference arm | peter |
| 4 | Design and analysis | Grid plan, noise floor, seed policy, attribution math, comparison against the reference arm | himani |
| 5 | Agent and page | The loop that launches and tracks runs, and the live results page | brendan (Member D) |

With three people: combine 1 with 2, and 3 with 4. With four: combine 1 with 2.
Members A to D are placeholders: replace each with a name when someone claims the role.

## Division of work (team remember US, four people)

**Member A: bench core and executor (1+2).**
- Add the three harness changes listed below: a pre-optimizer model hook, a loss-term hook,
  and per-domain LR re-warm.
- Write a frozen fast bench config: code then biomedical, LR 1e-4, about 30 minutes per run.
- Commit the run-spec and result-row formats early, with fake rows, so the other
  workstreams can build against them.
- Build a Modal executor with parallel fan-out, retries, idempotent run IDs and a budget
  guard.
- Add a Tinker backend that emits the same result row.
- Add `make reproduce`.

**Member B (peter): techniques (3).** In this order, each with a sanity run at matched gradient tokens:
1. The reference arm (replay plus per-domain LR re-warm).
2. Replay.
3. Example selection: random and loss-based controls, then the activation-based signal.
4. LoRA, then layer freezing.
5. Distillation from the base model.
6. Weight averaging.

**Member C: design and analysis (4).**
- Write `analysis/PREREG.md` before the grid launches. It covers the seed policy, the
  selection/report split of the protected set, and the decision rule against the reference
  arm.
- Measure the noise floor: the no-mitigation baseline at 3 to 5 seeds.
- Build the attribution math (main effects and pairwise interactions), bootstrap intervals
  and a cost-adjusted ranking, against fake rows first.
- Own the pitch figures and the prior-work credit wording.
- Also owns the time-boxed second task sequence (below).

**Member D: agent and page (5).**
- Build an agent loop that plans the screening grid, launches it, reads the rows and picks
  what to replicate or combine, logging every decision with its reason.
- Build the live results page.
- Set up the past.dev memory shared by the planner and analysis agents.
- Put Exa in the main loop once priorities 1 to 3 work.
- Own the submission: demo video, description and the challenges entered.

**Second task sequence (hybrid scope).** The top configurations are re-run on a published
sequence: Tinker LoRA on Qwen3-8B, SST-2 then MNLI then AG News, 5k train
and 1k eval per task. It runs naive, replay-200, joint and the top two techniques at 3 seeds,
scored as gap closed and average forgetting. Rules:
- It starts only after sync point 1.
- It gets about 4 hours of Member C's time.
- It is the first thing to cut.

**Budget caps** (70% of the team total, 30% held back for reruns): Modal $700, Tinker $280,
OpenAI $70.

**Cut order if behind:**
1. The second task sequence.
2. The Exa loop.
3. Weight averaging, then distillation.
4. Activation-based selection (keep random and loss-based).
5. Replication from 3 seeds to 2.

Never cut the reference arm, the noise floor, the decision log or the results page.

### Task dependencies

Read each row as "this task cannot finish until the tasks in *Needs* are done". A task
with nothing upstream can start now. Each task's details, open questions and options to try
are in [`TASKS.md`](TASKS.md).

| ID | Task | Owner | Needs | Unblocks |
|---|---|---|---|---|
| A0 | Commit the missing `src/clharness/data/` package (ignore rule fixed; the files are still needed) | A | The original files from whoever built the harness | Every run: A5, A6, B sanity runs |
| A1 | Run-spec and result-row formats, with fake rows | A | Team agrees the contract | A6, A7, C2, D1, D3 |
| A2 | Pre-optimizer model hook | A | (nothing upstream) | B4 (LoRA, layer freezing) |
| A3 | Loss-term hook | A | (nothing upstream) | B5 (distillation) |
| A4 | Per-domain LR re-warm | A | (nothing upstream) | B1 (reference arm) |
| A5 | Frozen fast bench config | A | One baseline run to confirm it | A6, B sanity runs, C3 |
| A6 | Modal executor (fan-out, retries, budget guard) | A | A1, A5 | Sync point 1, then C3, D2 |
| A7 | Tinker backend, same result row | A | A1 | C5 |
| A8 | `make reproduce` | A | A6, C4 | Submission |
| B1 | Reference arm: replay plus LR re-warm | B | A4, B2 | C1 decision rule, D2 grid, pitch |
| B2 | Replay | B | (nothing upstream; smoke data is enough) | B1, B3 |
| B3 | Selection: random and loss-based controls, then activation signal | B | B2 | Selection-signal test in the grid |
| B4 | LoRA, then layer freezing | B | A2 | Grid |
| B5 | Distillation from base | B | A3 | Grid |
| B6 | Weight averaging | B | (nothing upstream) | Grid |
| C1 | `analysis/PREREG.md` | C | B1 defined, C3 for the threshold | Sync point 2 (grid must not launch without it) |
| C2 | Attribution math, bootstrap, cost ranking on fake rows | C | A1 | C4, D3 uncertainty display |
| C3 | Noise floor: no-mitigation at 3 to 5 seeds | C | Sync point 1 | C1 threshold, D2 replication rule |
| C4 | Analysis on real rows, pitch figures | C | Sync point 2 grid results, C2 | Sync point 3, A8, D5 video |
| C5 | Second task sequence on Tinker | C | A7, sync point 1. The top-two-technique arms also need the screening results (Sunday morning) | Pitch backup slide |
| D1 | Agent v1: plan, launch, read, decide, log | D | A1 (fake rows); A6 for real launches | Sync point 2, D4 |
| D2 | Agent launches the screening grid | D | D1, A6, C1, C3, B1 plus at least four of B2 to B6 | Overnight grid, C4 |
| D3 | Live results page | D | A1, C2 | D5 video, finals demo |
| D4 | past.dev shared memory | D | D1, plus C's analysis agent writing to it | past.dev challenge |
| D5 | Submission: video, description, challenges | D | D3, C4, A8 | Sync point 4 |
| D6 | Exa in the main loop | D, B | Sync points 1 to 3 working, then B wraps the found technique | Exa challenge |

**Critical path:** A0 → A1 → A5 → A6 → sync point 1 → C3 and B1 → C1 → D2 (grid overnight) →
C4 → D3 → D5. Any slip on this chain moves the submission, so help the owner of the slipping
task before starting off-path work.

**Unblocked from the start:**
- A: A2 to A4.
- B: B2 and B6 on smoke data.
- C and D: they start on fake rows as soon as A1 lands.

A1 is the single biggest unblocker: do it first.

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
- techniques: which are on, with their parameters. In the harness config this is
  `techniques: [{name, params}]`, optional, where an empty list means no mitigation. Each name
  is a plugin in `src/clharness/techniques/`, which validates its own `params`

**Result row** (output of one run):

- the full run spec and the git commit that produced it
- retention: change in protected-set scores against base, per task and averaged
- acquisition: held-out perplexity on the trained domains against base
- cost: gradient tokens, wall-clock seconds, GPU type, size of the update that would ship
- answer-distribution check per multiple-choice task (see `docs/baselines.md` for why)
- status: finished, failed, or invalid, with a reason

With these fixed, the executor, analysis, agent and page can all be built against fake rows.

Both formats are defined in code in `src/clharness/spec.py` (`RunSpec`, `ResultRow`), with JSON
Schemas in `configs/schema/`. Fake rows with planted effects are in `results/fake/rows.jsonl`;
`results/fake/planted_effects.json` says what was planted. Scores and deltas are fractions
(a 5-point drop is -0.05). Check any rows file with `clh validate-row FILE`.

## Harness changes needed

The harness in `src/clharness/` runs a domain stream and measures at every checkpoint.
Techniques attach as hooks (`src/clharness/train/hooks.py`). Three things it could not do at
the start, all added on Saturday (tests in `tests/test_loop_hooks.py`):

- **Change the model before the optimizer is built.** Needed for LoRA and layer freezing.
  Done: `configure_model(ctx)` on a hook; return a wrapped model, or freeze in place.
- **Add a term to the loss.** Needed for distillation and weight penalties.
  Done: `extra_loss(ctx, batch, logits, labels)` on a hook; logged as `train/extra_loss/<name>`.
- **Re-warm the learning rate per domain.** The schedule was global.
  Done: `optim.schedule_scope: per_domain` with `optim.rewarm_steps` and `optim.min_lr_ratio`.

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

- A shared memory of what was tried and what failed, read and written by more than one agent.
- One contrast between supervised and reward-based fine-tuning on the same skill, measuring
  what each forgot. Kept separate from the main grid.
- The top configurations on a second, published task sequence.

## Sync points

Target times are Pacific.

1. One baseline run completes end to end on a cloud GPU. Member A, Saturday 18:00. This is
   the go/no-go: if it is missed, everyone helps Member A, and the Exa loop and the second
   task sequence are dropped.
2. Techniques are checked and the grid is launched. Members B and D, Saturday about 21:30.
   The grid runs overnight.
3. Analysis runs on real result rows. Member C, Sunday about 10:00.
4. Submission is ready: demo video, project description and repository link, before Sunday
   14:00. Member D.

## Where things are

- `README.md`: overview, quick start, what predates the hackathon.
- `docs/hackathon-brief.md`: the official event brief, with access codes removed.
- `docs/baselines.md`: the no-mitigation numbers and what the earlier runs showed.
- `docs/TASKS.md`: one card per task, with steps, open questions and options to try.
- `docs/harness.md`: harness reference.
- `docs/TECHNIQUES.md`: how a technique plugs in, the technique table and the open decisions.
- `docs/background/`: literature review and field overviews.
