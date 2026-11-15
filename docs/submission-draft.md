# Submission draft

For brendan, who owns the submission (due Sunday 14:00; plan to submit by 13:45). Drafted by
Member A's agent on Saturday evening so that Sunday's work is filling in numbers and recording,
not writing from scratch. Edit freely.

**Every "TO FILL" is a number or sentence that does not exist yet.** Do not submit with one left
in. Numbers already filled in come from committed rows and are marked with their file.

## What the platform asks for

| Item | Status |
|---|---|
| Track | Open Track (Track 5) |
| Challenges entered | Modal, Exa, past.dev: TO CONFIRM as a team. Originator: not entered unless the team agrees to upload agent sessions |
| Repository link | https://github.com/justinmilner1/AIResearchHackathonNov2026 |
| 2-minute demo video | Outline below. TO RECORD |
| Short project description | Draft below |

## Project description (draft, about 230 words)

**Remember: a test bench and research agent for forgetting in continually trained language
models.**

When a language model keeps training on new data, it forgets things it knew. Dozens of papers
propose fixes, each tested on a different model, data stream and compute budget, so their
results cannot be compared and some conflict.

Remember holds everything constant. One frozen bench trains Gemma 3 1B on code and then legal
text in about 20 minutes for about $1.30 on a single cloud GPU, and scores what the model kept,
what it learned and what the update cost. Six published techniques plug in as switches. An
agent launches the experiments on Modal and logs every decision; a pre-registration written
before the results fixes what counts as a win.

What we found:

- The published recipe (replay plus learning-rate re-warming) halves forgetting: −3.1 points
  against −6.4 with no mitigation.
- Simply lowering the learning rate does better: −1.1 points, with the same learning of the
  new domain. The recipe's schedule is, in effect, a learning-rate cut.
- TO FILL: one sentence on the screen and replication (which configurations came closest to the
  lower-learning-rate control, and by how much, with the interval).
- TO FILL: one sentence on the published-claims table (how many of the claims tested
  reproduced on the bench).
- TO FILL: one sentence on what the research agent proposed and concluded overnight.

Everything reported rebuilds from committed result rows with `make reproduce`.

The harness and four September baseline runs predate the event and are imported in a labelled
first commit; everything else was built this weekend.

## Facts behind the description

So that nothing in it is typed from memory. Check each against its file before submitting.

| Claim | Value | Where it comes from |
|---|---|---|
| No mitigation, report half | −6.44 points, SD 1.15, 5 seeds | `results/rows.phase0.remeasured.jsonl` |
| Reference arm, report half | −3.13 points, 3 seeds | `results/rows.phase0.jsonl`, `results/rows.reference.remeasured.jsonl` |
| No mitigation at lr 3e-5, report half | −1.07 points, 3 seeds | `results/rows.phase0.remeasured.jsonl`, `results/rows.phase0.jsonl` |
| Legal perplexity after legal training | 0.787 (1e-4), 0.744 (reference), 0.754 (3e-5) | same files, `acquisition.own_boundary_ppl_ratio` |
| Run length and cost | 17 to 27 minutes, about $1.20 to $1.45 | `cost` in any row |
| Runs on Modal so far | 19 full control and calibration runs, 38 in the grid, plus short checks | row files under `results/` |
| Mean learning rate of the reference arm | 5.54e-5 against 9.2e-5 for no mitigation | peter's review, posted in the team channel; TO VERIFY from `train/lr` in a run's metrics before quoting |

The pre-registered win rule cannot be met on this bench: a technique would have to beat the
3e-5 control by more than 2.30 points, and that control loses only 1.07. Say this plainly. The
honest result is "no technique beat lowering the learning rate", reported with the measured
differences and intervals, never as "no technique helps".

## How it maps to the five judging criteria

| Criterion | What to point at |
|---|---|
| Research rigour | Frozen bench; noise floor from 5 seeds; pre-registration with a logged amendment history; a published reference arm and lower-learning-rate controls; selection and report halves of the eval items kept apart; three rounds of independent review with the fixes in git |
| Novelty | All techniques under one protocol at matched compute; a table of published claims tested for reproduction; the finding that learning rate dominates and that the published recipe is partly a learning-rate cut |
| Results | The control table above; TO FILL: screen, replication, claims table. Live results page |
| Autonomy and self-improvement | The launcher runs the pre-registered grid unattended and resumably; the explore loop proposes hypotheses from results, a code gate accepts or refuses them, and verdicts feed the next cycle. Be exact: tonight's screen was launched by a deterministic launcher on a person's go, and the Conductor reads real data but launches nothing. TO FILL: what the overnight loop did |
| Interpretability and reproducibility | Hash-chained decision log; every row carries its spec, commit and cost; failed and invalid runs are rows too; `make reproduce`; this honest limits list |

## Sponsor challenges

- **Modal.** One GPU per experiment, 10 in parallel, launched by name from a deployed app that
  refuses callers at a different commit. TO FILL: total runs and total GPU-hours.
- **Exa.** The Scout searches with Exa inside the agent loop and writes claim cards with their
  sources; the explore loop turns them into gated hypotheses. Point to
  `src/clharness/agent/search.py` and to `read_inbox` in `src/clharness/agent/explore.py`. The
  demo must show sources: `results/agent/sources.jsonl` and the claim cards on the page.
  TO CONFIRM: that a claim card actually reached a launched run before claiming this.
- **past.dev.** `src/clharness/agent/pastdev.py`; README section "Shared memory between agents".
  The analyst, the Scout and the explorer write; the explorer reads what the other two wrote
  before it proposes. TO CONFIRM: a second reader exists before entering.

## Limits to state in the pitch

- One small model, a 20M-token stream, a harsh learning rate.
- The screen is one seed per configuration; only the top three and the controls are replicated.
- Two screen runs had been seen before the pre-registration froze, and several analysis fixes
  were logged after the grid launched (see the amendment log in `analysis/PREREG.md`).
- The harness and September baselines are prior work.

## Video outline (2:00)

| Time | On screen | Say |
|---|---|---|
| 0:00 to 0:15 | Title card, then one line of the bench config | The problem: models forget when they keep learning, and published fixes cannot be compared. We built a bench that holds everything constant: 20 minutes, about a dollar a run |
| 0:15 to 0:40 | Terminal: the launcher's dry run, then the decision log scrolling, then the Modal dashboard with 10 GPUs | One command launches the pre-registered grid on Modal. Every wave and every row is logged in a hash-chained file. It resumes if interrupted and stops at a budget cap |
| 0:40 to 1:05 | Results page: the runs table with the per-domain perplexity columns, then the frontier figure | What each configuration kept and what it learned. The controls: no mitigation loses 6.4 points, the published recipe 3.1, a lower learning rate 1.1 |
| 1:05 to 1:25 | The claims figure or table | TO FILL: published claims tested on the same bench, and how many reproduced |
| 1:25 to 1:45 | Hypotheses file and the explore decision log; a claim card with its Exa source | The research agent proposes its own follow-ups from the results and the literature. Code decides what runs. TO FILL: one hypothesis it proposed and what it found |
| 1:45 to 2:00 | `make reproduce` finishing; the limits list | Everything rebuilds from committed rows in seconds. One small model, a short stream: here is what we do not claim |

Record the screen segments on Sunday morning once the page shows real results. The first 40
seconds can be recorded tonight.

## Open questions for the team

- Which sponsor challenges are entered.
- Whether "tune the learning rate first" is the title line of the pitch.
- Who speaks in the 3-minute pitch and who takes harness questions.
