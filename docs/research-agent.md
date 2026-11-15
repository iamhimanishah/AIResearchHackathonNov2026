# Design: the research agent

**Status: draft for discussion, 14 November 2026.** Started by Member A's agent at justin's
request after the team talked about it. Nothing here is decided. The agent system is workstream
5 (brendan), so anything that changes it is his call; the pre-registration is himani's.

**How to take part.** Each workstream has a section under "Comments" at the end. Add your
points there and push; keep edits to other sections small and say what you changed in the
commit message. Answer the open questions addressed to you in place, the way task cards do
(strike the question through and write the decision next to it).

## 1. What we mean

A research agent uses the test bench to make progress on a question in continual learning:
it forms a hypothesis, runs the experiments that test it, and records what was learned in a form
someone else can check. The pre-registered screen answers questions we wrote down in advance.
The research agent is for the questions that come up afterwards.

## 2. What already exists

The agent system in [`agent/ARCHITECTURE.md`](agent/ARCHITECTURE.md) already does most of this:

- A deterministic Conductor owns control flow, budget and the decision log.
- A Planner proposes specs with a predicted effect; a Gatekeeper accepts or rejects them
  against the pre-registration, compute matching and the budget.
- An Analyst writes findings that cite rows; a checker confirms every number.
- Predictions are scored into lessons that the next plan reads.
- A Scout searches the literature (Exa) for a technique not in the catalog.
- Phases are screen, replicate, combine, then freeze and report.

So this document is about what to add, not a second agent.

## 3. Why not just a coding agent with the repo

A general coding agent can already read rows, write a spec and launch it. The question for the
pitch is what ours does better. The proposed answer:

| | General coding agent | Research agent |
|---|---|---|
| Can it look at the report half before choosing? | Yes, nothing stops it | No: the data is not in what it is handed |
| Can it change the bench, or pick the best of ten noisy runs? | Yes | No: the Gatekeeper rejects it |
| Is each decision checkable afterwards? | A chat transcript | A hash-chained log with the rows and prompt version behind each decision |
| Can it overspend or stall overnight? | Yes | Hard budget cap; a fallback rule after two rejections |
| Does it get better from its own results? | Only within one session's context | Predictions are scored; lessons and findings persist and are shared |

Section 6 proposes measuring this instead of only asserting it.

## 4. The gap

Three things the current design does not do:

1. **It has no place for a question it was not given.** Phases follow the pre-registered design.
   Today's surprise (no mitigation at a learning rate of 3e-5 forgets less than the published
   recipe at 1e-4, `results/rows.phase0*.jsonl`) suggests an obvious next run: the recipe at
   3e-5. Nothing in the loop would propose it.
2. **A hypothesis is not an object.** A finding records what was seen. Nothing records what was
   asked, what was predicted, what would count as an answer, and whether it was confirmed.
3. **The Scout looks for techniques, not claims.** Most of what the literature offers us is a
   claim we can test with the plugins we already have
   ([`published-claims.md`](published-claims.md)), which needs no new code and no human review.

## 5. Proposal

### 5.1 A hypothesis record

One new store, `results/agent/memory/hypotheses.jsonl`, one writer (Memory, through the
Conductor).

| Field | Meaning |
|---|---|
| `id`, `created_at`, `author` | As for findings |
| `origin` | `surprise`, `literature` or `follow_up` |
| `question` | One sentence |
| `claim` | The statement being tested |
| `source` | For `literature`: title and URL, from `sources.jsonl`. For `surprise`: the finding or lesson id that triggered it |
| `prediction` | Direction and rough size, stated before any run |
| `test_spec_hashes`, `control_spec_hashes` | The runs that test it and the runs it is compared with |
| `decision_rule` | Which rule in the pre-registration's exploratory section applies |
| `status` | `proposed`, `accepted`, `running`, `confirmed`, `refuted`, `inconclusive` or `rejected` |
| `evidence_row_hashes`, `effect`, `interval` | Filled when it is resolved |

### 5.2 An explore phase

A fourth phase after combine, reusing the existing cycle (plan, gate, launch, wait, read,
advance) with its own skill file.

- **Plan.** The Planner proposes a small batch of hypotheses with their specs. Inputs: findings,
  lessons, unresolved hypotheses, claims from the Scout, the catalog.
- **Gate.** New deterministic rules:
  - every spec uses only catalog techniques, within declared parameter ranges, on the frozen bench;
  - every hypothesis names its control, and the test runs share seeds with the control;
  - at least two seeds before a hypothesis can be resolved;
  - the phase has its own budget share and cannot spend past it;
  - a hypothesis already resolved is not reopened without new evidence.
- **Read.** The Analyst resolves each hypothesis by its decision rule, on the selection half.
- **Rows from this phase are exploratory.** They are marked as such and never enter the
  confirmatory analysis.

### 5.3 Two ways a hypothesis starts

- **From a surprise.** A deterministic trigger, not a model's taste: a prediction off by more
  than twice the noise floor, or a control arm that beats the reference arm. The trigger hands
  the Planner the finding and asks for at most one follow-up hypothesis.
- **From the literature.** The Scout gains a second output: a claim card (claim, source, the
  catalog arms that would test it). This puts the Exa call in the main loop, with sources shown.

### 5.4 What the reader sees

- A table of hypotheses on the results page: question, origin, prediction, verdict, evidence.
- The claims table from `published-claims.md`, filled by the agent instead of by hand.

## 6. Measuring it against a plain agent

The architecture already has a simulator with planted effects. Use it to answer section 3 with
a number: give a general coding agent and the research agent the same simulated world and the
same budget, and score each on how much of the planted truth it recovers and how many claims it
makes that are false. This is optional and comes after the explore phase works.

## 7. Not in scope

- The agent writing technique plugins on its own. A candidate technique still gets a human
  review.
- Changing the frozen bench, the pre-registered screen, or the confirmatory analysis.
- A general autonomous scientist. One model, one stream, the techniques in the catalog.

## 8. Order of work, if adopted

1. A real end-to-end launch of the existing agent through the deployed Modal app. Everything
   else depends on this.
2. The hypothesis record and the hypotheses table on the page.
3. The claim card from the Scout, feeding the claims table.
4. The explore phase with its gate rules, run on Sunday morning with whatever budget is left.
5. The surprise trigger.
6. The comparison in section 6.

Items 2 and 3 are small. Item 4 is the one that makes this a research agent.

## 9. Open questions

**For brendan (agent and page)**
- Is a fourth phase the right shape, or should explore be a budget slice inside the existing
  phases?
- Does the Scout's claim card fit the existing `Candidate` type, or is it a new one?
- Which of sections 5.1 to 5.4 fit before the submission deadline (Sunday 14:00)?
- Has the existing agent launched real runs yet? If not, what is in the way?

**For himani (design and analysis)**
- What is the decision rule for an exploratory hypothesis: sign and an interval excluding zero
  on the selection half, at two or more paired seeds?
- Should a confirmed exploratory hypothesis be eligible for a confirmatory rerun on the report
  half, or stay labelled exploratory?
- What budget share can explore have without threatening the screen and replication?

**For peter (techniques)**
- Which parameters of each technique are safe for an agent to vary, and over what range? The
  gate needs these declared in the catalog.
- Are there combinations that should be refused outright (for example two techniques that both
  set the learning rate)?

**For justin (bench and executor)**
- Rows need a way to be marked exploratory. Proposed: use the existing `notes` and
  `parent_spec_hash` fields of the run spec, with no format change. Is that enough for the
  analysis to filter on, or does the row need a field?

## Comments

### Member B (peter)

Answers from peter's agent, 14 November, for the two questions in section 9. Peter has not
reviewed the ranges line by line; treat them as the technique owner's starting point.

**Where the hard limits come from.** `clharness.techniques.available()` returns every technique's
name and the JSON Schema of its parameters, with types, allowed values and bounds. The catalog
can be built from that call, so it cannot drift from the plugins. The simplest gate check for
"is this spec runnable" is to call `config_from_spec(spec)` and `build_hooks(spec.techniques)`:
both raise `ValueError` with the reason, before any GPU is used.

**Ranges that make sense on the frozen bench** (153 optimizer steps per domain, 26 layers). The
schema allows more than this; outside these ranges a run works but is unlikely to mean much.

| Technique | Parameter | Range for the agent | Why |
|---|---|---|---|
| `replay` | `share` | 0.01 to 0.5 | The literature scan has 50% slowing adaptation |
| | `source` | `general`, or a domain earlier in the stream | Replay is off while the source itself is being trained |
| | `select` | `random`, `loss`, `act_nov` | `random` is the control |
| | `pool` | 2 to 8, with `share` x `pool` at most 1 | Scoring costs `share` x `pool` forward-only passes |
| | `keep` | `high`, `low` | Both directions are open questions |
| | `layer` | 0 to 25 (`act_nov` only) | |
| | `buffer_seqs` | unset, or 1,000 to 50,000 | A tiny buffer is memorised |
| `lr_rewarm` | `rewarm_steps` | 1 to 40 | Must stay well under the 153 steps of a domain |
| | `min_lr_ratio` | 0 to 1 | 1 with `schedule: constant` is warmup only |
| | `schedule` | `cosine`, `linear`, `constant` | |
| `wavg` | `alpha` | 0.05 to 0.75 | 0 does nothing; 1 returns the base model |
| | `mode` | `boundary`, `final`, `ema` | |
| | `ema_decay` | 0.9 to 0.995 | At 0.999 the average is still about 86% base after one domain |
| `freeze` | `bottom_layers`, `top_layers` | sum at most 25 | The plugin refuses a model with nothing left to train |
| | `embeddings` | true, false | Tied to the output head in Gemma |
| `lora` | `rank` | 4 to 256, jointly with `lr` two rows down | 256 fits on an H100; nothing above it has run |
| | `targets` | `attn`, `attn_mlp` | |
| | `lr` | 1e-4 up to 1e-3 x sqrt(8 / `rank`), never above 1e-3; or null for the bench rate | This sets the learning rate of the whole run. Corrected on 14 November: this table first gave `rank` and `lr` independent ranges up to 3e-3, and rank 256 at 1e-3 then diverged on three seeds (`results/agent/rows.jsonl`). With `alpha` at 2 x `rank` the size of a step grows roughly with `lr` x sqrt(`rank`), so the two are bounded together. The explore gate enforces this (`lora_lr_limit` in `agent/catalog_real.py`) |
| `distill` | `weight` | 0.1 to 10 | |
| | `temperature` | 0.5 to 2 | |
| | `top_k` | 32 to 1024 | Below about 25 hurt in the literature; null (full vocabulary) has not run on a GPU |
| | `rows` | `all`, `new`, `replay` | `replay` needs `replay` in the same run |

**Combinations to refuse.**

- `distill` with `rows: replay` and no `replay` in the spec. The plugin fails at run start; the
  gate should refuse it earlier.
- Two techniques that set the same config field to different values. `apply_config_overrides`
  already raises for this. Today only `lora` sets the learning rate and only `lr_rewarm` sets
  the schedule, so those two can run together.
- `wavg` with `alpha: 1` in `boundary` mode: it resets to the base model after every domain.
- Any spec whose forward-only tokens would exceed its gradient tokens (for example `replay`
  with `share` x `pool` above 1, or scored selection together with `distill` on all rows at a
  large pool). Not wrong, but it stops being a matched-compute comparison.

**Combinations to allow but label.**

- Anything with `lora` at a learning rate other than the bench's: it differs from its controls
  in learning rate as well as in the technique. The lower-LR controls matter most here.
- `lora` with `wavg`: in modes `boundary` and `final` only the B factors are averaged, so `alpha`
  scales the adapter's weight change exactly. Mode `ema` averages both factors, which is not the
  same as averaging the weight change.
- `freeze` with `lora`: adapters go only into the layers left trainable, so the pair is
  "adapters in the unfrozen half", not two independent effects.

Combinations of the six screened techniques all train on the tiny model
(`tests/test_screen_combos.py`). Only single techniques, and replay with distil-on-replay, have
run on a GPU so far.

### Member C (himani)

### Member D (brendan)

Answers from brendan's agent, 14 November, for the four questions in section 9. Brendan has not
reviewed them yet; treat them as the agent owner's starting point.

**Has the existing agent launched real runs?** No. What is in the way:

- `python -m clharness.agent.loop` accepts only `--executor sim`. `ModalExecutor`
  (`src/clharness/agent/modal_executor.py`) is written and tested against a fake Modal, but it
  is not wired into the loop's command line and has never called the deployed app.
- The deployed app accepts callers only at the commit it was deployed from, so the launch
  commit has to be fixed first and the checkout must not move during the grid.
- A small real test launch (2 or 3 specs) needs brendan's go-ahead to spend.
- The Conductor's SCOUT and READ steps are still minimal: no Scout runs behind SCOUT today. A
  Scout that searches with Exa and returns one candidate with its sources is written locally
  and under review, not pushed yet.

**A fourth phase or a budget slice?** A fourth phase. The state machine already runs the same
cycle per phase, so explore is one more phase with its own skill file and gate rules. A slice
inside screen or replicate would put exploratory and confirmatory rows in the same phase, and
the Gatekeeper would have to tell them apart spec by spec. The Conductor reserves explore's
budget share up front, so explore cannot eat into the screen or replication.

**Does the claim card fit `Candidate`?** No, it is a new type. `Candidate` describes a technique
to build: mechanism, parameter specs, a cost multiplier and a human review before use. A claim
card needs the claim, its `source_ids`, the catalog arms that test it and a prediction, and no
code review, since it runs only existing plugins. Both share the `Source` type and the same
sources file, and `ScoutResult` would carry either a candidate or a claim.

**What fits before Sunday 14:00?** In this order, stopping wherever time runs out:

1. The real launch through Modal (section 8, item 1). Nothing else counts until this works.
2. The Scout inside the loop at the SCOUT step, with its sources in the decision log. This is
   what the Exa challenge requires.
3. The claim card and the hypothesis record with its table on the page (5.1, 5.3 literature,
   5.4). Small, because they reuse the Scout and the existing stores.
4. The explore phase (5.2), only if items 1 and 2 work by Saturday night.

The surprise trigger and the comparison in section 6 do not fit.
