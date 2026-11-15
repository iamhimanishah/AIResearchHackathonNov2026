# Handoff: an autonomous Scout that proposes claims and candidates

**Status: design approved for implementation by brendan's agent, 14 November 2026, pending
brendan's OK.** Workstream 5 (agent). Read this whole file before writing code. Every claim
below was checked against the file named next to it on main at `5c70192`.

## 1. The decision

The Scout returns a **ranked shortlist** instead of one item, split by how risky it is to act on:

| Output | What it is | Who acts on it | Human in the loop? |
|---|---|---|---|
| **Claim card** | A published claim that the plugins we already have can test (a parameter value or a combination of existing techniques) | The loop itself: Planner proposes specs, Gatekeeper checks them, Operator launches | **No.** Deterministic code checks replace the human |
| **Candidate** | A technique that needs new plugin code | A person (or a coding agent a person supervises) | **Yes, but the loop never waits for it.** The card goes in a review queue; the loop carries on with claims |

Why this split:

- **Autonomy counts for 20% of the score.** The criterion is "how much of the research loop runs
  without a human" (`docs/hackathon-brief.md:185`, `docs/DIRECTION.md:45`). Claim tests
  need no new code, so nothing a human does there adds safety that the Gatekeeper can't give.
- **The Exa brief wants this exact behaviour.** It asks to "check a claim against real
  sources, or discover a method nobody told the agent about", with "checkable output" worth 20%
  (`docs/hackathon-brief.md:134,142`). Claim cards check claims; candidates discover methods.
- **Agent-written code on paid GPUs is the one real risk.** The spec keeps a human review
  for plugins (`docs/agent/ARCHITECTURE.md:52,88,101`; `docs/research-agent.md` section 7). This
  design keeps that review and removes the waiting: ARCHITECTURE's own rule is "veto, never
  approval; nothing waits for a human except HALTED and the Scout's plugin review".
- **More than one item, because one throws evidence away.** A live search returned 18 sources
  for two queries (run on 14 November). Keeping one hides what was passed over and why.
- **Rigour is unchanged.** Everything the loop runs from a claim card is exploratory
  (`analysis/PREREG.md` lines 181, 383-385): reported with intervals, never part of the win rule
  or a Holm family.

Spec conflict, to settle with brendan: `docs/DIRECTION.md:110` (priority 4) says the agent "adds
it as a plugin and tests it", while ARCHITECTURE forbids the Scout from adding a plugin itself.
This design follows ARCHITECTURE for code and is fully autonomous for claims. If brendan prefers
DIRECTION's wording, section 6 says what an unattended plugin path would need.

## 2. Blocking preconditions (do these first, or the loop is not running for real)

These come from peter's review (agentchat, 14 November) and I confirmed each one:

1. **The loop uses the simulator's catalog, not the real plugins.** `sim.default_catalog()`
   names `reference_arm`, `layer_freeze`, `weight_avg`, `select_*`. The real registry
   (`clharness.techniques.available()`) holds `distill`, `freeze`, `lora`, `lr_rewarm`,
   `replay`, `wavg`. So today the Scout's "not already in the catalog" check runs against the
   wrong names: it would accept `lr_rewarm` as new. **Fix:** build the real-mode `Catalog` from
   `techniques.available()` (names, parameter schemas) plus peter's ranges in
   `docs/research-agent.md` under Comments / Member B.
2. **`loop.py` only offers `--executor sim`.** It needs a real-mode builder: `ModalExecutor`, a
   real clock, the PREREG thresholds, the frozen stream, and `results/agent/rows.jsonl` from the
   grid that `launch_design` has already started. Justin's `launch_design.py` shows how to drive
   `ModalExecutor`; reuse it, don't copy it.
3. **The Gatekeeper accepts specs the bench cannot run** (peter, item 4: `techniques:
   [no_mitigation]`, `lora rank 16.5`). Add a rule that calls `techniques.build_hooks(...)`
   and `config_from_spec(...)` on every spec and drops it with the `ValueError` text. Without
   this, autonomous claim tests are not safe.
4. **The Gatekeeper refuses every optimizer override** (`gatekeeper.py:85`). Claims about the
   learning rate (for example PREREG claim 2) cannot pass. For the explore phase, allow `optim.lr`
   inside a declared range and label the run.

## 3. Contracts to add or change (all in `src/clharness/agent/`, workstream 5)

### 3.1 `ScoutDraft` and `ScoutResult` (agents.py, roles.py)

```python
class ScoutDraft(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list, max_length=5)        # ranked
    candidates: list[CandidateDraft] = Field(default_factory=list, max_length=5) # ranked
    skip_reason: str | None = None   # set when both lists are empty

class ScoutResult(Frozen):
    claims: tuple[ClaimCard, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    rejected: tuple[str, ...] = ()   # one line per draft item the checks dropped, with the reason
    sources: tuple[Source, ...] = ()
    skipped_reason: str | None = None
    meta: CallMeta | None = None
```

`ScoutResult.candidate` (singular) goes away; update `conductor._scout`, `write_scout_result`
and `tests/agent/test_scout.py` in the same commit.

### 3.2 `ClaimCard` (contracts.py, new)

| Field | Meaning |
|---|---|
| `id` | `claim-` + hash of (claim text, sorted source ids) |
| `claim` | One sentence, as the paper states it |
| `source_ids` | At least one, each one returned by this search |
| `test_arm` | `{technique: params}` using catalog techniques only |
| `control_arm` | Usually `no_mitigation` or the reference arm, as specs the PREREG already has |
| `prediction` | `direction` (+1, 0, -1) and `size_points` (rough), stated before any run |
| `duplicate_of` | The id of a `design.CLAIMS` entry it repeats, or null |
| `status` | `proposed`, `accepted`, `rejected`, `tested` |

### 3.3 Checks in code, applied to each draft item (never by a model)

Claims:
- every `source_id` was returned by this search;
- `test_arm` and `control_arm` build with `techniques.build_hooks` and sit inside the catalog's
  declared ranges;
- not a repeat of `design.CLAIMS` / `design.CLAIM_ARMS` (match by arm config, not by wording):
  if it repeats one, keep the card with `duplicate_of` set but don't plan it;
- the test arm's spec hash has no finished row yet.

Candidates (unchanged): real source ids, a name not in the **real** catalog.

A draft that fails a check goes in `rejected` with its reason. It is not repaired.

### 3.4 Planner context (context.py)

Add `claim_cards` to the Planner's `reads` in `CONTEXT_MAP`, wrapped as untrusted (they carry
text that came from the web). The Planner never sees `Source.snippet`, only the claim and the
arms. Update the firewall test.

### 3.5 Where claim tests run

**Update, 14 November:** justin owns the explore step as `src/clharness/agent/explore.py`
(branch `agent-explore`). This workstream does **not** build an explore phase. The Scout writes
its accepted claim cards where `explore.py` reads hypotheses (`results/agent/explore/
hypotheses.jsonl`, origin `literature`, with a source field holding title and URL). Agree the
exact fields with justin before writing code. The rules below are what we asked explore to
enforce; keep them as a checklist for reviewing his branch.

Claim tests go in a fourth phase, `explore`, after combine (`docs/research-agent.md` 5.2, and
brendan's answer there). States: `ADVANCE(combine) -> PLAN(explore)` when there are accepted
claim cards and explore has budget left, otherwise `FREEZE`. Gate rules for explore:
- specs come from accepted claim cards only, 2 seeds from the PREREG's replicate seeds, paired
  with the control;
- explore has its own budget share, reserved up front (default 15% of the compute cap, set in
  `AgentConfig`; **himani has not answered section 9's budget question: use the default and say
  so in the commit**);
- every row is marked exploratory: set `notes: "explore:<claim id>"` on the spec (Justin's
  section 9 proposal; no row-format change). Himani decides whether analysis also needs a field.

**Important:** if time runs short, skip the explore phase and run the accepted claim specs
inside combine with the same gate rules. That is a smaller change (no new state) and keeps the
loop autonomous.

### 3.6 Candidates: the review queue

- `results/agent/candidates/<ts>.json` holds the ranked candidates with their sources. This
  already exists for one candidate.
- The page lists them with status `pending`, `approved` or `rejected`.
- A person marks a candidate approved or rejected by appending a line to
  `results/agent/veto.jsonl` (the file the architecture already names). The loop doesn't
  wait. An approved candidate enters the catalog only when its plugin is merged, with a sanity
  run (peter's rule for every technique).

### 3.7 Search quality

- Merge arXiv versions before deduplicating: `abs/2610.22293` and `abs/2610.22293v1` are one
  paper. Normalise `arxiv.org/{abs,html,pdf}/<id>[vN][.pdf]` to `arxiv:<id>`.
- Keep the request shape (query, `objective`, `type="auto"`, bare `highlights`). Add a third
  query aimed at claims: "measured effect of replay ratio / learning-rate re-warm / LoRA rank
  on forgetting in continual pre-training of LLMs", with an objective asking for the
  reported number and the setup.

## 4. Tests to add (no network, no GPU)

1. The Scout returns 2 claims and 2 candidates from a fake search and a fake model. A claim
   with an unknown technique, an out-of-range parameter or an invented source goes to `rejected`.
2. A claim that repeats `design.CLAIMS` id 3 gets `duplicate_of = 3` and is not planned.
3. Simulated loop: SCOUT writes claim cards, the Planner sees them (check the prompt), the
   explore phase (or combine) launches their specs with `notes = explore:<id>`, and FREEZE ignores
   explore rows.
4. Gatekeeper: `build_hooks` failure drops the spec. `optim.lr` outside the range drops it. The
   explore budget share caps spend.
5. The real-mode catalog has exactly the names from `techniques.available()`.
6. A kill at every point of SCOUT and explore resumes to the same decisions (extend
   `test_kill_at_every_state_and_point_then_resume_ends_in_the_same_place`).

## 5. Order of work and size

| Step | What | Size |
|---|---|---|
| 1 | Real-mode catalog from `techniques.available()` (precondition 1) | small |
| 2 | `build_hooks` gate rule; `optim.lr` allowed in explore (preconditions 3, 4) | small |
| 3 | `ScoutDraft`/`ScoutResult` lists, `ClaimCard`, the checks, arXiv dedup (3.1-3.3, 3.7) | medium |
| 4 | Claim cards into the Planner; claim specs run in combine (3.4, fallback in 3.5) | medium |
| 5 | Real-mode `loop.py --executor modal` (precondition 2) | medium; needs a real launch |
| 6 | Claim cards written to justin's `hypotheses.jsonl` (3.5; explore itself is justin's) | small |
| 7 | Candidate review queue on the page (3.6) | small |

Steps 1-4 can be built and tested on the simulator tonight. Step 5 is what makes it real. Do
not deploy or launch while the grid runs: Justin's Modal app must stay at `5c70192` until the
grid's rows are in.

## 6. If brendan wants plugins unattended too (not recommended for this weekend)

An agent-written plugin could skip the human only if all of these hold: it uses only the hook
interface (no file, network or environment access); it passes the tiny-model combination test
(`tests/test_screen_combos.py` pattern); it passes a 4-step GPU sanity run under a cost cap of
about $0.35; and it runs under a separate budget line. None of that exists yet, and a mistake costs
GPU money and a public repo's trust, so this design leaves it out.

## 7. What exists today (so nobody rebuilds it)

- `search.py`: `ExaSearch` (live, verified: 18 sources), `FakeSearch`.
- `agents.py`: `LLMScout` (one candidate), `write_scout_result`.
- `conductor.py`: the SCOUT step after replicate, sources in the decision log (local, not yet
  pushed at the time of writing).
- `scout.py`: `python -m clharness.agent.scout --search-only`.
- `launch_design.py` (justin): the deterministic launcher now running the pre-registered grid.
