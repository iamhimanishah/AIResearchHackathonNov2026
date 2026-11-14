# Handoff: phase 0 for Member A's agent

From himani (workstream 4). This is a request, to be confirmed by Member A before any step
is taken. The plan it follows is [`PREREG.md` section 0](PREREG.md).

```text
You are Member A's agent (workstreams 1 and 2: bench core and executor). Read
docs/DIRECTION.md, then analysis/PREREG.md section 0: it is the plan for this task.

Context
- himani (workstream 4) has finished the pre-registration in PR #4
  https://github.com/justinmilner1/AIResearchHackathonNov2026/pull/4 (mergeable,
  `make test` green). Everything is decided except sigma_noise and the acquisition
  margins, which come from "phase 0": the controls only.
- PR #4 changes two things in your area:
  - executor/rows.py: half_indices reads manifests/eval-protected-v1-split.json (the
    pre-registered split, stratified by base correctness). Acquisition gains
    own_boundary_ppl_ratio / mean_own_boundary_ppl_ratio (perplexity right after each
    domain's own training). Answer collapse is recorded but no longer marks a row
    invalid.
  - fake.py follows the pre-registered screen (src/clharness/analysis/design.py).
- Your noise floor (results/rows.noise.jsonl) and the code->legal calibration rows were
  measured before PR #4: old random split, no own-boundary acquisition. Their spec
  hashes match the design, so they are reused after re-measuring, with no retraining.
- The technique comparison is ON HOLD until himani freezes the PREREG. Run controls
  only.

Do these in order. Stop and tell your human at the first failure.

1. Review and merge PR #4. `git pull --rebase origin main`, then `make test` must pass.

2. Redeploy the Modal app at the merged commit:
     modal deploy src/clharness/executor/modal_app.py

3. Re-measure 7 existing runs from their checkpoints (CPU only, about $0.10). Each
   run's files are at /vol/runs/<first 16 hex chars of spec_hash>/ on the clh-work
   volume:
     no-mitigation-s0 .. s4        rows in results/rows.noise.jsonl
     calibration-code-legal-lr5e-5-s0, calibration-code-legal-lr3e-5-s0
                                   rows in results/rows.calibration.jsonl
   The app has no entry point for this yet. Add a small CPU function, for example
   `remeasure(spec_hash)`:
   - call executor.rows.measure_run(run_dir, trained_domains=["code", "legal"]);
   - build a new ResultRow from the original row: keep spec, spec_hash, cost, backend
     and timestamps; replace retention, acquisition and answer_share with the new
     measurement; set git_sha to the deployed commit; set status from invalid_reason as
     run_spec does.
   Write the 7 rows to results/rows.phase0.remeasured.jsonl. Leave the original files
   unchanged.
   Also re-measure calibration-code-legal-lr1e-4-s0 into
   results/rows.reproducibility.jsonl. It is the same configuration as no-mitigation-s0
   with a different hash, so it checks GPU reproducibility. It is not part of the
   analysis.

4. Run the 7 phase-0 GPU specs (about 20 min in parallel, about $9; cap $12):
     clh run --modal $(grep -v '^#' configs/bench/phase0.txt) --out results/rows.phase0.jsonl
   That is reference-s0..s2, no-mitigation-lr5e-05-s1/s2 and
   no-mitigation-lr3e-05-s1/s2. Lines marked "# REUSE" are step 3, not GPU runs.

5. Check before you push:
   - `clh validate-row results/rows.phase0.jsonl results/rows.phase0.remeasured.jsonl`
   - all 14 rows are `finished`, and every one has
     acquisition.own_boundary_ppl_ratio for both code and legal;
   - the set of spec hashes in the two files equals design.phase0():
       uv run python -c "from clharness.analysis.design import phase0; print(sorted(s.spec_hash()[:12] for s in phase0()))"
   - `make test` still passes.
   Do NOT run `clh grid-analysis --confirmatory` and do not read report-half numbers to
   choose anything. himani computes sigma_noise and freezes the PREREG.

6. Commit the rows plus the remeasure function and its test. Pull with rebase, then push
   normally (never force). Then:
     agentchat merged "phase 0: 7 re-measured + 7 new control rows (results/rows.phase0*.jsonl); PREREG can freeze"
     agentchat touched himani "phase-0 rows are in; executor remeasure() added"

Rules: configs/bench/fast.yaml is frozen. Do not launch the screen or any technique
run: Brendan's agent does that after the freeze. Keys only in .env. Numbers come only
from result rows. End your report with REAL / STUBBED / HARDCODED.
```
