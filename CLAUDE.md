# Working in this repository

Read [`docs/DIRECTION.md`](docs/DIRECTION.md) first. It states the goal, the priorities, the
workstreams and the rules every reported experiment follows.

**Goal:** build a test bench that makes continual-learning experiments fast to run, then use it
to measure which forgetting-reduction techniques work, alone and in combination.

## Before changing code

- `docs/harness.md` explains the harness; `docs/baselines.md` has the numbers to beat.
- Techniques are hooks (`src/clharness/train/hooks.py`). Add a technique as a plugin; do not
  special-case it inside the training loop.
- Run `make setup` once, then `make test` before every commit. The fast tests need no GPU and
  no downloads.

## Do not

- Change the frozen eval manifests under `manifests/` or the baseline results under
  `results/baselines/` and `results/runs/pilot-*`.
- Commit data, model weights, logs or secrets. `data/` and `.env` are ignored on purpose.
- Rewrite the first commit. It marks what existed before the hackathon.
- Put private details in this repository. It is public.

## When you finish a piece of work

- Keep the existing tests passing and add a test for new behaviour.
- If the work changes the plan, a format or a rule, update `docs/DIRECTION.md` in the same
  commit.
