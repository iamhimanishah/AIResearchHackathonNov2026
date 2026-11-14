# Working in this repository

Read [`docs/DIRECTION.md`](docs/DIRECTION.md) first. It states the goal, the priorities, the
workstreams and the rules every reported experiment follows.

**Goal:** build a test bench that makes continual-learning experiments fast to run, then use it
to measure which forgetting-reduction techniques work, alone and in combination.

## Many agents work here at the same time

Several agents and people are changing this repository in parallel, on different workstreams.
Assume the remote has moved since you last looked.

- **Pull often.** Pull before you start, before every commit and before every push:
  `git pull --rebase origin main`. If you work on a branch, rebase it onto `main` just as often.
- **Commit small and push soon.** A change that sits unpushed for an hour will conflict with
  someone else's.
- **Stay inside your workstream.** Touch the files your task needs and nothing else. Do not
  reformat, rename or reorganize files you are not working on.
- **Re-read before you edit.** After a pull, look again at any file you are about to change.
  Someone may have changed it or already done the work.
- **Resolve conflicts by rebasing.** Never force-push `main`, and never discard someone else's
  change to make yours apply. If you cannot resolve a conflict safely, stop and say so.
- **Shared files need care.** `docs/DIRECTION.md`, the run-spec and result-row formats, and
  `src/clharness/train/loop.py` are used by every workstream. Keep edits to them small, and say
  what you changed in the commit message.

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
