"""Run one C5 arm at one seed on Tinker, and return one `TinkerRow`.

Flow: LoRA training client on Qwen/Qwen3-8B -> per stage, forward_backward on batches of
prompt+verbalizer sequences (loss on the verbalizer tokens only, the prompt masked) ->
save_weights_and_get_sampling_client -> score every seen task's eval items by verbalizer
log-likelihood (compute_logprobs, argmax over the label verbalizers) -> the R matrix and a
per-item correctness file for the bootstrap.

`tinker` is imported lazily; tests inject a stub service and types with the same method
names (clharness.tinker.fake). No function in this module touches the API unless a service
is asked for.

Cost: prefill/train/sample tokens are counted exactly as sent and priced from the C5 card
(docs/TASKS.md): $0.44/M train, $0.60/M sample, $0.195/M prefill. Every compute_logprobs
call is counted as a full uncached prefill, so est_usd is an upper bound — the server's
prefix cache would bill repeated prompt prefixes at $0.039/M.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from clharness.config import REPO_ROOT
from clharness.data.manifest import sha256_file
from clharness.tinker.arms import REPLAY_PER_TASK, Schedule, build_schedule
from clharness.tinker.data import (
    MANIFEST_PATH,
    TASK_ORDER,
    TASKS,
    load_task_examples,
    read_manifest,
)
from clharness.tinker.rows import TinkerRow, Tokens, spent_usd

# Pre-registered hyperparameters (PREREG section 11; himani confirms before any run).
MODEL_ID = "Qwen/Qwen3-8B"
LORA_RANK = 32
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
MAX_LEN = 256
EPOCHS = 2
EVAL_WINDOW = 40  # eval items scored concurrently per window

# Tinker list price for Qwen/Qwen3-8B, $/M tokens (thinkingmachines.ai/tinker/pricing,
# dated 2026-10-19, per docs/TASKS.md card C5).
PRICE_PER_MTOK = {"train": 0.44, "sample": 0.60, "prefill": 0.195}
BUDGET_USD = 70.0  # C5 cap: 70% of one member's Tinker credit


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def est_usd(tokens: dict[str, int]) -> float:
    return round(sum(tokens.get(k, 0) * p for k, p in PRICE_PER_MTOK.items()) / 1e6, 4)


@dataclass
class Encoded:
    prompt: list[int]
    answer: list[int]
    label: int

    @property
    def n_tokens(self) -> int:
        return len(self.prompt) + len(self.answer)


def _encode(tokenizer) -> Callable[[str], list[int]]:
    def enc(text: str) -> list[int]:
        try:
            return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
        except TypeError:
            return [int(t) for t in tokenizer.encode(text)]

    return enc


def encode_row(task, row: dict[str, Any], encode, max_len: int = MAX_LEN) -> Encoded:
    """Template + field tokens, fields right-truncated so prompt+answer fits `max_len`."""
    import re

    answer = encode(" " + task.labels[int(row["label"])])
    parts = re.split(r"\{(\w+)\}", task.template)  # static, field, static, field, ...
    static = [encode(p) for j, p in enumerate(parts) if j % 2 == 0]
    field_names = [parts[j] for j in range(len(parts)) if j % 2 == 1]
    assert field_names == list(task.fields), f"{task.name}: template fields {field_names}"
    static_tokens = sum(len(s) for s in static)
    budget = max_len - static_tokens - len(answer) - 1
    per_field = max(8, budget // max(1, len(field_names)))
    field_tokens = {f: encode(str(row[f]).strip())[:per_field] for f in field_names}
    prompt: list[int] = []
    si = 0
    for j, p in enumerate(parts):
        if j % 2 == 0:
            prompt += static[si]
            si += 1
        else:
            prompt += field_tokens[p]
    return Encoded(prompt=prompt, answer=answer, label=int(row["label"]))


def _datum(types, enc: Encoded):
    full = enc.prompt + enc.answer
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            # position i of the input predicts token i+1: mask every prompt target,
            # keep the loss on the verbalizer tokens only
            "weights": [0.0] * (len(enc.prompt) - 1) + [1.0] * len(enc.answer),
            "target_tokens": full[1:],
        },
    )


def _estimate(schedule: Schedule, encoded: dict[str, dict[str, list[Encoded]]],
              v_tokens: dict[str, list[list[int]]]) -> dict[str, int]:
    train = sum(enc.n_tokens
                for st in schedule.stages for b in st.batches for it in b
                for enc in [encoded[it.task]["train"][it.index]])
    prefill = 0
    for st in schedule.stages:
        for task in schedule.tasks_seen_at(st.index):
            for enc in encoded[task]["eval"]:
                prefill += sum(len(enc.prompt) + len(v) for v in v_tokens[task])
    return {"train": train, "prefill": prefill, "sample": 0}


def _verbalizer_tokens(tasks: list[str], encode) -> dict[str, list[list[int]]]:
    return {t: [encode(" " + lbl) for lbl in TASKS[t].labels] for t in tasks}


def _score_eval_items(sc, types, items: list[Encoded], v_tokens: list[list[int]],
                      window: int, counter: dict[str, int]) -> list[int]:
    correct: list[int] = []
    for lo in range(0, len(items), window):
        chunk = items[lo : lo + window]
        calls = [
            (i, l, sc.compute_logprobs(types.ModelInput.from_ints(e.prompt + vt),
                                       prompt_logprobs_last_n=len(vt)),
             len(vt))
            for i, e in enumerate(chunk) for l, vt in enumerate(v_tokens)
        ]
        scores = [[0.0] * len(v_tokens) for _ in chunk]
        for i, l, fut, n in calls:
            counter["prefill"] += len(chunk[i].prompt) + n
            lps = fut.result()
            scores[i][l] = sum(x for x in lps[-n:] if x is not None)
        correct += [int(np.argmax(s)) == e.label for s, e in zip(scores, chunk)]
    return correct


def run_arm(
    *,
    arm: str,
    seed: int,
    n_train: int,
    n_eval: int,
    examples: dict[str, dict[str, list[dict[str, Any]]]],
    service,
    types,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    lr: float = LEARNING_RATE,
    lora_rank: int = LORA_RANK,
    model: str = MODEL_ID,
    replay_per_task: int = REPLAY_PER_TASK,
    smoke: bool = False,
    manifest_path: str | Path = MANIFEST_PATH,
    items_dir: str | Path = REPO_ROOT / "results" / "tinker" / "items",
    git_sha: str | None = None,
    eval_window: int = EVAL_WINDOW,
    now: Callable[[], str] = _now,
) -> TinkerRow:
    """Train and score one arm. Never raises for a problem with the run itself: the
    exception becomes a `failed` row carrying the tokens spent so far."""
    from clharness.tinker.data import template_sha256
    from clharness.wandb_log import git_commit

    started = now()
    common: dict[str, Any] = {
        "arm": arm, "seed": seed, "model": model, "epochs": epochs, "smoke": smoke,
        "n_train": n_train, "n_eval": n_eval, "git_sha": git_sha or git_commit() or "unknown",
        "started_at": started,
        "lora": {"rank": lora_rank},
        "hyperparameters": {"lr": lr, "batch_size": batch_size, "max_len": max_len,
                            "replay_per_task": replay_per_task if arm == "replay-200" else None},
        "template_sha256": template_sha256(),
        "manifest_sha256": sha256_file(Path(manifest_path)) if Path(manifest_path).exists() else "",
        "manifest": _rel(manifest_path),
    }
    tokens = {"train": 0, "prefill": 0, "sample": 0}
    try:
        schedule = build_schedule(arm, n_train, epochs=epochs, batch_size=batch_size,
                                  seed=seed, replay_per_task=replay_per_task)
        tc = service.create_lora_training_client(
            base_model=model, rank=lora_rank, seed=seed).result()
        encode = _encode(tc.get_tokenizer())
        encoded = {
            t: {"train": [encode_row(TASKS[t], r, encode, max_len) for r in examples[t]["train"]],
                "eval": [encode_row(TASKS[t], r, encode, max_len) for r in examples[t]["eval"]]}
            for t in TASK_ORDER
        }
        v_tokens = _verbalizer_tokens(TASK_ORDER, encode)
        r_matrix: list[list[float | None]] = []
        items = np.full((len(schedule.stages), len(TASK_ORDER), n_eval), np.nan,
                        dtype=np.float32)
        replay_share_tokens: list[float] = []
        for st in schedule.stages:
            stage_replay = stage_total = 0
            for batch in st.batches:
                datums = [_datum(types, encoded[it.task]["train"][it.index]) for it in batch]
                fb = tc.forward_backward(datums, "cross_entropy")
                op = tc.optim_step(types.AdamParams(learning_rate=lr))
                fb.result()
                op.result()
                for it in batch:
                    n = encoded[it.task]["train"][it.index].n_tokens
                    tokens["train"] += n
                    stage_total += n
                    stage_replay += n if it.replay else 0
            replay_share_tokens.append(stage_replay / stage_total if stage_total else 0.0)
            sc = tc.save_weights_and_get_sampling_client().result()
            row: list[float | None] = [None] * len(TASK_ORDER)
            for task in schedule.tasks_seen_at(st.index):
                ti = TASK_ORDER.index(task)
                correct = _score_eval_items(sc, types, encoded[task]["eval"],
                                            v_tokens[task], eval_window, tokens)
                items[st.index, ti, : len(correct)] = correct
                row[ti] = float(np.mean(correct)) if correct else None
            r_matrix.append(row)
        items_path = _write_items(items_dir, arm, seed, epochs, smoke, items)
        return TinkerRow(
            status="finished", r_matrix=r_matrix, items_path=items_path,
            replay_share_examples=[st.replay_share for st in schedule.stages],
            replay_share_tokens=replay_share_tokens,
            tokens=Tokens(**tokens), est_usd=est_usd(tokens),
            finished_at=now(), **common)
    except Exception as e:  # noqa: BLE001 - the run's own failure is data, not a crash
        reason = f"{type(e).__name__}: {e}".strip()[:600]
        return TinkerRow(status="failed", reason=reason, tokens=Tokens(**tokens),
                         est_usd=est_usd(tokens), finished_at=now(), **common)


def _write_items(items_dir: str | Path, arm: str, seed: int, epochs: int, smoke: bool,
                 items: np.ndarray) -> str:
    items_dir = Path(items_dir)
    items_dir.mkdir(parents=True, exist_ok=True)
    name = f"{arm}-s{seed}-e{epochs}{'-smoke' if smoke else ''}.npy"
    np.save(items_dir / name, items)
    try:
        return str((items_dir / name).relative_to(REPO_ROOT))
    except ValueError:
        return str(items_dir / name)


# --- loading + estimating (the --dry-run path does this without creating any client) ------------


def load_examples(n_train: int | None, n_eval: int | None,
                  manifest: dict | None = None,
                  manifest_path: str | Path = MANIFEST_PATH,
                  load_split=None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    manifest = manifest or read_manifest(manifest_path)
    return {
        t: load_task_examples(TASKS[t], manifest, n_train=n_train, n_eval=n_eval,
                              load_split=load_split)
        for t in TASK_ORDER
    }


def estimate_run(
    *,
    arm: str,
    seed: int,
    examples: dict[str, dict[str, list[dict[str, Any]]]],
    encode,
    n_train: int,
    n_eval: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    replay_per_task: int = REPLAY_PER_TASK,
) -> dict[str, Any]:
    """Exact token counts for a would-be run: the schedule is deterministic and the encoding
    is the same one the runner uses, so the dry-run cost is the real cost."""
    schedule = build_schedule(arm, n_train, epochs=epochs, batch_size=batch_size,
                              seed=seed, replay_per_task=replay_per_task)
    encoded = {
        t: {"train": [encode_row(TASKS[t], r, encode, max_len) for r in examples[t]["train"]],
            "eval": [encode_row(TASKS[t], r, encode, max_len) for r in examples[t]["eval"]]}
        for t in TASK_ORDER
    }
    v_tokens = _verbalizer_tokens(TASK_ORDER, encode)
    tokens = _estimate(schedule, encoded, v_tokens)
    return {"arm": arm, "seed": seed, "n_train": n_train, "n_eval": n_eval, "epochs": epochs,
            "tokens": tokens, "est_usd": est_usd(tokens)}


def check_budget(estimate_usd: float, rows_path: str | Path, cap: float = BUDGET_USD) -> str | None:
    """The C5 budget guard: refuse to start if this run's estimate would push the arm past
    `cap`, counting the est_usd of every row already in rows.jsonl. Returns the refusal
    message, or None when the run fits."""
    spent = spent_usd(rows_path)
    if spent + estimate_usd > cap:
        return (f"budget guard: ${spent:.2f} already recorded + ${estimate_usd:.2f} estimated "
                f"for this run would exceed the ${cap:.0f} C5 cap "
                f"({rows_path}); refusing to start")
    return None


def real_service():
    """The real ServiceClient. `tinker` is an optional extra; fail with instructions."""
    try:
        import tinker

        import clharness.env  # noqa: F401 - loads .env (TINKER_API_KEY) into os.environ
    except ImportError as e:
        raise RuntimeError(
            "the tinker backend needs the optional extra: `uv sync --extra tinker` "
            "or `pip install clharness[tinker]`") from e
    return tinker.ServiceClient(), tinker.types
