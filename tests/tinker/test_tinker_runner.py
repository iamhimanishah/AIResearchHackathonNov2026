"""C5 runner against a stub Tinker service: valid R matrix, masked prompts, per-item
files, the budget guard, and a failed row when the backend dies. No network, no Tinker."""

import numpy as np

from clharness.tinker import runner
from clharness.tinker.data import TASK_ORDER
from clharness.tinker.fake import FakeFuture, FakeService, FakeTokenizer, FakeTypes
from clharness.tinker.rows import TinkerRow, Tokens, append_tinker_row


def _examples(n_train=12, n_eval=6):
    out = {}
    for t in TASK_ORDER:
        fields = {f: f"{t} {f} text" for f in runner.TASKS[t].fields}
        out[t] = {
            "train": [dict(fields, label=i % len(runner.TASKS[t].labels))
                      for i in range(n_train)],
            "eval": [dict(fields, label=i % len(runner.TASKS[t].labels))
                     for i in range(n_eval)],
        }
    return out


def _run(arm="naive", seed=0, service=None, **kw):
    return runner.run_arm(
        arm=arm, seed=seed, n_train=12, n_eval=6, examples=_examples(),
        service=service or FakeService(), types=FakeTypes,
        manifest_path=kw.pop("manifest_path", __file__),  # any real file for the sha
        items_dir=kw.pop("items_dir", None) or __import__("tempfile").mkdtemp(),
        epochs=kw.pop("epochs", 2), batch_size=kw.pop("batch_size", 4), **kw)


def test_full_naive_run_yields_a_valid_r_matrix(tmp_path):
    service = FakeService()
    row = _run(service=service, items_dir=tmp_path)
    assert row.status == "finished", row.reason
    assert len(row.r_matrix) == 3
    for s, r in enumerate(row.r_matrix):
        for t in range(3):
            if t <= s:
                assert r[t] is not None and 0.0 <= r[t] <= 1.0
            else:
                assert r[t] is None  # unseen tasks are null, not scored
    arr = np.load(tmp_path / row.items_path.split("/")[-1])
    assert arr.shape == (3, 3, 6)
    assert np.isnan(arr[0, 1:]).all() and np.isnan(arr[1, 2]).all()
    assert row.tokens and row.tokens.train > 0 and row.tokens.prefill > 0
    assert row.est_usd and row.est_usd > 0
    assert row.lora.rank == 32 and row.hyperparameters.lr == 3e-4


def test_prompt_tokens_are_masked_and_targets_shifted(tmp_path):
    service = FakeService()
    _run(service=service, items_dir=tmp_path)
    datums = service.clients[0].received
    assert datums, "the fake client saw no training data"
    enc = FakeTokenizer().encode(" negative")
    for d in datums:
        w = d.loss_fn_inputs["weights"]
        targets = d.loss_fn_inputs["target_tokens"]
        inp = d.model_input.tokens
        assert len(w) == len(targets) == len(inp)
        # all zeros before all ones: the loss covers only the trailing answer tokens
        n_on = int(sum(w))
        assert w == [0.0] * (len(w) - n_on) + [1.0] * n_on
        # input is full[:-1], targets full[1:]: every position predicts the next token
        assert targets[:-1] == inp[1:]
        # the single unmasked target is the verbalizer token (fake tokenizer: 1 token/label)
        assert n_on == len(enc) and targets[-1] in (
            FakeTokenizer().encode(" negative") + FakeTokenizer().encode(" positive")
            + FakeTokenizer().encode(" entailment") + FakeTokenizer().encode(" neutral")
            + FakeTokenizer().encode(" contradiction") + FakeTokenizer().encode(" World")
            + FakeTokenizer().encode(" Sports") + FakeTokenizer().encode(" Business")
            + FakeTokenizer().encode(" Sci/Tech"))


def test_estimate_matches_the_realized_token_counts(tmp_path):
    examples = _examples()
    encode = FakeTokenizer().encode
    est = runner.estimate_run(arm="naive", seed=0, examples=examples, encode=encode,
                            n_train=12, n_eval=6, batch_size=4)
    service = FakeService()
    row = _run(service=service, items_dir=tmp_path)
    assert est["tokens"]["train"] == row.tokens.train == service.train_tokens
    assert est["tokens"]["prefill"] == row.tokens.prefill == service.prefill_tokens


def test_replay_and_joint_rows(tmp_path):
    row = _run(arm="replay-200", service=FakeService(), items_dir=tmp_path)
    assert row.status == "finished", row.reason
    assert row.replay_share_examples[0] == 0.0
    assert row.replay_share_examples[2] > 0
    assert len(row.replay_share_tokens) == 3
    j = _run(arm="joint", service=FakeService(), items_dir=tmp_path)
    assert j.status == "finished" and len(j.r_matrix) == 1
    assert all(v is not None for v in j.r_matrix[0])


def test_budget_guard_refuses_over_cap(tmp_path):
    rows = tmp_path / "rows.jsonl"
    spent = TinkerRow(arm="naive", seed=0, model="m", status="finished",
                      r_matrix=[[0.5]], items_path="x.npy",
                      tokens=Tokens(train=1), est_usd=65.0)
    append_tinker_row(rows, spent)
    assert runner.check_budget(6.0, rows, cap=70.0) is not None
    assert runner.check_budget(4.0, rows, cap=70.0) is None
    assert runner.check_budget(69.0, tmp_path / "empty.jsonl", cap=70.0) is None


class _ExplodingService(FakeService):
    def create_lora_training_client(self, **kw):
        return FakeFuture(_ExplodingTraining())


class _ExplodingTraining:
    def get_tokenizer(self):
        return FakeTokenizer()

    def forward_backward(self, data, loss_fn):
        raise ConnectionError("tinker is down")


def test_a_backend_failure_becomes_a_failed_row(tmp_path):
    row = _run(service=_ExplodingService(), items_dir=tmp_path)
    assert row.status == "failed"
    assert "tinker is down" in row.reason


def test_unknown_arm_fails_cleanly(tmp_path):
    row = runner.run_arm(arm="wiggle", seed=0, n_train=4, n_eval=2,
                         examples=_examples(4, 2), service=FakeService(),
                         types=FakeTypes, manifest_path=__file__, items_dir=tmp_path)
    assert row.status == "failed" and "unknown arm" in row.reason
