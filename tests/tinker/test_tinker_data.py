"""C5 data: deterministic indices, per-task sha, verbalizer coverage, prompt encoding."""

import pytest

from clharness.tinker import data as td


class FakeDS:
    """Stands in for a datasets.Dataset: features + len + row access."""

    def __init__(self, labels, rows):
        self.rows = rows
        self.features = {"label": type("F", (), {"names": list(labels)})()}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _fake_load_split(n_train_rows: int, n_eval_rows: int):
    """A dataset per (task, split): distinct, content-deterministic rows."""
    stores = {}
    for name in td.TASK_ORDER:
        task = td.TASKS[name]
        stores[name, task.train_split] = FakeDS(
            task.labels,
            [{**{f: f"{name} {f} row {i}" for f in task.fields},
              "label": i % len(task.labels)} for i in range(n_train_rows)])
        stores[name, task.eval_split] = FakeDS(
            task.labels,
            [{**{f: f"{name} {f} eval {i}" for f in task.fields},
              "label": i % len(task.labels)} for i in range(n_eval_rows)])
    return lambda task, split: stores[task.name, split]


def test_verbalizers_cover_every_label():
    for task in td.TASKS.values():
        for i, label in enumerate(task.labels):
            assert td.verbalizer(task, i) == " " + label
        # the verbalizer strings are what the scorer arg-maxes over
        assert len(set(task.labels)) == len(task.labels)


def test_template_renders_and_hashes_are_stable():
    assert td.template_sha256() == td.template_sha256()
    row = {"sentence": "a great film", "label": 1}
    p = td.render_prompt(td.TASKS["sst2"], row)
    assert "a great film" in p and p.endswith("Answer:")


def test_manifest_is_deterministic(tmp_path):
    loader = _fake_load_split(800, 60)
    m1 = td.build_manifest(out=tmp_path / "m1.json", seed=20261114, n_train=100, n_eval=20,
                           load_split=loader)
    m2 = td.build_manifest(out=tmp_path / "m2.json", seed=20261114, n_train=100, n_eval=20,
                           load_split=loader)
    assert m1 == m2
    for name, e in m1["tasks"].items():
        assert len(e["train_indices"]) == 100 and len(e["eval_items"]) == 20
        assert len(set(e["train_indices"])) == 100
        assert e["sha256"] == td._sha_examples(
            [{**{f: f"{name} {f} row {i}" for f in td.TASKS[name].fields},
              "label": i % len(td.TASKS[name].labels)} for i in e["train_indices"]],
            [{**{f: f"{name} {f} eval {it['index']}" for f in td.TASKS[name].fields},
              "label": it["index"] % len(td.TASKS[name].labels)}
             if it["split"] == td.TASKS[name].eval_split else
             {**{f: f"{name} {f} row {it['index']}" for f in td.TASKS[name].fields},
              "label": it["index"] % len(td.TASKS[name].labels)}
             for it in e["eval_items"]])


def test_eval_shortfall_tops_up_from_train_without_overlap(tmp_path):
    loader = _fake_load_split(n_train_rows=800, n_eval_rows=60)
    m = td.build_manifest(out=tmp_path / "m.json", n_train=100, n_eval=80, load_split=loader)
    e = m["tasks"]["sst2"]
    assert len(e["eval_items"]) == 80  # 60 validation + 20 held-out train
    top_up = [it["index"] for it in e["eval_items"] if it["split"] == "train"]
    assert len(top_up) == 20
    assert not set(top_up) & set(e["train_indices"])


def test_label_mismatch_is_caught(tmp_path):
    loader = _fake_load_split(800, 60)
    task = td.TASKS["sst2"]
    stores = {(task.name, task.train_split): FakeDS(["wrong", "names"],
              [{"sentence": "x", "label": 0}] * 800)}
    stores[task.name, task.eval_split] = FakeDS(task.labels,
                                               [{"sentence": "y", "label": 0}] * 60)
    def load(t, split):
        if t.name == "sst2":
            return stores[t.name, split]
        return loader(t, split)
    with pytest.raises(ValueError, match="label names"):
        td.build_manifest(out=tmp_path / "m.json", n_train=10, n_eval=10, load_split=load)


def test_load_task_examples_uses_frozen_indices(tmp_path):
    loader = _fake_load_split(800, 60)
    m = td.build_manifest(out=tmp_path / "m.json", n_train=100, n_eval=20, load_split=loader)
    ex = td.load_task_examples(td.TASKS["sst2"], m, n_train=10, n_eval=5, load_split=loader)
    assert len(ex["train"]) == 10 and len(ex["eval"]) == 5
    first = td.TASKS["sst2"].train_split
    idx = m["tasks"]["sst2"]["train_indices"][0]
    assert ex["train"][0]["sentence"] == f"sst2 sentence row {idx}"
    assert first == "train"


def test_verify_task_sha_passes_on_frozen_data_and_names_both_hashes(tmp_path):
    loader = _fake_load_split(800, 60)
    m = td.build_manifest(out=tmp_path / "m.json", n_train=100, n_eval=20, load_split=loader)
    for name in td.TASK_ORDER:
        td.verify_task_sha(td.TASKS[name], m, load_split=loader)

    def moved(task, split):
        ds = loader(task, split)
        if task.name == "sst2" and split == task.train_split:
            i = m["tasks"]["sst2"]["train_indices"][0]
            ds.rows[i] = {**ds.rows[i], "sentence": "the review changed"}
        return ds

    with pytest.raises(ValueError) as e:
        td.verify_task_sha(td.TASKS["sst2"], m, load_split=moved)
    ex = td.load_task_examples(td.TASKS["sst2"], m, load_split=moved)
    got = td._sha_examples(ex["train"], ex["eval"])
    want = m["tasks"]["sst2"]["sha256"]
    assert want != got and "sst2" in str(e.value)
    assert want in str(e.value) and got in str(e.value)


def test_tinker_run_checks_the_data_before_creating_a_service(tmp_path, monkeypatch, capsys):
    """An injected loader that changes one frozen example makes tinker-run refuse:
    non-zero exit, the task name and both hashes on stderr, no service, no row.
    --dry-run runs the same check."""
    from types import SimpleNamespace

    from clharness import cli
    from clharness.tinker import runner
    from clharness.tinker.fake import FakeTokenizer

    loader = _fake_load_split(800, 60)
    mpath = tmp_path / "m.json"
    m = td.build_manifest(out=mpath, n_train=100, n_eval=20, load_split=loader)
    calls = []
    monkeypatch.setattr(runner, "real_service", lambda: calls.append("service"))
    monkeypatch.setattr(cli, "_estimate_encode",
                        lambda: calls.append("encode") or (FakeTokenizer().encode, False))
    args = SimpleNamespace(
        arm="naive", seed=0, epochs=2, smoke=True, dry_run=False, budget=70.0,
        rows=str(tmp_path / "rows.jsonl"), items_dir=str(tmp_path / "items"),
        manifest=str(mpath), model="m", rank=8, lr=1e-4, batch_size=4, max_len=64,
        replay=200, eval_window=4)

    bad = _fake_load_split(800, 60)  # same data as `loader`, then one example moves
    def moved(task, split):
        ds = bad(task, split)
        if task.name == "mnli" and split == task.train_split:
            i = m["tasks"]["mnli"]["train_indices"][0]
            ds.rows[i] = {**ds.rows[i], "premise": "the review changed"}
        return ds

    assert cli.cmd_tinker_run(args, load_split=moved) == 2
    assert calls == []  # the check refuses before even the tokenizer is built
    assert not (tmp_path / "rows.jsonl").exists()
    err = capsys.readouterr().err
    assert "mnli" in err and m["tasks"]["mnli"]["sha256"] in err
    args.dry_run = True
    assert cli.cmd_tinker_run(args, load_split=moved) == 2  # --dry-run checks too
    assert calls == []
    # unchanged data passes the check; dry-run stops at the estimate, still no service
    assert cli.cmd_tinker_run(args, load_split=loader) == 0
    assert calls == ["encode"]
    assert not (tmp_path / "rows.jsonl").exists()
