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
