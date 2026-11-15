"""C5 arms: naive stages, replay-by-replacement token matching, joint totalling naive."""

from clharness.tinker.arms import build_schedule
from clharness.tinker.data import TASK_ORDER


def _stage_items(stage):
    return [it for b in stage.batches for it in b]


def test_naive_stages_train_only_their_task():
    s = build_schedule("naive", n_train=500, epochs=2, batch_size=32, seed=0)
    assert len(s.stages) == 3
    for t, stage in enumerate(s.stages):
        items = _stage_items(stage)
        assert len(items) == 500 * 2  # 2 epochs
        assert {it.task for it in items} == {TASK_ORDER[t]}
        assert not any(it.replay for it in items)
        # each epoch covers every train index exactly once
        assert sorted(it.index for it in items) == sorted(list(range(500)) * 2)


def test_replay_is_token_matched_to_naive_per_stage():
    s = build_schedule("replay-200", n_train=5000, epochs=2, batch_size=32, seed=1)
    for t, stage in enumerate(s.stages):
        items = _stage_items(stage)
        assert len(items) == 5000 * 2  # same example count as naive: replay is by replacement
        replay = [it for it in items if it.replay]
        assert len(replay) == 200 * t * 2
        new = [it for it in items if not it.replay]
        assert {it.task for it in new} == {TASK_ORDER[t]}
        assert len({it.index for it in new}) == 5000 - 200 * t  # t*200 new examples replaced
        for et in TASK_ORDER[:t]:
            assert len([it for it in replay if it.task == et]) == 200 * 2
        assert abs(stage.replay_share - 200 * t / 5000) < 1e-9


def test_replay_buffer_is_fixed_across_stages():
    s = build_schedule("replay-200", n_train=5000, epochs=1, batch_size=32, seed=3)
    buf_stage1 = sorted(it.index for it in _stage_items(s.stages[1])
                        if it.replay and it.task == "sst2")
    buf_stage2 = sorted(it.index for it in _stage_items(s.stages[2])
                        if it.replay and it.task == "sst2")
    assert buf_stage1 == buf_stage2 and len(buf_stage1) == 200


def test_joint_is_token_matched_to_the_naive_total():
    naive = build_schedule("naive", n_train=5000, epochs=2, batch_size=32, seed=0)
    joint = build_schedule("joint", n_train=5000, epochs=2, batch_size=32, seed=0)
    assert len(joint.stages) == 1
    items = _stage_items(joint.stages[0])
    assert len(items) == naive.n_items() == 3 * 5000 * 2
    for t in TASK_ORDER:
        assert sorted(it.index for it in items if it.task == t) == sorted(list(range(5000)) * 2)


def test_schedules_are_deterministic_and_seeded():
    a = build_schedule("replay-200", n_train=500, epochs=2, batch_size=32, seed=7)
    b = build_schedule("replay-200", n_train=500, epochs=2, batch_size=32, seed=7)
    c = build_schedule("replay-200", n_train=500, epochs=2, batch_size=32, seed=8)
    aa = [[(it.task, it.index, it.replay) for it in bb] for st in a.stages for bb in st.batches]
    bb2 = [[(it.task, it.index, it.replay) for it in bb] for st in b.stages for bb in st.batches]
    cc = [[(it.task, it.index, it.replay) for it in bb] for st in c.stages for bb in st.batches]
    assert aa == bb2 and aa != cc


def test_smoke_replay_is_capped():
    s = build_schedule("replay-200", n_train=50, epochs=1, batch_size=32, seed=0)
    for t, stage in enumerate(s.stages):
        assert len(_stage_items(stage)) == 50
        replay = [it for it in _stage_items(stage) if it.replay]
        assert len(replay) == 16 * t  # min(200, 50 // 3) per earlier task
