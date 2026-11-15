"""Fixes for the open red-team findings (docs/reviews/red-team-57fa138.md), one block each."""

from __future__ import annotations

import math

import pytest

from clharness.agent import contracts as c
from clharness.agent import sim
from clharness.agent.loop import build_sim
from clharness.agent.roles import FakeClock

CAP = 700.0


# ---- RT-16: rows without a price are charged the estimate -------------------------------------


def _infra_row(spec: c.RunSpec, usd: float | None = None) -> c.ResultRow:
    cost = c.a1_cost(0, 0.0, "H100", usd) if usd is not None else None
    return c.from_a1_row(c.a1_failed_row(spec, "infra", "lost the call", cost=cost))


def test_rt16_an_unpriced_infra_row_is_charged_the_estimate(tmp_path):
    conductor = build_sim(tmp_path, 0, CAP, clock=FakeClock())
    spec = sim.make_spec(["replay"])
    before = conductor._budget().compute_usd_left
    (tmp_path / "rows.jsonl").write_text(
        c.a1_failed_row(spec, "infra", "lost the call").model_dump_json() + "\n")
    after = conductor._budget().compute_usd_left
    assert before - after == pytest.approx(conductor._estimate(spec))


@pytest.mark.parametrize("usd", [math.nan, math.inf, 0.0, -3.0])
def test_rt16_a_nan_or_negative_price_is_charged_the_estimate(tmp_path, usd):
    conductor = build_sim(tmp_path, 0, CAP, clock=FakeClock())
    spec = sim.make_spec(["replay"])
    row = _infra_row(spec)
    row = row.model_copy(update={"cost": row.cost.model_copy(update={"usd": usd})})
    assert conductor._spent([row]) == pytest.approx(conductor._estimate(spec))
    priced = _infra_row(spec, 0.42)
    assert conductor._spent([priced]) == pytest.approx(0.42)  # a real price is kept


def test_rt16_the_explore_share_counts_an_unpriced_infra_row(tmp_path):
    conductor = build_sim(tmp_path, 0, CAP, clock=FakeClock())
    spec = sim.make_spec(["replay"]).model_copy(update={"notes": "explore:claim-1"})
    assert conductor._explore_used([_infra_row(spec)]) == pytest.approx(
        conductor._estimate(spec))
