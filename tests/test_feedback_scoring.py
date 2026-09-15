"""
Closing the feedback loop: what a run recommended, against what then happened.

INTERFACE.md §7 item 5 says decisions belong in the store and that feedback learning is
the thing to build that for. Looking at it first turned up that **feedback learning has
never run** — 2,430 snapshots, 63 MB, none ever scored — because `record_actuals` asked
a person to assemble two frames by hand a month later, and the store already holds both.

So these pin the rewiring, and three properties that are easy to lose:

**The decision record is never written to.** Actuals are read at scoring time. The old
collector wrote them into the snapshot, which makes a decision mutable — and a record
that can be edited afterwards cannot answer what was decided at the time.

**The scoring period comes from `as_of`, not `planning_month`.** They are different
things and the snapshot carries both. Getting it wrong compares a forecast for one month
against demand in another and reports a confident number.

**A batch that does not cover the period is refused, not scored as zeroes.** Every SKU
"selling nothing" is what an out-of-period extract looks like, and it is indistinguishable
from a catastrophic forecast miss unless something refuses.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

pytest.importorskip("pyarrow", reason="the store writes parquet")

import pandas as pd                                                   # noqa: E402

from inventory_planning.feedback.actuals import (                     # noqa: E402
    ActualsUnavailable, actuals_for, candidates, scored_snapshot, scoring_period,
)
from inventory_planning.store.fact_store import FactStore             # noqa: E402


@pytest.fixture
def store(tmp_path):
    return FactStore(tmp_path / "store")


@pytest.fixture
def snapshot():
    """A decision record shaped as `SnapshotSaver` writes one."""
    return {
        "run_at": "2026-08-01T09:00:00",
        "planning_month": "2026-08",      # when the run executed
        "as_of": "2026-07-31",            # the newest date in its data
        "sku_count": 2,
        "skus": {
            "A-1": {"forecast_next_period": 100.0, "suggested_po_qty": 120.0,
                    "available_supply": 90.0, "safety_stock": 20.0,
                    "stocking_class": "stocking-high", "demand_pattern": "smooth"},
            "A-2": {"forecast_next_period": 50.0, "suggested_po_qty": 0.0,
                    "available_supply": 80.0, "safety_stock": 10.0,
                    "stocking_class": "stocking-med", "demand_pattern": "smooth"},
        },
    }


def _sales(store, rows):
    frame = pd.DataFrame(rows)
    return store.write_batch("sales_history", frame, valid_time="2026-08-31")


def _august(store):
    """An extract covering the month the plan was made for."""
    return _sales(store, [
        {"sku": "A-1", "demand_date": "2026-08-04", "qty": 60.0, "so_number": "1"},
        {"sku": "A-1", "demand_date": "2026-08-19", "qty": 55.0, "so_number": "2"},
        {"sku": "A-2", "demand_date": "2026-08-11", "qty": 40.0, "so_number": "3"},
        # Outside the period on purpose — it must not be counted.
        {"sku": "A-1", "demand_date": "2026-09-02", "qty": 999.0, "so_number": "4"},
    ])


class TestTheScoringPeriodComesFromAsOf:
    """
    `planning_month` is the wall clock when the run executed; `as_of` is the newest date
    in the data it anchored to, and `forecast_next_period` is the forecast for the period
    after it. Scoring by `planning_month` would compare a 2024-08 forecast against 2026-09
    demand — the shape of confident wrong number this pipeline exists to refuse.
    """

    def test_it_is_the_month_after_as_of(self, snapshot):
        start, end = scoring_period(snapshot)
        assert (start.year, start.month) == (2026, 8)
        assert (end.year, end.month) == (2026, 9)

    def test_it_ignores_planning_month(self, snapshot):
        moved = dict(snapshot, planning_month="2027-01")
        assert scoring_period(moved) == scoring_period(snapshot)

    def test_a_snapshot_without_as_of_is_refused_rather_than_guessed(self, snapshot):
        """
        The older snapshots have no `as_of`. Falling back to `planning_month` would
        score them against the wrong months while looking like it worked.
        """
        with pytest.raises(ActualsUnavailable, match="which period its forecast was for"):
            scoring_period({k: v for k, v in snapshot.items() if k != "as_of"})


class TestTheDecisionRecordIsNeverWrittenTo:

    def test_scoring_leaves_the_snapshot_exactly_as_it_was(self, store, snapshot):
        batch = _august(store)
        before = json.dumps(snapshot, sort_keys=True)
        actuals, _ = actuals_for(snapshot, store, batch.batch_id)
        assert json.dumps(snapshot, sort_keys=True) == before
        assert "actuals" not in snapshot

    def test_the_actuals_are_attached_to_a_copy(self, store, snapshot):
        batch = _august(store)
        actuals, _ = actuals_for(snapshot, store, batch.batch_id)
        scored = scored_snapshot(snapshot, actuals)
        assert scored["actuals"] == actuals
        assert "actuals" not in snapshot
        assert scored["skus"] is snapshot["skus"]     # a copy of the mapping, not a deep one


class TestWhatTheActualsAre:

    def test_demand_is_summed_within_the_period_only(self, store, snapshot):
        batch = _august(store)
        actuals, _ = actuals_for(snapshot, store, batch.batch_id)
        assert actuals["A-1"]["actual_demand"] == 115.0      # 60 + 55, not the 999
        assert actuals["A-2"]["actual_demand"] == 40.0

    def test_a_sku_the_extract_does_not_mention_is_none_not_zero(self, store, snapshot):
        """
        `loss.py` skips a SKU with no actual demand. A zero would score it as a total
        forecast miss instead — and "not in this extract" is not "sold nothing".
        """
        batch = _sales(store, [{"sku": "A-1", "demand_date": "2026-08-04", "qty": 10.0,
                                "so_number": "1"}])
        actuals, _ = actuals_for(snapshot, store, batch.batch_id)
        assert actuals["A-1"]["actual_demand"] == 10.0
        assert actuals["A-2"]["actual_demand"] is None

    def test_the_source_records_which_batch_and_which_period(self, store, snapshot):
        """So the number can be reproduced, the way a run records the facts it planned on."""
        batch = _august(store)
        _, source = actuals_for(snapshot, store, batch.batch_id)
        assert source.sales_batch == batch.batch_id
        assert source.period_start.startswith("2026-08")
        assert source.period_end.startswith("2026-09")
        assert any("2 planned SKUs had actual demand" in n for n in source.notes)

    def test_on_hand_comes_from_a_named_inventory_batch(self, store, snapshot):
        sales = _august(store)
        inventory = store.write_batch("inventory", pd.DataFrame([
            {"sku": "A-1", "location_id": "01", "qty_on_hand": 25.0},
            {"sku": "A-2", "location_id": "01", "qty_on_hand": 5.0},
        ]), valid_time="2026-08-31")
        actuals, source = actuals_for(snapshot, store, sales.batch_id,
                                      inventory.batch_id)
        assert actuals["A-1"]["actual_eom_inv"] == 25.0
        assert source.inventory_batch == inventory.batch_id

    def test_without_an_inventory_batch_it_says_what_is_missing(self, store, snapshot):
        batch = _august(store)
        _, source = actuals_for(snapshot, store, batch.batch_id)
        missing = [n for n in source.notes if "days of supply" in n]
        assert missing
        assert "forecast half of the score is unaffected" in missing[0]


class TestItRefusesRatherThanScoringZeroes:

    def test_an_extract_that_does_not_cover_the_period(self, store, snapshot):
        """
        Every SKU selling nothing is what an out-of-period extract looks like, and it is
        indistinguishable from a catastrophic forecast miss unless something refuses.
        """
        july = _sales(store, [{"sku": "A-1", "demand_date": "2026-07-04", "qty": 80.0,
                               "so_number": "1"}])
        with pytest.raises(ActualsUnavailable, match="no rows dated in 2026-08"):
            actuals_for(snapshot, store, july.batch_id)

    def test_an_extract_keyed_on_other_numbers(self, store, snapshot):
        """
        None of the planned SKUs present is usually two numbering systems, not a month
        with no sales — which is the failure the identity layer exists for.
        """
        other = _sales(store, [{"sku": "MS-FAC2513-0", "demand_date": "2026-08-04",
                                "qty": 80.0, "so_number": "1"}])
        with pytest.raises(ActualsUnavailable, match="different numbering systems"):
            actuals_for(snapshot, store, other.batch_id)

    def test_a_batch_that_is_not_there(self, store, snapshot):
        with pytest.raises(ActualsUnavailable, match="no sales_history batch"):
            actuals_for(snapshot, store, "not-a-batch")

    def test_a_snapshot_that_recommends_nothing(self, store, snapshot):
        batch = _august(store)
        with pytest.raises(ActualsUnavailable, match="recommends nothing"):
            actuals_for(dict(snapshot, skus={}), store, batch.batch_id)


class TestItScoresEndToEnd:
    """The loop, closed — which nothing in this repository had ever done."""

    def test_the_scorer_reads_an_in_memory_snapshot(self, store, snapshot, tmp_path):
        from inventory_planning.feedback.loss import LossCalculator

        batch = _august(store)
        actuals, _ = actuals_for(snapshot, store, batch.batch_id)

        history = tmp_path / "history" / "2026-08"
        history.mkdir(parents=True)
        path = history / "snapshot_20260801_090000-abc123.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")

        result = LossCalculator(path, snapshot=scored_snapshot(snapshot, actuals)
                                ).compute()
        assert result is not None
        # And the file on disk still holds no actuals.
        assert "actuals" not in json.loads(path.read_text(encoding="utf-8"))

    def test_the_forecast_error_is_the_plan_against_what_happened(self, store, snapshot,
                                                                 tmp_path):
        from inventory_planning.feedback.loss import LossCalculator

        batch = _august(store)
        actuals, _ = actuals_for(snapshot, store, batch.batch_id)
        history = tmp_path / "history" / "2026-08"
        history.mkdir(parents=True)
        path = history / "snapshot_x.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")

        calculator = LossCalculator(path, snapshot=scored_snapshot(snapshot, actuals))
        detail = calculator._compute_monthly_loss(calculator.snapshot)
        by_sku = detail.set_index("sku")
        # A-1 was forecast at 100 and sold 115 — under-forecast by 15.
        assert by_sku.loc["A-1", "forecast_error"] == -15.0
        # A-2 was forecast at 50 and sold 40 — over by 10.
        assert by_sku.loc["A-2", "forecast_error"] == 10.0


class TestListingTheCandidates:

    def test_the_largest_batches_come_first(self, store):
        """
        A store developed against is mostly the same sample file re-imported by every
        run, so load order buries the real extract under hundreds of them.
        """
        for _ in range(3):
            _sales(store, [{"sku": "A-1", "demand_date": "2026-08-01", "qty": 1.0,
                            "so_number": "s"}])
        big = _sales(store, [{"sku": f"A-{i}", "demand_date": "2026-08-01",
                              "qty": 1.0, "so_number": str(i)} for i in range(40)])

        listed = candidates(store, "sales_history", limit=4)
        assert listed[0]["batch_id"] == big.batch_id
        assert listed[0]["rows"] == 40
