"""
Whether the shelf lasts until the next delivery — which the position cannot say.

Everything in the projection compared one total against another. `should_be` is a
quantity, `effective_position` is a quantity, and an order committed for four months'
time counts in that position exactly as much as stock already on the rack. Nothing
asked *when*.

The case: a renumbered part with 37 units on the shelf against 14,398 a month, 79,500
on the way, 60,000 of it already past its committed date. On paper the position sits
24,939 over the reorder point, so the run called it excess and advised deferring the
delivery. Two hours of cover, and the advice was to slow the supply down.

The answer is not to distrust the position — it is right about the quantity — but to
ask the question it cannot answer. Two ways to run dry, and to a planner they are the
same morning: the shelf empties before anything lands, or what should have landed
already has not and there is nothing left to sell while it is chased. The second is the
more dangerous, because a late order goes on counting as inbound supply for exactly as
long as it stays late.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.inventory_projector import InventoryProjector
from inventory_planning.analytics.purchase_recommender import PurchaseRecommender
from inventory_planning.readers.open_po_reader import OpenPOReader

CONFIG_DIR = Path(__file__).parents[1] / "config"
AS_OF = pd.Timestamp("2026-08-02")


def _open_po_lines(rows):
    """Line-level open POs, as the reader receives them."""
    return pd.DataFrame(
        [{"sku": "SKU-A", "open_qty": q, "committed_delivery": pd.Timestamp(d),
          "location_id": "DC-01"} for q, d in rows]
    )


def _schedule(rows, as_of=AS_OF, horizon_days=30):
    return OpenPOReader(CONFIG_DIR).inbound_schedule(
        _open_po_lines(rows), as_of=as_of, horizon_days=horizon_days)


def _projection(on_hand, po_rows, demand=14397.6, safety_stock=29786.3, dlt=24811.8,
                as_of=AS_OF):
    """The projection frame. Kept as a frame, not a transposed row: `Series.to_frame().T`
    collapses every column to object dtype, and on pandas 2 the recommender's `.round()`
    then raises — a defect in the test, not in what it is testing."""
    schedule = _schedule(po_rows, as_of=as_of)
    total = float(schedule["total_open_po_qty"].iloc[0])
    stock = pd.DataFrame({
        "sku": ["SKU-A"], "qty_on_hand": [float(on_hand)], "qty_in_transit_adj": [0.0],
        "total_open_po_qty": [total], "effective_position": [on_hand + total],
    })
    ss = pd.DataFrame({
        "sku": ["SKU-A"], "location_id": ["DC-01"], "stocking_class": ["stocking-high"],
        "service_level": [0.95], "demand_mean_rolling": [demand],
        "wma_lead_time_days": [51.7], "safety_stock": [safety_stock],
        "demand_during_lt": [dlt],
    })
    return InventoryProjector(CONFIG_DIR).project(ss, stock, schedule)


def _project(*args, **kwargs):
    return _projection(*args, **kwargs).iloc[0]


# ── The inbound schedule keeps the dates ─────────────────────────────────────


class TestInboundIsPhasedNotJustTotalled:

    def test_the_quantity_is_split_by_when_it_is_due(self):
        row = _schedule([(60000, "2026-07-20"),    # past due
                         (19500, "2026-08-19"),    # inside the horizon
                         (10000, "2026-11-01")]).iloc[0]   # beyond it
        assert row["inbound_past_due_qty"] == 60000
        assert row["inbound_due_qty"] == 19500
        assert row["inbound_beyond_qty"] == 10000
        assert row["total_open_po_qty"] == 89500

    def test_the_next_arrival_ignores_what_is_already_late(self):
        """A date that has passed is not a date to plan the next delivery from."""
        row = _schedule([(60000, "2026-07-20"), (19500, "2026-08-19")]).iloc[0]
        assert row["next_arrival"] == pd.Timestamp("2026-08-19")
        assert row["days_to_next_arrival"] == 17

    def test_an_undated_line_is_not_near_term_supply(self):
        lines = _open_po_lines([(5000, "2026-08-10")])
        lines.loc[len(lines)] = {"sku": "SKU-A", "open_qty": 7000.0,
                                 "committed_delivery": pd.NaT, "location_id": "DC-01"}
        row = OpenPOReader(CONFIG_DIR).inbound_schedule(lines, as_of=AS_OF).iloc[0]
        assert row["inbound_due_qty"] == 5000
        assert row["inbound_beyond_qty"] == 7000

    def test_without_an_as_of_nothing_is_declared_late(self):
        """No boundary to split on, so the reading is the one the pipeline always had."""
        row = _schedule([(60000, "2026-07-20")], as_of=None).iloc[0]
        assert row["inbound_past_due_qty"] == 0
        assert row["inbound_due_qty"] == 60000


# ── The gap, and what it overrules ───────────────────────────────────────────


class TestAnEmptyShelfOutranksAFullPosition:

    def test_the_reported_case(self):
        """37 units, 14,398 a month, 60,000 past due — the position is not the answer."""
        row = _project(37.0, [(15000, "2026-07-01"), (15000, "2026-07-20"),
                              (15000, "2026-07-20"), (15000, "2026-07-20"),
                              (4500, "2026-08-06"), (15000, "2026-08-19")])
        assert row["surplus_deficit"] > 0          # the position really is over
        assert row["inventory_status"] == "EXCESS"  # and that reading is not wrong
        assert row["inbound_past_due_qty"] == 60000
        assert row["on_hand_cover_days"] < 1
        assert row["supply_gap"]

    def test_nothing_is_deferred_out_of_an_empty_shelf(self):
        row = _project(37.0, [(60000, "2026-07-20"), (19500, "2026-08-19")])
        assert not row["pushout_candidate"]
        assert row["pushout_open_po_qty"] == 0

    def test_the_action_is_to_chase_the_order_not_to_place_one(self):
        proj = _projection(37.0, [(60000, "2026-07-20"), (19500, "2026-08-19")])
        recs = PurchaseRecommender().recommend(
            proj, None,
            pd.DataFrame({"sku": ["SKU-A"], "backlog_qty": [0.0],
                          "backlog_due_qty": [0.0], "backlog_past_due_qty": [0.0]}),
            pd.DataFrame({"sku": ["SKU-A"], "total_open_po_qty": [79500.0]}),
        )
        assert recs["recommended_action"].iloc[0] == "EXPEDITE-INBOUND"
        assert recs["suggested_po_qty"].iloc[0] == 0

    def test_running_out_before_the_next_delivery_is_a_gap_even_with_nothing_late(self):
        """Nothing past due; the shelf simply does not reach the committed date."""
        row = _project(4000.0, [(50000, "2026-09-15")], demand=14397.6)
        assert row["inbound_past_due_qty"] == 0
        assert row["on_hand_cover_days"] < row["days_to_next_arrival"]
        assert row["supply_gap"]

    def test_a_late_order_against_a_full_shelf_is_not_an_emergency(self):
        """Lateness only matters when there is nothing to sell while it is chased."""
        row = _project(200000.0, [(60000, "2026-07-20")], demand=14397.6)
        assert row["inbound_past_due_qty"] == 60000
        assert row["on_hand_cover_days"] > 30
        assert not row["supply_gap"]

    def test_a_shelf_that_reaches_the_delivery_is_not_a_gap(self):
        row = _project(30000.0, [(50000, "2026-08-19")], demand=14397.6)
        assert row["on_hand_cover_days"] > row["days_to_next_arrival"]
        assert not row["supply_gap"]

    def test_a_gap_with_nothing_on_order_is_a_purchase_not_an_expedite(self):
        """There is no delivery to chase, so the answer is to raise one."""
        stock = pd.DataFrame({
            "sku": ["SKU-A"], "qty_on_hand": [10.0], "qty_in_transit_adj": [0.0],
            "total_open_po_qty": [0.0], "effective_position": [10.0],
        })
        ss = pd.DataFrame({
            "sku": ["SKU-A"], "location_id": ["DC-01"], "stocking_class": ["stocking-high"],
            "service_level": [0.95], "demand_mean_rolling": [14397.6],
            "wma_lead_time_days": [51.7], "safety_stock": [29786.3],
            "demand_during_lt": [24811.8],
        })
        proj = InventoryProjector(CONFIG_DIR).project(ss, stock, None)
        recs = PurchaseRecommender().recommend(
            proj, None,
            pd.DataFrame({"sku": ["SKU-A"], "backlog_qty": [0.0],
                          "backlog_due_qty": [0.0], "backlog_past_due_qty": [0.0]}),
            None)
        assert recs["recommended_action"].iloc[0] == "PURCHASE-REQUEST"


class TestSeverityIsReported:
    """
    A boolean cannot be triaged. Running out four hours early and running out for six
    weeks are the same flag and very different mornings.
    """

    def test_the_days_the_shelf_is_bare_are_counted(self):
        row = _project(4000.0, [(50000, "2026-09-15")], demand=14397.6)
        expected = row["days_to_next_arrival"] - row["on_hand_cover_days"]
        assert row["supply_gap_days"] == pytest.approx(expected, abs=0.1)

    def test_a_marginal_gap_ranks_below_a_severe_one(self):
        severe = _project(37.0, [(60000, "2026-07-20"), (19500, "2026-09-30")])
        marginal = _project(13800.0, [(50000, "2026-09-01")], demand=14397.6)
        assert severe["supply_gap"] and marginal["supply_gap"]
        assert severe["supply_gap_days"] > marginal["supply_gap_days"]

    def test_no_gap_means_no_days(self):
        row = _project(200000.0, [(60000, "2026-07-20")], demand=14397.6)
        assert row["supply_gap_days"] == 0
