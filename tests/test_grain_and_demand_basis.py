"""
Tests for two changes that alter what the pipeline recommends.

1. Inventory grain. A SKU held in a normal location and a quality-quarantine
   location arrives as two rows. Every downstream join is on `sku` alone, so the
   two rows used to fan out into two planning rows that each matched the same open
   PO and the same backlog — producing pull-in on one and push-out on the other.

2. Demand basis. The requirement used to be forecast + safety stock + backlog. The
   forecast is fitted on shipments, and shipments come from backlog, so the two
   terms count the same demand twice. It is now forecast consumption: the larger of
   the forecast and the realizable backlog due inside the horizon.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.backlog_realization import BacklogRealizationEstimator
from inventory_planning.analytics.inventory_projector import InventoryProjector
from inventory_planning.analytics.purchase_recommender import PurchaseRecommender
from inventory_planning.readers.inventory_reader import (
    InventoryReader,
    consolidate_to_planning_grain,
)
from inventory_planning.readers.open_so_reader import OpenSOReader

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def two_location_inventory():
    """SKU-A sits in 01 (sellable) and 02 (quality quarantine); SKU-B in 01 only."""
    return pd.DataFrame({
        "sku": ["SKU-A", "SKU-A", "SKU-B"],
        "location_id": ["01", "02", "01"],
        "qty_on_hand": [20.0, 500.0, 75.0],
        "qty_in_transit": [0.0, 0.0, 10.0],
        "unit_cost": [12.0, 12.0, 4.0],
    })


@pytest.fixture
def safety_stock_frame():
    return pd.DataFrame({
        "sku": ["SKU-A", "SKU-B"],
        "location_id": ["DC-01", "DC-01"],
        "stocking_class": ["stocking-high", "stocking-high"],
        "service_level": [0.95, 0.95],
        "demand_mean_rolling": [100.0, 30.0],
        "wma_lead_time_days": [60.0, 30.0],
        "safety_stock": [50.0, 15.0],
        "demand_during_lt": [200.0, 30.0],
    })


# ── 1. Grain ─────────────────────────────────────────────────────────────────


class TestInventoryGrain:

    def test_collapses_to_one_row_per_sku(self, two_location_inventory):
        out = consolidate_to_planning_grain(two_location_inventory, "DC-01", verbose=False)
        assert len(out) == 2
        assert not out["sku"].duplicated().any()

    def test_quantities_are_summed_across_locations(self, two_location_inventory):
        out = consolidate_to_planning_grain(two_location_inventory, "DC-01", verbose=False)
        assert out.loc[out["sku"] == "SKU-A", "qty_on_hand"].iloc[0] == 520.0

    def test_unit_cost_is_not_summed(self, two_location_inventory):
        """Cost belongs to the SKU, not the bin. Summing it doubles every valuation."""
        out = consolidate_to_planning_grain(two_location_inventory, "DC-01", verbose=False)
        assert out.loc[out["sku"] == "SKU-A", "unit_cost"].iloc[0] == 12.0

    def test_merged_locations_stay_visible(self, two_location_inventory):
        out = consolidate_to_planning_grain(two_location_inventory, "DC-01", verbose=False)
        assert out.loc[out["sku"] == "SKU-A", "stock_locations"].iloc[0] == "01,02"

    def test_consolidating_twice_is_the_same_as_once(self, two_location_inventory):
        """
        The orchestrator consolidates again in the policy and KPI phases, on a frame the
        reader may already have consolidated. A stale `stock_locations` column collided
        with the freshly computed one on merge.
        """
        once = consolidate_to_planning_grain(two_location_inventory, "DC-01", verbose=False)
        # Re-split it the way a caller passing the raw export a second time would.
        again = consolidate_to_planning_grain(
            pd.concat([once, once.assign(qty_on_hand=0.0)], ignore_index=True),
            "DC-01", verbose=False,
        )
        assert list(again["stock_locations"]) == list(once["stock_locations"])
        assert again["qty_on_hand"].sum() == once["qty_on_hand"].sum()

    def test_blank_provenance_does_not_mask_real_location_codes(self, two_location_inventory):
        """
        A frame loaded from an export with no location column carries an empty
        `stock_locations`. It must not then take precedence over real codes added
        later, or the multi-location warning goes silent on exactly the case it exists
        to catch.
        """
        df = two_location_inventory.assign(stock_locations="")
        out = consolidate_to_planning_grain(df, "DC-01", verbose=False)
        assert out.loc[out["sku"] == "SKU-A", "stock_locations"].iloc[0] == "01,02"

    def test_single_location_frame_is_untouched(self):
        df = pd.DataFrame({"sku": ["X"], "location_id": ["01"], "qty_on_hand": [5.0]})
        out = consolidate_to_planning_grain(df, "DC-01", verbose=False)
        assert len(out) == 1
        assert out["qty_on_hand"].iloc[0] == 5.0

    def test_open_po_is_counted_once_not_once_per_location(self, two_location_inventory):
        """The fan-out bug: the same 100-unit PO landing on both of a SKU's rows."""
        open_po = pd.DataFrame({"sku": ["SKU-A"], "total_open_po_qty": [100.0]})
        eff = InventoryReader(CONFIG_DIR).effective_inventory(two_location_inventory, open_po)

        rows = eff[eff["sku"] == "SKU-A"]
        assert len(rows) == 1
        assert rows["total_open_po_qty"].iloc[0] == 100.0
        assert rows["effective_position"].iloc[0] == 620.0

    def test_one_sku_gets_one_recommendation(self, two_location_inventory, safety_stock_frame):
        """
        End of the chain: one SKU, one action. Before the fix this SKU appeared twice
        — PURCHASE-REQUEST on the 20-unit row and PUSH-OUT-OPEN-PO on the 500-unit row.
        """
        open_po = pd.DataFrame({"sku": ["SKU-A"], "total_open_po_qty": [100.0]})
        eff = InventoryReader(CONFIG_DIR).effective_inventory(two_location_inventory, open_po)
        projection = InventoryProjector(CONFIG_DIR).project(safety_stock_frame, eff, open_po)

        backlog = pd.DataFrame({"sku": ["SKU-A"], "backlog_qty": [80.0],
                                "backlog_due_qty": [80.0], "backlog_past_due_qty": [0.0]})
        recs = PurchaseRecommender().recommend(projection, None, backlog, open_po)

        assert (recs["sku"] == "SKU-A").sum() == 1
        assert not recs["sku"].duplicated().any()

    def test_projector_refuses_a_frame_it_would_fan_out(self, safety_stock_frame):
        duplicated = pd.DataFrame({
            "sku": ["SKU-A", "SKU-A"],
            "qty_on_hand": [20.0, 500.0],
            "qty_in_transit_adj": [0.0, 0.0],
            "total_open_po_qty": [100.0, 100.0],
            "effective_position": [120.0, 600.0],
        })
        with pytest.raises(ValueError, match="more than one row"):
            InventoryProjector(CONFIG_DIR).project(safety_stock_frame, duplicated, None)


# ── 2. Backlog realization ───────────────────────────────────────────────────


class TestBacklogRealization:

    @staticmethod
    def _open_so(rows):
        return pd.DataFrame(rows)

    def test_uncollected_stock_lowers_the_rate(self):
        """Past the request date with the goods on the shelf: the customer did not pull."""
        open_so = self._open_so([
            # 600 of 1000 open units are past due with stock available
            {"sku": "S1", "open_qty": 300.0, "customer_request_date": "2024-01-01"},
            {"sku": "S1", "open_qty": 300.0, "customer_request_date": "2024-01-05"},
            {"sku": "S1", "open_qty": 400.0, "customer_request_date": "2024-09-01"},
        ])
        inventory = pd.DataFrame({"sku": ["S1"], "qty_on_hand": [600.0]})

        result = BacklogRealizationEstimator().estimate(
            open_so=open_so, inventory=inventory, as_of=date(2024, 6, 1)
        )
        assert result.measured
        assert result.global_rate == pytest.approx(0.4, abs=0.01)

    def test_no_inventory_means_no_measurement(self):
        open_so = self._open_so([
            {"sku": "S1", "open_qty": 100.0, "customer_request_date": "2024-01-01"},
        ])
        result = BacklogRealizationEstimator().estimate(open_so=open_so, inventory=None)
        assert not result.measured
        assert result.rate_for("S1") == 1.0

    def test_rate_never_falls_below_the_floor(self):
        """A collection problem must not zero out purchasing entirely."""
        open_so = self._open_so([
            {"sku": "S1", "open_qty": 100.0, "customer_request_date": "2024-01-01"},
        ])
        inventory = pd.DataFrame({"sku": ["S1"], "qty_on_hand": [100.0]})
        result = BacklogRealizationEstimator(floor=0.25).estimate(
            open_so=open_so, inventory=inventory, as_of=date(2024, 6, 1)
        )
        assert result.rate_for("S1") >= 0.25

    def test_thin_evidence_is_pulled_toward_the_portfolio(self):
        """One line should not produce a confident per-SKU rate."""
        open_so = self._open_so(
            # S1: many lines, all collected on time → high raw rate
            [{"sku": "S1", "open_qty": 100.0, "customer_request_date": "2024-09-01"}
             for _ in range(10)]
            # S2: a single uncollected line → raw rate 0
            + [{"sku": "S2", "open_qty": 100.0, "customer_request_date": "2024-01-01"}]
        )
        inventory = pd.DataFrame({"sku": ["S1", "S2"], "qty_on_hand": [0.0, 100.0]})
        result = BacklogRealizationEstimator(floor=0.0).estimate(
            open_so=open_so, inventory=inventory, as_of=date(2024, 6, 1)
        )
        s2 = result.per_sku.set_index("sku").loc["S2"]
        assert s2["raw_rate"] == 0.0
        assert s2["realization_rate"] > 0.0, "one line must not carry a 0% verdict alone"


# ── 3. Demand basis ──────────────────────────────────────────────────────────


@pytest.fixture
def projection():
    return pd.DataFrame({
        "sku": ["S1"],
        "location_id": ["DC-01"],
        "stocking_class": ["stocking-high"],
        "demand_mean_rolling": [100.0],
        "safety_stock": [50.0],
        "effective_position": [0.0],
        "inventory_status": ["SHORTAGE-RISK"],
        "pushout_candidate": [False],
        "pushout_open_po_qty": [0.0],
    })


@pytest.fixture
def forecast():
    return pd.DataFrame({
        "sku": ["S1"],
        "forecast_next_period": [100.0],
        "forecast_avg_monthly": [100.0],
    })


@pytest.fixture
def backlog():
    return pd.DataFrame({
        "sku": ["S1"],
        "backlog_qty": [400.0],
        "backlog_due_qty": [400.0],
        "backlog_past_due_qty": [0.0],
    })


class TestDemandBasis:

    def test_forecast_and_backlog_are_not_added(self, projection, forecast, backlog):
        recs = PurchaseRecommender("forecast_consumption").recommend(
            projection, forecast, backlog, None
        )
        row = recs.iloc[0]
        # Additive would be 100 + 400 + 50 = 550.
        assert row["gross_requirement"] == pytest.approx(450.0)
        assert row["demand_driver"] == "backlog"

    def test_forecast_wins_when_it_is_the_larger_estimate(self, projection, backlog):
        forecast = pd.DataFrame({"sku": ["S1"], "forecast_next_period": [900.0],
                                 "forecast_avg_monthly": [900.0]})
        recs = PurchaseRecommender("forecast_consumption").recommend(
            projection, forecast, backlog, None
        )
        assert recs.iloc[0]["demand_driver"] == "forecast"
        assert recs.iloc[0]["gross_requirement"] == pytest.approx(950.0)

    def test_realization_discounts_the_backlog(self, projection, forecast, backlog):
        class _HalfRate:
            global_rate = 0.5

            def apply(self, skus):
                return pd.Series(0.5, index=skus.index)

        recs = PurchaseRecommender("forecast_consumption").recommend(
            projection, forecast, backlog, None, realization=_HalfRate()
        )
        row = recs.iloc[0]
        assert row["firm_demand_qty"] == pytest.approx(200.0)
        assert row["gross_requirement"] == pytest.approx(250.0)

    def test_backlog_beyond_the_horizon_is_not_this_cycle_s_requirement(
        self, projection, forecast
    ):
        far_out = pd.DataFrame({"sku": ["S1"], "backlog_qty": [400.0],
                                "backlog_due_qty": [0.0], "backlog_past_due_qty": [0.0]})
        recs = PurchaseRecommender("forecast_consumption").recommend(
            projection, forecast, far_out, None
        )
        assert recs.iloc[0]["gross_requirement"] == pytest.approx(150.0)

    def test_legacy_additive_basis_is_still_reachable(self, projection, forecast, backlog):
        recs = PurchaseRecommender("forecast_plus_backlog").recommend(
            projection, forecast, backlog, None
        )
        assert recs.iloc[0]["gross_requirement"] == pytest.approx(550.0)

    def test_unknown_basis_is_rejected(self):
        with pytest.raises(ValueError, match="demand_basis"):
            PurchaseRecommender("whatever")

    def test_order_for_backlog_carries_a_quantity(self, forecast, backlog):
        """A recommendation to order that suggests zero units is not a recommendation."""
        non_stocking = pd.DataFrame({
            "sku": ["S1"], "location_id": ["DC-01"],
            "stocking_class": ["non-stocking"], "demand_mean_rolling": [0.0],
            "safety_stock": [0.0], "effective_position": [0.0],
            "inventory_status": ["non-stocking"], "pushout_candidate": [False],
            "pushout_open_po_qty": [0.0],
        })
        recs = PurchaseRecommender().recommend(non_stocking, forecast, backlog, None)
        row = recs.iloc[0]
        assert row["recommended_action"] == "ORDER-FOR-BACKLOG"
        assert row["suggested_po_qty"] == pytest.approx(400.0)


class TestBacklogHorizonSplit:

    def test_due_split_scopes_the_order_book(self):
        open_so = pd.DataFrame({
            "sku": ["S1", "S1", "S1"],
            "open_qty": [10.0, 20.0, 30.0],
            "customer_request_date": pd.to_datetime(
                ["2024-05-01", "2024-06-10", "2024-12-01"]
            ),
            "location_id": ["DC-01"] * 3,
        })
        out = OpenSOReader(CONFIG_DIR).backlog_summary(
            open_so, as_of=date(2024, 6, 1), horizon_days=30
        )
        row = out.iloc[0]
        assert row["backlog_qty"] == 60.0
        assert row["backlog_due_qty"] == 30.0     # past due 10 + due 2024-06-10
        assert row["backlog_past_due_qty"] == 10.0

    def test_undated_lines_count_as_due(self):
        open_so = pd.DataFrame({
            "sku": ["S1"], "open_qty": [10.0],
            "customer_request_date": [pd.NaT], "location_id": ["DC-01"],
        })
        out = OpenSOReader(CONFIG_DIR).backlog_summary(open_so, as_of=date(2024, 6, 1))
        assert out.iloc[0]["backlog_due_qty"] == 10.0
