"""
Exposure is what an order placed today cannot close — so it is measured over each
item's own lead time.

Two figures were measured over a fixed window. "Demand at risk" summed 180 days of
unserved demand, starting the clock where safety stock began and counting items held
to order; on an extract with no open POs that was half a year of demand, $6.1M against
$3.1M of stock, under a caption that said "within 30 days". The S&IOP gap ran across
the whole forecast horizon and summed each month's shortfall level into a total, so a
standing shortage counted once per month it stood.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from inventory_planning.analytics.siop import build_siop_plan
from inventory_planning.policy.diagnostics import DiagnosticsAnalyzer

AS_OF = date(2026, 11, 11)


def _risk(attributes, on_hand, safety=None, horizon=180):
    stock = pd.DataFrame({"sku": attributes["sku"], "qty_on_hand": on_hand})
    ss = pd.Series(safety, index=attributes["sku"]) if safety is not None else None
    return DiagnosticsAnalyzer(horizon_days=horizon).forward(
        attributes, stock, safety_stock=ss, as_of=AS_OF)


class TestDemandAtRiskIsInsideTheLeadTime:
    """30 units a month is 1 a day; 20 on hand is 20 days of cover."""

    def test_a_short_beyond_the_lead_time_is_a_purchase_not_a_risk(self):
        attributes = pd.DataFrame({"sku": ["A"], "demand_mean": [30.0],
                                   "unit_cost": [10.0], "lead_time_days": [14]})
        assert not len(_risk(attributes, [20.0]).stockout)

    def test_inside_the_lead_time_the_unserved_days_are_counted(self):
        attributes = pd.DataFrame({"sku": ["A"], "demand_mean": [30.0],
                                   "unit_cost": [10.0], "lead_time_days": [60]})
        row = _risk(attributes, [20.0]).stockout.iloc[0]
        assert row["days_to_stockout"] == 20
        assert row["horizon_days"] == 60
        assert row["qty_at_risk"] == pytest.approx(40.0)          # days 20..60
        assert row["value_at_risk"] == pytest.approx(400.0)

    def test_no_lead_time_falls_back_to_the_analyzer_horizon(self):
        attributes = pd.DataFrame({"sku": ["A"], "demand_mean": [30.0],
                                   "unit_cost": [10.0]})
        row = _risk(attributes, [20.0], horizon=90).stockout.iloc[0]
        assert row["horizon_days"] == 90
        assert row["qty_at_risk"] == pytest.approx(70.0)


class TestASaleIsLostAtAnEmptyShelf:
    def test_safety_stock_serves_demand_rather_than_being_written_off(self):
        """
        20 on hand, 15 of it safety stock, 30-day lead time. Measured to the safety
        level it ran out on day 5 and lost 25 days; it actually serves 20 days and
        loses 10.
        """
        attributes = pd.DataFrame({"sku": ["A"], "demand_mean": [30.0],
                                   "unit_cost": [10.0], "lead_time_days": [30]})
        row = _risk(attributes, [20.0], safety=[15.0]).stockout.iloc[0]
        assert row["days_to_stockout"] == 20
        assert row["qty_at_risk"] == pytest.approx(10.0)


class TestAnItemHeldToOrderHasNoShelfToEmpty:
    def test_non_stocking_items_are_not_projected_as_stockouts(self):
        attributes = pd.DataFrame({
            "sku": ["MTO", "MTS"], "demand_mean": [30.0, 30.0], "unit_cost": [10.0, 10.0],
            "lead_time_days": [60, 60], "stocking_class": ["non-stocking", "stocking-high"],
        })
        assert set(_risk(attributes, [0.0, 0.0]).stockout["sku"]) == {"MTS"}


# ── S&IOP ────────────────────────────────────────────────────────────────────

PERIODS = ["2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04"]


def _plan(lead_times, on_hand=150.0, monthly=100.0, safety=50.0):
    skus = list(lead_times)
    forecast = pd.DataFrame([{"sku": s, "period": p, "forecast_qty": monthly}
                             for s in skus for p in PERIODS])
    attributes = pd.DataFrame({"sku": skus, "unit_cost": [10.0] * len(skus),
                               "lead_time_days": [lead_times[s] for s in skus]})
    return build_siop_plan(
        forecast,
        inventory=pd.DataFrame({"sku": skus, "qty_on_hand": [on_hand] * len(skus)}),
        attributes=attributes,
        safety_stock=pd.DataFrame({"sku": skus, "safety_stock": [safety] * len(skus)}))


class TestTheSiopGapIsScopedToTheLeadTime:
    """150 on hand, 100 a month, floor 50: month 1 closes on the floor, then 100 lower each month."""

    def test_a_two_week_item_has_no_unrecoverable_gap(self):
        row = _plan({"FAST": 14}).per_sku_wide().iloc[0]
        assert row["lead_time_periods"] == 1
        assert row["gap qty within lead time"] == 0

    def test_a_four_month_item_carries_the_gap_its_lead_time_covers(self):
        row = _plan({"SLOW": 120}).per_sku_wide().iloc[0]
        assert row["lead_time_periods"] == 4
        assert row["gap qty within lead time"] == pytest.approx(300.0)
        assert row["gap amt within lead time"] == pytest.approx(3_000.0)

    def test_a_lead_time_one_day_past_a_month_reaches_the_second_month(self):
        assert _plan({"X": 31}).per_sku_wide().iloc[0]["lead_time_periods"] == 2

    def test_no_lead_time_counts_the_whole_horizon(self):
        """The cautious reading: unknown is treated as unrecoverable, not dropped."""
        row = _plan({"X": np.nan}).per_sku_wide().iloc[0]
        assert row["lead_time_periods"] == len(PERIODS)

    def test_the_plan_total_is_each_items_peak_inside_its_window(self):
        assert _plan({"FAST": 14, "SLOW": 120}).gap_within_lead_time == pytest.approx(3_000.0)


class TestAGapIsALevelNotAFlow:
    def test_the_lead_block_shows_the_peak_not_a_sum_of_months(self):
        """
        Short by 0, 100, 200, 300, 400, 500 across six months: the deepest shortfall is
        500 units ($5,000). Summing the months counted $15,000.
        """
        row = _plan({"SLOW": 120}).per_sku_wide().iloc[0]
        assert row["gap qty peak"] == pytest.approx(500.0)
        assert row["gap amt peak"] == pytest.approx(5_000.0)
        assert "gap amt total" not in row.index
