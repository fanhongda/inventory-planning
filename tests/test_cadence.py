"""
Whether the buying kept pace with the selling — the question order-size CV cannot ask.

`diagnostics.erratic` scores lot-size consistency. A planner can pass it perfectly and
still be a month behind demand every month of the year, because consistent lots placed
at the wrong times produce exactly the stockouts erratic ones do. The CV is blind to
timing, to direction, and to how many orders were spent.

The cumulative PO-minus-sales curve is not blind to any of them, and these tests hold
it to the four shapes it exists to separate — balanced, persistently behind,
persistently ahead, and swinging between the two — plus the two readings that sit
alongside it: how many orders were spent, and whether the item was critical enough to
justify spending them.

The fixtures are built month by month rather than randomly, because the whole point is
that the *sequence* carries the finding. A shuffled series with the same totals is a
different diagnosis, and a test built from summary statistics would not notice.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.policy.cadence import CadenceAnalyzer

AS_OF = date(2026, 7, 31)
MONTHS = pd.period_range(end=pd.Period(AS_OF, freq="M"), periods=12, freq="M")


def po_history(**series_by_sku) -> pd.DataFrame:
    """One PO line per non-zero month, so the order count is the cadence."""
    rows = []
    for sku, monthly in series_by_sku.items():
        for i, (month, qty) in enumerate(zip(MONTHS, monthly)):
            if qty:
                rows.append({
                    "sku": sku,
                    "po_number": f"{sku}-{i:02d}",
                    "po_date": month.to_timestamp() + pd.Timedelta(days=3),
                    "po_qty": float(qty),
                })
    return pd.DataFrame(rows)


def sales_history(**series_by_sku) -> pd.DataFrame:
    rows = []
    for sku, monthly in series_by_sku.items():
        for month, qty in zip(MONTHS, monthly):
            if qty:
                rows.append({
                    "sku": sku,
                    "so_number": f"S-{sku}-{month}",
                    "demand_date": month.to_timestamp() + pd.Timedelta(days=14),
                    "qty": float(qty),
                })
    return pd.DataFrame(rows)


def attributes(sku: str, unit_cost: float = 10.0, abc: str = "B") -> pd.DataFrame:
    return pd.DataFrame([{"sku": sku, "unit_cost": unit_cost, "abc_class": abc}])


def run(po, sales, attrs=None, review_days=30.0, **kwargs):
    return CadenceAnalyzer(**kwargs).analyze(
        po_history=po, sales_history=sales, sku_attributes=attrs,
        as_of=AS_OF, default_review_days=review_days,
    )


FLAT = [100] * 12


class TestTheShapeOfTheCurve:

    def test_matched_run_rates_on_the_cadences_own_orders_is_control(self):
        """Twelve orders of 100 against twelve months of 100. The target state."""
        res = run(po_history(A=FLAT), sales_history(A=FLAT), attributes("A"))
        row = res.frame.iloc[0]
        assert row["verdict"] == "controlled"
        assert row["closing_balance"] == pytest.approx(0.0)
        assert row["order_count"] == 12
        assert row["expected_orders"] == 12
        assert res.control_rate == 1.0

    def test_buying_persistently_behind_selling_is_chasing(self):
        res = run(po_history(A=[60] * 12), sales_history(A=FLAT), attributes("A"))
        row = res.frame.iloc[0]
        assert row["verdict"] == "chasing"
        assert row["deficit_months"] >= 10
        assert row["longest_deficit_run"] >= 10
        # 12 months at −40 a month bottoms out at −480, valued at the unit cost.
        assert row["max_deficit"] == pytest.approx(480.0)
        assert row["shortfall_value"] == pytest.approx(4800.0)

    def test_persistent_positive_bias_is_losing_control(self):
        """Nothing pulls the orders back down to the run rate."""
        res = run(po_history(A=[150] * 12), sales_history(A=FLAT), attributes("A"))
        row = res.frame.iloc[0]
        assert row["verdict"] == "losing_control"
        assert row["longest_surplus_run"] >= 3
        assert row["closing_balance"] == pytest.approx(600.0)
        assert row["imbalance_pct"] == pytest.approx(0.5)
        assert row["excess_value"] == pytest.approx(6000.0)

    def test_swinging_both_ways_is_over_correcting(self):
        """
        Filed as oscillation whichever side it happens to end on. The end of the window
        is an accident of when the extract was taken, so a curve that went deep short
        and then deep long must not be filed by its last month.
        """
        # Starved for half a year, then panic-bought past the run rate. The swing has
        # to clear a full lot in both directions — anything smaller is the ordinary
        # sawtooth of a periodic review, not over-correction.
        res = run(po_history(A=[20] * 6 + [200] * 4 + [700] * 2),
                  sales_history(A=FLAT), attributes("A"))
        row = res.frame.iloc[0]
        assert row["verdict"] == "oscillating"
        assert row["longest_deficit_run"] >= 2
        assert row["longest_surplus_run"] >= 2

    def test_behind_on_volume_with_orders_unused_is_under_ordered(self):
        """A distinct failure: the orders the review period allowed were never placed."""
        res = run(po_history(A=[300, 0, 0, 0, 300, 0, 0, 0, 0, 0, 0, 0]),
                  sales_history(A=FLAT), attributes("A"))
        row = res.frame.iloc[0]
        assert row["verdict"] == "under_ordered"
        assert row["order_count"] == 2
        assert row["expected_orders"] == 12
        assert row["closing_balance"] < 0

    def test_a_deficit_that_was_caught_up_is_not_under_ordered(self):
        """
        It was late, not under-bought, and the two have different fixes. Classifying by
        mid-window deficit alone filed items that ended square as under-ordered.
        """
        res = run(po_history(A=[0, 0, 0, 400, 0, 0, 400, 0, 0, 400, 0, 0]),
                  sales_history(A=FLAT), attributes("A"))
        row = res.frame.iloc[0]
        assert row["closing_balance"] == pytest.approx(0.0)
        assert row["verdict"] != "under_ordered"


class TestTheCadenceCount:

    def test_the_review_period_sets_what_the_cadence_allows(self):
        """A quarterly review entitles the planner to four orders, not twelve."""
        res = run(po_history(A=[0, 0, 300] * 4), sales_history(A=FLAT),
                  attributes("A"), review_days=90.0)
        row = res.frame.iloc[0]
        assert row["expected_orders"] == 4
        assert row["order_count"] == 4
        assert row["verdict"] == "controlled"

    def test_a_planner_worksheet_review_period_overrides_the_default(self):
        params = pd.DataFrame([{"sku": "A", "review_period_days": 90.0}])
        res = CadenceAnalyzer().analyze(
            po_history=po_history(A=[0, 0, 300] * 4), sales_history=sales_history(A=FLAT),
            sku_attributes=attributes("A"), parameters=params,
            as_of=AS_OF, default_review_days=30.0,
        )
        assert res.frame.iloc[0]["expected_orders"] == 4

    def test_multi_line_orders_count_once(self):
        """A PO number is one act of planning, however many lines it carries."""
        po = po_history(A=FLAT)
        doubled = pd.concat([po, po.assign(po_qty=0.0)], ignore_index=True)
        res = run(doubled, sales_history(A=FLAT), attributes("A"))
        assert res.frame.iloc[0]["order_count"] == 12

    def test_too_few_orders_to_have_a_cadence_is_not_judged(self):
        res = run(po_history(A=[1200] + [0] * 11), sales_history(A=FLAT), attributes("A"))
        assert not len(res.frame)
        assert any("too few orders" in n for n in res.notes)


class TestFrequencyHasToBeEarned:

    def test_over_ordering_a_non_critical_item_is_priced(self):
        """
        Balanced, but bought four times as often as a monthly review needs. Same result,
        three times the ordering cost — and on a B-class part it buys nothing back.
        """
        weekly = po_history(A=[25] * 12)
        # 48 separate orders across the year rather than 12.
        extra = pd.concat([
            weekly.assign(po_number=weekly["po_number"] + f"-{k}",
                          po_date=weekly["po_date"] + pd.Timedelta(days=7 * k))
            for k in range(4)
        ], ignore_index=True)
        res = run(extra, sales_history(A=FLAT), attributes("A", abc="B"),
                  order_cost=350.0)
        row = res.frame.iloc[0]

        assert row["order_count"] == 48
        assert row["expected_orders"] == 12
        assert row["verdict"] == "unearned_frequency"
        assert not row["frequency_earned"]
        assert row["avoidable_order_cost"] == pytest.approx(36 * 350.0)
        assert len(res.unearned_frequency) == 1

    def test_a_critical_item_earns_the_same_frequency_for_free(self):
        """High frequency is for critical parts. On an A item it is not an overspend."""
        weekly = po_history(A=[25] * 12)
        extra = pd.concat([
            weekly.assign(po_number=weekly["po_number"] + f"-{k}",
                          po_date=weekly["po_date"] + pd.Timedelta(days=7 * k))
            for k in range(4)
        ], ignore_index=True)
        res = run(extra, sales_history(A=FLAT), attributes("A", abc="A"))
        row = res.frame.iloc[0]

        assert row["frequency_earned"]
        assert row["avoidable_order_cost"] == 0.0
        assert not len(res.unearned_frequency)


class TestTheToleranceBand:

    def test_a_lumpy_low_volume_item_is_not_a_control_failure(self):
        """
        Ordering is lumpy. A part bought in lots of 40 cannot land its cumulative
        closer than about 20, and judging it against a band of 0.08 units filled the
        report with findings of "+1 unit long" on parts that sold two all year.
        """
        res = run(po_history(A=[40, 0, 0, 40, 0, 0, 40, 0, 0, 40, 0, 0]),
                  sales_history(A=[13] * 12), attributes("A"))
        row = res.frame.iloc[0]
        assert row["tolerance_qty"] >= 20.0
        assert row["verdict"] == "controlled"

    def test_the_band_scales_with_the_items_own_run_rate(self):
        """A 50-unit drift is a crisis at 20 a month and noise at 5,000."""
        small = run(po_history(A=[100] * 12), sales_history(A=[20] * 12), attributes("A"))
        large = run(po_history(B=[5050] * 12), sales_history(B=[5000] * 12), attributes("B"))
        assert small.frame.iloc[0]["verdict"] == "losing_control"
        assert large.frame.iloc[0]["verdict"] == "controlled"


class TestRankingAndCoverage:

    def test_findings_rank_by_money_not_by_ratio(self):
        """A wildly mistimed cheap part is not the finding an expensive one is."""
        po = pd.concat([po_history(CHEAP=[150] * 12), po_history(DEAR=[150] * 12)],
                       ignore_index=True)
        sales = pd.concat([sales_history(CHEAP=FLAT), sales_history(DEAR=FLAT)],
                          ignore_index=True)
        attrs = pd.DataFrame([
            {"sku": "CHEAP", "unit_cost": 0.10, "abc_class": "C"},
            {"sku": "DEAR", "unit_cost": 500.0, "abc_class": "A"},
        ])
        res = run(po, sales, attrs)
        assert res.frame.iloc[0]["sku"] == "DEAR"

    def test_undated_demand_is_reported_rather_than_absorbed(self):
        """
        A row with no date drops out of one side of the subtraction while the other
        keeps everything, so every SKU reads as over-bought by whatever went missing.
        """
        sales = sales_history(A=FLAT)
        sales.loc[sales.index[:6], "demand_date"] = pd.NaT
        res = run(po_history(A=FLAT), sales, attributes("A"))
        assert any("cannot be placed in a month" in n for n in res.notes)
        assert any("toward surplus" in n for n in res.notes)

    def test_missing_unit_cost_is_reported_not_read_as_no_exposure(self):
        attrs = pd.DataFrame([{"sku": "A", "abc_class": "B"}])
        res = run(po_history(A=[150] * 12), sales_history(A=FLAT), attrs)
        assert res.frame.iloc[0]["unpriced"]
        assert any("no unit cost" in n for n in res.notes)

    def test_a_sku_on_only_one_side_is_not_a_cadence_finding(self):
        """Bought but never sold is over-ordering; sold but never bought is a stockout."""
        po = pd.concat([po_history(A=FLAT), po_history(NEVERSOLD=FLAT)], ignore_index=True)
        res = run(po, sales_history(A=FLAT), attributes("A"))
        assert set(res.frame["sku"]) == {"A"}

    def test_no_demand_history_yields_a_reason_not_a_crash(self):
        res = run(po_history(A=FLAT), None, attributes("A"))
        assert not len(res.frame)
        assert any("demand" in n.lower() for n in res.notes)
        assert "cadence" in res.summary().lower()

    def test_a_precompiled_demand_series_works_when_sales_history_is_absent(self):
        pivot = pd.DataFrame({"A": [100.0] * 12}, index=MONTHS)
        res = CadenceAnalyzer().analyze(
            po_history=po_history(A=FLAT), sales_history=None, demand_pivot=pivot,
            sku_attributes=attributes("A"), as_of=AS_OF,
        )
        assert res.frame.iloc[0]["verdict"] == "controlled"
