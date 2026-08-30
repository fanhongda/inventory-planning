"""
The forecast has to survive a round trip through a sales review.

Out: a worksheet with history, forecast, and what both are worth, segmented the way the
business is organised. Back: the numbers sales committed to. In between, a spreadsheet
opened by people who will reorder columns, leave most cells blank, and occasionally
type a reason.

The invariants that matter are all about what an override does *not* touch. It replaces
a quantity; it does not replace the forecast error, because a number set by judgement
is not evidence that demand became more predictable, and letting it reset σ would cut
safety stock at the moment the plan started resting on an opinion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.sales_plan import (
    apply_sales_plan, read_sales_plan,
)
from inventory_planning.analytics.sop import (
    SOPWorksheet, average_selling_price, country_shares, value_series,
)
from inventory_planning.orchestrator import InventoryPlanner

SAMPLE = Path(__file__).parents[1] / "sample_data"


def _sales(n=600, skus=6, with_country=False):
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        sku = f"P-{i % skus:03d}"
        rows.append({
            "sku": sku,
            "demand_date": pd.Timestamp("2024-01-15") + pd.DateOffset(months=i % 18),
            "qty": float(rng.integers(5, 50)),
            "amount": float(rng.integers(5, 50)) * (40 + (i % skus) * 10),
            **({"country": ["India", "Singapore", "Singapore"][i % 3]}
               if with_country else {}),
        })
    return pd.DataFrame(rows)


# ── Price ────────────────────────────────────────────────────────────────────


class TestWhatTheForecastIsWorth:

    def test_the_price_is_revenue_over_units_not_a_mean_of_line_prices(self):
        """
        A mean of line prices weights a one-unit sample like a thousand-unit one. What
        the business achieved is the ratio of the two totals.
        """
        df = pd.DataFrame({
            "sku": ["A", "A"],
            "demand_date": [pd.Timestamp("2025-01-01")] * 2,
            "qty": [1.0, 99.0],
            "amount": [1000.0, 990.0],   # 1000/unit on one, 10/unit on ninety-nine
        })
        asp = average_selling_price(df).frame.set_index("sku")["asp"]["A"]
        assert asp == pytest.approx(19.9), "line-price mean would have said 505"

    def test_the_basis_is_recorded_not_blended(self):
        df = _sales()
        basis = average_selling_price(df)
        assert set(basis.frame["price_basis"]) == {"realised, trailing 12m"}
        assert "trailing 12m" in basis.summary()

    def test_standard_cost_is_labelled_as_not_a_price(self):
        """A reviewer seeing this against a line knows the margin in it is zero."""
        df = pd.DataFrame({"sku": ["A"], "demand_date": [pd.Timestamp("2025-01-01")],
                           "qty": [10.0], "amount": [np.nan]})
        cost = pd.DataFrame({"sku": ["A"], "unit_cost": [12.0]})
        basis = average_selling_price(df, unit_cost=cost)
        assert basis.frame.empty or "not a price" in " ".join(basis.frame["price_basis"])

    def test_history_is_valued_as_booked_not_at_todays_price(self):
        """
        A history restated at the current price hides the thing a review looks for:
        the units held up and the money did not.
        """
        df = pd.DataFrame({
            "sku": ["A", "A"],
            "demand_date": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-15")],
            "qty": [10.0, 10.0],
            "amount": [1000.0, 500.0],
        })
        series = value_series(df)
        assert list(series["A"]) == [1000.0, 500.0]


# ── The sheet ────────────────────────────────────────────────────────────────


class TestTheWorksheet:

    def _built(self, **kwargs):
        planner = InventoryPlanner(output_dir=Path("/tmp") / "sop_unit",
                                   interactive=False)
        results = planner.run_planning(**planner.load_all(sorted(SAMPLE.glob("*.csv"))))
        return results["sop"], results

    def test_it_carries_money_beside_units(self):
        sop, _ = self._built()
        assert any(c.startswith("hist amt ") for c in sop.sheet.columns)
        assert any(c.startswith("fcst amt ") for c in sop.sheet.columns)
        assert sop.sheet["forecast_amount_total"].notna().any()

    def test_the_family_is_joined_from_master_never_derived(self):
        """
        Deriving it here would put the prefix guess in front of the one audience least
        able to spot it — which is the failure `product_dimension` exists to stop.
        """
        sop, _ = self._built()
        assert set(sop.sheet["product_family"].dropna()) <= {
            "valve", "sprinkler", "fitting", "actuator"}

    def test_a_guessed_family_is_labelled_as_guessed(self):
        """
        The run no longer stops for an unclassified item, so the sheet has to carry the
        distinction instead. A count of missing families that a reader cannot reconcile
        against the report is a count they will stop believing.
        """
        sop, _ = self._built()
        assert "product_family_source" in sop.sheet.columns
        assert set(sop.sheet["product_family_source"].dropna()) <= {"master", "inferred"}

    def test_the_forecast_provenance_travels_with_the_numbers(self):
        sop, _ = self._built()
        for column in ("model_used", "vs_naive", "price_basis"):
            assert column in sop.sheet.columns

    def test_an_unpriced_row_is_blank_and_never_zero(self):
        """A zero sums into the total and understates it without saying so."""
        ts = pd.DataFrame({"A": [10.0] * 12},
                          index=pd.period_range("2024-01", periods=12, freq="M"))
        detail = pd.DataFrame({"sku": ["A"], "period": ["2025-01"],
                               "forecast_qty": [10.0], "is_next_period": [True],
                               "model_used": ["SMA"]})
        result = SOPWorksheet().build(ts, detail, sales_df=None)
        assert result.sheet["forecast_amount_total"].isna().all()

    def test_the_country_split_sums_back_to_the_sku_total(self):
        """
        The whole argument for top-down. The sales review and the replenishment plan
        cannot disagree, because the parts were apportioned from the total.
        """
        sales = _sales(with_country=True)
        ts = (sales.assign(period=lambda d: d["demand_date"].dt.to_period("M"))
              .groupby(["period", "sku"])["qty"].sum().unstack(fill_value=0.0))
        detail = pd.DataFrame([
            {"sku": sku, "period": "2025-08", "forecast_qty": 100.0,
             "is_next_period": True, "model_used": "SMA"}
            for sku in ts.columns])

        result = SOPWorksheet().build(ts, detail, sales_df=sales)
        assert result.split_by_country
        total = result.sheet.groupby("sku")["fcst qty 2025-08"].sum()
        assert np.allclose(total.values, 100.0), "the split lost or invented demand"

    def test_the_price_is_not_apportioned_by_the_split(self):
        """A price is a rate. Multiplying it by a share is plausible and wrong."""
        sales = _sales(with_country=True)
        ts = (sales.assign(period=lambda d: d["demand_date"].dt.to_period("M"))
              .groupby(["period", "sku"])["qty"].sum().unstack(fill_value=0.0))
        detail = pd.DataFrame([{"sku": "P-000", "period": "2025-08",
                                "forecast_qty": 100.0, "is_next_period": True,
                                "model_used": "SMA"}])
        sheet = SOPWorksheet().build(ts, detail, sales_df=sales).sheet
        assert sheet["asp"].nunique() == 1

    def test_a_slow_mover_keeps_its_country(self):
        """
        An item whose last sale predates the window still has a geography. Falling back
        only when the whole window is empty stranded these on the sheet as "(not
        stated)" — and a slow mover is exactly the item a reviewer is least sure about.
        """
        sales = pd.DataFrame({
            "sku": ["FAST", "SLOW"],
            "demand_date": [pd.Timestamp("2025-06-01"), pd.Timestamp("2023-01-01")],
            "qty": [10.0, 10.0],
            "country": ["India", "Singapore"],
        })
        shares = country_shares(sales).set_index("sku")["country"]
        assert shares["SLOW"] == "Singapore"

    def test_slivers_are_folded_not_dropped(self):
        """Dropping a 0.4% country would leave the parts not summing to the total."""
        sales = pd.DataFrame({
            "sku": ["A"] * 100,
            "demand_date": [pd.Timestamp("2025-01-15")] * 100,
            "qty": [10.0] * 99 + [0.1],
            "country": ["India"] * 99 + ["Nepal"],
        })
        shares = country_shares(sales)
        assert shares["share"].sum() == pytest.approx(1.0)
        assert set(shares["country"]) == {"India"}


# ── The round trip ───────────────────────────────────────────────────────────


class TestTheReviewComesBack:

    @staticmethod
    def _detail():
        return pd.DataFrame({
            "sku": ["A", "A", "B", "B"],
            "period": ["2025-01", "2025-02", "2025-01", "2025-02"],
            "forecast_qty": [100.0, 100.0, 50.0, 50.0],
            "forecast_rmse": [12.5, 12.5, 3.0, 3.0],
            "model_used": ["ETS", "ETS", "Croston", "Croston"],
            "is_next_period": [True, False, True, False],
        })

    def _sheet(self, tmp_path, name="reviewed.xlsx", **overrides):
        frame = pd.DataFrame({
            "sku": ["A", "B"],
            "REVIEWED qty 2025-01": [overrides.get("a1", np.nan),
                                     overrides.get("b1", np.nan)],
            "REVIEWED qty 2025-02": [overrides.get("a2", np.nan),
                                     overrides.get("b2", np.nan)],
            "REVIEWED by": ["Priya", ""],
            "REVIEWED reason": ["contract signed", ""],
        })
        path = tmp_path / name
        frame.to_excel(path, index=False)
        return path

    def test_a_blank_cell_means_no_change_not_zero(self, tmp_path):
        """
        The whole file rests on this. A reviewer types into two months of six; reading
        the blanks as zeroes plans four months of no demand for everything on the sheet.
        """
        path = self._sheet(tmp_path, a1=180.0)
        plan = read_sales_plan(path)
        assert len(plan) == 1
        result = apply_sales_plan(self._detail(), plan)
        after = result.forecast_detail.set_index(["sku", "period"])["forecast_qty"]
        assert after[("A", "2025-01")] == 180.0
        assert after[("A", "2025-02")] == 100.0, "an untouched month must not move"
        assert after[("B", "2025-01")] == 50.0

    def test_the_statistical_forecast_survives_beside_the_reviewed_one(self, tmp_path):
        """Without it there is no way to score the review next quarter."""
        result = apply_sales_plan(self._detail(),
                                  read_sales_plan(self._sheet(tmp_path, a1=180.0)))
        row = result.forecast_detail.set_index(["sku", "period"]).loc[("A", "2025-01")]
        assert row["forecast_qty"] == 180.0
        assert row["statistical_qty"] == 100.0
        assert row["forecast_source"] == "sales_review"

    def test_the_forecast_error_is_not_touched(self, tmp_path):
        """
        σDL is what safety stock absorbs. A sales adjustment is not evidence that
        demand became more predictable — if anything the opposite — so resetting the
        error here would cut safety stock exactly when the plan starts resting on an
        opinion.
        """
        before = self._detail()
        result = apply_sales_plan(before, read_sales_plan(
            self._sheet(tmp_path, a1=180.0, a2=200.0, b1=10.0)))
        assert list(result.forecast_detail["forecast_rmse"]) == list(
            before["forecast_rmse"])

    def test_the_model_that_was_argued_with_is_still_named(self, tmp_path):
        result = apply_sales_plan(self._detail(),
                                  read_sales_plan(self._sheet(tmp_path, a1=180.0)))
        assert list(result.forecast_detail["model_used"]) == [
            "ETS", "ETS", "Croston", "Croston"]

    def test_who_and_why_come_across(self, tmp_path):
        result = apply_sales_plan(self._detail(),
                                  read_sales_plan(self._sheet(tmp_path, a1=180.0)))
        row = result.forecast_detail.set_index(["sku", "period"]).loc[("A", "2025-01")]
        assert row["reviewed_by"] == "Priya"
        assert row["review_reason"] == "contract signed"

    def test_a_reviewed_sku_with_no_history_is_reported_not_invented(self, tmp_path):
        """
        A forecast conjured from a review alone has no error behind it, so there is
        nothing to size safety stock on. Say so rather than fabricate one.
        """
        frame = pd.DataFrame({"sku": ["Z"], "REVIEWED qty 2025-01": [500.0]})
        path = tmp_path / "r.xlsx"
        frame.to_excel(path, index=False)
        result = apply_sales_plan(self._detail(), read_sales_plan(path))
        assert result.unmatched == ["Z"]
        assert "not in the forecast" in result.summary()

    def test_a_country_split_sheet_sums_back_to_the_sku(self, tmp_path):
        """Stock is one pool, so the plan runs at SKU level whatever the sheet shows."""
        pd.DataFrame({
            "sku": ["A", "A"],
            "country": ["India", "Singapore"],
            "REVIEWED qty 2025-01": [120.0, 60.0],
        }).to_excel(tmp_path / "split.xlsx", index=False)
        plan = read_sales_plan(tmp_path / "split.xlsx")
        assert len(plan) == 1
        assert plan.adjustments["reviewed_qty"].iloc[0] == 180.0

    def test_a_sheet_without_an_item_column_is_refused(self, tmp_path):
        pd.DataFrame({"thing": ["A"], "REVIEWED qty 2025-01": [1.0]}).to_excel(
            tmp_path / "bad.xlsx", index=False)
        with pytest.raises(ValueError, match="no item column"):
            read_sales_plan(tmp_path / "bad.xlsx")

    def test_no_plan_leaves_the_forecast_alone(self):
        before = self._detail()
        result = apply_sales_plan(before, None)
        assert list(result.forecast_detail["forecast_qty"]) == list(
            before["forecast_qty"])
        assert set(result.forecast_detail["forecast_source"]) == {"statistical"}


class TestTheWholeLoop:

    def test_out_to_a_workbook_and_back_into_the_plan(self, tmp_path):
        """
        The loop the skill runs: forecast, hand it to sales, take back what they
        committed to, and plan on that.
        """
        planner = InventoryPlanner(output_dir=tmp_path / "run1", interactive=False)
        first = planner.run_planning(**planner.load_all(sorted(SAMPLE.glob("*.csv"))))

        workbook = next((tmp_path / "run1").glob("sop_worksheet_*.xlsx"))
        sheet = pd.read_excel(workbook, sheet_name="Review by SKU")
        assert "REVIEWED by" in sheet.columns

        # Sales double the first forecast month for one SKU and touch nothing else.
        period = sorted(first["forecast_detail"]["period"].unique())[0]
        column = f"REVIEWED qty {period}"
        target = sheet["sku"].iloc[0]
        original = float(first["forecast_detail"].query(
            "sku == @target and period == @period")["forecast_qty"].iloc[0])
        # The text columns come back as all-NaN floats, so pandas refuses a string
        # into them. Excel does not care; this is the test's problem, not a reviewer's.
        sheet["REVIEWED reason"] = sheet["REVIEWED reason"].astype(object)
        sheet.loc[sheet["sku"] == target, column] = original * 2
        sheet.loc[sheet["sku"] == target, "REVIEWED reason"] = "framework agreement"
        sheet.to_excel(tmp_path / "reviewed.xlsx", index=False)

        second = InventoryPlanner(output_dir=tmp_path / "run2", interactive=False)
        plan = second.load_sales_plan(tmp_path / "reviewed.xlsx")
        results = second.run_planning(
            **second.load_all(sorted(SAMPLE.glob("*.csv"))), sales_plan=plan)

        row = results["forecast_detail"].query(
            "sku == @target and period == @period").iloc[0]
        assert row["forecast_qty"] == pytest.approx(original * 2)
        assert row["statistical_qty"] == pytest.approx(original)
        assert row["forecast_source"] == "sales_review"

        # And it reached the plan, not just the frame.
        assert results["sales_plan_override"].applied == 1
        summary = results["forecast_summary"].set_index("sku")
        assert summary.loc[target, "forecast_next_period"] == pytest.approx(original * 2)
