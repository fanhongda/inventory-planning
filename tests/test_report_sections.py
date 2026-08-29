"""
The three analytics behind the new report sections, and the banner above them.

Each test here is about a way of aggregating that is easy to get wrong and impossible
to spot once it is wrong: a mean of ratios, a netted gap, an ageing figure that is
really a cover figure. All three produce numbers that look entirely reasonable.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.forecast_accuracy import build_forecast_accuracy
from inventory_planning.analytics.inventory_health import build_inventory_health
from inventory_planning.analytics.siop import build_siop_plan
from inventory_planning.quality import SEVERE, Finding, GateReport, assess
from inventory_planning.reporting.kpi_report import KPIReport


# ── S&IOP period balance ─────────────────────────────────────────────────────


def _forecast(skus, periods, qty=10.0):
    rows = [{"sku": s, "period": p, "forecast_qty": qty, "is_next_period": i == 0}
            for s in skus for i, p in enumerate(periods)]
    return pd.DataFrame(rows)


class TestTheGapIsNeverNetted:

    PERIODS = ["2026-09", "2026-10", "2026-11"]

    def test_one_skus_surplus_cannot_cover_anothers_shortage(self):
        """
        The failure this exists to prevent: aggregate first and the business looks
        covered while it cannot ship a single order of the item that is short.
        Different materials are not interchangeable and neither is the cash.
        """
        forecast = _forecast(["A", "B"], self.PERIODS, qty=10.0)
        inventory = pd.DataFrame({"sku": ["A", "B"], "qty_on_hand": [1000.0, 0.0]})
        attributes = pd.DataFrame({"sku": ["A", "B"], "unit_cost": [5.0, 5.0]})

        plan = build_siop_plan(forecast, inventory=inventory, attributes=attributes)
        # B is short from the first period; A is swimming in stock. The aggregate
        # position is hugely positive, and there must still be a gap.
        assert plan.by_period["gap_value"].sum() > 0
        assert plan.by_period["closing_value"].sum() > 0

    def test_the_gap_is_measured_against_safety_stock_not_zero(self):
        """
        A period is short where service is at risk, which is above an empty shelf.
        Measuring against zero would report nothing until the stockout had happened.
        """
        forecast = _forecast(["A"], self.PERIODS, qty=10.0)
        inventory = pd.DataFrame({"sku": ["A"], "qty_on_hand": [100.0]})
        attributes = pd.DataFrame({"sku": ["A"], "unit_cost": [1.0]})
        safety = pd.DataFrame({"sku": ["A"], "safety_stock": [95.0]})

        without = build_siop_plan(forecast, inventory=inventory, attributes=attributes)
        with_ss = build_siop_plan(forecast, inventory=inventory, attributes=attributes,
                                  safety_stock=safety)
        assert without.by_period["gap_value"].sum() == 0, "never runs out of stock"
        assert with_ss.by_period["gap_value"].sum() > 0, "but falls below policy at once"

    def test_past_due_supply_is_counted_in_the_first_period(self):
        """
        It has no forward date any more, but the goods are still coming. Dropping it
        reports a shortfall the receiving dock is about to close.
        """
        forecast = _forecast(["A"], self.PERIODS, qty=10.0)
        attributes = pd.DataFrame({"sku": ["A"], "unit_cost": [2.0]})
        open_po = pd.DataFrame({
            "sku": ["A"], "open_qty": [50.0],
            "committed_delivery": [pd.Timestamp("2020-01-01")],   # long overdue
        })
        plan = build_siop_plan(forecast, open_po=open_po, attributes=attributes)
        first = plan.by_period.iloc[0]
        assert first["supply_qty"] == 50.0

    def test_supply_beyond_the_horizon_cannot_serve_demand_inside_it(self):
        forecast = _forecast(["A"], self.PERIODS, qty=10.0)
        attributes = pd.DataFrame({"sku": ["A"], "unit_cost": [2.0]})
        open_po = pd.DataFrame({
            "sku": ["A"], "open_qty": [50.0],
            "committed_delivery": [pd.Timestamp("2030-01-01")],
        })
        plan = build_siop_plan(forecast, open_po=open_po, attributes=attributes)
        assert plan.by_period["supply_qty"].sum() == 0

    def test_an_uncosted_sku_is_excluded_and_counted_not_valued_at_zero(self):
        """
        Zero would sum into the total and understate the plan without saying so.
        """
        forecast = _forecast(["A", "B"], self.PERIODS, qty=10.0)
        attributes = pd.DataFrame({"sku": ["A", "B"], "unit_cost": [5.0, np.nan]})
        plan = build_siop_plan(forecast, attributes=attributes)
        assert plan.costed_skus == 1 and plan.uncosted_skus == 1
        assert "no unit cost" in plan.summary()


# ── DIOH ─────────────────────────────────────────────────────────────────────


class _ShouldBe:
    def __init__(self, frame):
        self.frame = frame


class TestDIOHIsValueWeighted:

    def _should_be(self):
        # One dead item and two healthy ones, all in the same line. The dead one holds
        # almost no money and scores an enormous per-SKU DIOH.
        return _ShouldBe(pd.DataFrame({
            "sku": ["DEAD", "LIVE-1", "LIVE-2"],
            "actual_qty": [40.0, 500.0, 500.0],
            "actual_value": [40.0, 20000.0, 20000.0],
            "demand_mean": [0.1, 250.0, 250.0],
            "actual_dioh": [12000.0, 60.0, 60.0],
        }))

    def _attributes(self):
        return pd.DataFrame({
            "sku": ["DEAD", "LIVE-1", "LIVE-2"],
            "product_family": ["valve"] * 3,
            "unit_cost": [1.0, 40.0, 40.0],
        })

    def test_one_dead_part_does_not_speak_for_the_line(self):
        """
        The mean of per-SKU DIOH here is 4,040 days and the line looks terminal. The
        value-weighted figure is what the money actually does.
        """
        health = build_inventory_health(should_be=self._should_be(),
                                        attributes=self._attributes())
        line = health.by_family.iloc[0]
        naive_mean = self._should_be().frame["actual_dioh"].mean()
        assert naive_mean > 4000
        assert line["dioh"] < 100, f"value-weighted DIOH came out {line['dioh']}"

    def test_slow_movers_are_found_from_the_demand_series(self):
        ts = pd.DataFrame({
            "DEAD": [5.0] + [0.0] * 11,          # nothing for eleven months
            "LIVE-1": [200.0] * 12,
            "LIVE-2": [200.0] * 12,
        })
        health = build_inventory_health(should_be=self._should_be(), time_series=ts,
                                        attributes=self._attributes())
        assert set(health.slow_moving["sku"]) == {"DEAD"}

    def test_without_an_ageing_column_the_figure_is_named_as_cover(self):
        """
        Cover and age are wrong about each other in both directions. Labelling cover as
        age puts the wrong items on a write-off list.
        """
        health = build_inventory_health(should_be=self._should_be(),
                                        attributes=self._attributes())
        assert not health.aging_measured
        assert "cover" in health.aging_basis.lower()
        assert "days of cover, not age" in health.summary()

    def test_a_real_ageing_column_is_measured_as_age(self):
        inventory = pd.DataFrame({
            "sku": ["DEAD", "LIVE-1", "LIVE-2"],
            "inventory_age_days": [900.0, 20.0, 15.0],
        })
        health = build_inventory_health(should_be=self._should_be(),
                                        attributes=self._attributes(),
                                        inventory=inventory)
        assert health.aging_measured
        assert set(health.long_aging["sku"]) == {"DEAD"}
        assert "cover" not in health.aging_basis.lower()


# ── Forecast accuracy ────────────────────────────────────────────────────────


class TestThePublishedPlanIsScoredAgainstWhatSold:
    """
    Not the backtest. The backtest asks whether the model was the best available for a
    series; this asks whether the number the business committed to — after sales
    reviewed it, and often after they changed it — turned out to be right. A model can
    win its backtest and still be wrong about next March, and a reviewed forecast that
    overrode the model is not in the backtest at all.
    """

    PERIODS = ["2026-01", "2026-02", "2026-03"]

    def _history(self, tmp_path, plan_rows, run_at="2026-01-01T00:00:00",
                 name="snapshot_20260101_0000.json"):
        month = tmp_path / "2026-01"
        month.mkdir(parents=True, exist_ok=True)
        (month / name).write_text(
            json.dumps({"run_at": run_at, "plan": plan_rows}), encoding="utf-8")
        return tmp_path

    def _plan(self, sku, qty):
        return [{"sku": sku, "period": p, "forecast_qty": qty} for p in self.PERIODS]

    def _fixture(self, tmp_path):
        # A large line planned 43% over, and a tiny one planned 300% over.
        root = self._history(tmp_path,
                             self._plan("BIG", 1000.0) + self._plan("TINY", 40.0))
        ts = pd.DataFrame({"BIG": [700.0] * 3, "TINY": [10.0] * 3}, index=self.PERIODS)
        attrs = pd.DataFrame({"sku": ["BIG", "TINY"],
                              "product_family": ["sprinkler", "odds"],
                              "business_unit": ["Water", "Water"]})
        price = pd.Series({"BIG": 100.0, "TINY": 2.0})
        return build_forecast_accuracy(time_series=ts, history_root=root,
                                       attributes=attrs, price=price)

    def test_ranking_puts_value_before_percentage(self, tmp_path):
        accuracy = self._fixture(tmp_path)
        assert list(accuracy.by_sku["sku"]) == ["BIG", "TINY"]
        assert accuracy.by_sku.iloc[0]["bias_pct"] < accuracy.by_sku.iloc[1]["bias_pct"]

    def test_direction_is_carried_not_just_magnitude(self, tmp_path):
        """Size decides whether it is worth discussing; direction decides what about."""
        accuracy = self._fixture(tmp_path)
        assert accuracy.bias_direction == "over-forecast"
        assert accuracy.bias_value > 0

    def test_a_first_run_says_so_rather_than_showing_the_backtest(self, tmp_path):
        """
        Substituting a statistic that answers a different question is worse than an
        empty section — it looks like an answer.
        """
        ts = pd.DataFrame({"A": [10.0] * 3}, index=self.PERIODS)
        accuracy = build_forecast_accuracy(time_series=ts, history_root=tmp_path)
        assert not accuracy.measured
        assert "No earlier plan" in accuracy.reason
        assert "different question" in accuracy.reason

    def test_a_plan_whose_periods_have_not_closed_is_not_scored(self, tmp_path):
        root = self._history(tmp_path, [{"sku": "A", "period": "2027-06",
                                         "forecast_qty": 10.0}])
        ts = pd.DataFrame({"A": [10.0] * 3}, index=self.PERIODS)
        accuracy = build_forecast_accuracy(time_series=ts, history_root=root)
        assert not accuracy.measured
        assert "still in the future" in accuracy.reason

    def test_the_earliest_plan_wins_when_a_period_was_planned_twice(self, tmp_path):
        """
        The first plan is the one the business acted on. Scoring a revision published a
        fortnight later would flatter every forecast by measuring it after the fact.
        """
        root = self._history(tmp_path, self._plan("A", 100.0))
        self._history(root, self._plan("A", 20.0), run_at="2026-02-01T00:00:00",
                      name="snapshot_20260201_0000.json")
        ts = pd.DataFrame({"A": [20.0] * 3}, index=self.PERIODS)
        price = pd.Series({"A": 1.0})
        accuracy = build_forecast_accuracy(time_series=ts, history_root=root,
                                           price=price)
        # Scored against the 100 it originally promised, not the 20 it later corrected to.
        assert accuracy.bias_value == pytest.approx(240.0)

    def test_both_sides_are_valued_at_one_price(self, tmp_path):
        """
        Valuing actuals at their own realised price would make a discount look like a
        demand error.
        """
        accuracy = self._fixture(tmp_path)
        big = accuracy.by_sku.set_index("sku").loc["BIG"]
        assert big["planned_value"] == pytest.approx(3000 * 100.0)
        assert big["actual_value"] == pytest.approx(2100 * 100.0)

    def test_this_runs_review_is_reported_but_not_scored(self, tmp_path):
        """Nothing has happened to it yet — it becomes scoreable next period."""
        detail = pd.DataFrame({
            "sku": ["A", "A"], "period": self.PERIODS[:2],
            "forecast_qty": [20.0, 10.0], "statistical_qty": [10.0, 10.0],
            "forecast_source": ["sales_review", "statistical"],
        })
        ts = pd.DataFrame({"A": [10.0] * 3}, index=self.PERIODS)
        accuracy = build_forecast_accuracy(time_series=ts, history_root=tmp_path,
                                           forecast_detail=detail)
        assert len(accuracy.adjustments) == 1
        assert accuracy.adjustments.iloc[0]["delta_qty"] == 10.0


# ── The banner ───────────────────────────────────────────────────────────────


class TestTheHealthBannerLeadsTheReport:

    def _page(self, health):
        return KPIReport().render(health=health)

    def test_a_severe_finding_appears_before_the_kpi_tiles(self):
        gate = GateReport(stage="intake", findings=[Finding(
            stage="intake", check="no_product_dimension", severity=SEVERE,
            what="No product family anywhere", why="the guess is arbitrary",
            fix="map it", impacts=["Revenue by product line: unusable",
                                   "Not affected: safety stock and purchase quantities"])])
        page = self._page(assess("r1", [gate]))
        # Before the body of the report, whatever the first section of it turns out to
        # be. A reader who meets the numbers first has formed a view by the time the
        # caveat arrives.
        assert page.index("No product family anywhere") < page.index("Chapter 1")

    def test_the_unaffected_half_is_shown_apart_from_the_damage(self):
        """
        A warning that does not bound itself is read as "the whole report is suspect",
        and the recommendations get discarded with the rollup they had nothing to do
        with.
        """
        gate = GateReport(stage="intake", findings=[Finding(
            stage="intake", check="no_product_dimension", severity=SEVERE,
            what="w", why="y", fix="f",
            impacts=["Revenue by line: unusable",
                     "Not affected: safety stock and purchase quantities"])])
        page = self._page(assess("r1", [gate]))
        assert "class='safe'" in page
        assert "✓ Not affected" in page

    def test_a_clean_run_says_so_rather_than_showing_nothing(self):
        page = self._page(assess("r1", [GateReport(stage="intake")]))
        assert "Nothing in this report rests on a fallback" in page

    def test_no_health_renders_no_banner(self):
        assert "class='health" not in self._page(None)
