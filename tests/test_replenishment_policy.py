"""
Tests for the replenishment quantity and the policy that decides it.

The defect these exist to keep closed: the order quantity did not cover the lead
time. The requirement was one horizon of demand — thirty days, from a constructor
default — plus safety stock, and nothing in it accounted for the weeks the supplier
takes to deliver. Meanwhile the policy layer had always sized safety stock on R + LT.
One pipeline, two definitions of one exposure period, and the buy was the one that was
wrong.

Alongside that, which arithmetic applies is now a per-SKU decision rather than a
constant: periodic review orders the gap up to S, an (s, Q) item orders whole lots, and
an order-on-demand item holds no policy stock at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.purchase_recommender import PurchaseRecommender
from inventory_planning.analytics.safety_stock import SafetyStockCalculator
from inventory_planning.lot_sizing import economic_order_quantity, round_to_lot
from inventory_planning.policy.profile import (
    POLICY_MAKE_TO_ORDER,
    POLICY_PERIODIC,
    POLICY_REORDER_POINT,
    build_policy_profile,
    canonical_policy,
)

CONFIG_DIR = Path(__file__).parents[1] / "config"


def projection(sku="S1", demand=14398.0, safety_stock=0.0, position=0.0,
               lead_time=51.7, **extra):
    row = {
        "sku": [sku],
        "location_id": ["DC-01"],
        "stocking_class": ["stocking-high"],
        "demand_mean_rolling": [demand],
        "wma_lead_time_days": [lead_time],
        "safety_stock": [safety_stock],
        "effective_position": [position],
        "inventory_status": ["SHORTAGE-RISK"],
        "pushout_candidate": [False],
        "pushout_open_po_qty": [0.0],
    }
    row.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(row)


def forecast(sku="S1", qty=14398.0):
    return pd.DataFrame({"sku": [sku], "forecast_next_period": [qty],
                         "forecast_avg_monthly": [qty]})


def params(sku="S1", **overrides):
    row = {"sku": [sku], "review_period_days": [30.0],
           "replenishment_method": [POLICY_PERIODIC],
           "min_order_qty": [0.0], "order_multiple": [1.0], "unit_cost": [np.nan]}
    row.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(row)


class TestTheOrderCoversTheLeadTime:
    """The headline defect, with the numbers from the report that found it."""

    def test_a_month_of_demand_does_not_cover_a_fifty_day_lead_time(self):
        recs = PurchaseRecommender().recommend(
            projection(), forecast(), None, None, parameters=params()
        )
        row = recs.iloc[0]
        # R + LT = 30 + 51.7 = 81.7 days of a 14,398/month rate.
        assert row["coverage_days"] == pytest.approx(81.7)
        assert row["period_demand"] == pytest.approx(14398 * 81.7 / 30, rel=1e-3)
        # The old arithmetic sized this at one month. It is 2.7x that.
        assert row["suggested_po_qty"] / 14398 == pytest.approx(2.72, abs=0.02)

    def test_a_shorter_review_period_buys_less(self):
        weekly = PurchaseRecommender().recommend(
            projection(), forecast(), None, None,
            parameters=params(review_period_days=7.0),
        ).iloc[0]
        monthly = PurchaseRecommender().recommend(
            projection(), forecast(), None, None, parameters=params()
        ).iloc[0]
        assert weekly["coverage_days"] == pytest.approx(58.7)
        assert weekly["suggested_po_qty"] < monthly["suggested_po_qty"]

    def test_the_exposure_the_safety_stock_used_is_the_one_the_order_covers(self):
        """
        Two clocks is the defect. Where the safety stock states its own exposure the
        order takes that number rather than re-deriving R + LT and hoping they agree.
        """
        recs = PurchaseRecommender().recommend(
            projection(ss_exposure_days=120.0), forecast(), None, None,
            parameters=params(),
        )
        row = recs.iloc[0]
        assert row["coverage_basis"] == "safety_stock_exposure"
        assert row["coverage_days"] == pytest.approx(120.0)

    def test_without_parameters_the_run_says_so_rather_than_inventing_a_policy(self):
        recs = PurchaseRecommender().recommend(projection(), forecast(), None, None)
        assert recs.iloc[0]["policy_source"] == "default"
        assert recs.iloc[0]["replenishment_method"] == POLICY_PERIODIC


class TestSafetyStockUsesTheSameExposure:

    def _classified(self, demand=100.0, sigma=20.0):
        return pd.DataFrame({
            "sku": ["S1"], "stocking_class": ["stocking-high"], "z_score": [1.645],
            "demand_mean_rolling": [demand], "demand_std_rolling": [sigma],
        })

    def _lt(self, days=30.0, sigma=0.0):
        return pd.DataFrame({"sku": ["S1"], "wma_lead_time_days": [days],
                             "lt_std_days": [sigma], "order_count": [10]})

    def test_review_period_lengthens_the_exposure(self):
        calc = SafetyStockCalculator()
        lt_only = calc.calculate(self._classified(), self._lt(), review_period_days=None)
        with_review = calc.calculate(self._classified(), self._lt(),
                                     review_period_days=30)
        assert lt_only.iloc[0]["ss_exposure_days"] == pytest.approx(30.0)
        assert with_review.iloc[0]["ss_exposure_days"] == pytest.approx(60.0)
        # √2 more exposure, √2 more safety stock — the 41% the docstring names.
        ratio = with_review.iloc[0]["safety_stock"] / lt_only.iloc[0]["safety_stock"]
        assert ratio == pytest.approx(np.sqrt(2), rel=0.01)

    def test_continuous_review_keeps_the_lead_time_alone(self):
        calc = SafetyStockCalculator()
        out = calc.calculate(self._classified(), self._lt(), review_period_days=30,
                             exposure="lt_only")
        assert out.iloc[0]["ss_exposure_basis"] == "lt_only"
        assert out.iloc[0]["ss_exposure_days"] == pytest.approx(30.0)

    def test_a_per_sku_review_period_is_honoured(self):
        calc = SafetyStockCalculator()
        out = calc.calculate(
            self._classified(), self._lt(),
            review_period_days=pd.DataFrame({"sku": ["S1"], "review_period_days": [7]}),
        )
        assert out.iloc[0]["ss_exposure_days"] == pytest.approx(37.0)


class TestLotSizing:

    def test_the_order_up_to_level_is_the_reorder_point_plus_a_lot(self):
        recs = PurchaseRecommender().recommend(
            projection(), forecast(), None, None,
            parameters=params(unit_cost=10.0),
        )
        row = recs.iloc[0]
        assert row["eoq_qty"] > 0
        assert row["order_up_to"] == pytest.approx(row["reorder_point"] + row["order_lot"])

    def test_no_lot_size_leaves_the_policy_ordering_up_to_the_reorder_point(self):
        """S is allowed to be absent. That is the degenerate case, not a missing input."""
        row = PurchaseRecommender().recommend(
            projection(), forecast(), None, None, parameters=params()
        ).iloc[0]
        assert row["order_lot"] == 0
        assert row["order_up_to"] == pytest.approx(row["reorder_point"])

    def test_the_supplier_minimum_applies_even_without_a_priced_eoq(self):
        row = PurchaseRecommender().recommend(
            projection(), forecast(), None, None,
            parameters=params(min_order_qty=5000.0),
        ).iloc[0]
        assert row["order_lot"] == pytest.approx(5000.0)

    def test_a_lot_is_rounded_up_to_the_order_multiple_never_down(self):
        assert round_to_lot([101.0], min_order_qty=0, order_multiple=25).iloc[0] == 125.0
        assert round_to_lot([100.0], min_order_qty=0, order_multiple=25).iloc[0] == 100.0

    def test_an_unpriced_item_has_no_eoq_rather_than_an_eoq_of_zero(self):
        eoq = economic_order_quantity([1200.0], [None], order_cost=350, holding_rate=0.22)
        assert pd.isna(eoq.iloc[0])


class TestThePolicyDecidesTheQuantity:

    def test_periodic_review_closes_the_gap_to_the_order_up_to_level(self):
        row = PurchaseRecommender().recommend(
            projection(position=1000.0), forecast(), None, None,
            parameters=params(min_order_qty=2000.0),
        ).iloc[0]
        expected = row["order_up_to"] - row["available_supply"]
        assert row["suggested_po_qty"] == pytest.approx(expected, abs=0.1)

    def test_an_s_q_item_orders_whole_lots(self):
        row = PurchaseRecommender().recommend(
            projection(demand=100.0, lead_time=30.0, position=0.0), forecast(qty=100.0),
            None, None,
            parameters=params(replenishment_method=POLICY_REORDER_POINT,
                              min_order_qty=100.0),
        ).iloc[0]
        assert row["order_lot"] == pytest.approx(100.0)
        assert row["suggested_po_qty"] % 100 == pytest.approx(0.0)
        # One lot would leave it short of the reorder point; it buys enough lots to clear.
        assert row["suggested_po_qty"] >= row["net_requirement"]

    def test_a_position_above_the_reorder_point_buys_nothing(self):
        row = PurchaseRecommender().recommend(
            projection(position=1_000_000.0, inventory_status="OK"), forecast(),
            None, None, parameters=params(min_order_qty=500.0),
        ).iloc[0]
        assert row["net_requirement"] == 0
        assert row["order_quantity"] == 0
        assert row["suggested_po_qty"] == 0

    def test_an_order_on_demand_item_holds_no_policy_stock(self):
        row = PurchaseRecommender().recommend(
            projection(stocking_class="non-stocking", inventory_status="non-stocking"),
            forecast(), None, None,
            parameters=params(replenishment_method=POLICY_MAKE_TO_ORDER,
                              min_order_qty=500.0),
        ).iloc[0]
        assert row["order_quantity"] == 0
        assert row["recommended_action"] in ("NO-ACTION", "ORDER-FOR-BACKLOG")

    def test_the_supply_gap_still_outranks_the_requirement(self):
        """
        Ranked ahead on purpose: where the shelf empties before the next delivery the
        order already exists, and a second one is not the answer.
        """
        row = PurchaseRecommender().recommend(
            projection(supply_gap=True, total_open_po_qty=5000.0), forecast(),
            None, None, parameters=params(),
        ).iloc[0]
        assert row["recommended_action"] == "EXPEDITE-INBOUND"
        assert row["suggested_po_qty"] == 0

    def test_min_and_max_are_emitted_for_the_erp_to_run_on(self):
        row = PurchaseRecommender().recommend(
            projection(), forecast(), None, None,
            parameters=params(replenishment_method=POLICY_REORDER_POINT,
                              min_order_qty=1000.0),
        ).iloc[0]
        assert row["suggested_min_qty"] == pytest.approx(row["reorder_point"])
        assert row["suggested_max_qty"] == pytest.approx(row["order_up_to"])


class TestPolicyProfile:

    def _attributes(self, **overrides):
        row = {
            "sku": ["S1"], "demand_cv": [0.62], "total_cycles_evaluated": [12],
            "active_cycles_rolling": [11], "demand_pattern": ["smooth"],
            "demand_mean": [14398.0], "stocking_class": ["stocking-high"],
            "lead_time_days": [51.7], "lt_sigma_days": [12.4], "lt_samples": [9],
            "review_period_days": [30.0], "replenishment_method": [POLICY_PERIODIC],
            "min_order_qty": [0.0], "order_multiple": [1.0],
            "abc_class": ["A"], "unit_cost": [40.0], "applied_rules": ["R-001"],
        }
        row.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(row)

    def test_every_axis_carries_its_evidence_and_where_it_came_from(self):
        profile = build_policy_profile(self._attributes())
        row = profile.frame.iloc[0]
        assert row["demand_variability"] == "variable"
        assert "measured" in row["demand_variability_evidence"]
        assert row["lead_time_behaviour"] == "variable"
        assert "9 receipts" in row["lead_time_behaviour_evidence"]
        assert "R-001" in row["review_mode_evidence"]

    def test_a_thin_lead_time_sample_is_not_presented_as_a_distribution(self):
        profile = build_policy_profile(self._attributes(lt_samples=2))
        row = profile.frame.iloc[0]
        assert row["lead_time_behaviour"] == "thinly-sampled"
        assert "stated" in row["lead_time_behaviour_evidence"]

    def test_an_unobservable_axis_says_so_rather_than_guessing(self):
        """No order book supplied is not the same finding as no backorders."""
        profile = build_policy_profile(self._attributes())
        row = profile.frame.iloc[0]
        assert row["excess_demand"] == "unknown"
        assert "not observable" in row["excess_demand_evidence"]

    def test_a_backlog_makes_the_backorder_assumption_measured(self):
        backlog = pd.DataFrame({"sku": ["S1"], "backlog_due_qty": [20000.0]})
        row = build_policy_profile(self._attributes(), backlog=backlog).frame.iloc[0]
        assert row["excess_demand"] == "backordered"
        assert row["demand_certainty"] == "known"

    def test_a_lot_constraint_is_recorded_as_a_capacity_limit(self):
        row = build_policy_profile(
            self._attributes(min_order_qty=500.0)).frame.iloc[0]
        assert row["capacity"] == "lot-constrained"

    def test_merged_storage_locations_are_visible_on_the_profile(self):
        inventory = pd.DataFrame({"sku": ["S1"], "stock_locations": ["01,02"]})
        row = build_policy_profile(self._attributes(), inventory=inventory).frame.iloc[0]
        assert row["location_scope"] == "merged-multi"
        assert "2 storage locations" in row["location_scope_evidence"]

    def test_the_rule_governs_and_the_disagreement_is_reported(self):
        """
        A planner's deliberate exception outranks an inference drawn from history.
        Overwriting the rule would destroy the finding.
        """
        profile = build_policy_profile(
            self._attributes(stocking_class="non-stocking"))
        row = profile.frame.iloc[0]
        assert row["policy_in_force"] == POLICY_PERIODIC
        assert row["policy_implied"] == POLICY_MAKE_TO_ORDER
        assert not row["policy_agrees"]
        assert len(profile.disagreements) == 1
        assert "policy" in profile.summary().lower()

    def test_the_cheap_tail_is_implied_onto_s_q(self):
        row = build_policy_profile(
            self._attributes(abc_class="C", unit_cost=4.0)).frame.iloc[0]
        assert row["policy_implied"] == POLICY_REORDER_POINT


class TestPolicyVocabulary:

    def test_min_max_is_still_read_as_the_s_q_policy(self):
        """An existing parameter file keeps working; the name it resolves to is (s, Q)."""
        assert canonical_policy("min_max") == POLICY_REORDER_POINT
        assert canonical_policy(None) == POLICY_PERIODIC

    def test_the_shipped_rules_only_use_the_current_names(self):
        text = (CONFIG_DIR / "planning_parameters.md").read_text(encoding="utf-8")
        rules = text.split("## 覆盖规则", 1)[1]
        assert "replenishment_method: min_max" not in rules
