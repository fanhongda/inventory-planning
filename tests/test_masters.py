"""
Tests for the two optional master inputs.

An item master and a planner's worksheet each carry parameters that the transaction
history also speaks to. The three things that must hold:

  fill         a parameter with no measurement behind it takes the best stated value,
               and the row records which source it came from
  cross-check  where sources overlap and disagree materially, the disagreement is
               recorded rather than resolved silently
  compare      the planner's own numbers are never consumed as inputs — they are the
               benchmark the computed ones are measured against
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.registry import AdapterRegistry
from inventory_planning.orchestrator import InventoryPlanner
from inventory_planning.policy.assemble import build_sku_attributes
from inventory_planning.policy.crosscheck import (
    ITEM_MASTER,
    MEASURED,
    PLANNING_MASTER,
    SourceResolver,
)
from inventory_planning.policy.parameters import PlanningParameters
from inventory_planning.policy.suggestions import SuggestionBuilder

CONFIG_DIR = Path(__file__).parents[1] / "config"
SAMPLE_DIR = Path(__file__).parents[1] / "sample_data"


@pytest.fixture
def classified():
    """Three SKUs: one measured well, one measured thinly, one never bought."""
    return pd.DataFrame({
        "sku": ["MEASURED-1", "THIN-1", "NEVER-1"],
        "location_id": ["DC-01"] * 3,
        "demand_mean_rolling": [100.0, 50.0, 20.0],
        "demand_std_rolling": [30.0, 15.0, 6.0],
        "demand_cv": [0.3, 0.3, 0.3],
        "stocking_class": ["stocking-high"] * 3,
        "demand_pattern": ["smooth"] * 3,
        "service_level": [0.95] * 3,
        "z_score": [1.645] * 3,
    })


@pytest.fixture
def supplier_lt():
    """MEASURED-1 has nine receipts; THIN-1 has two; NEVER-1 has none."""
    return pd.DataFrame({
        "sku": ["MEASURED-1", "THIN-1"],
        "supplier": ["ACME", "ACME"],
        "wma_lead_time_days": [60.0, 20.0],
        "lt_std_days": [8.0, 2.0],
        "sample_count": [9, 2],
        "incoterm": ["FOB", "FOB"],
    })


@pytest.fixture
def item_master():
    return pd.DataFrame({
        "sku": ["MEASURED-1", "THIN-1", "NEVER-1"],
        "description": ["Measured item", "Thin item", "Never bought"],
        "supplier": ["ACME", "ACME", "GLOBEX"],
        "lead_time_days": [40.0, 75.0, 90.0],   # 40 vs a measured 60 is a −33% gap
        "min_order_qty": [100.0, 50.0, 25.0],
        "order_multiple": [25.0, 25.0, 5.0],
        "unit_cost": [40.0, 12.0, 300.0],
        "product_family": ["valve", "valve", "actuator"],
        "item_status": ["active", "active", "active"],
    })


@pytest.fixture
def planning_master():
    return pd.DataFrame({
        "sku": ["MEASURED-1", "THIN-1", "NEVER-1"],
        "planner_safety_stock": [900.0, 30.0, 200.0],
        "planner_reorder_point": [1500.0, 90.0, 300.0],
        "planner_lead_time_days": [50.0, 70.0, 95.0],
        "planner_review_period_days": [7.0, 30.0, 30.0],
        "planner_service_level": [98.0, 95.0, 95.0],   # percentage form
        "planner_monthly_demand": [105.0, 400.0, 21.0],  # THIN-1 disagrees badly
        "planner_stocking_class": ["A", "C", "B"],
    })


def _attributes(classified, supplier_lt, **kwargs):
    return build_sku_attributes(
        classified_demand=classified,
        supplier_lt=supplier_lt,
        params=PlanningParameters(CONFIG_DIR / "planning_parameters.md"),
        **kwargs,
    )


# ── Routing ──────────────────────────────────────────────────────────────────


class TestRouting:

    def test_item_master_is_recognised(self, item_master):
        raw = item_master.rename(columns={
            "sku": "Material", "lead_time_days": "Planned Deliv. Time",
            "min_order_qty": "MOQ", "unit_cost": "Standard Cost",
            "order_multiple": "Rounding Value",
        }).astype(str)
        route = AdapterRegistry().route(raw, source_name="master.csv")
        assert route.doc_type == "item_master"

    def test_planning_master_is_recognised(self, planning_master):
        raw = planning_master.rename(columns={
            "sku": "Item Code", "planner_safety_stock": "Safety Stock",
            "planner_reorder_point": "ROP", "planner_lead_time_days": "Lead Time",
            "planner_review_period_days": "Review Cycle",
        }).astype(str)
        route = AdapterRegistry().route(raw, source_name="planner sheet.csv")
        assert route.doc_type == "planning_master"

    @pytest.mark.parametrize("doc", ["sales_history", "po_history", "open_so",
                                     "open_po", "inventory"])
    def test_core_documents_still_route_correctly(self, doc):
        """The new contracts must not capture documents that already had a home."""
        raw = pd.read_csv(SAMPLE_DIR / f"{doc}.csv", dtype=str)
        assert AdapterRegistry().route(raw, source_name=f"{doc}.csv").doc_type == doc


# ── Fill ─────────────────────────────────────────────────────────────────────


class TestFill:

    def test_measured_lead_time_wins_over_stated(self, classified, supplier_lt,
                                                 item_master, planning_master):
        attrs, _ = _attributes(classified, supplier_lt, item_master=item_master,
                               planning_master=planning_master)
        row = attrs.set_index("sku").loc["MEASURED-1"]
        assert row["lead_time_days"] == 60.0
        assert row["lead_time_days_source"] == MEASURED

    def test_thin_measurement_yields_to_the_master(self, classified, supplier_lt,
                                                   item_master, planning_master):
        """Two receipts is a coincidence, not a distribution."""
        row = _attributes(classified, supplier_lt, item_master=item_master,
                          planning_master=planning_master)[0].set_index("sku").loc["THIN-1"]
        assert row["lead_time_days"] == 75.0
        assert row["lead_time_days_source"] == ITEM_MASTER

    def test_unbought_sku_gets_a_stated_lead_time(self, classified, supplier_lt,
                                                  item_master, planning_master):
        row = _attributes(classified, supplier_lt, item_master=item_master,
                          planning_master=planning_master)[0].set_index("sku").loc["NEVER-1"]
        assert row["lead_time_days"] == 90.0
        assert row["lead_time_days_source"] == ITEM_MASTER

    def test_planner_lead_time_used_when_no_item_master(self, classified, supplier_lt,
                                                        planning_master):
        row = _attributes(classified, supplier_lt,
                          planning_master=planning_master)[0].set_index("sku").loc["NEVER-1"]
        assert row["lead_time_days"] == 95.0
        assert row["lead_time_days_source"] == PLANNING_MASTER

    def test_cost_and_moq_come_from_the_master(self, classified, supplier_lt, item_master):
        attrs, _ = _attributes(classified, supplier_lt, item_master=item_master)
        row = attrs.set_index("sku").loc["NEVER-1"]
        assert row["unit_cost"] == 300.0
        assert row["min_order_qty"] == 25.0
        assert row["order_multiple"] == 5.0

    def test_master_product_family_beats_the_prefix_guess(self, classified, supplier_lt,
                                                          item_master):
        attrs, _ = _attributes(classified, supplier_lt, item_master=item_master)
        assert attrs.set_index("sku").loc["NEVER-1", "product_family"] == "actuator"

    def test_no_masters_leaves_behaviour_unchanged(self, classified, supplier_lt):
        attrs, crosscheck = _attributes(classified, supplier_lt)
        assert attrs.set_index("sku").loc["MEASURED-1", "lead_time_days"] == 60.0
        assert crosscheck.resolutions == {}

    def test_duplicate_sku_in_a_master_does_not_multiply_rows(self, classified,
                                                              supplier_lt, item_master):
        dupes = pd.concat([item_master, item_master.head(1)], ignore_index=True)
        attrs, _ = _attributes(classified, supplier_lt, item_master=dupes)
        assert len(attrs) == len(classified)


# ── Cross-check ──────────────────────────────────────────────────────────────


class TestCrossCheck:

    def test_lead_time_disagreement_is_recorded(self, classified, supplier_lt,
                                                item_master, planning_master):
        _, crosscheck = _attributes(classified, supplier_lt, item_master=item_master,
                                    planning_master=planning_master)
        lt = [d for d in crosscheck.all_disagreements if d.attribute == "lead_time_days"]
        assert any(d.sku == "MEASURED-1" and d.other_source == ITEM_MASTER for d in lt)

    def test_agreement_within_tolerance_is_not_reported(self, classified, supplier_lt,
                                                        item_master, planning_master):
        """MEASURED-1's planner monthly demand of 105 against a measured 100 is noise."""
        _, crosscheck = _attributes(classified, supplier_lt, item_master=item_master,
                                    planning_master=planning_master)
        demand = [d for d in crosscheck.all_disagreements
                  if d.attribute == "monthly_demand" and d.sku == "MEASURED-1"]
        assert demand == []

    def test_demand_definition_gap_is_surfaced(self, classified, supplier_lt,
                                               planning_master):
        """THIN-1: history says 50/month, the planner's sheet says 400."""
        _, crosscheck = _attributes(classified, supplier_lt,
                                    planning_master=planning_master)
        demand = [d for d in crosscheck.all_disagreements
                  if d.attribute == "monthly_demand" and d.sku == "THIN-1"]
        assert len(demand) == 1
        assert demand[0].other_value == 400.0

    def test_planner_demand_never_overrides_the_history(self, classified, supplier_lt,
                                                        planning_master):
        attrs, _ = _attributes(classified, supplier_lt, planning_master=planning_master)
        assert attrs.set_index("sku").loc["THIN-1", "demand_mean"] == 50.0

    def test_crosscheck_frame_is_writable(self, classified, supplier_lt, item_master,
                                          planning_master, tmp_path):
        _, crosscheck = _attributes(classified, supplier_lt, item_master=item_master,
                                    planning_master=planning_master)
        frame = crosscheck.frame()
        assert set(frame.columns) >= {"sku", "attribute", "chosen_source", "other_source"}
        frame.to_csv(tmp_path / "xc.csv", index=False)

    def test_summary_names_a_systematic_gap(self, classified, supplier_lt,
                                            planning_master):
        _, crosscheck = _attributes(classified, supplier_lt,
                                    planning_master=planning_master)
        assert "planner worksheet" in crosscheck.summary()


class TestSourceResolver:

    def test_config_is_the_last_resort(self):
        resolver = SourceResolver()
        skus = pd.Series(["A", "B"])
        out = resolver.resolve(
            "lead_time_days", skus,
            {MEASURED: pd.Series([np.nan, np.nan]), ITEM_MASTER: pd.Series([30.0, np.nan])},
        )
        assert out["value"].tolist()[0] == 30.0
        assert pd.isna(out["value"].tolist()[1])

    def test_tolerance_boundary_is_exclusive(self):
        """Exactly at tolerance is not a disagreement — only beyond it is."""
        resolver = SourceResolver(tolerance=0.25)
        resolver.resolve(
            "lead_time_days", pd.Series(["A", "B"]),
            {MEASURED: pd.Series([60.0, 60.0]), ITEM_MASTER: pd.Series([45.0, 44.0])},
            sample_counts=pd.Series([9, 9]),
        )
        flagged = {d.sku for d in resolver.result.all_disagreements}
        assert flagged == {"B"}

    def test_all_sources_absent_yields_nothing(self):
        out = SourceResolver().resolve("lead_time_days", pd.Series(["A"]), {})
        assert out["value"].isna().all()
        assert out["source"].isna().all()

    def test_thin_measurement_is_rescued_when_nothing_else_exists(self):
        """Demoting a thin measurement must not throw it away."""
        out = SourceResolver(min_samples=5).resolve(
            "lead_time_days", pd.Series(["A"]),
            {MEASURED: pd.Series([40.0])},
            sample_counts=pd.Series([1]),
        )
        assert out["value"].iloc[0] == 40.0
        assert out["source"].iloc[0] == MEASURED


# ── Compare ──────────────────────────────────────────────────────────────────


class TestPlannerComparison:

    @pytest.fixture
    def suggestions(self, classified, supplier_lt, item_master, planning_master):
        attrs, _ = _attributes(classified, supplier_lt, item_master=item_master,
                               planning_master=planning_master)
        params = PlanningParameters(CONFIG_DIR / "planning_parameters.md")
        return SuggestionBuilder(CONFIG_DIR).build(params.resolve(attrs))

    def test_planner_safety_stock_is_compared_not_consumed(self, suggestions):
        row = suggestions.frame.set_index("sku").loc["MEASURED-1"]
        assert row["planner_safety_stock"] == 900.0
        # The justified figure is computed, so it must differ from what was handed in.
        assert row["suggested_safety_stock"] != 900.0
        assert row["ss_vs_planner_qty"] == pytest.approx(
            row["suggested_safety_stock"] - 900.0, abs=0.5
        )

    def test_overstock_and_understock_get_different_verdicts(self, suggestions):
        verdicts = set(suggestions.frame["ss_verdict"])
        assert verdicts <= {"planner holds more than justified", "in line",
                            "planner holds less than justified", ""}
        assert any(v for v in verdicts if v)

    def test_percentage_service_level_is_read_as_a_fraction(self, classified,
                                                            supplier_lt, planning_master):
        """A worksheet saying 98 means 98%, not a probability of 98."""
        attrs, _ = _attributes(classified, supplier_lt, planning_master=planning_master)
        assert attrs.set_index("sku").loc["MEASURED-1", "planner_service_level"] == 0.98

    def test_comparison_columns_absent_without_a_planning_master(self, classified,
                                                                 supplier_lt, item_master):
        attrs, _ = _attributes(classified, supplier_lt, item_master=item_master)
        params = PlanningParameters(CONFIG_DIR / "planning_parameters.md")
        result = SuggestionBuilder(CONFIG_DIR).build(params.resolve(attrs))
        assert "ss_verdict" not in result.frame.columns

    def test_summary_reports_the_planner_gap(self, suggestions):
        assert "safety stock the planner set by hand" in suggestions.summary()


# ── Lead-time backfill in the pipeline ───────────────────────────────────────


class TestLeadTimeBackfill:

    def test_unbought_skus_get_a_master_lead_time(self, item_master, tmp_path):
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        measured = pd.DataFrame({
            "sku": ["MEASURED-1"], "supplier": ["ACME"],
            "wma_lead_time_days": [60.0], "lt_std_days": [8.0], "sample_count": [9],
        })
        out = planner._fill_lead_time_from_masters(
            measured, ["MEASURED-1", "THIN-1", "NEVER-1"], item_master, None
        )
        never = out.set_index("sku").loc["NEVER-1"]
        assert never["wma_lead_time_days"] == 90.0
        assert never["sample_count"] == 0
        assert never["lt_source"] == "item_master"

    def test_measured_skus_are_not_overwritten(self, item_master, tmp_path):
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        measured = pd.DataFrame({
            "sku": ["MEASURED-1"], "supplier": ["ACME"],
            "wma_lead_time_days": [60.0], "lt_std_days": [8.0], "sample_count": [9],
        })
        out = planner._fill_lead_time_from_masters(
            measured, ["MEASURED-1"], item_master, None
        )
        assert out.set_index("sku").loc["MEASURED-1", "wma_lead_time_days"] == 60.0
        assert len(out) == 1

    def test_no_masters_leaves_the_frame_alone(self, tmp_path):
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        measured = pd.DataFrame({
            "sku": ["A"], "supplier": ["X"], "wma_lead_time_days": [30.0],
            "lt_std_days": [1.0], "sample_count": [5],
        })
        out = planner._fill_lead_time_from_masters(measured, ["A"], None, None)
        assert len(out) == 1

    def test_nonpositive_stated_lead_time_is_ignored(self, tmp_path):
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        bad = pd.DataFrame({"sku": ["A"], "lead_time_days": [0.0]})
        out = planner._fill_lead_time_from_masters(
            pd.DataFrame(columns=["sku", "wma_lead_time_days"]), ["A"], bad, None
        )
        assert len(out) == 0
