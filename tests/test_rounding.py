"""
Whole units — and the direction of the test that decides whether it ever fires.

A quantity the business counts cannot be fractional, and 33.4 travels from the forecast
into the safety stock and the reorder point built on it. What makes this worth a test
file is not the arithmetic but the direction: rounding *unless* the unit says otherwise
is the only version that does anything on the extracts this plans from, none of which
carry a unit of measure at all.
"""

import numpy as np
import pandas as pd
import pytest

from inventory_planning.analytics.rounding import (
    DEFAULT_CONTINUOUS_UOM, INTEGER, NONE, Rounding, uom_by_sku,
)
from inventory_planning.policy.parameters import PlanningParameters


@pytest.fixture
def rounding():
    return Rounding.from_conventions({})


class TestTheDirectionOfTheTest:

    def test_no_unit_at_all_is_rounded(self, rounding):
        """
        The case the whole convention exists for. Of the four exports this plans from,
        the 83-column master carries no unit, the stock export carries none and the
        sales history carries none — so a rule that fired only on a stated unit would
        fire on nothing.
        """
        out = rounding.apply(pd.Series([33.4, 33.6]), None)
        assert out.tolist() == [33.0, 34.0]
        assert rounding.report["rounded_with_no_unit"] == 2
        assert rounding.report["rounded_on_a_stated_unit"] == 0

    def test_a_counted_unit_is_rounded_and_counted_as_stated(self, rounding):
        out = rounding.apply(pd.Series([33.4, 12.5]), pd.Series(["EA", "PC"]))
        assert out.tolist() == [33.0, 12.0]     # 12.5 → 12 under banker's rounding
        assert rounding.report["rounded_on_a_stated_unit"] == 2

    def test_a_measured_unit_keeps_its_decimals(self, rounding):
        out = rounding.apply(pd.Series([12.45, 3.5]), pd.Series(["KG", "L"]))
        assert out.tolist() == [12.4, 3.5]
        assert rounding.report["left_fractional"] == 2

    def test_an_unfamiliar_code_errs_towards_a_whole_number(self, rounding):
        """
        Not in the continuous list, so countable — a whole number nobody can misread
        beats a decimal nobody can order.
        """
        assert rounding.apply(pd.Series([7.6]), pd.Series(["ZZ"])).tolist() == [8.0]

    def test_a_blank_unit_counts_as_no_unit_not_as_a_code(self, rounding):
        rounding.apply(pd.Series([1.4, 1.4]), pd.Series(["", None]))
        assert rounding.report["rounded_with_no_unit"] == 2

    def test_nothing_is_rounded_when_the_convention_is_off(self):
        off = Rounding.from_conventions({"quantity_rounding": NONE})
        assert off.apply(pd.Series([33.44]), None).tolist() == [33.4]
        assert off.report["rounded_with_no_unit"] == 0

    def test_an_unknown_mode_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="quantity_rounding"):
            Rounding.from_conventions({"quantity_rounding": "maybe"})

    def test_a_missing_quantity_stays_missing(self, rounding):
        out = rounding.apply(pd.Series([np.nan, 2.6]), None)
        assert bool(pd.isna(out.iloc[0])) and out.iloc[1] == 3.0


class TestTheConventionLivesInTheParameterFile:

    def test_the_shipped_file_turns_it_on(self):
        params = PlanningParameters("config/planning_parameters.md")
        assert params.conventions["quantity_rounding"] == INTEGER
        assert "KG" in params.conventions["continuous_uom"]

    def test_a_declared_unit_list_replaces_the_default(self):
        only_kg = Rounding.from_conventions({"continuous_uom": ["kg"]})
        assert only_kg.continuous == frozenset({"KG"})
        assert only_kg.apply(pd.Series([1.4]), pd.Series(["L"])).tolist() == [1.0]

    def test_the_default_list_is_used_when_none_is_declared(self, rounding):
        assert rounding.continuous == frozenset(DEFAULT_CONTINUOUS_UOM)


class TestWhichDocumentSuppliesTheUnit:

    def test_the_first_frame_given_wins(self):
        """
        A master's unit is the item's; a stock row's is whatever that warehouse
        happened to record. The caller passes them in that order.
        """
        units = uom_by_sku(
            pd.DataFrame({"sku": ["A"], "uom_raw": ["EA"]}),
            pd.DataFrame({"sku": ["A", "B"], "uom": ["KG", "PC"]}),
        )
        assert units["A"] == "EA" and units["B"] == "PC"

    def test_frames_without_a_unit_column_are_skipped(self):
        units = uom_by_sku(pd.DataFrame({"sku": ["A"], "qty": [1]}),
                           pd.DataFrame({"sku": ["A"], "uom_raw": ["EA"]}))
        assert units["A"] == "EA"

    def test_nothing_supplied_gives_an_empty_answer_not_an_error(self):
        assert uom_by_sku(None, pd.DataFrame()).empty


class TestItReachesTheFiguresAPlannerActsOn:

    def _run(self, tmp_path, inventory="sample_data/inventory.csv"):
        from inventory_planning.orchestrator import InventoryPlanner

        planner = InventoryPlanner(output_dir=str(tmp_path), interactive=False)
        sales, _ = planner.load_sales_history("sample_data/sales_history.csv")
        po, _ = planner.load_po_history("sample_data/po_history.csv")
        so, _ = planner.load_open_so("sample_data/open_so.csv")
        open_po, _ = planner.load_open_po("sample_data/open_po.csv")
        inv, _ = planner.load_inventory(inventory)
        from inventory_planning.ingest_bridge import IngestBridge
        masters = IngestBridge(verbose=False).load(["sample_data/item_master.csv"])
        planner.absorb_intake(masters)
        results = planner.run_planning(sales, po, so, open_po, inv,
                                       item_master_df=masters.get("item_master_df"))
        return planner, results

    def test_the_forecast_and_the_reorder_point_come_back_whole(self, tmp_path):
        planner, results = self._run(tmp_path)
        forecast = results["forecast_detail"]
        assert (forecast["forecast_qty"].dropna() % 1 == 0).all()

        recommendations = results["recommendations"]
        for column in ("reorder_point", "suggested_min_qty", "suggested_max_qty"):
            if column in recommendations.columns:
                values = pd.to_numeric(recommendations[column], errors="coerce").dropna()
                assert (values % 1 == 0).all(), f"{column} still fractional"

    def test_one_convention_object_counts_the_whole_run(self, tmp_path):
        """
        Every stage that produces a quantity reports into the same counters. Three of
        five producers reporting would state a number that is simply wrong, in the
        report whose purpose is figures a reader can trust.
        """
        planner, _ = self._run(tmp_path)
        report = planner.rounding.report
        assert report["rounded_on_a_stated_unit"] > 0
        assert "Quantities rounded to whole units" in planner.rounding.summary()

    def test_an_extract_with_no_unit_says_what_it_assumed(self, tmp_path):
        frame = pd.read_csv("sample_data/inventory.csv").drop(columns=["Base Unit"])
        path = tmp_path / "inventory_no_uom.csv"
        frame.to_csv(path, index=False)

        planner, _ = self._run(tmp_path, inventory=str(path))
        assert planner.rounding.report["rounded_with_no_unit"] > 0
        assert planner.rounding.report["rounded_on_a_stated_unit"] == 0
        assert "no unit of measure" in planner.rounding.summary()
