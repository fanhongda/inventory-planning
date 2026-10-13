"""
The run as one workbook.

A run used to leave sixteen CSVs named for the stage that produced them. The question a
planner arrives with — what do I buy, and is this item in trouble — was answered by
joining four of them, and nobody joins four CSVs in a review meeting.

What these hold: the five sheets exist and are keyed so a planner can read a row without
decoding it against another file; the curated columns lead and nothing is dropped behind
them; and the S&IOP sheet sums to the rollup it replaced, because a detail sheet that
disagrees with its own total is worse than the two files it merged.
"""

import glob
import io
import contextlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.orchestrator import InventoryPlanner
from inventory_planning.reporting.workbook import (
    LEAD_COLUMNS, build_workbook, collect_sheets,
)

SAMPLE = Path(__file__).parents[1] / "sample_data"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("workbook")
    with contextlib.redirect_stdout(io.StringIO()):
        planner = InventoryPlanner(output_dir=out, interactive=False)
        loaded = planner.load_all(sorted(glob.glob(str(SAMPLE / "*.csv"))))
        results = planner.run_planning(**loaded)
        policy = planner.run_policy_analysis(
            results, inventory_df=results.get("inventory_consolidated"))
        planner.run_kpi_review(
            policy,
            sales_df=loaded.get("sales_df"),
            open_so_df=loaded.get("open_so_df"),
            open_po_df=loaded.get("open_po_df"),
            inventory_df=loaded.get("inventory_df"),
            po_history_df=loaded.get("po_history_df"),
        )
    return planner, results, policy, out


class TestTheFiveSheets:

    def test_every_sheet_is_built(self, run):
        _, results, policy, _ = run
        assert set(collect_sheets(results, policy)) == {
            "Forecast", "Parameters", "Purchase", "Inventory", "S&IOP"}

    def test_each_one_is_keyed_on_the_item(self, run):
        _, results, policy, _ = run
        for name, frame in collect_sheets(results, policy).items():
            assert frame.columns[0] == "sku", f"{name} does not lead with the item"

    def test_a_row_can_be_read_without_another_file(self, run):
        """A sheet keyed only on a material number is one a planner has to decode."""
        _, results, policy, _ = run
        for name, frame in collect_sheets(results, policy).items():
            assert "product_family" in frame.columns, name

    def test_the_curated_columns_lead(self, run):
        _, results, policy, _ = run
        for name, frame in collect_sheets(results, policy).items():
            wanted = [c for c, _ in LEAD_COLUMNS[name] if c in frame.columns]
            assert list(frame.columns[:len(wanted)]) == wanted

    def test_nothing_behind_them_is_dropped(self, run):
        """
        Curating the front is what makes a sheet readable. Dropping the rest would make
        this a lossier report than the CSVs it replaces, and the column somebody's
        spreadsheet depends on is never the one you would have guessed.
        """
        _, results, policy, _ = run
        sheets = collect_sheets(results, policy)
        for source, sheet in ((results["recommendations"], "Purchase"),
                              (policy["should_be"].frame, "Parameters")):
            missing = set(source.columns) - set(sheets[sheet].columns)
            assert not missing, f"{sheet} dropped {sorted(missing)}"


class TestTheSiopSheetIsTheRollup:
    """
    The by-family rollup existed and was the wrong shape: a gap by product line is a
    number to be explained, not acted on. Per item, summed, it *is* the rollup — and a
    detail sheet that disagrees with its own total is worse than the two files it
    merged.

    And one row per item, not one per item per period. The long form was the grain the
    projection is computed on, published unchanged: one item spread over six rows of
    positions, which with a real catalogue is thousands of rows for hundreds of
    decisions. These pin the reshape as lossless, because the only honest defence of a
    narrower-looking sheet is that nothing left it.
    """

    def test_it_is_one_row_per_item(self, run):
        _, results, policy, _ = run
        sheet = collect_sheets(results, policy)["S&IOP"]
        assert "period" not in sheet.columns, \
            "the period is the header here, not a column that repeats the item"
        assert len(sheet) == sheet["sku"].nunique() == len(policy["siop"].per_sku["sku"].unique())

    def test_every_cell_of_the_long_frame_is_still_there(self, run):
        """
        The reshape is a reshape. Each measure x period lands in its own column, and the
        value in it is the value the projection computed for that SKU in that period.
        """
        from inventory_planning.analytics.siop import SIOPPlan

        _, results, policy, _ = run
        sheet = collect_sheets(results, policy)["S&IOP"].set_index("sku")
        long = policy["siop"].per_sku
        for column, label in SIOPPlan.PHASED:
            for _, row in long.iterrows():
                cell = f"{label} {row['period']}"
                assert cell in sheet.columns, f"{cell} is not on the sheet"
                assert sheet.at[str(row["sku"]), cell] == pytest.approx(
                    row[column], rel=1e-9, abs=1e-9), cell

    def test_the_safety_floor_is_one_column_not_one_per_period(self, run):
        """The projection holds one floor per SKU; N copies of it would be N-1 of nothing."""
        _, results, policy, _ = run
        sheet = collect_sheets(results, policy)["S&IOP"]
        assert "safety_stock" in sheet.columns
        assert not [c for c in sheet.columns if str(c).startswith("safety qty ")]
        long = policy["siop"].per_sku
        assert (long.groupby("sku")["safety_stock"].nunique() == 1).all()

    def test_it_sums_to_the_period_totals(self, run):
        _, results, policy, _ = run
        sheet = collect_sheets(results, policy)["S&IOP"]
        by_period = policy["siop"].by_period.set_index("period")
        for period, row in by_period.iterrows():
            assert sheet[f"demand amt {period}"].sum() == pytest.approx(
                row["demand_cogs"], rel=1e-6)
            assert sheet[f"closing amt {period}"].sum() == pytest.approx(
                row["closing_value"], rel=1e-6)

    def test_the_purchase_behind_the_position_is_on_the_row(self, run):
        """
        Projected on-hand without the buy that produces it is not actionable. Committed
        supply is phased across the header; the action this cycle is one column.
        """
        _, results, policy, _ = run
        sheet = collect_sheets(results, policy)["S&IOP"]
        assert [c for c in sheet.columns if str(c).startswith("supply qty ")]
        assert {"recommended_action", "suggested_po_qty"} <= set(sheet.columns)

    def test_the_first_short_month_leads_the_row(self, run):
        """The one date on the row that decides whether anything has to happen."""
        _, results, policy, _ = run
        sheet = collect_sheets(results, policy)["S&IOP"].set_index("sku")
        long = policy["siop"].per_sku
        short = long[long["gap_qty"] > 0]
        expected = short.groupby("sku")["period"].min().astype(str)
        for sku, period in expected.items():
            assert sheet.at[str(sku), "first_short_period"] == period
        covered = set(long["sku"].astype(str)) - set(expected.index.astype(str))
        for sku in covered:
            assert pd.isna(sheet.at[sku, "first_short_period"]), \
                "an item covered across the horizon must read blank, not as month one"


class TestTheFileItself:

    def test_it_writes_and_carries_the_notes(self, run):
        import openpyxl

        _, results, policy, out = run
        path = build_workbook(out / "wb.xlsx", results, policy, currency="CNY")
        book = openpyxl.load_workbook(path)
        assert book.sheetnames == ["Forecast", "Parameters", "Purchase", "Inventory",
                                   "S&IOP", "How to read this"]
        notes = pd.read_excel(path, sheet_name="How to read this")
        assert notes["note"].str.contains("in CNY").any(), \
            "a workbook that does not name its currency is how one gets read as another"

    def test_every_sheet_freezes_its_header(self, run):
        import openpyxl

        _, results, policy, out = run
        book = openpyxl.load_workbook(build_workbook(out / "wb2.xlsx", results, policy))
        assert all(ws.freeze_panes == "A2" for ws in book.worksheets)

    def test_a_run_with_no_policy_stage_still_writes_what_it_has(self, run):
        """A run that stops at the gate should leave something readable, not a folder."""
        _, results, _, out = run
        sheets = collect_sheets(results, None)
        assert "Forecast" in sheets and "Purchase" in sheets
        assert build_workbook(out / "wb3.xlsx", results, None) is not None

    def test_nothing_to_report_is_not_an_empty_file(self, run):
        _, _, _, out = run
        assert build_workbook(out / "wb4.xlsx", {}, {}) is None
        assert not (out / "wb4.xlsx").exists()

    def test_the_run_writes_it_and_records_it(self, run):
        planner, _, _, out = run
        path = out / f"planning_{planner.run.run_id}.xlsx"
        assert path.exists()
        assert any(Path(o.name).name == path.name for o in planner.run.outputs), \
            "an output the manifest does not name has no provenance"


class TestOneNumberPerRow:

    def test_a_folded_frame_does_not_bring_a_second_copy_of_a_column(self, run):
        """
        Two `unit_cost` columns on one row is not extra information — it is a question
        about which one is real, on every row, for a reader who cannot answer it.
        """
        _, results, policy, _ = run
        for name, frame in collect_sheets(results, policy).items():
            duplicated = [c for c in frame.columns if list(frame.columns).count(c) > 1]
            assert not duplicated, f"{name} carries {duplicated} twice"


class TestTheRunAlsoLeavesThePicture:
    """
    The workbook is the artefact a meeting is handed; the review is the one it is read
    from. It was written to disk by a method nothing called, and the one output that
    could not be recovered from the numbers was therefore the one output a run did not
    produce.

    Recording it on the manifest is the other half: the results screen reads the
    manifest rather than listing the directory, deliberately, so a file that is on disk
    and off the index is a file that screen cannot show.
    """

    def test_the_review_is_written(self, run):
        planner, _, _, out = run
        reports = sorted(out.glob("kpi_review_*.html"))
        assert len(reports) == 1, "a run produced no visual review"
        body = reports[0].read_text(encoding="utf-8")
        assert body.count("<svg") >= 3, "a review with no charts is a second workbook"

    def test_it_is_on_the_run_s_own_index(self, run):
        from inventory_planning.provenance import RunRegistry

        planner, _, _, out = run
        manifest = RunRegistry(out).get(planner.run.run_id)
        names = [o["name"] for o in manifest["outputs"]]
        assert any(n.startswith("kpi_review_") and n.endswith(".html") for n in names)
