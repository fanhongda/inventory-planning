"""
The results screen: one artefact, two renderings.

INTERFACE.md §7 revised the argument against this screen. The objection — that a page
showing the same numbers becomes a second place they are formatted, rounded and subtly
disagreed about — holds only if the page re-derives them, and it does not: it renders the
run's own workbook, located from the manifest.

So what is pinned here is the absence of a second opinion. The cells come back as the
workbook holds them. The numbers are displayed under the workbook's own number formats,
with Excel's rounding rather than Python's, because a page that showed 0 where the file
shows 1 would be the disagreement in its purest form. And the formats the writer emits
are exactly the ones the reader renders — the one test that keeps the two files honest
without making them one file.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

openpyxl = pytest.importorskip("openpyxl")

from inventory_planning.reporting import workbook as writer          # noqa: E402
from inventory_planning.reporting.read_workbook import (             # noqa: E402
    WorkbookUnreadable, read_sheet, render_number, sheets,
)


def _workbook(path, rows, formats=None, sheet="Purchase"):
    """A workbook shaped like the real one: a header, values, formats per column."""
    book = openpyxl.Workbook()
    page = book.active
    page.title = sheet
    page.append(list(rows[0]))
    for row in rows[1:]:
        page.append(list(row))
    for index, fmt in (formats or {}).items():
        for cell in list(page.iter_rows(min_row=2, min_col=index + 1,
                                        max_col=index + 1)):
            cell[0].number_format = fmt
    book.save(path)
    return path


class TestTheWriterAndTheReaderAgree:
    """
    The one test that keeps two files honest without making them one file. Add a fifth
    number format to `workbook.py` and this fails here, which is where someone would
    otherwise find out months later, from a column that reads differently on screen
    than in the file.
    """

    def test_every_format_the_writer_emits_is_one_the_reader_renders(self):
        emitted = {value for name, value in vars(writer).items()
                   if name.isupper() and isinstance(value, str) and "0" in value}
        assert emitted, "the writer's format constants moved; this guard is looking "\
                        "in the wrong place"
        unrendered = [f for f in emitted if render_number(1234.5678, f) is None]
        assert unrendered == []

    def test_every_format_named_in_a_lead_column_is_one_of_them(self):
        used = {fmt for columns in writer.LEAD_COLUMNS.values()
                for _, fmt in columns if fmt}
        unrendered = [f for f in used if render_number(1.5, f) is None]
        assert unrendered == []


class TestNumbersAreShownAsTheWorkbookShowsThem:

    @pytest.mark.parametrize("value,fmt,expected", [
        (12345.678, "#,##0", "12,346"),
        (12345.678, "#,##0.0", "12,345.7"),
        (0.2327, "0.00", "0.23"),
        (0.45, "0%", "45%"),
        (-172669.7, "#,##0", "-172,670"),
        (0, "#,##0", "0"),
    ])
    def test_the_four_formats_the_pipeline_writes(self, value, fmt, expected):
        assert render_number(value, fmt) == expected

    def test_rounding_is_excels_and_not_pythons(self):
        """
        `round(0.5)` is 0 in Python — half to even — and 1 in Excel. A screen that
        showed 0 where the file shows 1 would be the disagreement this screen exists
        to avoid, arrived at by being clever rather than by computing.
        """
        assert round(0.5) == 0
        assert render_number(0.5, "#,##0") == "1"
        assert render_number(1.5, "#,##0") == "2"
        assert render_number(-0.5, "#,##0") == "-1"

    def test_a_format_it_does_not_understand_is_not_guessed(self, tmp_path):
        """
        None, and the column is named, so the screen can say the display differs. A
        wrong rendering is worse than an unrendered one, because the first is believed.
        """
        assert render_number(1234.5, '"$"#,##0.00;[Red]-"$"#,##0.00') is None

        path = _workbook(tmp_path / "w.xlsx", [["sku", "value"], ["A", 1234.5]],
                         formats={1: '[Red]0.000'})
        sheet = read_sheet(path, "Purchase")
        assert sheet.unformatted == ["value"]
        assert sheet.rows[0][1] == 1234.5          # the raw value, not a guess


class TestItReadsAndDoesNotRecompute:

    def test_the_rows_are_the_rows(self, tmp_path):
        path = _workbook(tmp_path / "w.xlsx", [
            ["sku", "qty"], ["A", 1], ["B", 2], ["C", 3]], formats={1: "#,##0"})
        sheet = read_sheet(path, "Purchase")
        assert sheet.columns == ["sku", "qty"]
        assert sheet.rows == [["A", "1"], ["B", "2"], ["C", "3"]]
        assert sheet.total == 3

    def test_a_blank_cell_stays_blank(self, tmp_path):
        """
        Not zero, and not "—". A missing figure and a zero are different findings, and
        filling one in would be the screen deciding something the run did not.
        """
        path = _workbook(tmp_path / "w.xlsx", [["sku", "qty"], ["A", None]],
                         formats={1: "#,##0"})
        assert read_sheet(path, "Purchase").rows == [["A", None]]

    def test_the_column_order_is_the_files(self, tmp_path):
        """
        `workbook.py` leads each sheet with a curated order and puts everything else to
        the right. Re-ordering here would be a second view of what matters.
        """
        path = _workbook(tmp_path / "w.xlsx", [["z", "a", "m"], [1, 2, 3]])
        assert read_sheet(path, "Purchase").columns == ["z", "a", "m"]

    def test_a_formula_is_reported_and_not_evaluated(self, tmp_path):
        """
        `workbook.py` writes none, so this never fires on a real one — but evaluating
        one would be the screen computing, which is the line not to cross.
        """
        book = openpyxl.Workbook()
        page = book.active
        page.title = "Purchase"
        page.append(["sku", "total"])
        page.append(["A", "=1+1"])
        book.save(tmp_path / "w.xlsx")
        assert read_sheet(tmp_path / "w.xlsx", "Purchase").rows == [["A", "=1+1"]]

    def test_paging_is_a_window_and_not_a_sample(self, tmp_path):
        rows = [["sku", "qty"]] + [[f"S-{i:03d}", i] for i in range(10)]
        path = _workbook(tmp_path / "w.xlsx", rows, formats={1: "#,##0"})

        first = read_sheet(path, "Purchase", offset=0, limit=4)
        second = read_sheet(path, "Purchase", offset=4, limit=4)
        assert [r[0] for r in first.rows] == ["S-000", "S-001", "S-002", "S-003"]
        assert [r[0] for r in second.rows] == ["S-004", "S-005", "S-006", "S-007"]
        assert first.total == second.total == 10
        assert second.offset == 4


class TestWhenTheFileIsNotThere:

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(WorkbookUnreadable, match="not there any more"):
            sheets(tmp_path / "gone.xlsx")

    def test_a_file_that_is_not_a_workbook(self, tmp_path):
        path = tmp_path / "notes.xlsx"
        path.write_text("this is not a zip", encoding="utf-8")
        with pytest.raises(WorkbookUnreadable, match="could not be opened"):
            sheets(path)

    def test_a_sheet_that_is_not_there_names_the_ones_that_are(self, tmp_path):
        path = _workbook(tmp_path / "w.xlsx", [["sku"], ["A"]])
        with pytest.raises(WorkbookUnreadable, match="Purchase"):
            read_sheet(path, "Forecast")


class TestTheRealWorkbookReadsBack:
    """
    Against a workbook `workbook.py` actually wrote, not a hand-built stand-in — the
    stand-in cannot catch a change to how the writer lays a sheet out.
    """

    @pytest.fixture
    def built(self, tmp_path):
        import pandas as pd

        frame = pd.DataFrame({
            "sku": ["A-1", "A-2"],
            "actual_value": [52385.8, 17061.03],
            "should_be_dioh": [87.6118795768918, 88.3838383838384],
            "coverage_ratio": [0.2327683615819209, 0.2076923076923077],
        })
        path = tmp_path / "planning.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as excel:
            frame.to_excel(excel, sheet_name="Inventory", index=False)
            writer._format_sheet(excel.book["Inventory"], frame, "Inventory")
        return path

    def test_the_sheet_is_listed_with_its_size(self, built):
        listed = {s["name"]: s for s in sheets(built)}
        assert listed["Inventory"]["rows"] == 2

    def test_the_money_and_ratio_columns_read_as_the_file_displays_them(self, built):
        sheet = read_sheet(built, "Inventory")
        by_name = dict(zip(sheet.columns, sheet.rows[0]))
        assert by_name["actual_value"] == "52,386"
        assert by_name["should_be_dioh"] == "87.6"
        assert by_name["coverage_ratio"] == "0.23"
        assert sheet.unformatted == []

    def test_the_header_row_is_not_counted_as_a_row(self, built):
        assert read_sheet(built, "Inventory").total == 2
