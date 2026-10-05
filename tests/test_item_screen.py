"""
One item, read out of a run's own workbook.

The screen is the results screen's constraint one layer down: it computes nothing. Every
figure is a cell the run wrote, and the chart plots those cells. A screen that
re-forecast to draw a line would be a second forecast, and the first time the two
disagreed the planner would have no way to tell which one was the plan.

Two lines, not the three that were intended. Nothing in this pipeline cleanses demand
history — the only outlier trim anywhere is on lead time, in `_prepare_po_history` — so
a "history as the model saw it" line would be the first line drawn twice, implying a
step that does not happen. The endpoint says so rather than leaving it to be noticed,
and that is asserted here, because the next person to build this screen will reach for
the third line again.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

pytest.importorskip("openpyxl")

from inventory_planning.reporting.read_workbook import (  # noqa: E402
    WorkbookUnreadable, read_row,
)


@pytest.fixture
def book(tmp_path):
    import openpyxl

    path = tmp_path / "planning_x.xlsx"
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Forecast"
    sheet.append(["sku", "model_used", "hist 2026-01", "fcst 2026-02"])
    sheet.append(["SKU-1", "SMA", 100, 110])
    # A material number Excel has made a number of, which is the ordinary case on a
    # real export and the one a strict comparison gets wrong.
    sheet.append([1003524, "Croston", 7, 8])
    wb.create_sheet("Parameters").append(["sku", "safety_stock", "safety_stock_suggested"])
    wb["Parameters"].append(["SKU-1", 40, 55])
    wb.save(path)
    return path


class TestFindingOneRow:

    def test_a_row_comes_back_keyed_by_column(self, book):
        row = read_row(book, "Forecast", "sku", "SKU-1")
        assert row["model_used"] == "SMA"
        assert row["hist 2026-01"] == 100

    def test_a_numeric_item_number_is_matched_as_text(self, book):
        """
        `1003524 == "1003524"` is False, and a material number is an identifier that
        happens to be digits often enough that Excel will have made some of them
        numbers. Matching strictly loses exactly the items a planner looks up by hand.
        """
        assert read_row(book, "Forecast", "sku", "1003524")["model_used"] == "Croston"

    def test_an_item_that_is_not_there_is_none_rather_than_an_error(self, book):
        assert read_row(book, "Forecast", "sku", "NOPE") is None

    def test_a_missing_sheet_says_which_sheets_there_are(self, book):
        with pytest.raises(WorkbookUnreadable) as refused:
            read_row(book, "Nope", "sku", "SKU-1")
        assert "Forecast" in str(refused.value)

    def test_a_missing_key_column_says_which_columns_there_are(self, book):
        with pytest.raises(WorkbookUnreadable) as refused:
            read_row(book, "Forecast", "item", "SKU-1")
        assert "sku" in str(refused.value)


class TestTheScreenSaysWhatItCannotShow:
    """
    The third line was in the design and is not in the data. A screen quietly drawing
    two where three were promised teaches the reader that the cleansing step happened.
    """

    def test_nothing_in_code_cleanses_demand(self):
        """
        Checked against the pipeline rather than trusted from a comment: if a cleansing
        step is ever added, this fails and the screen's note has to be rewritten.

        Matched on definitions and calls, not on the word. Three docstrings discuss
        outliers — the lead-time trim explains itself at length, and so it should — and
        a scan that read prose would have to be loosened until it caught nothing.
        """
        import re

        named = re.compile(
            r"^\s*(?:def\s+\w*(?:outlier|winsor|cleanse)\w*"
            r"|\w*(?:outlier|winsor|cleanse)\w*\s*=(?!=)"
            r"|\w*\.(?:winsorize|remove_outliers)\s*\()",
            re.I)
        hits = []
        for path in (Path(__file__).parents[1] / "inventory_planning").rglob("*.py"):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if named.match(line):
                    hits.append(f"{path.name}:{n} {line.strip()[:60]}")
        assert hits == [], (
            f"something now cleanses, so the item screen's note is wrong: {hits}")

    def test_the_only_trim_is_on_lead_time_and_it_blanks_rather_than_drops(self):
        """
        The one trim that does exist, and the shape of it. Dropping the row instead of
        blanking the value threw 63% of the ordered quantity off a real extract, and
        every analysis that counts what was bought read the remainder as the whole.
        """
        from types import SimpleNamespace

        import pandas as pd

        from inventory_planning.ingest_bridge import IngestBridge

        frame = pd.DataFrame({"sku": ["A", "B"], "po_qty": [10, 20],
                              "lead_time_days": [30, 900]})
        # Called unbound on a stub: constructing a bridge reads the config and the FX
        # table, and neither has anything to do with what this asserts.
        out = IngestBridge._prepare_po_history(SimpleNamespace(verbose=False), frame)
        assert len(out) == 2, "the row survives"
        assert pd.isna(out.loc[1, "lead_time_days"]), "the implausible lead time is gone"
        assert out.loc[1, "po_qty"] == 20, "and nothing else on the line is implicated"


class TestWhatCountsAsADisagreement:
    """
    The grid's default view. Every flag restates a column the run wrote — its own
    verdict on the policy, the suggestion engine's sentence, should-be against actual —
    because a screen that decided for itself which items matter would be a second
    opinion kept beside the engine's, and nothing would be comparing the two.

    No threshold anywhere. A cutoff would be a number invented by the interface, and
    the one thing this repository is consistent about is not inventing numbers.
    """

    @staticmethod
    def _flags(**row):
        from inventory_planning.api.app import _disagreements

        return _disagreements(row)

    def test_the_run_s_own_verdict_decides_the_policy_flag(self):
        assert "policy" in self._flags(policy_agrees="False")
        assert "policy" not in self._flags(policy_agrees="True")

    def test_a_workbook_boolean_may_arrive_as_text(self):
        """
        Cells come back through the workbook reader, which renders under the sheet's
        own number formats — so `False` is the string "False" by the time it is here,
        and `if not value` would have flagged every item in the catalogue.
        """
        for falsey in ("False", "false", "FALSE", "no", "0"):
            assert "policy" in self._flags(policy_agrees=falsey), falsey

    def test_an_absent_verdict_is_not_a_disagreement(self):
        """A run that wrote no policy column has not disagreed about anything."""
        assert self._flags(sku="A") == []
        assert self._flags(policy_agrees=None) == []

    def test_a_gap_of_zero_is_not_a_position_finding(self):
        assert "position" not in self._flags(gap_value=0)
        assert "position" in self._flags(gap_value=-75405.5)

    def test_the_word_none_is_not_a_suggested_change(self):
        """
        `changes_suggested` is a sentence, and a sheet writes an absent one as the text
        "None". Taken literally that is every item in the catalogue needing attention.
        """
        assert "parameters" not in self._flags(changes_suggested="None")
        assert "parameters" not in self._flags(changes_suggested="")
        assert "parameters" in self._flags(changes_suggested="review 7d → 30d")

    def test_a_number_survives_the_formatting_the_workbook_applied(self):
        from inventory_planning.api.app import _number

        assert _number("-75,405.55") == pytest.approx(-75405.55)
        assert _number("95%") == pytest.approx(95.0)
        assert _number("periodic") is None
        assert _number(True) is None, "a flag is not a quantity"
