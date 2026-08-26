"""
The natural key as an identity, not just a grain hint.

Until now the key was only ever asked one question — "does this frame have more rows
than distinct keys, and therefore need a rollup?" — and every call site answered it on
whatever part of the key happened to be present. That is the right answer for planning:
a sales history with no order number still forecasts.

It is the wrong answer for a store. Superseding a row by its key needs the key to
identify one row, and a key assembled from whatever mapped identifies a different
number of rows in each file. These tests pin the two behaviours apart: the frame still
loads and plans on a partial key, and `KeyStatus` says so rather than the loss going
unrecorded.
"""

import pandas as pd
import pytest

from inventory_planning.ingest.contract import (
    KEY_COMPLETE, KEY_DEGRADED, KEY_UNAVAILABLE, default_registry,
)
from inventory_planning.ingest.contract_tests import ContractTester, SEVERITY_WARN
from inventory_planning.ingest.intake import Intake


@pytest.fixture
def registry():
    return default_registry()


@pytest.fixture
def sap_schedule_lines():
    """One PO line (Item 10) delivered in two tranches, plus two unrelated lines."""
    return pd.DataFrame({
        "Purch.Doc.":      ["4500001", "4500001", "4500002", "4500003"],
        "Item":            ["10", "10", "10", "20"],
        "Sched. Line":     ["1", "2", "1", "1"],
        "Material":        ["ABC-1", "ABC-1", "ABC-2", "ABC-3"],
        "Vendor":          ["V100", "V100", "V200", "V100"],
        "Scheduled Qty":   ["600", "600", "300", "450"],
        "Qty Delivered":   ["0", "0", "0", "450"],
        "Del. Ind.":       ["", "", "", "CLSD"],
        "Doc Date":        ["05.03.2026", "05.03.2026", "17.03.2026", "22.03.2026"],
        "Stat. Del. Date": ["20.09.2026", "25.09.2026", "28.09.2026", "30.09.2026"],
        "Net Value":       ["1.200,50", "1.200,50", "900,00", "1.350,00"],
        "Incoterms":       ["EXW", "EXW", "FOB", "CIF"],
    })


class TestTheKeyIsCountedInLines:
    def test_the_document_line_leads_the_key(self, registry):
        """`po_line` is counted in lines, so the line number is part of its identity."""
        assert registry.get("open_po").natural_key == ["po_number", "po_line_number", "sku"]

    def test_the_receipt_keeps_its_date(self, registry):
        contract = registry.get("po_history")
        assert contract.natural_key == ["po_number", "po_line_number", "sku", "receive_date"]

    def test_a_master_still_keys_on_the_item_alone(self, registry):
        assert registry.get("item_master").natural_key == ["sku"]


class TestScheduleLineIsNotLineNumber:
    """
    EKET-ETENR is a finer grain than EKPO-EBELP. While both answered to
    `po_line_number` the winner was decided by whichever column the profile favoured,
    so the same export keyed on `Item` at full size and on `Sched. Line` when sampled.
    """

    def test_each_column_lands_in_its_own_field(self, registry):
        contract = registry.get("open_po")
        assert "item" in contract.field("po_line_number").aliases
        assert "sched. line" in contract.field("po_schedule_line").aliases
        assert "sched. line" not in contract.field("po_line_number").aliases

    def test_the_key_column_does_not_move_when_the_file_is_sampled(
        self, tmp_path, sap_schedule_lines
    ):
        sap_schedule_lines.head(2).to_excel(tmp_path / "sample.xlsx", index=False)
        sap_schedule_lines.to_excel(tmp_path / "full.xlsx", index=False)

        mapped = {}
        for name in ("sample.xlsx", "full.xlsx"):
            doc = Intake(verbose=False).load_files([tmp_path / name]).get("open_po")
            mapped[name] = doc.adapter.column_map.get("po_line_number")

        assert mapped["sample.xlsx"] == mapped["full.xlsx"] == "Item"

    def test_the_two_tranches_still_roll_up_into_one_line(self, tmp_path, sap_schedule_lines):
        """The finer grain is summed away, as before — 600 + 600 on one PO line."""
        sap_schedule_lines.to_excel(tmp_path / "full.xlsx", index=False)
        doc = Intake(verbose=False).load_files([tmp_path / "full.xlsx"]).get("open_po")
        line = doc.frame[doc.frame["po_number"] == "4500001"]
        assert len(line) == 1
        assert line["open_qty"].iloc[0] == 1200


class TestKeyStatus:
    def test_a_full_export_is_storable(self, registry):
        status = registry.get("open_po").key_status(
            ["po_number", "po_line_number", "sku", "open_qty"]
        )
        assert status.verdict == KEY_COMPLETE
        assert status.storable

    def test_a_missing_line_number_degrades_rather_than_fails(self, registry):
        """The common export. It plans fine; it just cannot be superseded by key."""
        status = registry.get("open_po").key_status(["po_number", "sku", "open_qty"])
        assert status.verdict == KEY_DEGRADED
        assert status.missing == ["po_line_number"]
        assert not status.storable

    def test_nothing_mapped_is_its_own_verdict(self, registry):
        status = registry.get("open_po").key_status(["open_qty"])
        assert status.verdict == KEY_UNAVAILABLE
        assert not status.storable

    def test_it_names_what_it_lost(self, registry):
        status = registry.get("open_po").key_status(["po_number", "sku"])
        assert "po_line_number did not map" in status.describe()


class TestTheGrainTestStopsPassingWhatItDidNotCheck:
    def test_an_unverifiable_grain_is_not_a_pass(self, registry):
        contract = registry.get("open_po")
        df = pd.DataFrame({"open_qty": [10.0, 5.0]})
        report = ContractTester().run(df, contract)
        grain = next(r for r in report.results if r.name.startswith("grain:"))
        assert not grain.passed
        assert "grain unverified" in grain.detail

    def test_an_unverifiable_grain_is_a_warning_not_an_error(self, registry):
        """
        Flipping it to a failure must not start blocking runs. The frame below fails
        anyway — a document with no `sku` breaks a required field — so the assertion is
        on the grain result itself, which is what governs whether it can block.
        """
        contract = registry.get("open_po")
        df = pd.DataFrame({"open_qty": [10.0, 5.0]})
        report = ContractTester().run(df, contract)
        grain = next(r for r in report.results if r.name.startswith("grain:"))
        assert grain.severity == SEVERITY_WARN
        assert not grain.is_error

    def test_the_report_carries_the_key_verdict(self, registry):
        contract = registry.get("open_po")
        df = pd.DataFrame({"po_number": ["P1", "P2"], "sku": ["A", "B"],
                           "open_qty": [10.0, 5.0]})
        report = ContractTester().run(df, contract)
        assert report.key_status.verdict == KEY_DEGRADED
        assert report.to_dict()["key"]["storable"] is False

    def test_a_degraded_key_does_not_inflate_the_warning_count(self, registry):
        """It is a property of the load, not an assertion about the rows."""
        contract = registry.get("open_po")
        df = pd.DataFrame({"po_number": ["P1", "P2"], "sku": ["A", "B"],
                           "open_qty": [10.0, 5.0]})
        report = ContractTester().run(df, contract)
        assert not any(w.name.startswith("key") for w in report.warnings)
