"""
Purchase-order history arrives in two shapes, and both have to work.

  with receipt dates     the common case — lead time is measurable, and everything
                         downstream that depends on it keeps working
  without receipt dates  an SAP purchase-record export: order date, order quantity,
                         no goods receipt, often mixed with recent open orders. A
                         real record of *what was ordered, how often and how much*,
                         and no record at all of how long anything took

The second used to be rejected outright — `lead_time_days` was a required field, and
without a receipt date it is neither present nor derivable. Rejected there, it was
picked up by `open_po` instead and concatenated into the genuine open-PO extract,
counting hundreds of long-closed orders as inbound supply. The position inflates and
the pipeline stops recommending purchases, with no error anywhere.

So: admit it, route it to `po_history`, and withhold the one capability it cannot
back. A document type declares what it *can* supply; only the loaded frame says what
actually arrived.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.registry import AdapterRegistry

SAMPLE_DIR = Path(__file__).parents[1] / "sample_data"


@pytest.fixture(scope="module")
def registry():
    return AdapterRegistry()


def _sap_purchase_records(rows: int = 120) -> pd.DataFrame:
    """SAP purchase-record export: no goods-receipt date, no delivery schedule."""
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "Material": rng.choice([f"M-{i:04d}" for i in range(30)], rows),
        "Vendor Name": rng.choice(["ACME", "GLOBEX"], rows),
        "Purch.Doc.": [f"45{i:06d}" for i in range(rows)],
        "Doc. Date": pd.to_datetime("2025-01-01")
                     + pd.to_timedelta(rng.integers(0, 500, rows), "D"),
        "Order Quantity": rng.integers(10, 500, rows),
        "Net Value": rng.integers(100, 9000, rows),
    }).astype(str)


def _open_po_extract(rows: int = 40) -> pd.DataFrame:
    """A genuine open-PO extract: says something about what is still outstanding."""
    rng = np.random.default_rng(12)
    return pd.DataFrame({
        "Material": rng.choice([f"M-{i:04d}" for i in range(30)], rows),
        "Vendor": rng.choice(["ACME", "GLOBEX"], rows),
        "PO No.": [f"46{i:06d}" for i in range(rows)],
        "Ordered Qty": rng.integers(10, 300, rows),
        "Delivered Qty": rng.integers(0, 50, rows),
        "Planned Del Date": pd.to_datetime("2026-08-15")
                            + pd.to_timedelta(rng.integers(0, 90, rows), "D"),
    }).astype(str)


# ── With receipt dates: nothing may regress ──────────────────────────────────


class TestWithReceiptDates:

    def test_still_routes_to_po_history(self, registry):
        raw = pd.read_csv(SAMPLE_DIR / "po_history.csv", dtype=str)
        assert registry.route(raw, source_name="po_history.csv").doc_type == "po_history"

    def test_lead_time_signal_is_supplied(self):
        result = Intake(verbose=False).load_files([SAMPLE_DIR / "po_history.csv"])
        assert result.plan.has("lead_time_signal")
        assert result.plan.source_of("lead_time_signal") == "po_history"

    def test_nothing_is_withheld(self):
        result = Intake(verbose=False).load_files([SAMPLE_DIR / "po_history.csv"])
        withheld = [c for d, c in result.plan.withheld if d == "po_history"]
        assert "lead_time_signal" not in withheld


# ── Without receipt dates: admitted, but honest about it ─────────────────────


class TestWithoutReceiptDates:

    def test_routes_to_po_history_not_open_po(self, registry):
        assert registry.route(_sap_purchase_records(),
                              source_name="Purchase History.xlsx").doc_type == "po_history"

    def test_it_still_supplies_the_ordering_pattern(self, tmp_path):
        """The reason the planner exported it in the first place."""
        path = tmp_path / "Purchase History.xlsx"
        _sap_purchase_records().to_excel(path, index=False)
        result = Intake(verbose=False).load_files([path])
        assert result.plan.has("order_pattern_signal")

    def test_lead_time_signal_is_withheld(self, tmp_path):
        """
        Claiming it would report a measured lead time that was never measured, and
        suppress the item-master fallback that should cover for it.
        """
        path = tmp_path / "Purchase History.xlsx"
        _sap_purchase_records().to_excel(path, index=False)
        result = Intake(verbose=False).load_files([path])
        assert not result.plan.has("lead_time_signal")
        assert ("po_history", "lead_time_signal") in result.plan.withheld

    def test_the_shortfall_is_stated_in_the_summary(self, tmp_path):
        path = tmp_path / "Purchase History.xlsx"
        _sap_purchase_records().to_excel(path, index=False)
        summary = Intake(verbose=False).load_files([path]).summary()
        assert "carries no data for lead_time_signal" in summary

    def test_the_master_can_still_supply_lead_time(self, tmp_path):
        """With a receipt-less history, the item master is the remaining source."""
        _sap_purchase_records().to_excel(tmp_path / "Purchase History.xlsx", index=False)
        pd.DataFrame({
            "Material": [f"M-{i:04d}" for i in range(30)],
            "Planned Deliv. Time": [45] * 30,
            "MOQ": [50] * 30,
        }).to_excel(tmp_path / "master.xlsx", index=False)

        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert result.plan.source_of("lead_time_signal") == "item_master"


# ── The collision that inflated inbound supply ───────────────────────────────


class TestPurchaseHistoryDoesNotBecomeInboundSupply:

    @pytest.fixture
    def both_files(self, tmp_path):
        _sap_purchase_records().to_excel(tmp_path / "Purchase History.xlsx", index=False)
        _open_po_extract().to_excel(tmp_path / "open PO.xlsx", index=False)
        return sorted(tmp_path.iterdir())

    def test_they_land_on_different_document_types(self, both_files):
        result = Intake(verbose=False).load_files(both_files)
        assert set(result.documents) == {"po_history", "open_po"}

    def test_the_open_po_extract_is_the_inbound_source(self, both_files):
        result = Intake(verbose=False).load_files(both_files)
        assert "open PO" in result.documents["open_po"].source_name

    def test_inbound_is_not_inflated_by_purchase_history(self, both_files):
        """
        The failure this guards: closed orders counted as stock on the way.

        The count is bounded by the open-PO extract rather than equal to it — the
        contract's default filters legitimately drop fully-delivered lines. What
        matters is that the 120 purchase records are nowhere in it.
        """
        result = Intake(verbose=False).load_files(both_files)
        rows = result.documents["open_po"].row_count
        assert 0 < rows <= 40, f"{rows} rows — purchase history has leaked into inbound"

    def test_an_open_po_extract_must_say_something_about_openness(self, registry):
        """
        Order date and quantity alone describe what was ordered, not what is still
        coming. Without a delivered qty, open qty, schedule or status it is history.
        """
        contract = registry.contracts.get("open_po")
        assert contract.identifying_any
        assert set(contract.identifying_any[0]) >= {"delivered_qty", "committed_delivery"}


# ── Different reports must never be concatenated ─────────────────────────────


class TestPartitionGuard:

    def test_files_with_different_layouts_are_not_partitions(self, tmp_path):
        """
        Two files claiming one doc_type with barely-overlapping keys used to read as a
        clean partition and get concatenated, doubling every quantity.
        """
        _sap_purchase_records(60).to_excel(tmp_path / "a history.xlsx", index=False)
        _open_po_extract(30).to_excel(tmp_path / "b open.xlsx", index=False)
        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        for doc in result.documents.values():
            assert " + " not in doc.source_name, "two different reports were welded together"

    def test_genuine_partitions_of_one_export_still_concatenate(self, tmp_path):
        """An AU tab and an NZ tab share a layout and must not be split apart."""
        au = _open_po_extract(20)
        nz = _open_po_extract(20)
        nz["Material"] = [f"NZ-{i:04d}" for i in range(20)]
        nz["PO No."] = [f"99{i:06d}" for i in range(20)]
        au.to_excel(tmp_path / "open PO AU.xlsx", index=False)
        nz.to_excel(tmp_path / "open PO NZ.xlsx", index=False)

        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert result.documents["open_po"].row_count == 40

    def test_a_dropped_file_says_why_and_how_to_fix_it(self, tmp_path):
        same_shape = _open_po_extract(30)
        subset = same_shape.head(10)
        same_shape.to_excel(tmp_path / "full.xlsx", index=False)
        subset.to_excel(tmp_path / "sample.xlsx", index=False)
        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert result.failures, "the redundant file must be reported, not silently used"
