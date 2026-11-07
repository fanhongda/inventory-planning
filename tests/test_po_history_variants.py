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

A third shape, finer than the contract's `po_receipt_line` grain: a *confirmation*
export, where one PO line arrives once per vendor confirmation. SAP under confirmation
control gives two rows — `AB`, the order acknowledgement, with no receipt against it,
and `LA`, the shipping notification, which is what the goods receipt is posted to. The
lead time survives that untouched, because the acknowledgement row carries no receipt
date and the weighted average already drops it. The quantities do not: `po_qty` is a
line-level value repeated on both rows, so every ordering-pattern figure doubles, and
the grain test cannot see it because `receive_date` is in the natural key and the two
rows differ on exactly that column.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.registry import AdapterRegistry
from inventory_planning.readers.po_history_reader import (
    POHistoryReader,
    consolidate_to_receipt_grain,
)

SAMPLE_DIR = Path(__file__).parents[1] / "sample_data"
CONFIG_DIR = Path(__file__).parents[1] / "config"


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


# ── Third shape: a confirmation-grain export ──────────────────────────────────


def _confirmation_rows() -> pd.DataFrame:
    """
    Three received PO lines as a confirmation-grain export leaves them: each line twice,
    an `AB` row with no receipt and an `LA` row carrying it.

    `po_qty` is the same on both rows of a line, because it is a property of the line
    and not of the confirmation. That is the whole defect: it sums twice.
    """
    rows = []
    for po, sku, qty, ordered, received in (
        ("4500001", "SKU-001", 100, "2026-01-05", "2026-03-12"),
        ("4500002", "SKU-001", 200, "2026-02-05", "2026-04-10"),
        ("4500003", "SKU-002", 50, "2026-02-20", "2026-03-26"),
    ):
        common = {"po_number": po, "po_line_number": "10", "sku": sku,
                  "supplier": "V-ALPHA", "po_date": pd.Timestamp(ordered),
                  "po_qty": qty, "open_qty": 0}
        rows.append({**common, "confirmation_category": "AB", "receive_date": pd.NaT})
        rows.append({**common, "confirmation_category": "LA",
                     "receive_date": pd.Timestamp(received)})
    return pd.DataFrame(rows)


class TestAConfirmationGrainExport:

    def test_the_line_quantity_is_counted_once(self):
        """
        The defect: 350 units ordered across three lines read as 700, and every figure
        built on the ordering pattern — bought against demand, the over-ordered
        finding, lot-size conformance, the cadence — moved with it.
        """
        df = _confirmation_rows()
        assert df["po_qty"].sum() == 700
        out = consolidate_to_receipt_grain(df, verbose=False)
        assert out["po_qty"].sum() == 350
        assert len(out) == 3

    def test_the_row_that_carries_the_receipt_is_the_one_kept(self):
        out = consolidate_to_receipt_grain(_confirmation_rows(), verbose=False)
        assert out["receive_date"].notna().all()
        assert set(out["confirmation_category"]) == {"LA"}

    def test_the_lead_time_is_unchanged_by_the_consolidation(self):
        """
        It was already right, and it has to stay right: the acknowledgement row has no
        receipt date, so `lead_time_days` is null on it and the weighted average drops
        it. A consolidation that moved this number would be fixing one signal by
        breaking the one that was working.
        """
        reader = POHistoryReader(CONFIG_DIR)
        df = _confirmation_rows()
        df["lead_time_days"] = (df["receive_date"] - df["po_date"]).dt.days
        before = reader.compute_supplier_lt(df).set_index("sku")["wma_lead_time_days"]
        after = reader.compute_supplier_lt(
            consolidate_to_receipt_grain(df, verbose=False)
        ).set_index("sku")["wma_lead_time_days"]
        assert before.to_dict() == after.to_dict()
        assert after["SKU-002"] == 34.0

    def test_a_line_still_awaiting_its_shipment_is_kept(self):
        """
        An acknowledgement with no shipping notification yet is the only record of that
        line. Dropping every row without a receipt would delete it — which is why the
        rule is per line and not per row.
        """
        df = pd.concat([
            _confirmation_rows(),
            pd.DataFrame([{"po_number": "4500004", "po_line_number": "10",
                           "sku": "SKU-003", "supplier": "V-BETA",
                           "po_date": pd.Timestamp("2026-06-01"), "po_qty": 75,
                           "open_qty": 75, "confirmation_category": "AB",
                           "receive_date": pd.NaT}]),
        ], ignore_index=True)
        out = consolidate_to_receipt_grain(df, verbose=False)
        assert "SKU-003" in set(out["sku"])
        assert out[out["sku"] == "SKU-003"]["po_qty"].sum() == 75

    def test_partial_deliveries_against_one_line_are_both_kept(self):
        """
        Several receipts against one PO line are partial deliveries — legitimately
        separate rows at `po_receipt_line` grain, and not a duplication to collapse.
        """
        df = pd.DataFrame({
            "po_number": ["4500001", "4500001"], "po_line_number": ["10", "10"],
            "sku": ["SKU-001", "SKU-001"], "supplier": ["V-ALPHA", "V-ALPHA"],
            "po_date": [pd.Timestamp("2026-01-05")] * 2,
            "receive_date": pd.to_datetime(["2026-03-01", "2026-03-20"]),
            "po_qty": [100, 100],
        })
        out = consolidate_to_receipt_grain(df, verbose=False)
        assert len(out) == 2

    def test_an_export_with_no_receipts_at_all_is_left_alone(self):
        """
        The second shape this module is about. It is a valid record of what was ordered,
        how often and how much — and every one of its rows lacks a receipt, so a rule
        that dropped those would empty the file.
        """
        df = _sap_purchase_records(rows=40)
        out = consolidate_to_receipt_grain(df, verbose=False)
        assert len(out) == len(df)

    def test_nothing_is_dropped_without_a_line_to_reason_about(self):
        """No `po_number`, no line — and a guess here deletes real receipts."""
        df = pd.DataFrame({
            "sku": ["SKU-001", "SKU-001"], "supplier": ["V", "V"],
            "po_date": [pd.Timestamp("2026-01-05")] * 2,
            "receive_date": [pd.NaT, pd.Timestamp("2026-03-01")],
            "po_qty": [100, 100],
        })
        out = consolidate_to_receipt_grain(df, verbose=False)
        assert len(out) == 2

    def test_the_finding_names_the_codes_it_found(self, capsys):
        """
        Reported, not silent, and worded off the data: a planner whose system spells
        the category differently should see their own code, not the SAP one asserted
        at them.
        """
        consolidate_to_receipt_grain(_confirmation_rows(), verbose=True)
        out = capsys.readouterr().out
        assert "coded AB" in out
        assert "350" in out        # the quantity that would have doubled

    def test_the_reader_applies_it(self):
        """
        Both entry points have to clean, because the intake path does not run
        `_post_process` — the contract derives `lead_time_days` itself.
        """
        df = _confirmation_rows()
        processed = POHistoryReader(CONFIG_DIR)._post_process(df)
        assert len(processed) == 3
        assert processed["po_qty"].sum() == 350
