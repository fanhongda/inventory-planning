"""
Telling a purchase history from an open-PO extract.

Reported from production: an SAP purchase-record export kept landing on `open_po`,
where its long-closed orders were counted as inbound supply — inflating the stock
position and suppressing purchase recommendations. It scored 88% against open_po, 85%
against sales_history and only 68% against po_history, which is the document it is.

Headers cannot settle this. Both exports carry a material, a vendor, a PO number, a
date and a quantity. What separates them is what is *in* the rows, and the planner's
own rule is the right one: if the lines are closed — received, or with nothing
outstanding — it is history, whatever the columns are called.

Two shapes have to work, because both occur:
  with a goods-receipt date     the receipt itself is the signal
  without one                   ordered minus received is, and many SAP extracts omit
                                the receipt date entirely
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.registry import AdapterRegistry


@pytest.fixture(scope="module")
def registry():
    return AdapterRegistry()


def _as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Render a fixture the way the real intake sees one: every cell a string, every
    blank a genuine null.

    `.astype(str)` alone is not equivalent, and the difference is version-dependent.
    Under pandas 2.x a NaT becomes the literal string "NaT", which is not null — so a
    content test asking `is_null(receive_date)` reads every row as received and the
    document routes to the wrong contract. Under pandas 3 the same call preserves NaN
    and the test passes. `load_sheets` reads Excel with `dtype=str`, which yields NaN
    for a blank cell on every version, so that is what a fixture has to reproduce.
    """
    text = frame.astype(str)
    return text.replace({"NaT": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})


def _purchase_records(with_gr_date: bool, closed_share: float = 0.85, n: int = 300):
    """An SAP purchase-record extract: mostly closed lines, a few recent open ones."""
    rng = np.random.default_rng(5)
    closed = rng.random(n) < closed_share
    po_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 540, n), "D")
    ordered = rng.integers(10, 500, n)
    received = np.where(closed, ordered, (ordered * 0.3).astype(int))

    frame = pd.DataFrame({
        "Material": [f"M-{i % 60:04d}" for i in range(n)],
        "Vendor": rng.choice(["ACME", "GLOBEX"], n),
        "Purchasing Document": [f"45{i:06d}" for i in range(n)],
        "Document Date": po_date,
        "Order Quantity": ordered,
        "Quantity Received": received,
        "Net Value": rng.integers(100, 9000, n),
    })
    if with_gr_date:
        frame["GR Date"] = np.where(closed, po_date + pd.Timedelta(days=40), pd.NaT)
    return _as_text(frame)


def _open_po_extract(n: int = 120):
    """A genuine open-PO extract: nothing received, a delivery schedule ahead."""
    rng = np.random.default_rng(6)
    return _as_text(pd.DataFrame({
        "Material": [f"M-{i % 60:04d}" for i in range(n)],
        "Vendor": rng.choice(["ACME", "GLOBEX"], n),
        "PO No.": [f"46{i:06d}" for i in range(n)],
        "Ordered Qty": rng.integers(10, 300, n),
        "Delivered Qty": [0] * n,
        "Planned Del Date": pd.to_datetime("2026-09-01") + pd.to_timedelta(
            rng.integers(0, 90, n), "D"
        ),
    }))


class TestPurchaseHistoryIsNotInboundSupply:

    def test_with_a_goods_receipt_date(self, registry):
        route = registry.route(_purchase_records(with_gr_date=True), "Purchase History.xlsx")
        assert route.doc_type == "po_history"

    def test_without_a_goods_receipt_date(self, registry):
        """
        The harder case, and the one reported. No receipt date anywhere — the only
        evidence is that ordered equals received on most lines.
        """
        route = registry.route(_purchase_records(with_gr_date=False), "Purchase History.xlsx")
        assert route.doc_type == "po_history"

    def test_the_decision_is_made_on_content_not_headers(self, registry):
        route = registry.route(_purchase_records(with_gr_date=False), "ph.xlsx")
        assert "content decided" in route.reason

    def test_a_genuinely_open_extract_still_reads_as_open(self, registry):
        assert registry.route(_open_po_extract(), "open PO.xlsx").doc_type == "open_po"

    def test_a_mostly_open_purchase_extract_reads_as_open(self, registry):
        """The rule is closed-ness, not the filename. Mostly-open lines are supply."""
        route = registry.route(
            _purchase_records(with_gr_date=False, closed_share=0.05), "Purchase History.xlsx"
        )
        assert route.doc_type == "open_po"


class TestTheTwoCoexist:

    def test_each_file_keeps_its_own_document_type(self, registry, tmp_path):
        """
        Previously both landed on open_po and were concatenated, so 300 closed orders
        were added to the inbound pipeline.
        """
        from inventory_planning.ingest.intake import Intake

        _purchase_records(with_gr_date=True).to_excel(
            tmp_path / "Purchase History.xlsx", index=False)
        _open_po_extract().to_excel(tmp_path / "open PO.xlsx", index=False)

        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert set(result.documents) == {"po_history", "open_po"}
        assert result.documents["open_po"].row_count < 130, "inbound must not absorb history"


class TestGoodsReceiptIsNotADemandSignal:

    def test_a_receipt_report_does_not_read_as_sales_history(self, registry):
        """
        `posting date` is a demand_date alias, so a goods-receipt report satisfied every
        required field of sales_history and scored 85% against it. A sales history has
        to have a customer side.
        """
        frame = _purchase_records(with_gr_date=True)
        frame["Posting Date"] = frame["GR Date"]
        assert registry.route(frame, "gr report.xlsx").doc_type != "sales_history"


class TestContentTestsAreEvidenceNotAssumption:

    def test_a_single_column_derivation_cannot_decide(self, registry):
        """
        `open_qty` falls back to `order_qty` when no received quantity exists — which
        asserts nothing was delivered. Useful when transforming a known open PO,
        invalid as evidence: it would make every order book read as fully open.
        """
        no_received = _purchase_records(with_gr_date=False).drop(columns=["Quantity Received"])
        route = registry.route(no_received, "ambiguous.xlsx")
        assert "content decided" not in route.reason, (
            "with no received quantity there is no evidence either way; the router "
            "must say so rather than manufacture a verdict"
        )


class TestOverlappingPoNumbers:
    """
    A PO that is not fully closed legitimately appears in both extracts — pulling the
    open-PO and purchase-history reports from the ERP returns it twice, by design.
    That is reported and then left alone: the open extract supplies an outstanding
    balance, the history supplies the order and receipt behind lead time and ordering
    behaviour. The same line counted once in each is not the same quantity twice.
    """

    @staticmethod
    def _pair(tmp_path):
        rng = np.random.default_rng(9)
        n = 200
        po_numbers = [f"45{i:06d}" for i in range(n)]
        closed = rng.random(n) < 0.8
        ordered = rng.integers(10, 400, n)
        received = np.where(closed, ordered, (ordered * 0.4).astype(int))
        order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 200, n), "D")

        pd.DataFrame({
            "Material": [f"M-{i % 50:04d}" for i in range(n)],
            "Vendor": ["ACME"] * n,
            "Purchasing Document": po_numbers,
            "Document Date": order_date,
            "Order Quantity": ordered,
            "Quantity Received": received,
            "GR Date": np.where(closed, order_date + pd.Timedelta(days=45), pd.NaT),
        }).to_excel(tmp_path / "Purchase History.xlsx", index=False)

        pd.DataFrame({
            "Material": [f"M-{i % 50:04d}" for i in range(n)],
            "Vendor": ["ACME"] * n,
            "PO No.": po_numbers,
            "Ordered Qty": ordered,
            "Delivered Qty": received,
            "Planned Del Date": pd.to_datetime("2026-09-15"),
        })[~closed].to_excel(tmp_path / "open PO.xlsx", index=False)
        return int((~closed).sum())

    def _load(self, tmp_path):
        from inventory_planning.ingest.intake import Intake
        return Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))

    def test_overlap_is_reported(self, tmp_path):
        self._pair(tmp_path)
        notes = " ".join(self._load(tmp_path).notes)
        assert "appear in both open_po and po_history" in notes

    def test_the_note_says_it_is_expected_not_a_fault(self, tmp_path):
        self._pair(tmp_path)
        notes = " ".join(self._load(tmp_path).notes)
        assert "nothing is double-counted" in notes

    def test_no_row_is_dropped_or_deduplicated(self, tmp_path):
        expected_open = self._pair(tmp_path)
        result = self._load(tmp_path)
        assert result.documents["open_po"].row_count == expected_open
        assert result.documents["po_history"].row_count == 200

    def test_neither_file_is_reported_as_a_duplicate(self, tmp_path):
        self._pair(tmp_path)
        result = self._load(tmp_path)
        assert result.failures == []
        assert set(result.documents) == {"open_po", "po_history"}

    def test_no_note_when_the_extracts_do_not_overlap(self, tmp_path):
        """Silence when there is nothing to say — the note must not become wallpaper."""
        self._pair(tmp_path)
        import openpyxl  # noqa: F401
        frame = pd.read_excel(tmp_path / "open PO.xlsx")
        frame["PO No."] = [f"99{i:06d}" for i in range(len(frame))]
        frame.to_excel(tmp_path / "open PO.xlsx", index=False)
        assert self._load(tmp_path).notes == []


class TestRealSapColumnShapes:
    """
    The column sets from a live SAP run, which the synthetic fixtures above did not
    reproduce. Three defects only showed up here:

      every frame is read as text, so `open_qty > 0` raised TypeError and the test was
      silently skipped -- removing the one signal able to separate the two documents

      the first applicable test decided, so a 52%-received extract was judged on that
      alone when 85% of its lines were closed and said so

      `open_qty > 0` was open_po's first test, and an open *sales* order satisfies it
      too, so a sales-order export began routing to open_po
    """

    @staticmethod
    def _purchase_history(n: int = 800):
        """Received on 52% of lines, closed on 85% — the reported proportions."""
        rng = np.random.default_rng(21)
        received = rng.random(n) < 0.52
        closed = rng.random(n) < 0.85
        po_date = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 2000, n), "D")
        return _as_text(pd.DataFrame({
            "Company code": "PL30",
            "Vendor Name": rng.choice(["Sentinel Safety", "Aurora Industrial Gases"], n),
            "PO Number": [f"44000{i % 99999:05d}" for i in range(n)],
            "PO Date": po_date,
            "Material": [f"0000000000054{i % 9999:04d}" for i in range(n)],
            "PO quantity": rng.integers(1, 500, n),
            "PO del date": po_date + pd.Timedelta(days=45),
            "Net Price": np.round(rng.random(n) * 100, 2),
            "Open Quantity": np.where(closed, 0, rng.integers(1, 200, n)),
            "Status": np.where(closed, "C", "O"),
            "GR Date": np.where(received, po_date + pd.Timedelta(days=40), pd.NaT),
        }))

    @staticmethod
    def _open_po(n: int = 400):
        """GR Date blank throughout — the mark of an open-PO extract."""
        rng = np.random.default_rng(22)
        closed = rng.random(n) < 0.80
        po_date = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D")
        return _as_text(pd.DataFrame({
            "Company code": "PL30", "Vendor Name": "Aurora Industrial Gases Limited",
            "PO Number": [f"45082{i % 99999:05d}" for i in range(n)],
            "PO Date": po_date,
            "Material": [f"60168{i % 999:03d}" for i in range(n)],
            "PO quantity": rng.integers(1, 100, n),
            "PO del date": po_date + pd.Timedelta(days=14),
            "Open Quantity": np.where(closed, 0, rng.integers(1, 50, n)),
            "Status": np.where(closed, "C", "O"),
            "GR Document": "", "GR Date": pd.NaT,
            "ETA Date": po_date + pd.Timedelta(days=317),
        }))

    def test_purchase_history_routes_to_po_history(self, registry):
        route = registry.route(self._purchase_history(), "Purchase History 10th Aug 2026.XLSX")
        assert route.doc_type == "po_history"

    def test_it_is_decided_by_closed_ness_not_by_the_receipt_share(self, registry):
        """
        Receipts cover 52% of rows, below the confidence floor. Closed lines cover 85%
        and are what carries the verdict.
        """
        route = registry.route(self._purchase_history(), "ph.XLSX")
        assert "content decided" in route.reason
        # The verdict must rest on the closed-line share (~85%), not the receipt
        # share (~52%) — which is below the confidence floor and would decide nothing.
        share = int(re.search(r"holds for (\d+)% of rows", route.reason).group(1))
        assert share > 70, f"decided on {share}%, so the receipt share won again"

    def test_the_open_extract_still_routes_to_open_po(self, registry):
        assert registry.route(self._open_po(), "Regional 1030 Jul open PO.xlsx").doc_type == "open_po"

    def test_a_numeric_test_survives_text_columns(self, registry):
        """`open_qty > 0` on string data raised TypeError and was skipped silently."""
        route = registry.route(self._purchase_history(), "ph.XLSX")
        assert "errored" not in route.reason

    def test_both_keep_their_own_type_together(self, tmp_path):
        from inventory_planning.ingest.intake import Intake

        self._purchase_history().to_excel(tmp_path / "Purchase History.xlsx", index=False)
        self._open_po().to_excel(tmp_path / "Regional open PO.xlsx", index=False)
        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert set(result.documents) == {"po_history", "open_po"}

    def test_lead_time_is_measured_not_assumed(self, tmp_path):
        """
        The point of getting this right: with the history correctly identified, lead
        time comes from receipts instead of falling back to the item master.
        """
        from inventory_planning.ingest.intake import Intake

        self._purchase_history().to_excel(tmp_path / "Purchase History.xlsx", index=False)
        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert result.plan.source_of("lead_time_signal") == "po_history"


class TestFixturesMatchWhatIntakeSees:
    """
    A fixture that differs from a real read proves nothing about a real read.

    These three tests passed on pandas 3 and failed on pandas 2 because
    `pd.DataFrame({...}).astype(str)` turns a NaT into the literal string "NaT" on the
    older version and preserves NaN on the newer one. A content test asking
    `is_null(receive_date)` then read every row as received, and the open-PO extract
    routed to po_history — on the reporter's machine only.

    `load_sheets` reads Excel with `dtype=str`, which yields NaN for a blank cell on
    every version. That is the contract a fixture has to honour, so it is asserted
    rather than assumed.
    """

    @pytest.mark.parametrize("build", [
        lambda: _purchase_records(with_gr_date=True),
        lambda: _purchase_records(with_gr_date=False),
        TestRealSapColumnShapes._open_po,
        TestRealSapColumnShapes._purchase_history,
    ])
    def test_a_blank_cell_is_null_not_the_word_naught(self, build):
        frame = build()
        for column in frame.columns:
            rendered = frame[column].astype(str)
            offenders = {"NaT", "nan", "None", "NaN"} & set(rendered)
            assert not offenders, (
                f"{column} carries {offenders} as text; a real read would give NaN, so "
                f"any is_null() test in a contract sees the opposite of the truth"
            )

    def test_the_open_extract_has_no_receipts_at_all(self):
        """The property the routing verdict turns on, pinned so it cannot drift."""
        assert TestRealSapColumnShapes._open_po()["GR Date"].isna().all()

    def test_the_history_is_part_received_and_mostly_closed(self):
        """
        Both proportions matter: receipts alone (~52%) are below the confidence floor,
        and it is the closed share (~85%) that has to carry the decision.
        """
        frame = TestRealSapColumnShapes._purchase_history()
        received = frame["GR Date"].notna().mean()
        closed = (pd.to_numeric(frame["Open Quantity"], errors="coerce") == 0).mean()
        assert 0.40 < received < 0.65, f"receipts at {received:.0%} — floor no longer probed"
        assert closed > 0.75, f"closed at {closed:.0%} — no longer the deciding signal"
