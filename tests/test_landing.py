"""
The landing layer.

Every test here is a failure that has already happened on a real extract, or that the
design review found waiting to happen:

  a planner told the SKU column was wrong renames three columns to `Material`, pandas
  hands back `Material`, `Material.1`, `Material.2`, and a payload keyed by header
  keeps one of them — the layer whose whole job is losing nothing loses two columns
  on its first day

  a master carries `5051 Std Cost(CNY)` beside `5052 Std Cost(CNY)`; a PO history
  carries `Material` beside `PartNo.`; the adapter picks one of each and the other is
  gone from the stored frame, so a mis-mapping costs a re-ingest rather than a re-read

  an export has a title banner above the header row, and the offset between "row 0 of
  the landed rows" and "row 0 of the canonical frame" becomes the height of the banner
  — silently, since both are valid row numbers and neither errors

  `dropna(subset=required)` removes rows and nobody can say afterwards how many or
  which
"""

import json

import pandas as pd
import pytest

from inventory_planning.store.landing import (
    LandingStore, REJECT_REQUIRED, find_header_row, read_verbatim,
)

pytest.importorskip("pyarrow", reason="the landing layer stores parquet")


@pytest.fixture
def store(tmp_path):
    return LandingStore(tmp_path / "store")


def _write_xlsx(path, rows):
    """Write rows verbatim — no header handling, so duplicates survive to the file."""
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    return path


class TestDuplicateHeadersSurvive:
    """The failure that motivated keying on position instead of name."""

    @pytest.fixture
    def renamed(self, tmp_path):
        # What an export looks like after someone "unifies the SKU column".
        return _write_xlsx(tmp_path / "renamed.xlsx", [
            ["Material", "Material", "Material", "Qty"],
            ["1003524", "MS-FAC2513-0", "V46AS-9301", "8"],
            ["419348", "FX-ATV7003-0", "M9320-HGA-4", "2"],
        ])

    def test_pandas_would_have_mangled_them(self, renamed):
        """The premise. If pandas ever stops doing this, this layer can relax."""
        mangled = list(pd.read_excel(renamed, dtype=str).columns)
        assert mangled != ["Material", "Material", "Material", "Qty"]

    def test_every_column_is_kept(self, store, renamed):
        record = store.land(renamed, doc_type="po_history", batch_id="b1")
        assert [c["header"] for c in record["columns"]] == [
            "Material", "Material", "Material", "Qty"]

    def test_the_values_are_not_collapsed(self, store, renamed):
        store.land(renamed, doc_type="po_history", batch_id="b1")
        payload = json.loads(store.rows("po_history", "b1", named=False).iloc[0]["payload"])
        # Keyed by position, so all three survive under distinct keys.
        assert payload["0"] == "1003524"
        assert payload["1"] == "MS-FAC2513-0"
        assert payload["2"] == "V46AS-9301"

    def test_the_duplication_is_reported(self, store, renamed):
        record = store.land(renamed, doc_type="po_history", batch_id="b1")
        assert record["duplicate_headers"] == ["Material"]

    def test_reading_back_named_keeps_all_three(self, store, renamed):
        store.land(renamed, doc_type="po_history", batch_id="b1")
        wide = store.rows("po_history", "b1", named=True)
        # The first keeps the plain name; the others are suffixed by position rather
        # than overwriting it, so the round trip loses nothing either.
        assert "Material" in wide.columns
        assert sum(c.startswith("Material") for c in wide.columns) == 3


class TestNothingIsDroppedForNotBeingMapped:
    """The columns `explain` lists as matching nothing are the ones that turned out
    to matter. They have to be here after landing, not after a second read."""

    @pytest.fixture
    def master(self, tmp_path):
        return _write_xlsx(tmp_path / "master.xlsx", [
            ["Material", "Product code", "Planning LT (D)", "MOQ",
             "5051 Std Cost(CNY)", "5052 Std Cost(CNY)"],
            ["5391348", "MS-FAC2513-0", "63", "300", "298.25", "301.10"],
        ])

    def test_the_unmapped_columns_are_all_there(self, store, master):
        store.land(master, doc_type="item_master", batch_id="m1")
        row = store.source_row("item_master", "m1", 0)
        assert row["Planning LT (D)"] == "63"
        assert row["5051 Std Cost(CNY)"] == "298.25"
        assert row["5052 Std Cost(CNY)"] == "301.10"

    def test_both_identifier_systems_are_kept(self, store, master):
        """Neither `Material` nor `Product code` is chosen here. Choosing is a later,
        replayable step; this layer is not entitled to an opinion."""
        store.land(master, doc_type="item_master", batch_id="m1")
        row = store.source_row("item_master", "m1", 0)
        assert row["Material"] == "5391348"
        assert row["Product code"] == "MS-FAC2513-0"


class TestNoTypeInference:
    """A padded material number and a numeric-looking code both come back as written."""

    def test_leading_zeros_survive(self, store, tmp_path):
        path = _write_xlsx(tmp_path / "stock.xlsx", [
            ["Material #", "Qty"],
            ["000000000005757504", "20"],
        ])
        store.land(path, doc_type="sales_history", batch_id="s1")
        assert store.source_row("sales_history", "s1", 0)["Material #"] \
            == "000000000005757504"


class TestBannerRows:
    """Row numbers have to agree with the canonical frame, or every pointer from a
    fact back to its source row is off by the height of a title banner."""

    def test_the_header_is_found_under_a_banner(self):
        raw = pd.DataFrame([
            ["Controls SDC Stock Report", None, None],
            ["Generated 2026-07-31", None, None],
            ["Material", "Plant", "Qty"],
            ["1003524", "5051", "8"],
        ])
        assert find_header_row(raw) == 2

    def test_row_numbering_starts_below_the_header(self, store, tmp_path):
        path = _write_xlsx(tmp_path / "banner.xlsx", [
            ["Controls SDC Stock Report", None, None],
            ["Material", "Plant", "Qty"],
            ["1003524", "5051", "8"],
            ["419348", "5052", "2"],
        ])
        store.land(path, doc_type="inventory", batch_id="i1")
        assert store.source_row("inventory", "i1", 0)["Material"] == "1003524"
        assert store.source_row("inventory", "i1", 1)["Material"] == "419348"

    def test_a_plain_sheet_has_no_banner(self, tmp_path):
        raw = pd.DataFrame([["Material", "Qty"], ["1003524", "8"]])
        assert find_header_row(raw) == 0


class TestRejectsAreCounted:
    """Isolated, not dropped — and the count is answerable afterwards."""

    def test_rejected_rows_are_stored_with_a_reason(self, store, tmp_path):
        path = _write_xlsx(tmp_path / "po.xlsx", [
            ["Material", "Vendor", "Order_date"],
            ["1003524", "ACME", "2026-01-05"],
            ["", "ACME", "2026-01-06"],
        ])
        store.land(path, doc_type="po_history", batch_id="p1")
        written = store.reject("po_history", "p1", [
            {"row_no": 1, "stage": "resolve", "reason_code": REJECT_REQUIRED,
             "field": "sku", "detail": {"column": 0}},
        ])
        assert written == 1
        rejects = pd.read_parquet(store.reject_path("po_history", "p1"))
        assert rejects.iloc[0]["reason_code"] == REJECT_REQUIRED
        assert rejects.iloc[0]["row_no"] == 1

    def test_the_rejected_row_is_still_readable_in_full(self, store, tmp_path):
        """Nothing about the row is stored twice — it is already landed, so only the
        reason goes in the reject file."""
        path = _write_xlsx(tmp_path / "po.xlsx", [
            ["Material", "Vendor"],
            ["", "ACME"],
        ])
        store.land(path, doc_type="po_history", batch_id="p2")
        store.reject("po_history", "p2", [
            {"row_no": 0, "stage": "resolve", "reason_code": REJECT_REQUIRED},
        ])
        assert store.source_row("po_history", "p2", 0)["Vendor"] == "ACME"

    def test_no_rejects_writes_nothing(self, store):
        assert store.reject("po_history", "p3", []) == 0
        assert not store.reject_path("po_history", "p3").exists()


class TestBatchIdIsShared:
    """`(batch_id, row_no)` is the pointer from a fact to its source. A landing layer
    minting its own ids would be a second archive, not the source of the first."""

    def test_the_caller_supplies_the_id(self, store, tmp_path):
        path = _write_xlsx(tmp_path / "x.xlsx", [["Material"], ["1003524"]])
        record = store.land(path, doc_type="inventory", batch_id="20260901_120000-abc123")
        assert record["batch_id"] == "20260901_120000-abc123"
        assert store.raw_path("inventory", "20260901_120000-abc123").exists()


class TestCsvToo:
    def test_a_csv_lands_the_same_way(self, store, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("Material,Material,Qty\n1003524,MS-FAC2513-0,8\n", encoding="utf-8")
        record = store.land(path, doc_type="inventory", batch_id="c1")
        assert record["duplicate_headers"] == ["Material"]
        payload = json.loads(store.rows("inventory", "c1", named=False).iloc[0]["payload"])
        assert payload["1"] == "MS-FAC2513-0"


class TestEmptySheet:
    def test_an_empty_sheet_lands_nothing_and_does_not_raise(self, store, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        record = store.land(path, doc_type="inventory", batch_id="e1")
        assert record["rows"] == 0
        assert record["columns"] == []
