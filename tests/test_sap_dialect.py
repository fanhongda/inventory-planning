"""
`Item` means the line number in SAP and the material almost everywhere else.

`test_join_keys.py` already covers the open_so case, where the fix was to give `Item`
a field of its own so it stopped displacing `sku`. What that missed is that the fix
was applied to two contracts out of eight. A colleague loading a different SAP extract
hit the same failure from the other six, and three separate holes let it through:

  po_history, sales_history and inventory had no line-number field at all, so `Item`
  had nowhere to go but `sku`

  even where a home existed, it only helped while `Material` was there to outrank
  `Item` — an export that omits the material column, carries it empty, or spells it
  `Matl` leaves `Item` the last candidate standing, and it wins by default

  the guard written to catch exactly this skipped any document with fewer than ten
  distinct keys as too small a sample. A document keyed on 10, 20, 30 has six. It
  then counted the corrupt key as evidence against the other files and warned about
  the one whose mapping was right.

So the guard now runs at three levels: every transactional contract has somewhere to
put a line number, a contract can declare which headers a field must *never* take
under a named source system, and a key too coarse to be an item number is reported
loudly whether or not there is a second document to compare it against.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.contract import FieldContract
from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.profiler import (
    SYSTEM_SAP,
    SYSTEM_UNKNOWN,
    Profiler,
)
from inventory_planning.ingest.registry import AdapterRegistry


@pytest.fixture(scope="module")
def registry():
    return AdapterRegistry()


def _sap_lines(n=360, material=True, material_header="Material", lines=6):
    """
    An SAP transactional extract: document + item + material, one row per line.

    Unmistakably SAP by its headers — `Sold-to Party` and `Purchasing Document` are
    phrases no other ERP produces — which is what the dialect rules key off.
    """
    frame = pd.DataFrame({
        "Purchasing Document": [f"45{i // lines:08d}" for i in range(n)],
        "Item": [str(10 * (i % lines + 1)) for i in range(n)],
        "Sold-to Party": "ACME",
        "Vendor": [f"V{i % 12:03d}" for i in range(n)],
        "Document Date": pd.Timestamp("2025-03-04"),
        "Posting Date": pd.Timestamp("2025-04-11"),
        "Order Quantity": np.arange(1, n + 1) % 50 + 1,
        "Net Value": np.arange(1, n + 1) * 3.5,
    })
    if material:
        frame[material_header] = [f"600{i % 60:04d}" for i in range(n)]
    return frame.astype(str)


class TestEveryTransactionalContractHasSomewhereToPutALineNumber:
    """
    The fix that worked for open_so, applied to the contracts it was never applied to.

    Without a line-number field the greedy assignment has no losing move to offer
    `Item`: it is a candidate for `sku` and for nothing else, so it takes `sku` or
    goes unmapped, and `sku` is required.
    """

    @pytest.mark.parametrize("doc_type, line_field", [
        ("open_po", "po_line_number"),
        ("open_so", "so_line_number"),
        ("po_history", "po_line_number"),
        ("sales_history", "so_line_number"),
        ("inventory", "line_number"),
    ])
    def test_the_contract_declares_one(self, registry, doc_type, line_field):
        contract = registry.contracts.get(doc_type)
        assert line_field in contract.fields
        assert "item" in [a.lower() for a in contract.fields[line_field].aliases]

    @pytest.mark.parametrize("doc_type, line_field", [
        ("po_history", "po_line_number"),
        ("sales_history", "so_line_number"),
        ("inventory", "line_number"),
    ])
    def test_item_lands_there_and_material_takes_the_sku(
        self, registry, doc_type, line_field
    ):
        profile = Profiler().profile(_sap_lines(), source_name="extract.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get(doc_type))
        assert mapping.get("sku") == "Material"
        assert mapping.get(line_field) == "Item"


class TestTheSourceSystemIsIdentified:
    """
    Which ERP wrote the file, decided once, from evidence rather than per column.

    Biased toward `unknown` on purpose. A false positive turns on rules that forbid
    mappings, so misreading a non-SAP export as SAP can leave a genuinely `Item`-keyed
    file with no item key at all.
    """

    def test_sap_display_phrases_are_enough(self):
        profile = Profiler().profile(_sap_lines(), source_name="extract.xlsx")
        assert profile.system == SYSTEM_SAP

    def test_a_table_download_is_recognised_by_its_field_names(self):
        frame = pd.DataFrame({
            "EBELN": [f"45{i // 5:08d}" for i in range(100)],
            "EBELP": [str(10 * (i % 5 + 1)) for i in range(100)],
            "MATNR": [f"600{i % 20:04d}" for i in range(100)],
            "MENGE": range(100),
        }).astype(str)
        assert Profiler().profile(frame, source_name="se16.csv").system == SYSTEM_SAP

    def test_zero_padded_keys_alone_settle_it(self):
        """The signal that survives translation — no other ERP pads keys like this."""
        frame = pd.DataFrame({
            "Artikel": [f"{i:018d}" for i in range(60)],
            "Menge": range(60),
        }).astype(str)
        profile = Profiler().profile(frame, source_name="de.csv")
        assert profile.system == SYSTEM_SAP
        assert profile.column("Artikel").zero_padded_code

    @pytest.mark.parametrize("name", [
        "inventory.csv", "open_po.csv", "open_so.csv",
        "po_history.csv", "sales_history.csv",
    ])
    def test_the_projects_own_samples_are_not_mistaken_for_sap(self, name):
        """
        These are the files the dialect rules must not touch. `open_po.csv` keys on
        `Item`, and it is the material there.
        """
        path = Path(__file__).parents[1] / "sample_data" / name
        frame = pd.read_csv(path, dtype=str)
        assert Profiler().profile(frame, source_name=name).system == SYSTEM_UNKNOWN

    def test_a_non_sap_item_keyed_source_still_maps(self, registry):
        """The regression the ban would cause if detection were loose."""
        path = Path(__file__).parents[1] / "sample_data" / "open_po.csv"
        frame = pd.read_csv(path, dtype=str)
        profile = Profiler().profile(frame, source_name="open_po.csv")
        mapping = registry._assign_columns(profile, registry.contracts.get("open_po"))
        assert mapping.get("sku") == "Item"


class TestSkuRefusesItemOnceTheDialectIsKnown:
    """
    Outranking `Item` only works while there is something to outrank.

    Each of these is an export the scorer handled correctly right up until the
    material column stopped being comparable — absent, empty, or spelled something
    the aliases do not list. In every one of them `Item` was the last candidate
    standing for a required field, and won by default.
    """

    @pytest.mark.parametrize("doc_type", ["po_history", "sales_history", "open_so"])
    def test_a_missing_material_column_leaves_sku_unmapped(self, registry, doc_type):
        profile = Profiler().profile(_sap_lines(material=False), source_name="x.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get(doc_type))
        assert mapping.get("sku") is None

    def test_an_empty_material_column_does_not_hand_sku_to_item(self, registry):
        frame = _sap_lines()
        frame["Material"] = None
        profile = Profiler().profile(frame, source_name="x.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("sku") is None

    def test_an_unmapped_key_is_better_than_a_wrong_one(self, registry):
        """
        Refusing is the point. `required_field:sku` then fails and the run stops,
        which is a question a human can answer in a minute — where a backlog keyed on
        10, 20, 30 produces a full report of confident zeroes and no error at all.
        """
        profile = Profiler().profile(_sap_lines(material=False), source_name="x.xlsx")
        contract = registry.contracts.get("po_history")
        mapping = registry._assign_columns(profile, contract)
        assert contract.fields["sku"].required
        assert "sku" not in mapping
        assert mapping.get("po_line_number") == "Item"

    def test_the_ban_is_scoped_to_the_dialect(self, registry):
        """
        Same file with the SAP wording taken out of the headers, and an `Item` column
        holding part numbers rather than a series: it is the material again.
        """
        frame = _sap_lines(material=False).rename(columns={
            "Purchasing Document": "Order Ref",
            "Sold-to Party": "Customer",
            "Posting Date": "Received On",
        })
        frame["Item"] = [f"M-{i % 60:04d}" for i in range(len(frame))]
        profile = Profiler().profile(frame, source_name="x.xlsx")
        assert profile.system == SYSTEM_UNKNOWN
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("sku") == "Item"

    def test_the_ban_holds_even_where_the_values_look_like_a_key(self, registry):
        """
        Under SAP the header decides, not the data. A one-line-per-document extract
        gives `Item` the value 10 on every row and no series to detect, and a schedule
        line can run to five figures — neither is a material, and the dialect knows
        that without having to be shown.
        """
        frame = _sap_lines(material=False, lines=1)
        profile = Profiler().profile(frame, source_name="x.xlsx")
        assert profile.system == SYSTEM_SAP
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("sku") is None

    def test_spelling_variants_are_banned_too(self, registry):
        """`ITEM_NO` must not walk around a ban written as `item no`."""
        frame = _sap_lines(material=False).rename(columns={"Item": "ITEM_NO"})
        profile = Profiler().profile(frame, source_name="x.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("sku") is None

    def test_never_accepts_a_bare_list_as_meaning_every_system(self):
        spec = FieldContract.parse("sku", {"never": ["item"]})
        assert spec.forbidden_headers(SYSTEM_UNKNOWN)[0] == {"item"}

    def test_a_malformed_never_block_is_rejected_at_load(self):
        with pytest.raises(ValueError, match="must be a list of headers"):
            FieldContract.parse("sku", {"never": "item"})


class TestALineNumberLooksLikeOne:
    """
    The shape signal, for the sources the dialect rules cannot reach.

    Cardinality alone missed this. A PO running to line 60 has six distinct values —
    past the `distinct <= 3` floor, and a distinct_rate that costs it two hundredths
    of a point, nowhere near enough to lose. What separates a position number from a
    part number is that it is a *series*.
    """

    @staticmethod
    def _column(values, rows=None):
        frame = pd.DataFrame({"x": [str(v) for v in (rows or values)]})
        return Profiler().profile(frame, source_name="x.csv").column("x")

    def test_sap_increments_of_ten_are_recognised(self):
        assert self._column(None, [10 * (i % 6 + 1) for i in range(300)]).is_small_ordinal

    def test_a_dense_count_from_one_is_too(self):
        assert self._column(None, [i % 8 + 1 for i in range(300)]).is_small_ordinal

    def test_a_numeric_part_number_is_not(self):
        assert not self._column(None, [6013900 + i % 60 for i in range(300)]).is_small_ordinal

    def test_a_short_catalogue_of_round_numbers_is_not(self):
        """One row per item: no repetition, so nothing is a position within anything."""
        assert not self._column(None, [10 * i for i in range(1, 61)]).is_small_ordinal

    def test_a_plant_code_is_not(self):
        assert not self._column(None, [3000 + 17 * (i % 4) for i in range(300)]).is_small_ordinal

    def test_a_required_key_is_not_rescued_with_a_series(self, registry):
        """
        The hole the line-number field alone does not close, and the only place the
        shape verdict changes an outcome.

        `Item` reaches `po_line_number` on score and is assigned there — and then the
        rescue pass, which exists so a required field is never starved by an optional
        one, takes it straight back for the unmapped `sku`. That pass was written on
        the assumption that a lone `Item` column must be the material. Here the data
        says otherwise, and refusing the rescue is what turns a silent wrong key into
        a failed required-field test.
        """
        frame = _sap_lines(material=False).rename(columns={
            "Purchasing Document": "Order Ref",
            "Sold-to Party": "Customer",
            "Posting Date": "Received On",
        })
        profile = Profiler().profile(frame, source_name="x.xlsx")
        assert profile.system == SYSTEM_UNKNOWN      # the dialect ban cannot help here
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("sku") is None
        assert mapping.get("po_line_number") == "Item"

    def test_a_real_item_key_is_still_rescued(self, registry):
        """The rescue pass must keep working for the case it was written for."""
        frame = _sap_lines(material=False).rename(columns={
            "Purchasing Document": "Order Ref",
            "Sold-to Party": "Customer",
            "Posting Date": "Received On",
        })
        frame["Item"] = [f"M-{i % 60:04d}" for i in range(len(frame))]
        profile = Profiler().profile(frame, source_name="x.xlsx")
        assert not profile.column("Item").is_small_ordinal
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("sku") == "Item"

    def test_the_line_number_field_is_not_penalised(self, registry):
        """Demoting the shape everywhere would evict `Item` from its own home."""
        profile = Profiler().profile(_sap_lines(), source_name="x.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get("po_history"))
        assert mapping.get("po_line_number") == "Item"


class TestAKeyTooCoarseToBeAnItemNumberIsReported:
    """
    The last line, for whatever gets past the first two.

    `_check_sku_agreement` was written for exactly this failure and declined to look
    at it: it skips any document with fewer than ten distinct keys, and a line-number
    key has six. Worse, it kept counting the corrupt key as evidence, so the only
    warning printed named the file whose mapping was correct.

    The layers above now catch the `Item` spelling, so what reaches here is the shape
    they cannot see coming: a column named exactly like an item key and holding a
    series anyway. A frozen adapter pointing `sku` at the wrong column produces it, and
    so does an export built with the wrong field under the `Material` heading — which
    is how a CS extract carrying the notification item arrives.
    """

    @staticmethod
    def _corrupt_sales(tmp_path, n=300, lines=6):
        """A billing extract whose `Material` column holds the line number."""
        pd.DataFrame({
            "Billing Document": [f"90{i // lines:06d}" for i in range(n)],
            "Material": [str(10 * (i % lines + 1)) for i in range(n)],
            "Billing Date": [pd.Timestamp("2025-01-01") + pd.DateOffset(days=i)
                             for i in range(n)],
            "Billed Quantity": np.arange(n) % 40 + 1,
            "Net Value": np.arange(n) * 7.0,
        }).to_excel(tmp_path / "billing.xlsx", index=False)

    @staticmethod
    def _healthy_stock(tmp_path, n=60):
        pd.DataFrame({
            "Material": [f"600{i:04d}" for i in range(n)],
            "Closing Stock": range(n),
            "Std cost": np.linspace(5, 90, n).round(2),
        }).to_excel(tmp_path / "stock.xlsx", index=False)

    def _notes(self, tmp_path):
        return "\n".join(Intake(verbose=False)
                         .load_files(sorted(tmp_path.glob("*.xlsx"))).notes)

    def test_the_corrupt_document_is_named(self, tmp_path):
        self._corrupt_sales(tmp_path)
        self._healthy_stock(tmp_path)
        notes = self._notes(tmp_path)
        assert "keyed on only 6 distinct item number" in notes
        assert "billing.xlsx" in notes

    def test_the_healthy_document_is_not_blamed(self, tmp_path):
        """
        The regression that made the old warning worse than useless: a key agreeing
        with nothing makes every correctly-mapped file look like the odd one out.
        """
        self._corrupt_sales(tmp_path)
        self._healthy_stock(tmp_path)
        notes = self._notes(tmp_path)
        assert "stock.xlsx" not in notes

    def test_the_reason_is_given_not_just_the_symptom(self, tmp_path):
        self._corrupt_sales(tmp_path)
        self._healthy_stock(tmp_path)
        assert "document line number" in self._notes(tmp_path)

    def test_a_lone_document_is_still_diagnosed(self, tmp_path):
        """No second file to compare against, and the shape says it on its own."""
        self._corrupt_sales(tmp_path)
        assert "keyed on only 6 distinct item number" in self._notes(tmp_path)

    def test_a_small_catalogue_ordered_repeatedly_is_not_accused(self):
        """
        The project's own sample data, which is exactly the shape a bare cardinality
        rule gets wrong: eight SKUs across eighty backlog lines. Coarse is a symptom,
        not a diagnosis — either the values are a series or nothing else in the run
        recognises them, and here neither holds.
        """
        samples = sorted((Path(__file__).parents[1] / "sample_data").glob("*.csv"))
        notes = "\n".join(Intake(verbose=False).load_files(samples).notes)
        assert "keyed on only" not in notes

    def test_a_small_file_with_few_skus_is_left_alone(self, tmp_path):
        """Eight SKUs in a twelve-row file is a small file, not a broken key."""
        pd.DataFrame({
            "Material": [f"600{i % 8:04d}" for i in range(12)],
            "Closing Stock": range(12),
            "Std cost": np.linspace(5, 90, 12).round(2),
        }).to_excel(tmp_path / "stock.xlsx", index=False)
        assert "keyed on only" not in self._notes(tmp_path)

    def test_healthy_documents_stay_silent(self, tmp_path):
        """The warning must not become wallpaper."""
        self._healthy_stock(tmp_path)
        skus = [f"600{i:04d}" for i in range(60)]
        pd.DataFrame({
            "Part Number": [s for s in skus for _ in range(5)],
            "Billto Customer Name": "ACME",
            "Invdate Date": pd.Timestamp("2025-01-10"),
            "Shipped Quantity": 10,
        }).to_excel(tmp_path / "sales.xlsx", index=False)
        assert "keyed on only" not in self._notes(tmp_path)
