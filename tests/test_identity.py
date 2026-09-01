"""
The identity layer.

The failure being closed is the one that produced a full report of confident zeroes on
the real Controls SDC extract: four documents keyed on the SAP material number, two on
a commercial part number, and nothing anywhere able to say the two halves named the
same materials. The gate then blocked the file whose key was right — a whole-warehouse
stock snapshot, whose codes legitimately appear nowhere else in the transactions — and
passed the file whose key was wrong, because the two wrongly-keyed documents agreed
with each other.

The acceptance criterion for this module is one sentence: **entering the commercial
part number and entering the SAP material number must return the same item.**

The other tests are the ways that criterion can be met dishonestly — by merging things
that are not the same, by chaining the whole catalogue together through a column that
repeats, or by producing an answer that depends on which file was read first.
"""

import pandas as pd
import pytest

from inventory_planning.store.identity import (
    IdentityBuilder, SYSTEM_MATNR, SYSTEM_PARTNO, SYSTEM_UNKNOWN, classify_code,
)
from inventory_planning.store.landing import LandingStore

pytest.importorskip("pyarrow", reason="the identity layer stores parquet")


@pytest.fixture
def store(tmp_path):
    return tmp_path / "store"


@pytest.fixture
def landing(store):
    return LandingStore(store)


@pytest.fixture
def builder(store, landing):
    return IdentityBuilder(landing, store)


def _land(landing, tmp_path, name, doc_type, batch_id, rows):
    """Write a sheet verbatim and land it, headers and all."""
    path = tmp_path / name
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    landing.land(path, doc_type=doc_type, batch_id=batch_id)
    return batch_id


def _master(n=30):
    """A planning master carrying both numbering systems on one row — the crosswalk."""
    rows = [["Material", "Product code", "Description", "MOQ"]]
    for i in range(n):
        rows.append([f"53913{i:02d}", f"MS-FAC25{i:02d}-0", f"Controller {i}", "300"])
    return rows


def _po_history(n=30):
    """A purchase history keyed on the part number, with the material number beside it."""
    rows = [["PO_NO.", "Material", "PartNo.", "Qty", "Vendor_Name"]]
    for i in range(n):
        rows.append([f"45110209{i:02d}", f"53913{i:02d}", f"MS-FAC25{i:02d}-0",
                     "8", "ACME"])
    return rows


class TestCodeSystems:
    """Shape, not column name. The column name is the evidence that already failed."""

    @pytest.mark.parametrize("value,expected", [
        ("1003524", SYSTEM_MATNR),
        ("000000000005757504", SYSTEM_MATNR),
        ("419348", SYSTEM_MATNR),
        ("MS-FAC2513-0", SYSTEM_PARTNO),
        ("V46AS-9301", SYSTEM_PARTNO),
        ("FX-ATV7003-0", SYSTEM_PARTNO),
        ("8", SYSTEM_UNKNOWN),            # a quantity
        ("2026-07-13", SYSTEM_UNKNOWN),   # a date
        ("", SYSTEM_UNKNOWN),
    ])
    def test_shape_decides(self, value, expected):
        assert classify_code(value) == expected


class TestTheAcceptanceCriterion:
    """Either code must reach the same item."""

    @pytest.fixture
    def built(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1", _master())
        return builder.build([("item_master", "m1")])

    def test_both_codes_resolve(self, built):
        assert built.resolve("5391300") is not None
        assert built.resolve("MS-FAC2500-0") is not None

    def test_both_codes_resolve_to_the_same_item(self, built):
        assert built.resolve("5391300") == built.resolve("MS-FAC2500-0")

    def test_the_item_carries_both_codes(self, built):
        uid = built.resolve("MS-FAC2500-0")
        assert built.codes_for(uid) == ["5391300", "MS-FAC2500-0"]

    def test_the_uid_is_anchored_on_the_material_number(self, built):
        """Not a hash of the component — adding an alias must not re-identify the item."""
        assert built.resolve("5391300").endswith(f"{SYSTEM_MATNR}:5391300")

    def test_sap_padding_is_not_a_different_item(self, built):
        assert built.resolve("000000005391300") == built.resolve("5391300")


class TestLoadOrderDoesNotMatter:
    """The reason this is a separate phase with a version, rather than a side effect
    of resolving. Greedy building made the answer depend on which file came first."""

    def test_the_same_inputs_give_the_same_version(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1", _master())
        _land(landing, tmp_path, "po.xlsx", "po_history", "p1", _po_history())
        forward = builder.build([("item_master", "m1"), ("po_history", "p1")])
        reverse = builder.build([("po_history", "p1"), ("item_master", "m1")])
        assert forward.version == reverse.version

    def test_the_same_inputs_give_the_same_items(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1", _master())
        _land(landing, tmp_path, "po.xlsx", "po_history", "p1", _po_history())
        forward = builder.build([("item_master", "m1"), ("po_history", "p1")])
        reverse = builder.build([("po_history", "p1"), ("item_master", "m1")])
        assert forward.resolve("MS-FAC2500-0") == reverse.resolve("MS-FAC2500-0")

    def test_a_different_input_set_is_a_different_version(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1", _master())
        _land(landing, tmp_path, "po.xlsx", "po_history", "p1", _po_history())
        one = builder.build([("item_master", "m1")])
        both = builder.build([("item_master", "m1"), ("po_history", "p1")])
        assert one.version != both.version


class TestTheErpNamespace:
    """The same string in two ERP instances is two materials. Merging them on string
    equality is the original two-code-system bug, one level up."""

    def test_the_same_code_in_two_erps_stays_separate(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "cn.xlsx", "item_master", "cn", _master())
        _land(landing, tmp_path, "us.xlsx", "item_master", "us", _master())
        built = builder.build(
            [("item_master", "cn"), ("item_master", "us")],
            erp_of={"cn": "CN01", "us": "US01"},
        )
        assert built.resolve("5391300", erp_id="CN01") \
            != built.resolve("5391300", erp_id="US01")


class TestAmbiguityIsRecordedNotGuessed:
    """One part number reaching two SAP materials is real. Any automatic choice is a
    coin flip that reports itself as a fact."""

    @pytest.fixture
    def built(self, builder, landing, tmp_path):
        rows = [["Material", "Product code"]]
        for i in range(12):
            rows.append([f"70000{i:02d}", f"AB-{i:03d}-X"])
        # One part number claimed by a second material number.
        rows.append(["7000099", "AB-000-X"])
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1", rows)
        return builder.build([("item_master", "m1")])

    def test_the_ambiguous_component_is_a_conflict(self, built):
        reasons = set(built.conflicts["reason"])
        assert any("distinct sap_matnr codes reach one item" in r for r in reasons)

    def test_the_conflicting_codes_are_not_merged(self, built):
        assert built.resolve("7000000") is None
        assert built.resolve("7000099") is None

    def test_the_conflict_names_the_candidates(self, built):
        row = built.conflicts[built.conflicts["code"] == "AB-000-X"].iloc[0]
        assert "sap_matnr:7000000" in row["candidates"]
        assert "sap_matnr:7000099" in row["candidates"]

    def test_unaffected_items_still_resolve(self, built):
        """A conflict quarantines its own component, not the catalogue."""
        assert built.resolve("AB-005-X") == built.resolve("7000005")


class TestRenumberingIsProposedNotApplied:
    """A material that changed number carries both on one row. Two codes of the *same*
    system is a renumbering, not an alias — merging them is a planning decision."""

    @pytest.fixture
    def built(self, builder, landing, tmp_path):
        rows = [["Material", "Old Material"]]
        for i in range(15):
            rows.append([f"10035{i:02d}", f"99035{i:02d}"])
        _land(landing, tmp_path, "so.xlsx", "open_so", "s1", rows)
        return builder.build([("open_so", "s1")])

    def test_the_pair_is_recorded_as_a_substitution_candidate(self, built):
        pairs = built.substitution_candidates
        assert not pairs.empty
        assert set(pairs["system_id"]) == {SYSTEM_MATNR}
        assert {"1003500", "9903500"} == set(
            pairs[pairs["code_a"].isin(["1003500", "9903500"])].iloc[0]
            [["code_a", "code_b"]])

    def test_it_is_not_applied_as_an_alias(self, built):
        """Old and new stay separate items until someone approves the supersession."""
        assert built.resolve("1003500") != built.resolve("9903500")

    def test_neither_is_quarantined(self, built):
        """Not merged is not the same as not usable — both still name an item."""
        assert built.resolve("1003500") is not None
        assert built.resolve("9903500") is not None


class TestALegacyCodeBesideACurrentOne:
    """One material number carrying two part numbers. Its identity is not in doubt,
    so refusing the whole component would quarantine an unambiguous material."""

    @pytest.fixture
    def built(self, builder, landing, tmp_path):
        rows = [["Material", "Old Material Number", "Product code"]]
        for i in range(15):
            rows.append([f"10035{i:02d}", f"FX-ATV70{i:02d}-0", f"PC-{i:03d}-Z"])
        _land(landing, tmp_path, "so.xlsx", "open_so", "s1", rows)
        return builder.build([("open_so", "s1")])

    def test_the_material_number_still_resolves(self, built):
        assert built.resolve("1003500") is not None

    def test_both_part_numbers_reach_it(self, built):
        uid = built.resolve("1003500")
        assert built.resolve("FX-ATV7000-0") == uid
        assert built.resolve("PC-000-Z") == uid

    def test_nothing_is_quarantined(self, built):
        assert built.conflicts.empty


class TestRunawayMerging:
    """A column that repeats would co-occur with everything and chain the whole
    catalogue into one material. That has to be impossible, not unlikely."""

    def test_a_low_cardinality_column_is_not_an_identifier(self, builder, landing, tmp_path):
        rows = [["Material", "Product code", "Plant"]]
        for i in range(40):
            rows.append([f"53913{i:02d}", f"MS-FAC25{i:02d}-0", "5051"])
        _land(landing, tmp_path, "m.xlsx", "item_master", "m1", rows)
        built = builder.build([("item_master", "m1")])
        # 40 items, each with exactly its own two codes — not one item with 81 codes.
        assert built.manifest["items"] == 40
        assert len(built.codes_for(built.resolve("5391300"))) == 2

    def test_an_oversized_component_is_refused(self, builder, landing, tmp_path):
        """A shared code on every row would merge everything. It is quarantined."""
        rows = [["Material", "Product code", "Alt"]]
        for i in range(60):
            rows.append([f"88000{i:02d}", f"ZZ-{i:03d}-Q", "SHARED-CODE-1"])
        # `Alt` repeats, so it is rejected as an identifier before it can chain anything.
        _land(landing, tmp_path, "m.xlsx", "item_master", "m1", rows)
        built = builder.build([("item_master", "m1")])
        assert built.manifest["items"] == 60


class TestDocumentNumbersAreNotMaterials:
    """
    Found by running the six-document shape rather than by unit test, which is why it
    is pinned here: a PO number is all digits and highly distinct, so on shape alone it
    is indistinguishable from a material number — and it sits on the same row as one.

    Left in, every PO number becomes an alias of whatever material it was ordered
    against, each material then reaches two "material numbers", and every component is
    quarantined. The first end-to-end run produced 120 conflicts and not one resolvable
    item, with all thirty-four unit tests green.
    """

    @pytest.fixture
    def built(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1", _master())
        _land(landing, tmp_path, "po.xlsx", "po_history", "p1", _po_history())
        return builder.build([("item_master", "m1"), ("po_history", "p1")])

    def test_the_run_resolves_at_all(self, built):
        assert built.manifest["items"] == 30
        assert built.manifest["conflicts"] == 0

    def test_the_po_number_is_not_an_alias(self, built):
        assert built.resolve("4511020900") is None

    def test_the_material_keeps_exactly_its_two_codes(self, built):
        assert built.codes_for(built.resolve("5391300")) == ["5391300", "MS-FAC2500-0"]

    @pytest.mark.parametrize("header", [
        "PO_NO.", "Purchasing Document", "Sales Order", "Delivery #",
        "Sold-To Customer", "Vendor/supplying plant", "Customer Po Number",
    ])
    def test_document_and_party_headers_are_excluded(self, header):
        from inventory_planning.store.identity import names_a_document
        assert names_a_document(header)

    @pytest.mark.parametrize("header", [
        "Material", "Material #", "PartNo.", "Product code", "Old Material Number",
    ])
    def test_item_headers_are_not(self, header):
        from inventory_planning.store.identity import names_a_document
        assert not names_a_document(header)


class TestCorroborationIsMeasuredBothWays:
    """
    A whole-warehouse stock snapshot holds far more materials than any transaction
    extract, and is supposed to. Judging its key by "how much of mine appears
    elsewhere" is the direction that made the intake gate block the one document whose
    mapping was right, so it must not be the direction used here.
    """

    @pytest.fixture
    def built(self, builder, landing, tmp_path):
        # 200 materials in stock; only the first 30 were ever ordered or sold.
        stock = [["Material", "Plant", "Qty"]] + [
            [f"53913{i:03d}", "5051", "5"] for i in range(200)]
        _land(landing, tmp_path, "stock.xlsx", "inventory", "i1", stock)
        _land(landing, tmp_path, "master.xlsx", "item_master", "m1",
              [["Material", "Product code", "MOQ"]] +
              [[f"53913{i:03d}", f"MS-FAC25{i:02d}-0", "300"] for i in range(30)])
        return builder.build([("inventory", "i1"), ("item_master", "m1")])

    def test_the_snapshot_column_is_kept(self, built):
        """Only 15% of its codes appear anywhere else — and it is still the item key."""
        assert built.manifest["key_columns"]["i1"] == ["Material"]

    def test_stock_only_materials_still_get_an_identity(self, built):
        assert built.resolve("53913199") is not None

    def test_the_crosswalk_still_works_for_the_transacted_ones(self, built):
        assert built.resolve("MS-FAC2500-0") == built.resolve("53913000")


class TestPromotion:
    """Building is cheap and safe; promoting changes what the next run means by an
    item number. They are separate acts."""

    def test_nothing_is_current_until_promoted(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "m.xlsx", "item_master", "m1", _master())
        builder.build([("item_master", "m1")])
        assert builder.current() is None

    def test_promoting_names_the_version(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "m.xlsx", "item_master", "m1", _master())
        built = builder.build([("item_master", "m1")])
        builder.promote(built)
        assert builder.current().version == built.version
        assert builder.current().resolve("MS-FAC2500-0") == built.resolve("MS-FAC2500-0")


class TestManifest:
    def test_it_records_which_columns_were_used_as_keys(self, builder, landing, tmp_path):
        _land(landing, tmp_path, "po.xlsx", "po_history", "p1", _po_history())
        built = builder.build([("po_history", "p1")])
        used = built.manifest["key_columns"]["p1"]
        assert "Material" in used and "PartNo." in used
        # The PO number varies per row but is not an item code; the quantity and the
        # vendor name are neither distinct enough nor code-shaped.
        assert "Qty" not in used and "Vendor_Name" not in used

    def test_an_empty_build_is_not_an_error(self, builder):
        built = builder.build([])
        assert built.manifest["items"] == 0
        assert built.aliases.empty
