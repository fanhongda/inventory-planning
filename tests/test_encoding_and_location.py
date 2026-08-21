"""
Three defects reported against a run on a different SAP dataset.

  the readers/ path recognised `Item` as the SKU — the same failure the ingest path
  had been fixed for, in the half of the codebase the fix never reached

  CSV output rendered Chinese supplier names as mojibake

  location_id was DC-01 on every row regardless of what plant the export named

They look unrelated and share a shape: each is a default that was never stated. The
alias list happened to rank `item` second, `to_csv` happened to omit the BOM, and
`location_id` happened to fall through to the placeholder — and none of the three
announced itself, so all three arrived as numbers and names that looked like data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.encoding import (
    CSV_ENCODING,
    decode_bytes,
    describe_choice,
    sniff_encoding,
    write_csv,
)
from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest_bridge import IngestBridge
from inventory_planning.node import PLACEHOLDER_LOCATION, load_planning_node
from inventory_planning.readers.base_reader import (
    _detect_mapping,
    _disqualified_item_keys,
)
from inventory_planning.schema import ALL_SCHEMAS, OPEN_SO_SCHEMA, PO_HISTORY_SCHEMA

SUPPLIERS = ["深圳市华强电子", "东莞立信精密", "上海浦东物流", "北京中科仪器"]


def _sap_frame(n=300, lines=5, material=True):
    frame = pd.DataFrame({
        "Purchasing Document": [f"45{i // lines:08d}" for i in range(n)],
        "Item": [str(10 * (i % lines + 1)) for i in range(n)],
        "Sold-to Party": "ACME",
        "Vendor": [f"V{i % 9:03d}" for i in range(n)],
        "Document Date": "2025-01-05",
        "Order Quantity": np.arange(n) % 50 + 1,
    })
    if material:
        frame["Material"] = [f"600{i % 40:04d}" for i in range(n)]
    return frame.astype(str)


# ── 1. the readers/ path ─────────────────────────────────────────────────────

class TestTheLegacyReadersPathRefusesItemToo:
    """
    The fix landed in `ingest/` and the report came from `readers/`.

    Two code paths read the same files with two different sets of rules, and only one
    of them had been taught. This one is worse than the original: `_detect_mapping`
    takes the first alias that matches, and `item` was ranked second for `sku` — ahead
    of `material` — so it did not even need `Material` to be missing.
    """

    def test_material_wins_over_item(self):
        frame = _sap_frame()
        mapping, _ = _detect_mapping(
            list(frame.columns), PO_HISTORY_SCHEMA,
            _disqualified_item_keys(frame, "po_history"),
        )
        assert mapping.get("sku") == "Material"

    def test_alias_order_alone_already_settles_it(self):
        """Without the profile there is no dialect and no shape — order must still hold."""
        mapping, _ = _detect_mapping(list(_sap_frame().columns), PO_HISTORY_SCHEMA)
        assert mapping.get("sku") == "Material"

    @pytest.mark.parametrize("doc_type", sorted(ALL_SCHEMAS))
    def test_item_is_last_in_every_sku_list(self, doc_type):
        aliases = ALL_SCHEMAS[doc_type]["sku"]
        assert aliases[-1] == "item"

    def test_an_sap_item_column_is_refused_outright(self):
        """With no `Material` to lose to, ranking cannot help and the ban must."""
        frame = _sap_frame(material=False)
        refused = _disqualified_item_keys(frame, "po_history")
        assert "Item" in refused
        mapping, _ = _detect_mapping(list(frame.columns), PO_HISTORY_SCHEMA, refused)
        assert mapping.get("sku") is None

    def test_a_non_sap_item_column_is_still_the_material(self):
        frame = pd.DataFrame({
            "Item": [f"M-{i % 40:04d}" for i in range(300)],
            "Supplier": "S1", "PO Date": "2025-01-05", "PO Qty": 1,
        }).astype(str)
        refused = _disqualified_item_keys(frame, "po_history")
        mapping, _ = _detect_mapping(list(frame.columns), PO_HISTORY_SCHEMA, refused)
        assert mapping.get("sku") == "Item"

    def test_the_open_so_schema_behaves_the_same(self):
        frame = _sap_frame().rename(columns={"Purchasing Document": "Sales Document"})
        mapping, _ = _detect_mapping(
            list(frame.columns), OPEN_SO_SCHEMA,
            _disqualified_item_keys(frame, "open_so"),
        )
        assert mapping.get("sku") == "Material"

    def test_the_two_paths_use_one_list_of_banned_headers(self):
        """
        Not a style point. The bug was two copies of the same knowledge, one of which
        had been updated, so the reader imports the contract's rules rather than
        restating them.
        """
        from inventory_planning.ingest.contract import default_registry
        spec = default_registry().get("po_history").fields["sku"]
        banned, _ = spec.forbidden_headers("sap")
        assert "item" in banned
        assert _disqualified_item_keys(_sap_frame(), "po_history")["Item"]


# ── 2. encoding ──────────────────────────────────────────────────────────────

class TestChineseNamesSurviveTheRoundTrip:
    """
    Written correctly and read back as mojibake, which is the whole complaint.

    `to_csv` defaults to UTF-8 with no BOM, and Excel without a BOM decodes using the
    system codepage — GBK on a Chinese install. The file is right; every planner who
    opens it sees 娣卞湷甯傚崕寮虹數瀛.
    """

    def test_the_csv_carries_a_bom(self, tmp_path):
        path = tmp_path / "out.csv"
        write_csv(pd.DataFrame({"supplier": SUPPLIERS, "qty": range(4)}), path)
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_excel_under_a_chinese_locale_reads_the_names_back(self, tmp_path):
        path = tmp_path / "out.csv"
        write_csv(pd.DataFrame({"supplier": SUPPLIERS, "qty": range(4)}), path)
        # What Excel does when it sees the BOM: honour it rather than use the codepage.
        assert pd.read_csv(path, encoding=CSV_ENCODING)["supplier"].tolist() == SUPPLIERS

    def test_the_helper_defaults_are_the_ones_that_were_forgotten(self, tmp_path):
        path = tmp_path / "out.csv"
        write_csv(pd.DataFrame({"a": [1]}), path)
        assert path.read_text(encoding=CSV_ENCODING).splitlines()[0] == "a"   # no index column

    @pytest.mark.parametrize("codec, names", [
        ("gb18030", SUPPLIERS),
        ("big5", ["台灣積體電路", "鴻海精密工業", "聯發科技股份", "統一超商股份"]),
        ("cp932", ["株式会社日立製作所", "三菱電機株式会社", "ソニーグループ", "東芝テック"]),
        ("cp949", ["삼성전자주식회사", "엘지이노텍", "에스케이하이닉스", "현대모비스"]),
        ("cp1252", ["Société Générale", "Zürich Handels", "Müller & Co", "Rhône-Poulenc"]),
        ("utf-8", SUPPLIERS),
    ])
    def test_a_legacy_export_is_read_with_the_codec_it_was_written_with(self, codec, names):
        """
        The ladder this replaces was `utf-8, utf-8-sig, latin-1, cp1252`, tried until
        one did not raise — and latin-1 maps all 256 byte values, so it never raises.
        Every one of these files decoded as latin-1, silently.
        """
        text = "supplier,qty\n" + "".join(f"{n},{i}\n" for i, n in enumerate(names))
        decoded, chosen = decode_bytes(text.encode(codec))
        assert decoded == text, f"{codec} read as {chosen}"

    def test_latin1_would_have_swallowed_all_of_them(self):
        """The property that made the old ladder unable to reach any CJK codec."""
        assert bytes(range(256)).decode("latin-1")

    def test_a_guessed_codec_is_reported(self):
        assert "gb18030" in describe_choice("gb18030", "po.csv")

    def test_a_verified_codec_is_not(self):
        """UTF-8 was checked, not guessed. A note on every file is a note nobody reads."""
        assert describe_choice("utf-8", "po.csv") is None

    def test_intake_reads_a_gbk_export_and_says_so(self, tmp_path):
        n = 320
        frame = pd.DataFrame({
            "Part No": [f"600{i % 40:04d}" for i in range(n)],
            "Vendor Name": [SUPPLIERS[i % 4] for i in range(n)],
            "PO Date": "2025-01-05",
            "Goods Receipt Date": "2025-02-10",
            "PO Qty": np.arange(n) % 50 + 1,
        })
        (tmp_path / "po.csv").write_bytes(frame.to_csv(index=False).encode("gb18030"))

        result = Intake(verbose=False).load_files([tmp_path / "po.csv"])
        assert sniff_encoding(tmp_path / "po.csv") == "gb18030"
        assert set(result.documents["po_history"].frame["supplier"]) == set(SUPPLIERS)
        assert any("read as gb18030" in note for note in result.notes)


# ── 3. location_id ───────────────────────────────────────────────────────────

class TestLocationComesFromTheSourceWhereTheSourceHasOne:
    """
    `DC-01` on every row of every output, whatever plant the export named.

    Two causes. Only the inventory contract declared `location_id`, so on the other
    four documents the plant column was dropped at mapping time and the configured node
    stamped unconditionally. And `DC-01` is what `node_config.json` ships with — a
    placeholder that nobody had to change and nothing pointed at.
    """

    @staticmethod
    def _files(tmp_path, with_plant=True):
        skus = [f"600{i:04d}" for i in range(40)]
        n = 320
        po = pd.DataFrame({
            "Material": [skus[i % 40] for i in range(n)],
            "Vendor": [f"V{i % 9}" for i in range(n)],
            "Document Date": "2025-01-05",
            "Posting Date": "2025-02-10",
            "Order Quantity": np.arange(n) % 50 + 1,
            "Net Order Value": np.arange(n) * 3.0,
        })
        if with_plant:
            po["Plant"] = [["PL30", "PL10", "PL20", "PL40"][i % 4] for i in range(n)]
        po.to_csv(tmp_path / "po_hist.csv", index=False)
        pd.DataFrame({
            "Material": skus, "Plant": "PL30", "Unrestricted": range(40),
            "Std cost": np.linspace(5, 90, 40).round(2),
        }).to_csv(tmp_path / "stock.csv", index=False)
        return sorted(tmp_path.glob("*.csv"))

    def test_a_purchase_history_keeps_its_own_plants(self, tmp_path):
        out = IngestBridge(verbose=False).load(self._files(tmp_path))
        assert sorted(out["po_history_df"]["location_id"].unique()) == \
            ["PL20", "PL30", "PL40", "PL10"]

    @pytest.mark.parametrize("doc_type", ["open_po", "open_so", "po_history",
                                          "sales_history", "inventory"])
    def test_every_transactional_contract_can_carry_one(self, doc_type):
        from inventory_planning.ingest.contract import default_registry
        spec = default_registry().get(doc_type).fields.get("location_id")
        assert spec is not None
        assert "plant" in spec.aliases

    @pytest.mark.parametrize("header", ["Plant", "Plnt", "Werks", "Storage Location"])
    def test_sap_spells_plant_several_ways(self, header):
        """
        `Plnt` is what MB52 and most stock ALV layouts print, and it was not an alias —
        so the one file that names its plant that way had `location_id` stamped from
        config on all 14,442 rows, hiding that the extract covers two plants.
        """
        from inventory_planning.ingest.profiler import Profiler
        from inventory_planning.ingest.registry import AdapterRegistry
        frame = pd.DataFrame({
            "Material": [f"600{i % 40:04d}" for i in range(120)],
            header: [["5790", "5792"][i % 2] for i in range(120)],
            "Closing Stock": range(120),
        }).astype(str)
        profile = Profiler().profile(frame, source_name="stock.xlsx")
        mapping = AdapterRegistry()._assign_columns(
            profile, AdapterRegistry().contracts.get("inventory"))
        assert mapping.get("location_id") == header

    def test_the_placeholder_is_named_as_one(self, tmp_path):
        out = IngestBridge(verbose=False).load(self._files(tmp_path, with_plant=False))
        notes = "\n".join(out["_intake"].notes)
        assert PLACEHOLDER_LOCATION in notes
        assert "placeholder" in notes

    def test_it_is_silent_when_the_source_supplied_one(self, tmp_path):
        out = IngestBridge(verbose=False).load(self._files(tmp_path))
        assert "placeholder" not in "\n".join(out["_intake"].notes)

    def test_a_configured_node_is_not_a_placeholder(self, tmp_path):
        (tmp_path / "node_config.json").write_text(
            '{"location_id": "PL10", "currency": "USD"}', encoding="utf-8")
        node = load_planning_node(tmp_path)
        assert node.location_id == "PL10"
        assert not node.is_placeholder
        assert node.stamp_note(["po_history"]) is None

    def test_the_shipped_default_is_a_placeholder(self):
        node = load_planning_node(Path(__file__).parents[1] / "config")
        assert node.location_id == PLACEHOLDER_LOCATION
        assert node.is_placeholder

    def test_no_reader_hardcodes_the_placeholder(self):
        """
        Four readers wrote the literal instead of reading the config, so setting
        `location_id` in node_config.json changed some outputs and not others.
        """
        readers = (Path(__file__).parents[1] / "inventory_planning" / "readers")
        for path in readers.glob("*.py"):
            source = "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if not line.lstrip().startswith("#")
            )
            assert f'"{PLACEHOLDER_LOCATION}"' not in source, path.name
