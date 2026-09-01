"""
The declaration layer.

Two operational failures, both from the same real run:

  the SKU-agreement check blocked a whole-warehouse stock snapshot, whose codes
  legitimately appear nowhere else, and the only way past it was `allow_degraded=True`
  — a single switch that would also have waved through a purchase order whose open
  quantity had been mapped to a money column

  correcting a column meant editing headers in Excel or hand-freezing an adapter, and
  the edit broke the next run outright, because a frozen adapter reads its source
  column by name and the rename removed it

What is being pinned is not only that a declaration works, but that it cannot rot in
silence: a waiver has to expire, an expiry has to announce itself as an expiry, and a
declaration that matched nothing has to say so — someone believing they fixed something
they did not is worse than not having tried, because they have stopped looking.
"""

from datetime import date

import pytest
import yaml

from inventory_planning.quality.gates import BLOCK, Finding, GateReport, WARN
from inventory_planning.store.declarations import (
    DeclarationError, Declarations, GateWaiver, Override, SCOPE_MAPPING,
)

TODAY = date(2026, 9, 1)


def _write(tmp_path, payload) -> str:
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "declarations.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return config


def _finding(check="sku_agreement", doc_type="inventory", severity=BLOCK):
    return Finding(
        stage="intake", check=check, severity=severity,
        what=f"{doc_type} keys on something the other documents do not use.",
        why="Every join in the pipeline is on `sku`.",
        fix="Check which column the rest of the business uses.",
        evidence={"doc_type": doc_type, "agreement": 0.11},
    )


# ── Mapping overrides ────────────────────────────────────────────────────────

class TestAMappingOverrideCorrectsOneField:
    """The middle setting that did not exist: correct one column, infer the rest."""

    @pytest.fixture
    def declarations(self, tmp_path):
        return Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "mapping",
                "target": {"doc_type": "po_history",
                           "headers": ["PO_NO.", "Material", "PartNo."]},
                "field": "sku",
                "value": "Material",
                "reason": "PartNo. is the commercial part number; the rest of the "
                          "business joins on Material",
                "by": "jfanhon",
            }],
        }), today=TODAY)

    def test_it_names_the_column(self, declarations):
        assert declarations.column_map_for(
            "po_history", ["Vendor", "PO_NO.", "Material", "PartNo.", "Qty"]
        ) == {"sku": "Material"}

    def test_it_leaves_every_other_field_alone(self, declarations):
        """One key, not a whole hand-authored adapter."""
        assert list(declarations.column_map_for(
            "po_history", ["PO_NO.", "Material", "PartNo."])) == ["sku"]

    def test_it_does_not_apply_to_another_document(self, declarations):
        assert declarations.column_map_for(
            "open_so", ["PO_NO.", "Material", "PartNo."]) == {}

    def test_it_does_not_apply_when_the_headers_are_absent(self, declarations):
        assert declarations.column_map_for("po_history", ["Material", "Qty"]) == {}


class TestItSurvivesNextMonthsExport:
    """A file is identified by its bytes; next month's export of the same report is
    different bytes under a different name. A declaration tied to the file expires the
    moment it becomes useful."""

    @pytest.fixture
    def declarations(self, tmp_path):
        return Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "mapping",
                "target": {"doc_type": "po_history", "headers": ["Material", "PartNo."]},
                "field": "sku", "value": "Material", "reason": "the ERP number",
            }],
        }), today=TODAY)

    def test_a_renamed_re_export_still_matches(self, declarations):
        headers = ["Vendor", "Material", "PartNo.", "Qty"]
        assert declarations.column_map_for(
            "po_history", headers, source_name="Controls PO History 202607.xlsx")
        assert declarations.column_map_for(
            "po_history", headers, source_name="Controls PO History 202608.xlsx")

    def test_a_file_scoped_override_is_the_narrower_choice(self, tmp_path):
        declarations = Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "mapping",
                "target": {"doc_type": "po_history", "source": "one-off.xlsx"},
                "field": "sku", "value": "Material", "reason": "a one-time fix",
            }],
        }), today=TODAY)
        headers = ["Material", "PartNo."]
        assert declarations.column_map_for("po_history", headers, "one-off.xlsx")
        assert not declarations.column_map_for("po_history", headers, "next-month.xlsx")


# ── Gate waivers ─────────────────────────────────────────────────────────────

class TestAWaiverIsPerCheckPerDocument:
    """`allow_degraded=True` waves the whole run through. This does not."""

    @pytest.fixture
    def declarations(self, tmp_path):
        return Declarations.load(_write(tmp_path, {
            "gate_waivers": [{
                "check": "sku_agreement", "doc_type": "inventory",
                "reason": "Whole-warehouse snapshot — 11% is its expected shape",
                "by": "jfanhon", "expires": "2026-12-01",
            }],
        }), today=TODAY)

    def test_the_waived_finding_stops_blocking(self, declarations):
        report = GateReport(stage="intake", findings=[_finding()])
        assert report.blocking
        assert not declarations.waive(report).blocking

    def test_the_same_check_on_another_document_still_blocks(self, declarations):
        report = GateReport(stage="intake",
                            findings=[_finding(doc_type="item_master")])
        assert declarations.waive(report).blocking

    def test_another_check_on_the_same_document_still_blocks(self, declarations):
        report = GateReport(stage="intake",
                            findings=[_finding(check="semantic_failures")])
        assert declarations.waive(report).blocking

    def test_the_finding_is_still_reported(self, declarations):
        """A waiver that hid the finding would be indistinguishable from a check that
        never fired."""
        waived = declarations.waive(GateReport(stage="intake", findings=[_finding()]))
        assert len(waived.findings) == 1
        assert waived.findings[0].severity == WARN

    def test_it_says_who_waived_it_and_why(self, declarations):
        waived = declarations.waive(GateReport(stage="intake", findings=[_finding()]))
        finding = waived.findings[0]
        assert "jfanhon" in finding.why and "Whole-warehouse snapshot" in finding.why
        assert finding.evidence["waived_until"] == "2026-12-01"

    def test_a_run_with_both_a_waived_and_a_real_finding_still_stops(self, declarations):
        """The case the single switch got wrong: waiving the snapshot must not wave
        through an open quantity mapped to a money column."""
        report = GateReport(stage="intake", findings=[
            _finding(),
            _finding(check="semantic_failures", doc_type="open_po"),
        ])
        waived = declarations.waive(report)
        assert len(waived.blocking) == 1
        assert waived.blocking[0].check == "semantic_failures"


class TestWaiversMustExpire:
    def test_a_waiver_without_an_expiry_is_refused(self, tmp_path):
        with pytest.raises(DeclarationError, match="permanently disabled check"):
            Declarations.load(_write(tmp_path, {
                "gate_waivers": [{"check": "sku_agreement", "doc_type": "inventory",
                                  "reason": "it is fine"}],
            }))

    def test_an_expired_waiver_stops_waiving(self, tmp_path):
        declarations = Declarations.load(_write(tmp_path, {
            "gate_waivers": [{"check": "sku_agreement", "doc_type": "inventory",
                              "reason": "expected shape", "expires": "2026-08-01"}],
        }), today=TODAY)
        assert declarations.waive(GateReport("intake", [_finding()])).blocking

    def test_the_expiry_announces_itself_as_an_expiry(self, tmp_path):
        """Not as a fresh discovery. The reader needs to know the check is back."""
        declarations = Declarations.load(_write(tmp_path, {
            "gate_waivers": [{"check": "sku_agreement", "doc_type": "inventory",
                              "reason": "expected shape", "expires": "2026-08-01"}],
        }), today=TODAY)
        notes = "\n".join(declarations.notes())
        assert "expired 2026-08-01" in notes
        assert "applies again" in notes


# ── Hygiene ──────────────────────────────────────────────────────────────────

class TestADeclarationThatMatchedNothingSaysSo:
    """Believing you fixed something you did not is worse than not having tried."""

    @pytest.fixture
    def declarations(self, tmp_path):
        return Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "mapping",
                "target": {"doc_type": "po_history", "headers": ["Part Number"]},
                "field": "sku", "value": "Part Number",
                "reason": "the column was renamed after this was written",
            }],
        }), today=TODAY)

    def test_it_is_unused_when_nothing_matched(self, declarations):
        declarations.column_map_for("po_history", ["Material", "PartNo."])
        assert len(declarations.unused()) == 1

    def test_the_note_names_the_headers_it_was_looking_for(self, declarations):
        declarations.column_map_for("po_history", ["Material", "PartNo."])
        notes = "\n".join(declarations.notes())
        assert "matched no document" in notes and "Part Number" in notes

    def test_an_applied_override_is_not_reported(self, declarations):
        declarations.column_map_for("po_history", ["Part Number", "Material"])
        assert declarations.unused() == []


class TestTheFileMustBeTrustworthy:
    """Absent is fine. Half-understood is not — the half that was dropped is the
    correction someone is relying on."""

    def test_no_file_is_not_an_error(self, tmp_path):
        declarations = Declarations.load(tmp_path / "config")
        assert declarations.overrides == [] and declarations.waivers == []

    def test_malformed_yaml_is_refused(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "declarations.yaml").write_text("overrides: [{", encoding="utf-8")
        with pytest.raises(DeclarationError, match="not valid YAML"):
            Declarations.load(config)

    def test_an_unknown_scope_is_refused(self, tmp_path):
        with pytest.raises(DeclarationError, match="scope"):
            Declarations.load(_write(tmp_path, {
                "overrides": [{"scope": "colour", "field": "sku", "value": "Material",
                               "reason": "why not"}],
            }))

    def test_a_reason_is_required(self, tmp_path):
        """A declaration overrules the pipeline's own evidence. Whoever reads it in six
        months needs to know on what grounds."""
        with pytest.raises(DeclarationError, match="say why"):
            Declarations.load(_write(tmp_path, {
                "overrides": [{"scope": "mapping", "field": "sku", "value": "Material",
                               "target": {"doc_type": "po_history"}}],
            }))

    def test_a_bad_date_is_refused(self, tmp_path):
        with pytest.raises(DeclarationError, match="not a date"):
            Declarations.load(_write(tmp_path, {
                "gate_waivers": [{"check": "sku_agreement", "reason": "x",
                                  "expires": "next tuesday"}],
            }))


class TestOtherScopes:
    def test_a_parameter_override_is_found_by_target(self, tmp_path):
        declarations = Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "parameter",
                "target": {"item_uid": "sap_matnr:1003524", "location": "5051"},
                "field": "min_order_qty", "value": 500,
                "reason": "supplier raised the MOQ, master not yet updated",
            }],
        }), today=TODAY)
        assert declarations.values_for(
            "parameter", item_uid="sap_matnr:1003524", location="5051"
        ) == {"min_order_qty": 500}

    def test_it_does_not_leak_to_another_item(self, tmp_path):
        declarations = Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "parameter", "target": {"item_uid": "sap_matnr:1003524"},
                "field": "min_order_qty", "value": 500, "reason": "raised MOQ",
            }],
        }), today=TODAY)
        assert declarations.values_for("parameter", item_uid="sap_matnr:419348") == {}


class TestSummary:
    def test_nothing_declared_says_nothing(self, tmp_path):
        assert Declarations.load(tmp_path / "config").summary() == ""
        assert Declarations.load(tmp_path / "config").notes() == []


# ── The seam ─────────────────────────────────────────────────────────────────

class TestItReachesIntake:
    """
    A declaration that only works in isolation is a declaration nobody can use. These
    go through `Intake`, which is where the corrected column has to actually arrive.

    The document is the shape that failed: a purchase history carrying both the ERP
    material number and a commercial part number, where the scorer picks the part
    number because `Material` is emptier — a six-hundredth of a point of lexical lead,
    erased by the null rate.
    """

    @staticmethod
    def _po_history(n=40):
        import pandas as pd
        return pd.DataFrame({
            "Vendor_Name": ["ACME"] * n,
            "Order_date": ["2026-01-05"] * n,
            "PO_NO.": [f"45110209{i:02d}" for i in range(n)],
            # Emptier than the part number, which is what loses it the field.
            "Material": [f"53913{i:02d}" if i % 3 else "" for i in range(n)],
            "PartNo.": [f"MS-FAC25{i:02d}-0" for i in range(n)],
            "Qty": ["8"] * n,
        })

    @pytest.fixture
    def declared(self, tmp_path):
        return Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "mapping",
                "target": {"doc_type": "po_history",
                           "headers": ["Material", "PartNo."]},
                "field": "sku", "value": "Material",
                "reason": "PartNo. is the commercial part number; the rest of the "
                          "business joins on Material",
                "by": "jfanhon",
            }],
        }), today=TODAY)

    def _load(self, declarations=None):
        from inventory_planning.ingest.intake import Intake
        intake = Intake(verbose=False, declarations=declarations)
        return intake.load_frame(self._po_history(), source_name="poh.xlsx",
                                 doc_type_hint="po_history")

    def test_without_a_declaration_the_scorer_decides(self):
        """The premise. If the scorer ever stops preferring the fuller column, this
        test is what will say so."""
        assert self._load().route.adapter.column_map["sku"] == "PartNo."

    def test_the_declaration_wins(self, declared):
        assert self._load(declared).route.adapter.column_map["sku"] == "Material"

    def test_the_corrected_column_is_what_lands_in_the_frame(self, declared):
        frame = self._load(declared).frame
        assert "5391301" in set(frame["sku"].dropna().astype(str))
        assert "MS-FAC2501-0" not in set(frame["sku"].dropna().astype(str))

    def test_every_other_field_is_still_inferred(self, declared):
        mapped = self._load(declared).route.adapter.column_map
        assert mapped["po_number"] == "PO_NO."
        assert mapped["supplier"] == "Vendor_Name"
        assert mapped["po_qty"] == "Qty"

    def test_the_shared_adapter_is_not_mutated(self, declared):
        """Adapters are cached per type. Editing one in place would carry a
        declaration written for the purchase history into the open PO report."""
        declared_route = self._load(declared).route
        plain_route = self._load().route
        assert declared_route.adapter.column_map["sku"] == "Material"
        assert plain_route.adapter.column_map["sku"] == "PartNo."

    def test_a_column_the_file_lacks_is_reported_not_applied(self, tmp_path):
        from inventory_planning.ingest.intake import Intake
        declarations = Declarations.load(_write(tmp_path, {
            "overrides": [{
                "scope": "mapping",
                "target": {"doc_type": "po_history", "headers": ["Material"]},
                "field": "sku", "value": "Part Number",
                "reason": "written before the column was renamed",
            }],
        }), today=TODAY)
        intake = Intake(verbose=False, declarations=declarations)
        doc = intake.load_frame(self._po_history(), source_name="poh.xlsx",
                                doc_type_hint="po_history")
        # Not applied — blanking the field would read downstream as an absent measure
        # rather than as a mistake.
        assert doc.route.adapter.column_map["sku"] == "PartNo."
        assert any("does not have" in note for note in intake._declaration_notes)

