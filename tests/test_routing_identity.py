"""
Regression tests for a production misroute.

A real run handed six files to intake and got back:

    item_master  3,018 rows  <- Regional PL30 master.xlsx + Regional PL30 sales history.xlsx

The sales history had been absorbed into the item master, taking the demand signal
with it, and the run aborted on `missing required input(s): demand_signal`. Every
contract test passed — a misroute produces correct arithmetic on the wrong premise.

The cause was in the contract, not the router. `item_master` declared only `sku`
required, so its required-coverage score was a full 1.0 against any file with an item
column, which outranked the document that genuinely fitted. `identifying_any` states
the missing invariant: a master earns the name by carrying at least one planning
parameter.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.contract import DocContract
from inventory_planning.ingest.registry import AdapterRegistry

SAMPLE_DIR = Path(__file__).parents[1] / "sample_data"


@pytest.fixture(scope="module")
def registry():
    return AdapterRegistry()


def _route(registry, frame, name="file.xlsx"):
    return registry.route(frame.astype(str), source_name=name).doc_type


# ── The reported failure ─────────────────────────────────────────────────────


class TestSalesHistoryIsNotAnItemMaster:

    def test_erp_shaped_sales_history_routes_to_sales_history(self, registry):
        sales = pd.DataFrame({
            "Material": ["A-1", "A-2"],
            "Material Description": ["Valve", "Actuator"],
            "Delivered  Date": ["2026-07-01", "2026-07-02"],
            "Delivered  Quantity": ["10", "20"],
            "Net Value": ["100", "200"],
            "Sold-to": ["C1", "C2"],
        })
        assert _route(registry, sales, "Regional PL30 sales history.xlsx") == "sales_history"

    def test_sales_history_with_an_unrecognised_date_still_is_not_a_master(self, registry):
        """
        The production case. With no mappable date the sales contract cannot score
        full marks — but that must not hand the file to a contract that requires
        nothing beyond an item number.
        """
        sales = pd.DataFrame({
            "Material": ["A-1", "A-2"],
            "Material Description": ["Valve", "Actuator"],
            "Fiscal Per": ["2026007", "2026007"],
            "Qty Shipped": ["10", "20"],
            "Sold-to": ["C1", "C2"],
        })
        assert _route(registry, sales, "sales.xlsx") != "item_master"

    def test_a_bare_item_list_is_not_a_master_either(self, registry):
        bare = pd.DataFrame({
            "Material": ["A-1", "A-2"],
            "Material Description": ["Valve", "Actuator"],
        })
        assert _route(registry, bare, "items.xlsx") != "item_master"


# ── The invariant itself ─────────────────────────────────────────────────────


class TestIdentifyingAny:

    def test_item_master_declares_what_makes_it_one(self, registry):
        contract = registry.contracts.get("item_master")
        assert contract.identifying_any, "sku alone cannot identify an item master"

    def test_planning_master_declares_it_too(self, registry):
        assert registry.contracts.get("planning_master").identifying_any

    def test_a_contract_finding_none_of_its_identity_scores_zero(self, registry):
        """Not a weak match — the wrong document."""
        po_history = pd.read_csv(SAMPLE_DIR / "po_history.csv", dtype=str)
        from inventory_planning.ingest.profiler import Profiler
        profile = Profiler().profile(po_history, source_name="po_history.csv")
        _, _, _ = registry.classify(profile, po_history)

        contract = registry.contracts.get("item_master")
        hit = set(registry._assign_columns(profile, contract))
        assert not (set(contract.identifying_any[0]) & hit), (
            "a PO history carries no planning parameter, so it cannot be a master"
        )

    def test_incoterm_alone_does_not_identify_a_master(self, registry):
        """
        It did at first, which let a PO history qualify. An incoterm is a commercial
        term that appears on transactions too; only a planning parameter is evidence.
        """
        assert "incoterm" not in registry.contracts.get("item_master").identifying_any[0]

    def test_unknown_field_in_identifying_any_is_rejected_at_load(self):
        with pytest.raises(ValueError, match="identifying_any"):
            DocContract.parse({
                "doc_type": "bogus",
                "fields": {"sku": {"type": "string", "required": True}},
                "identifying_any": [["no_such_field"]],
            })


# ── A real master must still be recognised ───────────────────────────────────


class TestMastersStillRoute:

    def test_item_master_with_lead_time(self, registry):
        master = pd.DataFrame({
            "Material": ["A-1", "A-2"],
            "Material Description": ["Valve", "Actuator"],
            "Vendor Name": ["ACME", "GLOBEX"],
            "Planned Deliv. Time": ["45", "90"],
            "MOQ": ["50", "25"],
            "Standard Cost": ["120", "340"],
        })
        assert _route(registry, master, "Regional PL30 master.xlsx") == "item_master"

    def test_item_master_with_only_moq_and_multiple(self, registry):
        """A master need not carry a lead time — any planning parameter identifies it."""
        master = pd.DataFrame({
            "Material": ["A-1", "A-2"],
            "Material Description": ["Valve", "Actuator"],
            "MOQ": ["50", "25"],
            "Rounding Value": ["25", "5"],
        })
        assert _route(registry, master, "master.xlsx") == "item_master"

    def test_planner_worksheet(self, registry):
        sheet = pd.DataFrame({
            "Item Code": ["A-1", "A-2"],
            "Safety Stock": ["500", "100"],
            "ROP": ["900", "300"],
            "Lead Time": ["45", "90"],
            "Review Cycle": ["30", "30"],
        })
        assert _route(registry, sheet, "planner.xlsx") == "planning_master"


# ── No regression on the documents that already worked ───────────────────────


class TestCoreDocumentsUnaffected:

    @pytest.mark.parametrize("doc", ["sales_history", "po_history", "open_so",
                                     "open_po", "inventory"])
    def test_sample_document_routes_to_itself(self, registry, doc):
        raw = pd.read_csv(SAMPLE_DIR / f"{doc}.csv", dtype=str)
        assert registry.route(raw, source_name=f"{doc}.csv").doc_type == doc
