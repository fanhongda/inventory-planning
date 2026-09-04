"""
Exposure — that the money behind an assumption is measured, and measured once.

The failure being guarded is not a wrong total. It is a list of assumptions in file
order, where the one governing twelve million of stock sits below the one governing
four rows of a sample, and the reader stops before reaching it.
"""

import numpy as np
import pandas as pd
import pytest

from inventory_planning.ingest.contract import default_registry
from inventory_planning.ingest.exposure import (
    Assumption, BASIS_ASSUMED, BASIS_DECLARED, GOVERNS_ALL, GOVERNS_BLANKS,
    assemble, measure, measure_one, money_field, qty_field,
)


class _Field:
    def __init__(self, unit=None):
        self.unit = unit


class _Contract:
    """The only part of a DocContract this module reads: field name -> declared unit."""

    def __init__(self, **units):
        self.fields = {name: _Field(unit) for name, unit in units.items()}


def _ledger(*items):
    from inventory_planning.ingest.exposure import RestingLedger
    return RestingLedger(items=list(items))


def _assumption(**kw):
    base = dict(doc_type="inventory", field="currency", value="CNY",
                basis=BASIS_DECLARED, governs=GOVERNS_ALL)
    base.update(kw)
    return Assumption(**base)


class TestWhichColumnIsMoney:

    def test_the_unit_decides_not_the_name(self):
        """
        `Still to be delivered (value)` mapped as an open quantity is the defect the
        intake summary exists for. A module that picked its money column by reading the
        header would be the second place to make it.
        """
        contract = _Contract(open_qty="currency", open_amount="base_uom")
        frame = pd.DataFrame({"open_qty": [1.0, 2.0], "open_amount": [10.0, 20.0]})
        assert money_field(frame, contract) == "open_qty"
        assert qty_field(frame, contract) == "open_amount"

    def test_two_money_fields_are_not_added_together(self):
        """po_history carries two `unit: currency` fields. Summing them doubles it."""
        contract = _Contract(net_value="currency", gross_value="currency",
                             qty="base_uom")
        frame = pd.DataFrame({
            "net_value": [100.0, 200.0, 300.0],
            "gross_value": [110.0, np.nan, np.nan],
            "qty": [1.0, 1.0, 1.0],
        })
        rest = measure_one(frame, contract, _assumption(doc_type="po_history"))
        assert rest.exposure.money_field == "net_value"
        assert rest.exposure.money == pytest.approx(600.0)

    def test_a_document_with_no_measure_reports_unknown_not_zero(self):
        contract = _Contract(sku="identifier", supplier="identifier")
        frame = pd.DataFrame({"sku": ["A", "B"], "supplier": ["S1", "S2"]})
        rest = measure_one(frame, contract, _assumption(doc_type="item_master"))
        assert rest.exposure.rows == 2
        assert rest.exposure.money is None
        assert not rest.exposure.known


class TestWhatTheAssumptionGoverns:

    def test_an_unmapped_field_governs_every_row_exactly(self):
        contract = _Contract(inventory_value="currency", qty_on_hand="base_uom")
        frame = pd.DataFrame({"inventory_value": [1_000.0] * 3,
                              "qty_on_hand": [5.0] * 3})
        rest = measure_one(frame, contract, _assumption(governs=GOVERNS_ALL))
        assert rest.exposure.rows == 3
        assert rest.exposure.money == pytest.approx(3_000.0)
        assert rest.exposure.upper_bound is False

    def test_a_default_over_a_mapped_column_reaches_only_its_own_rows(self):
        """
        Where the export does carry the column, the declaration filled the blanks and
        the rows it filled cannot be recovered afterwards. The exposure is the rows now
        reading that value — an upper bound, and marked as one.
        """
        contract = _Contract(inventory_value="currency")
        frame = pd.DataFrame({
            "currency": ["CNY", "USD", "CNY"],
            "inventory_value": [100.0, 900.0, 100.0],
        })
        rest = measure_one(frame, contract, _assumption(governs=GOVERNS_BLANKS))
        assert rest.exposure.rows == 2
        assert rest.exposure.money == pytest.approx(200.0)
        assert rest.exposure.upper_bound is True

    def test_an_exact_assumption_is_not_row_matched(self):
        """
        The adapter normalises what it writes, so the stored value need not equal the
        declared one. Row-matching an assumption that governs everything would report
        zero exposure on exactly the assumption most worth reading.
        """
        contract = _Contract(inventory_value="currency")
        frame = pd.DataFrame({"currency": ["cny", "cny"],
                              "inventory_value": [50.0, 50.0]})
        rest = measure_one(frame, contract, _assumption(value="CNY", governs=GOVERNS_ALL))
        assert rest.exposure.rows == 2
        assert rest.exposure.money == pytest.approx(100.0)


class TestDerivedExposure:

    def test_quantity_times_price_when_the_export_dropped_the_value_column(self):
        """The contract says this document carries money; this export did not."""
        contract = _Contract(open_amount="currency", open_qty="base_uom",
                             unit_cost="currency_per_base_uom")
        frame = pd.DataFrame({"open_qty": [10.0, 5.0], "unit_cost": [3.0, 4.0]})
        rest = measure_one(frame, contract, _assumption(doc_type="open_po"))
        assert rest.exposure.money == pytest.approx(50.0)
        assert rest.exposure.money_derived is True
        assert "×" in rest.exposure.money_field

    def test_a_document_whose_contract_holds_no_money_is_given_no_total(self):
        """
        The first real run reported an item master's currency assumption as
        `Σ min_order_qty × unit_cost 150,058` — the value of one minimum order of every
        item. It computes, it sorts, and it refers to nothing. A master has no extended
        value in its contract, so it gets none here.
        """
        contract = _Contract(min_order_qty="base_uom", order_multiple="base_uom",
                             unit_cost="currency_per_base_uom")
        frame = pd.DataFrame({"min_order_qty": [100.0, 250.0],
                              "unit_cost": [3.0, 4.0]})
        rest = measure_one(frame, contract, _assumption(doc_type="item_master"))
        assert rest.exposure.money is None
        assert rest.exposure.qty is None
        assert rest.exposure.priced_rows == 2
        assert rest.exposure.priced_field == "unit_cost"
        assert "a rate, not a total" in _ledger(rest).summary()

    def test_a_priced_master_still_sorts_below_any_measured_money(self):
        """
        Unsizeable is not the same as unimportant — a standard cost read in the wrong
        currency misprices every EOQ — but it cannot outrank a number.
        """
        contracts = {"item_master": _Contract(unit_cost="currency_per_base_uom"),
                     "open_so": _Contract(order_value="currency")}
        frames = {"item_master": pd.DataFrame({"unit_cost": [3.0] * 500}),
                  "open_so": pd.DataFrame({"order_value": [1.0]})}
        ledger = measure([_assumption(doc_type="item_master"),
                          _assumption(doc_type="open_so")], frames, contracts)
        assert [r.assumption.doc_type for r in ledger.ordered] == [
            "open_so", "item_master"]


class TestTheLedgerIsSortedByWhatIsAtStake:

    def test_largest_money_first_and_unmeasurable_last(self):
        contracts = {
            "inventory": _Contract(inventory_value="currency"),
            "open_so": _Contract(order_value="currency"),
            "item_master": _Contract(sku="identifier"),
        }
        frames = {
            "inventory": pd.DataFrame({"inventory_value": [12_400_000.0]}),
            "open_so": pd.DataFrame({"order_value": [4_000.0]}),
            "item_master": pd.DataFrame({"sku": ["A"]}),
        }
        assumptions = [
            _assumption(doc_type="item_master"),
            _assumption(doc_type="open_so"),
            _assumption(doc_type="inventory"),
        ]
        ledger = measure(assumptions, frames, contracts, reporting_currency="USD")
        assert [r.assumption.doc_type for r in ledger.ordered] == [
            "inventory", "open_so", "item_master"]

    def test_the_summary_names_the_document_the_value_and_the_money(self):
        contracts = {"inventory": _Contract(inventory_value="currency")}
        frames = {"inventory": pd.DataFrame({"inventory_value": [12_400_000.0]})}
        ledger = measure([_assumption(by="jfanhon")], frames, contracts,
                         reporting_currency="USD")
        text = ledger.summary()
        assert "inventory" in text and "CNY" in text
        assert "12,400,000" in text
        assert "jfanhon" in text

    def test_an_empty_ledger_prints_nothing(self):
        assert measure([], {}, {}).summary() == ""


class TestAssemble:

    def test_a_currency_declaration_says_what_it_replaced(self):
        out = assemble([_assumption()], [], reporting_currency="USD")
        assert len(out) == 1
        assert "USD" in out[0].instead_of

    def test_a_defaulted_document_joins_the_list_unattributed(self):
        out = assemble([], ["open_po"], reporting_currency="USD")
        assert [a.basis for a in out] == [BASIS_ASSUMED]
        assert out[0].value == "USD"
        assert out[0].by == ""

    def test_a_declared_document_is_not_also_reported_as_assuming(self):
        """
        Both labels on one document would show the same exposure twice, under two
        statements that contradict each other.
        """
        out = assemble([_assumption(doc_type="inventory")], ["inventory", "open_po"],
                       reporting_currency="USD")
        assert len(out) == 2
        inventory = [a for a in out if a.doc_type == "inventory"]
        assert len(inventory) == 1
        assert inventory[0].basis == BASIS_DECLARED


class TestAgainstTheRealContracts:

    def test_the_shipped_contracts_expose_the_units_this_module_reads(self):
        """
        The stub above stands in for a DocContract. If the real one stopped carrying
        `unit`, every exposure would silently become unmeasurable — the failure mode
        this whole module is meant to prevent, arriving through its own dependency.
        """
        contract = default_registry().get("inventory")
        frame = pd.DataFrame({"inventory_value": [1.0], "qty_on_hand": [2.0]})
        assert money_field(frame, contract) == "inventory_value"
        assert qty_field(frame, contract) == "qty_on_hand"
