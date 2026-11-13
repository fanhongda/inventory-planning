"""
A fingerprint hit has to be a document the adapter can actually read.

A signature is a sample of an adapter's mapped columns matched at 60%, and two exports
from one ERP share most of their vocabulary. So an inventory adapter learned from an
on-hand report recognised a planning sheet by `Item` and `Currency` alone — two of three
signature columns, the missing one being the on-hand quantity — and the sheet was landed
as stock. Its twenty-one months of demand were never looked at, because the fingerprint
path returns before the wide layout is ever seen.

`classify` already refused the equivalent: a contract that finds none of its identifying
columns scores zero. These tests hold the fingerprint path to the same standard.
"""

import numpy as np
import pandas as pd
import pytest

from inventory_planning.ingest.registry import BASIS_FINGERPRINT, AdapterRegistry

MONTHS = [f"{y} {m}" for y, ms in (
    (2018, ["Oct", "Nov", "Dec"]),
    (2019, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
) for m in ms]


@pytest.fixture
def store(tmp_path):
    """An adapter store holding only what each test puts in it."""
    return tmp_path / "adapters"


def _registry(store):
    return AdapterRegistry(adapters_dir=store)


def _on_hand_export(extra=None):
    frame = pd.DataFrame({
        "Item": ["206.02", "206.03", "130.91"],
        "Description": ["FIRERAY 50R", "FIRERAY 100R", "Terminal clamp"],
        "Subinventory": ["1001", "1004", "5700"],
        "On hand qty": [5, 0, 12],
        "std cst": [310.22, 368.88, 34.06],
        "Currency": ["USD"] * 3,
        "Status": ["ACT"] * 3,
    })
    return frame.assign(**(extra or {}))


def _planning_sheet():
    """Item attributes and a lead time, then monthly demand — and no stock at all."""
    head = ["Item", "Description", "std cst", "Currency", "Status",
            "S&OP Classification", "Lead time"]
    rows = [
        ["206.02", "FIRERAY 50R", 310.22, "USD", "OBS", "Sensors & Bases", np.nan],
        ["1603-9821", "SIMSYNC", 4.47, "USD", "ACT", "Parts & Accessories", 30],
        ["FX-1", "FLEX CABLE", 12.0, "USD", "ACT", "Systems", 45],
    ]
    return pd.DataFrame([r + [3] * len(MONTHS) for r in rows], columns=head + MONTHS)


def _learn(store, frame, name):
    registry = _registry(store)
    adapter = registry.route(frame, source_name=name).adapter
    registry.save(adapter)
    return adapter


class TestAnAdapterDoesNotClaimADocumentItCannotRead:
    def test_a_sheet_with_no_on_hand_column_is_not_stock(self, store):
        learned = _learn(store, _on_hand_export(), "Inventory on-hand 4th Nov-26.XLSX")
        assert learned.doc_type == "inventory"
        # The precondition that made this a defect rather than a theory: the
        # signature on its own does match.
        from inventory_planning.ingest.profiler import Profiler
        matched, _, _ = learned.fingerprint.match(Profiler().profile(_planning_sheet()))
        assert matched

        routed = _registry(store).route(_planning_sheet(), source_name="SG master.xlsx")
        assert routed.adapter.doc_type != "inventory"

    def test_the_wide_layout_is_then_seen(self, store):
        """Rejected by the adapter, it reaches `classify`, which reads the months."""
        _learn(store, _on_hand_export(), "Inventory on-hand 4th Nov-26.XLSX")
        routed = _registry(store).route(_planning_sheet(), source_name="SG master.xlsx")
        assert routed.adapter.doc_type == "demand_timeseries"

    def test_the_reason_names_what_was_passed_over_and_why(self, store):
        """Someone who expected the old route has to be able to see why it changed."""
        _learn(store, _on_hand_export(), "Inventory on-hand 4th Nov-26.XLSX")
        reason = _registry(store).route(_planning_sheet(), source_name="SG.xlsx").reason
        assert "passed over" in reason
        assert "qty_on_hand" in reason and "On hand qty" in reason

    def test_an_identifying_group_with_nothing_in_it_is_refused(self, store):
        """
        `planning_master` requires only `sku`, so the required-field test alone lets
        any item list through. What makes a planner's sheet one is that it carries a
        planner's parameter — and an item master with a lead time carries none.
        """
        worksheet = pd.DataFrame({
            "Item": ["A", "B", "C"], "Description": ["a", "b", "c"],
            "std cst": [1.0, 2.0, 3.0], "Currency": ["USD"] * 3,
            "Safety Stock": [5, 6, 7], "Reorder Point": [9, 9, 9],
        })
        assert _learn(store, worksheet, "planner worksheet.xlsx").doc_type == \
            "planning_master"

        item_master = worksheet.drop(columns=["Safety Stock", "Reorder Point"]) \
            .assign(**{"Lead time": [30, 45, 60]})
        routed = _registry(store).route(item_master, source_name="item master.xlsx")
        assert routed.adapter.doc_type == "item_master"
        assert "carries none of" in routed.reason


class TestTheSourceItWasLearnedFromStillMatches:
    def test_a_reexport_with_an_added_column_keeps_its_adapter(self, store):
        """The resilience `header_contains` exists for must survive the new check."""
        learned = _learn(store, _on_hand_export(), "Inventory on-hand 4th Nov-26.XLSX")
        again = _on_hand_export({"Lot": ["L1", "L2", "L3"]})
        routed = _registry(store).route(again, source_name="Inventory on-hand 11th Nov-26.XLSX")
        assert routed.confidence_basis == BASIS_FINGERPRINT
        assert routed.adapter.slug == learned.slug

    def test_a_field_the_adapter_derives_counts_as_carried(self, store):
        """A derivation stands in for its column; refusing it would undo the adapter."""
        learned = _learn(store, _on_hand_export(), "Inventory on-hand 4th Nov-26.XLSX")
        learned.derivations["qty_on_hand"] = "0"
        _registry(store).save(learned)
        routed = _registry(store).route(_on_hand_export().drop(columns=["On hand qty"]),
                                        source_name="Inventory on-hand.XLSX")
        assert routed.confidence_basis == BASIS_FINGERPRINT
        assert routed.adapter.slug == learned.slug
