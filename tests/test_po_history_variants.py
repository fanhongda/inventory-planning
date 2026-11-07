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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.adapter import Adapter
from inventory_planning.orchestrator import InventoryPlanner
from inventory_planning.ingest.contract import DocContract
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


# ── Fourth shape: a dispatch date and no goods receipt ────────────────────────


def _dispatch_rows(confirmations: bool = False) -> pd.DataFrame:
    """
    A purchase export with no goods receipt: the supplier's confirmed delivery date is
    the furthest the clock can be measured.

    One row per PO line by default — the shape the frame has by the time the lead time
    is computed, because the intake collapses the confirmation rows first. Pass
    `confirmations=True` for the raw shape, two rows per line carrying the same date,
    which is what `Adapter._collapse_repeats` is given.
    """
    rows = []
    for po, sku, vendor, qty, ordered, confirmed in (
        ("4500001", "SKU-001", "V-ALPHA", 100, "2026-01-05", "2026-02-20"),
        ("4500002", "SKU-001", "V-ALPHA", 200, "2026-02-05", "2026-03-20"),
        ("4500003", "SKU-002", "V-BETA", 50, "2026-02-20", "2026-03-18"),
    ):
        common = {"po_number": po, "po_line_number": "10", "sku": sku,
                  "supplier": vendor, "po_date": pd.Timestamp(ordered),
                  "po_qty": qty, "open_qty": 0,
                  "confirmed_delivery_date": pd.Timestamp(confirmed)}
        if confirmations:
            rows.append({**common, "confirmation_category": "AB"})
            rows.append({**common, "confirmation_category": "LA"})
        else:
            rows.append({**common, "confirmation_category": "LA"})
    return pd.DataFrame(rows)


class TestTheSameLineToldTwice:
    """
    With no goods receipt the two confirmation rows of a line differ on nothing but the
    category, the natural key degrades to the line, and `Adapter._apply_rollup` sums
    their measures — 100 + 100 into 200, before anything downstream can see there was
    ever more than one row. The sum is not recoverable, so this has to run inside the
    intake rather than after it, which is why the receipt-grain consolidation in the
    reader cannot cover this case.
    """

    @staticmethod
    def _collapse(df):
        contract = DocContract.load(
            Path(__file__).parents[1] / "inventory_planning" / "ingest" /
            "contracts" / "po_history.yaml")
        adapter = Adapter.__new__(Adapter)
        log = []
        return adapter._collapse_repeats(df, contract, log), log

    def test_the_repeat_is_dropped_before_the_rollup_can_sum_it(self):
        out, log = self._collapse(_dispatch_rows(confirmations=True))
        assert len(out) == 3
        assert out["po_qty"].sum() == 350
        assert any("told twice" in str(s) for s in log)

    def test_a_row_differing_anywhere_else_is_a_different_row(self):
        """
        What keeps partial deliveries and schedule lines intact: only a row identical
        outside the marker is a repeat. "Identical measures within the group" would
        also describe two schedule lines that happen to be equal halves of a line, and
        summing those is correct.
        """
        df = _dispatch_rows(confirmations=True)
        df.loc[df["confirmation_category"] == "LA", "po_qty"] = 999
        out, _ = self._collapse(df)
        assert len(out) == 6

    def test_a_file_with_no_category_column_is_untouched(self):
        df = _dispatch_rows(confirmations=True).drop(columns=["confirmation_category"])
        out, log = self._collapse(df)
        assert len(out) == len(df)
        assert not log


def _transit_config(tmp_path: Path, suppliers: dict, default=None,
                    modes: dict = None) -> Path:
    """
    A config directory whose `supplier_transit.csv` holds `suppliers`.

    `suppliers` maps a supplier prefix to transit days; `modes` optionally gives each
    one a transport mode. `default` becomes the `*` row — the blanket for everyone not
    named, the same convention `location_status.json` uses.
    """
    out = tmp_path / "config"
    out.mkdir(parents=True, exist_ok=True)
    for src in CONFIG_DIR.glob("*.json"):
        (out / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    modes = modes or {}
    lines = ["supplier,transport_mode,transit_days"]
    if default is not None:
        lines.append(f"*,{modes.get('*', 'AIR')},{default}")
    for name, days in suppliers.items():
        lines.append(f"{name},{modes.get(name, '')},{days}")
    (out / "supplier_transit.csv").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    return out


class TestADispatchDateAndNoReceipt:

    def test_the_lead_time_is_the_dispatch_leg_plus_the_stated_transit(self, tmp_path):
        """
        Before this, such a file measured nothing: `receive_date` refuses the bare
        "delivery date" spellings — rightly, since ending the clock at a date the goods
        had not arrived on understates every lead time by the whole transit leg — so
        every item fell back to the item master's planned delivery time.
        """
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21, "V-BETA": 7}))
        out = reader._post_process(_dispatch_rows())
        rows = out.set_index("po_number")
        # 2026-01-05 -> 2026-02-20 is 46 days to dispatch, plus 21 in transit.
        assert rows.loc["4500001", "lead_time_days"] == 67.0
        assert rows.loc["4500003", "lead_time_days"] == 33.0
        assert set(out["lt_basis"]) == {"promised_dispatch_plus_transit"}

    def test_the_stated_leg_is_named_on_the_row(self, tmp_path):
        """
        Part measured, part stated, and both halves visible. `transit_days` is also what
        `should_be._transit_days` reads in place of its `transit_share_of_lt` guess.
        """
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21, "V-BETA": 7}))
        out = reader._post_process(_dispatch_rows())
        assert out.set_index("po_number").loc["4500001", "transit_days"] == 21.0

    def test_a_supplier_with_no_transit_time_gets_no_lead_time(self, tmp_path):
        """
        Not a short one. Treating a missing transit time as zero understates the reorder
        point, so the order goes out late and the shortage lands before the goods — the
        item falls back to a stated planned delivery time instead, labelled as such.
        """
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21}))
        out = reader._post_process(_dispatch_rows())
        beta = out[out["supplier"] == "V-BETA"]
        assert len(beta) == 1
        assert pd.isna(beta["lead_time_days"].iloc[0])
        assert pd.isna(beta["transit_days"].iloc[0])

    def test_a_default_covers_the_unlisted_suppliers_when_one_is_set(self, tmp_path):
        reader = POHistoryReader(
            _transit_config(tmp_path, {"V-ALPHA": 21}, default=14))
        out = reader._post_process(_dispatch_rows())
        assert out[out["supplier"] == "V-BETA"]["transit_days"].iloc[0] == 14.0

    def test_the_longest_matching_prefix_wins(self):
        """
        Export columns carry the code and the name in one field, so matching is on the
        prefix — the rule `supplier_incoterm.json` already uses. Longest wins, so a
        specific vendor can override a group.
        """
        table = {"SIM": 99.0, "SIM001": 30.0}
        assert POHistoryReader._transit_for("SIM001 TYCO", table, None) == 30.0
        assert POHistoryReader._transit_for("SIM002 OTHER", table, None) == 99.0
        assert POHistoryReader._transit_for("ACME", table, None) is None
        assert POHistoryReader._transit_for("ACME", table, 12.0) == 12.0

    def test_a_receipt_date_still_wins_where_the_file_has_one(self, tmp_path):
        """
        A measured order-to-arrival lead time beats a measured order-to-dispatch one
        plus a number from a config file, so the receipt path stays the default.
        """
        df = _dispatch_rows()
        df["receive_date"] = pd.NaT
        df.loc[df["confirmation_category"] == "LA", "receive_date"] = pd.Timestamp("2026-03-01")
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21, "V-BETA": 7}))
        out = reader._post_process(df)
        assert set(out["lt_basis"]) == {"receipt"}

    def test_it_is_a_commitment_and_is_not_labelled_measured(self, tmp_path):
        """
        The date is what the supplier *committed* to. A supplier who always promises 45
        days and is always ten days late looks punctual in it, so calling the result
        `measured` would let an optimistic promise outrank the planned delivery time a
        person maintains — the one place the promise is most likely to be wrong.
        """
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21, "V-BETA": 7}))
        lt = reader.compute_supplier_lt(reader._post_process(_dispatch_rows()))
        assert set(lt["lt_source"]) == {"supplier_confirmed"}
        assert set(lt["lt_sigma_basis"]) == {"promise_spread_floor"}

    def test_a_receipt_still_outranks_a_confirmation(self, tmp_path):
        """
        The half of the ordering that is not a judgement call: a measurement of what
        arrived beats a commitment about what would, whatever the commitment says.
        """
        from inventory_planning.policy.assemble import build_sku_attributes

        classified = pd.DataFrame([{
            "sku": "SKU-001", "demand_mean_rolling": 10.0, "demand_std_rolling": 2.0,
            "demand_cv": 0.2, "stocking_class": "stocking-high",
            "service_level": 0.95, "z_score": 1.65,
        }])
        measured = pd.DataFrame([{
            "sku": "SKU-001", "supplier": "V-ALPHA", "wma_lead_time_days": 88.0,
            "lt_std_days": 9.0, "sample_count": 9, "lt_basis": "receipt",
            "lt_source": "measured",
        }])
        item_master = pd.DataFrame([{"sku": "SKU-001", "lead_time_days": 95.0}])
        attributes, _ = build_sku_attributes(
            classified, measured, inventory=None, item_master=item_master)
        row = attributes.set_index("sku").loc["SKU-001"]
        assert row["lead_time_days_source"] == "measured"
        assert row["lead_time_days"] == 88.0

    def test_a_receipt_based_pair_keeps_the_measured_label(self, tmp_path):
        """The relabelling must not reach the case that really is measured."""
        df = _dispatch_rows()
        df["receive_date"] = pd.Timestamp("2026-04-01")
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21}))
        lt = reader.compute_supplier_lt(reader._post_process(df))
        assert set(lt["lt_source"]) == {"measured"}

    def test_a_confirmed_lead_time_lands_in_its_own_tier(self, tmp_path):
        """
        End of the chain: `assemble` splits the promise out of the measured channel, so
        `crosscheck` ranks it in its own tier. Without the split it arrived as
        `measured` and beat every stated value in the run — including a planned
        delivery time maintained by a person who can see lateness the system never
        recorded.

        `AUTHORITY_ORDER` then puts it above the masters: per order and current beats a
        standing field, and on an extract with no goods receipt nobody has been
        checking the promise against anything, so a stale master is the likelier of the
        two to be wrong. It stays below a receipt, always.
        """
        from inventory_planning.policy.assemble import build_sku_attributes

        classified = pd.DataFrame([{
            "sku": "SKU-001", "demand_mean_rolling": 10.0, "demand_std_rolling": 2.0,
            "demand_cv": 0.2, "stocking_class": "stocking-high",
            "service_level": 0.95, "z_score": 1.65,
        }])
        supplier_lt = pd.DataFrame([{
            "sku": "SKU-001", "supplier": "V-ALPHA", "wma_lead_time_days": 65.0,
            "lt_std_days": 1.5, "sample_count": 9,
            "lt_basis": "promised_dispatch_plus_transit",
            "lt_source": "supplier_confirmed", "transit_days": 21.0,
        }])
        item_master = pd.DataFrame([{"sku": "SKU-001", "lead_time_days": 95.0}])
        attributes, _ = build_sku_attributes(
            classified, supplier_lt, inventory=None, item_master=item_master)
        row = attributes.set_index("sku").loc["SKU-001"]
        assert row["lead_time_days_source"] == "supplier_confirmed"
        assert row["lead_time_days"] == 65.0
        # And the defect this closes: it is no longer in the measured channel, where it
        # outranked every stated value in the run.
        assert row["lead_time_days_source"] != "measured"

    def test_a_present_but_empty_receipt_column_does_not_win(self, tmp_path):
        """
        It used to: the branch tested whether the column existed, not whether anything
        was in it, so a file carrying an empty `receive_date` beside a populated
        dispatch date produced a column of nulls and no lead time at all.
        """
        df = _dispatch_rows()
        df["receive_date"] = pd.NaT
        reader = POHistoryReader(_transit_config(tmp_path, {"V-ALPHA": 21, "V-BETA": 7}))
        out = reader._post_process(df)
        assert set(out["lt_basis"]) == {"promised_dispatch_plus_transit"}


class TestTheSupplierTransitTable:
    """
    The purchase export carries no transport mode, so the mode and the transit days are
    both declared here. A CSV rather than JSON for one reason: it is filled in for every
    supplier bought from, and a spreadsheet is the quickest way to fill a table.
    """

    def test_the_wildcard_row_answers_for_suppliers_not_named(self, tmp_path):
        reader = POHistoryReader(
            _transit_config(tmp_path, {"V-BETA": 35}, default=5,
                            modes={"*": "AIR", "V-BETA": "SEA"}))
        out = reader._post_process(_dispatch_rows())
        rows = out.set_index("supplier")
        assert rows.loc["V-ALPHA", "transit_days"].iloc[0] == 5.0
        assert rows.loc["V-BETA", "transit_days"] == 35.0

    def test_the_mode_is_carried_because_the_export_has_none(self, tmp_path):
        """
        Nothing is inferred from it — it changes no arithmetic. It is carried because a
        reader confronted with a 35-day transit leg needs to know why, and because
        `diagnostics._chronic_air` already looks for the column.
        """
        reader = POHistoryReader(
            _transit_config(tmp_path, {"V-BETA": 35}, default=5,
                            modes={"*": "AIR", "V-BETA": "SEA"}))
        out = reader._post_process(_dispatch_rows())
        rows = out.set_index("supplier")
        assert rows.loc["V-BETA", "transport_mode"] == "SEA"
        assert rows.loc["V-ALPHA", "transport_mode"].iloc[0] == "AIR"

    def test_changing_a_mode_moves_the_lead_time_on_the_next_run(self, tmp_path):
        """
        The property that makes a mode change cheap: only the dispatch leg is measured,
        and that leg does not change with the mode. So moving a supplier from air to sea
        is one edit, and the lead time follows immediately rather than waiting for a
        measurement window to refill — which is what it would have to do if the transit
        time were buried inside a receipt-based measurement.
        """
        rows = _dispatch_rows()
        air = POHistoryReader(_transit_config(tmp_path / "a", {"V-ALPHA": 5}))
        sea = POHistoryReader(_transit_config(tmp_path / "b", {"V-ALPHA": 35}))
        before = air._post_process(rows.copy())
        after = sea._post_process(rows.copy())
        gap = (after.set_index("po_number").loc["4500001", "lead_time_days"]
               - before.set_index("po_number").loc["4500001", "lead_time_days"])
        assert gap == 30.0

    def test_a_starter_table_is_written_for_the_suppliers_not_named(self, tmp_path):
        """
        Compiling the supplier list is the slow half of filling this file in, and it is
        the half the run can do. Written even when a `*` row is covering for them: a
        blanket is a reason to finish the table, not a reason to stop.
        """
        out_dir = tmp_path / "out"
        reader = POHistoryReader(_transit_config(tmp_path, {"V-BETA": 35}, default=5))
        reader.output_dir = out_dir
        reader._post_process(_dispatch_rows())
        written = out_dir / "supplier_transit_template.csv"
        assert written.exists()
        table = pd.read_csv(written, comment="#")
        assert list(table["supplier"]) == ["V-ALPHA"]
        # Pre-filled from the blanket, and the header says that is what it is.
        assert "blanket" in written.read_text(encoding="utf-8-sig")

    def test_the_run_records_it_so_the_screen_can_offer_it(self, tmp_path):
        """
        `/runs/{id}/outputs` lists the run manifest and not the directory — so a
        folder holding a month of runs can still say which files are this one's. A
        starter table written to disk and not recorded is therefore invisible to the
        screen and cannot be downloaded, while the console line promises it exists.
        """
        import glob
        import json

        out = tmp_path / "run"
        store = tmp_path / "store"
        data = tmp_path / "data"
        data.mkdir()
        for src in SAMPLE_DIR.glob("*.csv"):
            (data / src.name).write_text(src.read_text(encoding="utf-8"),
                                         encoding="utf-8")
        # An extract with a confirmed dispatch date, no goods receipt, and suppliers
        # that the (empty) shipped transit table does not name.
        po = pd.read_csv(data / "po_history.csv")
        po.insert(0, "Purchasing Document", [f"45{i:05d}" for i in range(len(po))])
        po.insert(1, "Item", 10)
        po = po.rename(columns={"Goods Receipt Date": "Confirmed Delivery Date"})
        po["Delivered Qty"] = po["PO Qty"]
        po["Still to be delivered"] = 0
        po.to_csv(data / "po_history.csv", index=False)

        planner = InventoryPlanner(output_dir=out, store_root=store,
                                   interactive=False)
        loaded = planner.load_all(sorted(str(f) for f in data.glob("*.csv")))
        planner.run_planning(**loaded)

        written = out / "supplier_transit_template.csv"
        assert written.exists()
        newest = sorted(glob.glob(str(out / "runs" / "*.json")))[-1]
        manifest = json.loads(Path(newest).read_text(encoding="utf-8"))
        names = [o.get("name") for o in manifest.get("outputs") or []]
        assert "supplier_transit_template.csv" in names

    def test_nothing_is_written_when_every_supplier_is_named(self, tmp_path):
        out_dir = tmp_path / "out"
        reader = POHistoryReader(
            _transit_config(tmp_path, {"V-ALPHA": 5, "V-BETA": 35}))
        reader.output_dir = out_dir
        reader._post_process(_dispatch_rows())
        assert not (out_dir / "supplier_transit_template.csv").exists()

    def test_it_never_writes_into_the_config_directory(self, tmp_path):
        """
        That would either overwrite a table someone is maintaining or look like it had
        been accepted. A reader with nowhere to write simply does not write.
        """
        config = _transit_config(tmp_path, {"V-BETA": 35}, default=5)
        reader = POHistoryReader(config)
        assert reader.output_dir is None
        reader._post_process(_dispatch_rows())
        assert not (config / "supplier_transit_template.csv").exists()

    def test_the_longest_prefix_still_wins_with_modes(self):
        table = {"SIM": ("AIR", 5.0), "SIM001": ("SEA", 35.0)}
        assert POHistoryReader._transit_for("SIM001 TYCO", table, (None, None)) == ("SEA", 35.0)
        assert POHistoryReader._transit_for("SIM002 OTHER", table, (None, None)) == ("AIR", 5.0)
