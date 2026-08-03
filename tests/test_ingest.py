"""
Tests for the contract-driven ingest layer.

The cases here are drawn from real ERP failure modes rather than invented edge cases.
Each of the "silent corruption" tests represents a bug that produced plausible-looking
numbers with no error — the class of failure that contract tests exist to catch.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.capabilities import CapabilityResolver
from inventory_planning.ingest.contract import default_registry
from inventory_planning.ingest.contract_tests import ContractTester
from inventory_planning.ingest.expressions import Expression, ExpressionError
from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.profiler import (
    detect_date_order,
    detect_decimal_style,
    mask_value,
    parse_period_header,
    profile_frame,
)
from inventory_planning.ingest.registry import AdapterRegistry


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sap_open_po():
    """SAP flavour: no open qty, schedule-line grain, DMY dates, EU decimals."""
    return pd.DataFrame({
        "Purch.Doc.":      ["4500001", "4500001", "4500002", "4500003"],
        "Item":            ["10", "10", "10", "20"],
        "Sched. Line":     ["1", "2", "1", "1"],
        "Material":        ["ABC-1", "ABC-1", "ABC-2", "ABC-3"],
        "Vendor":          ["V100", "V100", "V200", "V100"],
        "Scheduled Qty":   ["600", "600", "300", "450"],
        "Qty Delivered":   ["0", "0", "0", "450"],
        "Del. Ind.":       ["", "", "", "CLSD"],
        "Doc Date":        ["05.03.2026", "05.03.2026", "17.03.2026", "22.03.2026"],
        "Stat. Del. Date": ["20.09.2026", "25.09.2026", "28.09.2026", "30.09.2026"],
        "Net Value":       ["1.200,50", "1.200,50", "900,00", "1.350,00"],
        "Incoterms":       ["EXW", "EXW", "FOB", "CIF"],
    })


@pytest.fixture
def planner_timeseries():
    """What a planner hands over once they have already bucketed demand themselves."""
    months = pd.period_range("2024-01", "2026-06", freq="M")
    data = {
        "Item Code":   ["ABC-1", "ABC-2", "ABC-3"],
        "Description": ["Widget A", "Widget B", "Widget C"],
        "SOPC Class":  ["A", "B", "C"],
    }
    rng = np.random.default_rng(42)
    for m in months:
        data[m.strftime("%b-%y")] = rng.integers(0, 200, 3)
    return pd.DataFrame(data)


# ── Expressions ──────────────────────────────────────────────────────────────

class TestExpressions:
    def test_derivation(self):
        df = pd.DataFrame({"order_qty": [100.0, 50.0], "delivered_qty": [40.0, np.nan]})
        result = Expression("order_qty - coalesce(delivered_qty, 0)").evaluate(df)
        assert list(result) == [60.0, 50.0]

    def test_division_by_zero_is_unknown_not_infinite(self):
        """An undefined unit rate must be null, or it poisons every downstream mean."""
        df = pd.DataFrame({"amount": [100.0, 50.0], "qty": [10.0, 0.0]})
        result = Expression("amount / qty").evaluate(df)
        assert result.iloc[0] == 10.0
        assert pd.isna(result.iloc[1])
        assert not np.isinf(result).any()

    def test_chained_comparison(self):
        df = pd.DataFrame({"lt": [10, 500, 45]})
        assert list(Expression("0 <= lt <= 400").evaluate(df)) == [True, False, True]

    def test_can_evaluate_reports_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        assert Expression("a + 1").can_evaluate(df)
        assert not Expression("a + b").can_evaluate(df)

    @pytest.mark.parametrize("hostile", [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "order_qty.apply(print)",
        "[x for x in range(10)]",
        "(lambda: 1)()",
    ])
    def test_rejects_code_execution(self, hostile):
        """Adapter YAML may be model-generated; it must not reach the interpreter."""
        with pytest.raises(ExpressionError):
            Expression(hostile)


# ── Profiler ─────────────────────────────────────────────────────────────────

class TestProfiler:
    @pytest.mark.parametrize("header,expected", [
        ("2026-01", "2026-01"), ("Jan-26", "2026-01"), ("202601", "2026-01"),
        ("2026/1", "2026-01"), ("Jan 2026", "2026-01"), ("01-2026", "2026-01"),
        ("2026-Q1", "2026-01"),
    ])
    def test_recognises_period_headers(self, header, expected):
        assert str(parse_period_header(header)) == expected

    @pytest.mark.parametrize("header", ["SKU", "Item Code", "2026", "M1", "Qty", "Total"])
    def test_rejects_non_period_headers(self, header):
        """Conservative on purpose — a bare year or code must not trigger a reshape."""
        assert parse_period_header(header) is None

    def test_detects_wide_timeseries(self, planner_timeseries):
        profile = profile_frame(planner_timeseries, "demand.xlsx")
        assert profile.shape == "wide_periods"
        assert len(profile.period_columns) == 30
        assert set(profile.id_columns) == {"Item Code", "Description", "SOPC Class"}

    def test_detects_long_transactional(self, sap_open_po):
        assert profile_frame(sap_open_po, "po.xlsx").shape == "long"

    def test_detects_day_first_dates(self):
        """Evidence must come from the text; DMY values fail MDY parsing outright."""
        assert detect_date_order(pd.Series(["13.09.2026", "25.09.2026"])) is True
        assert detect_date_order(pd.Series(["09/13/2026", "09/25/2026"])) is False
        assert detect_date_order(pd.Series(["05/06/2026"])) is None

    def test_detects_european_decimals(self):
        assert detect_decimal_style(pd.Series(["1.200,50", "900,00"])) == "eu"
        assert detect_decimal_style(pd.Series(["1,200.50", "900.00"])) == "us"

    def test_flags_total_rows(self):
        df = pd.DataFrame({"Material": ["A-1", "A-2", "TOTAL"], "Qty": ["10", "20", "30"]})
        assert profile_frame(df, "x.csv").suspected_total_rows == 1

    def test_discovers_status_vocabulary(self, sap_open_po):
        """Low-cardinality columns are surfaced so a value domain can be written."""
        col = profile_frame(sap_open_po, "po.xlsx").column("Del. Ind.")
        assert "CLSD" in col.distinct_values

    def test_masks_values(self):
        assert mask_value("ACME-10023") == "AAAA-99999"

    def test_portrait_excludes_raw_values_by_default(self, sap_open_po):
        """Compliance floor: nothing identifying may leave without an explicit opt-in."""
        portrait = profile_frame(sap_open_po, "po.xlsx").to_dict()
        assert "V100" not in str(portrait)
        assert all("samples" not in c for c in portrait["columns"])

    def test_header_hash_survives_column_reordering(self, sap_open_po):
        reordered = sap_open_po[list(reversed(sap_open_po.columns))]
        assert (profile_frame(sap_open_po, "a").header_hash
                == profile_frame(reordered, "b").header_hash)


# ── Routing and adapters ─────────────────────────────────────────────────────

class TestRouting:
    def test_routes_wide_file_to_timeseries(self, planner_timeseries):
        """The case from the brief: a planner-supplied series needs no declaration."""
        result = AdapterRegistry().route(planner_timeseries, "demand.xlsx")
        assert result.doc_type == "demand_timeseries"

    def test_routes_open_po_by_content(self, sap_open_po):
        assert AdapterRegistry().route(sap_open_po, "ZMM.xlsx").doc_type == "open_po"

    def test_sku_beats_line_number_on_profile_evidence(self, sap_open_po):
        """
        `Item` and `Material` both match a `sku` alias. Name alone picks the wrong
        one; cardinality picks the right one.
        """
        adapter = AdapterRegistry().route(sap_open_po, "ZMM.xlsx").adapter
        assert adapter.column_map["sku"] == "Material"
        assert adapter.column_map["po_line_number"] == "Item"

    def test_matches_reordered_alias_tokens(self):
        """`QUANTITY_ORDERED` must reach `ordered quantity` without a new alias."""
        df = pd.DataFrame({
            "ITEM_NUMBER": ["A-1"], "VENDOR_NAME": ["V1"], "PO_NUM": ["P1"],
            "CREATION_DATE": ["2026-01-01"], "LEAD TIME DAYS": ["30"],
            "QUANTITY_ORDERED": ["100"], "PO_LINE_AMOUNT": ["500"],
        })
        adapter = AdapterRegistry().route(df, "oracle.csv", doc_type_hint="po_history").adapter
        assert adapter.column_map["po_qty"] == "QUANTITY_ORDERED"


class TestAdapterApply:
    def test_derives_open_qty_when_absent(self, sap_open_po):
        """Failure mode 2: the field must be synthesised, not renamed."""
        route = AdapterRegistry().route(sap_open_po, "ZMM.xlsx")
        assert "open_qty" in route.adapter.derivations
        frame, _ = route.adapter.apply(sap_open_po, route.contract, route.profile)
        assert (frame["open_qty"] >= 0).all()

    def test_translates_status_vocabulary(self, sap_open_po):
        """Failure mode 3: `CLSD` is closed, and must not be counted as open supply."""
        route = AdapterRegistry().route(sap_open_po, "ZMM.xlsx")
        frame, _ = route.adapter.apply(sap_open_po, route.contract, route.profile)
        assert "ABC-3" not in set(frame["sku"])       # the CLSD line is filtered out
        assert set(frame["po_status"]) == {"open"}

    def test_parses_day_first_dates(self, sap_open_po):
        route = AdapterRegistry().route(sap_open_po, "ZMM.xlsx")
        frame, _ = route.adapter.apply(sap_open_po, route.contract, route.profile)
        assert frame["order_date"].min() == pd.Timestamp("2026-03-05")

    def test_parses_european_decimals(self, sap_open_po):
        """`900,00` must read as 900.00, not as NaN or as 90000."""
        route = AdapterRegistry().route(sap_open_po, "ZMM.xlsx")
        frame, _ = route.adapter.apply(sap_open_po, route.contract, route.profile)
        # ABC-2 has a single schedule line, so its value is unaffected by the rollup
        assert frame.loc[frame["sku"] == "ABC-2", "open_amount"].iloc[0] == pytest.approx(900.00)

    def test_rolls_schedule_lines_up_to_po_lines(self, sap_open_po):
        """Without the rollup every quantity is inflated and nothing downstream notices."""
        route = AdapterRegistry().route(sap_open_po, "ZMM.xlsx")
        assert route.adapter.rollup_to == "po_line"
        frame, _ = route.adapter.apply(sap_open_po, route.contract, route.profile)
        abc1 = frame[frame["sku"] == "ABC-1"]
        assert len(abc1) == 1
        assert abc1["order_qty"].iloc[0] == 1200      # 600 + 600 summed once

    def test_transform_log_traces_every_column(self, sap_open_po):
        route = AdapterRegistry().route(sap_open_po, "ZMM.xlsx")
        _, log = route.adapter.apply(sap_open_po, route.contract, route.profile)
        assert any(s.field == "open_qty" and s.action == "derived" for s in log)
        assert any(s.action == "value_mapped" for s in log)


# ── Contract tests ───────────────────────────────────────────────────────────

class TestContractTests:
    def test_detects_grain_mismatch(self):
        contract = default_registry().get("open_po")
        df = pd.DataFrame({
            "po_number": ["P1", "P1"], "sku": ["A", "A"],
            "order_qty": [10.0, 10.0], "open_qty": [10.0, 10.0],
        })
        report = ContractTester().run(df, contract)
        grain = next(r for r in report.results if r.name.startswith("grain:"))
        assert not grain.passed

    def test_detects_negative_open_qty(self):
        contract = default_registry().get("open_po")
        df = pd.DataFrame({
            "po_number": ["P1", "P2"], "sku": ["A", "B"],
            "order_qty": [10.0, 10.0], "open_qty": [10.0, -5.0],
        })
        report = ContractTester().run(df, contract)
        assert not report.passed

    def test_null_operands_are_not_violations(self):
        """
        `NaN >= 0` is False in pandas. Counting that as a breach would report an
        error for every row where the value is simply unknown.
        """
        contract = default_registry().get("open_po")
        df = pd.DataFrame({
            "po_number": ["P1", "P2"], "sku": ["A", "B"],
            "order_qty": [10.0, 10.0], "open_qty": [10.0, 5.0],
            "unit_cost": [3.0, np.nan],
        })
        report = ContractTester().run(df, contract)
        cost = [r for r in report.results if "unit_cost" in r.name]
        assert all(r.passed for r in cost)

    def test_missing_required_field_fails(self):
        contract = default_registry().get("inventory")
        report = ContractTester().run(pd.DataFrame({"sku": ["A"]}), contract)
        assert not report.passed

    def test_detects_distribution_drift(self):
        """A column's meaning can change while its name does not."""
        contract = default_registry().get("inventory")
        df = pd.DataFrame({"sku": ["A", "B"], "qty_on_hand": [1000.0, 1200.0]})
        baseline = {"row_count": 2, "column_means": {"qty_on_hand": 10.0}}
        report = ContractTester().run(df, contract, baseline=baseline)
        drift = next(r for r in report.results if r.name.startswith("distribution_drift"))
        assert not drift.passed


# ── Capabilities ─────────────────────────────────────────────────────────────

class TestCapabilities:
    def test_timeseries_alone_satisfies_demand(self):
        """The brief's case: a pre-bucketed series makes sales history unnecessary."""
        plan = CapabilityResolver().resolve({"demand_timeseries": "d.xlsx",
                                             "inventory": "i.csv"})
        assert plan.has("demand_signal")
        assert plan.source_of("demand_signal") == "demand_timeseries"
        assert plan.can_run

    def test_sales_history_also_satisfies_demand(self):
        plan = CapabilityResolver().resolve({"sales_history": "s.csv", "inventory": "i.csv"})
        assert plan.source_of("demand_signal") == "sales_history"

    def test_missing_demand_blocks_the_run(self):
        plan = CapabilityResolver().resolve({"inventory": "i.csv"})
        assert not plan.can_run
        assert "demand_signal" in [c.name for c in plan.missing_required]

    def test_names_specific_consequences_of_gaps(self):
        """Degradation must be stated, not silently absorbed into a thinner number."""
        plan = CapabilityResolver().resolve({"demand_timeseries": "d.xlsx",
                                             "inventory": "i.csv"})
        assert plan.can_run
        assert any("lead-time variability" in d for d in plan.degradations)
        assert any("backlog" in d for d in plan.degradations)


# ── End to end ───────────────────────────────────────────────────────────────

class TestIntake:
    def test_identifies_mixed_formats_without_declaration(self, tmp_path,
                                                          sap_open_po, planner_timeseries):
        sap_open_po.to_excel(tmp_path / "ZMM_export.xlsx", index=False)
        planner_timeseries.to_excel(tmp_path / "demand_36m.xlsx", index=False)
        pd.DataFrame({
            "Stock Code": ["ABC-1", "ABC-2"], "Warehouse": ["SG01", "SG01"],
            "On Hand Qty": ["100", "250"], "Std Cost": ["12.50", "8.00"],
        }).to_csv(tmp_path / "wms.csv", index=False)

        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))

        assert set(result.documents) == {"open_po", "demand_timeseries", "inventory"}
        assert result.can_run

    def test_melts_wide_series_to_long_grain(self, tmp_path, planner_timeseries):
        planner_timeseries.to_excel(tmp_path / "demand.xlsx", index=False)
        result = Intake(verbose=False).load_files([tmp_path / "demand.xlsx"])
        frame = result.frame("demand_timeseries")

        assert set(frame.columns) >= {"sku", "period", "qty"}
        assert len(frame) == 3 * 30
        assert frame["qty"].notna().all()

    def test_pivots_back_to_forecastable_matrix(self, tmp_path, planner_timeseries):
        planner_timeseries.to_excel(tmp_path / "demand.xlsx", index=False)
        intake = Intake(verbose=False)
        result = intake.load_files([tmp_path / "demand.xlsx"])
        pivot = intake.to_pivot(result.frame("demand_timeseries"))

        assert pivot.shape == (30, 3)
        assert isinstance(pivot.index, pd.PeriodIndex)
        assert pivot.index.is_monotonic_increasing

    def test_blank_periods_read_as_zero_demand(self, tmp_path):
        """
        Zero and unknown are different: one makes a SKU intermittent, the other makes
        it short-history, and they route to different forecasting methods.
        """
        df = pd.DataFrame({
            "Item": ["A-1"],
            "Jan-26": [10], "Feb-26": [np.nan], "Mar-26": [30], "Apr-26": [40],
        })
        df.to_excel(tmp_path / "d.xlsx", index=False)
        frame = Intake(verbose=False).load_files([tmp_path / "d.xlsx"]).frame("demand_timeseries")
        assert frame["qty"].notna().all()
        assert frame.loc[frame["period"] == "2026-02", "qty"].iloc[0] == 0.0

    def test_keeps_the_fuller_of_two_duplicate_documents(self, tmp_path, sap_open_po):
        """A sample overlaps the full export's keys, so it is a copy, not a partition."""
        sap_open_po.head(2).to_excel(tmp_path / "a_sample.xlsx", index=False)
        sap_open_po.to_excel(tmp_path / "b_full.xlsx", index=False)
        result = Intake(verbose=False).load_files(
            [tmp_path / "a_sample.xlsx", tmp_path / "b_full.xlsx"]
        )
        full_alone = Intake(verbose=False).load_files([tmp_path / "b_full.xlsx"])
        assert result.get("open_po").source_name == "b_full.xlsx"
        # The sample's rows must not be added on top of the full export's.
        assert result.get("open_po").row_count == full_alone.get("open_po").row_count

    def test_concatenates_partitions_across_sheets(self, tmp_path, sap_open_po):
        """
        Two country tabs describe different orders and must be stacked. Discarding one
        as a "duplicate" would quietly drop a whole country from the plan.
        """
        au = sap_open_po.copy()
        nz = sap_open_po.copy()
        nz["Purch.Doc."] = ["4600001", "4600001", "4600002", "4600003"]
        nz["Material"] = ["NZ-1", "NZ-1", "NZ-2", "NZ-3"]

        path = tmp_path / "open_po_anz.xlsx"
        with pd.ExcelWriter(path) as writer:
            au.to_excel(writer, sheet_name="AUS", index=False)
            nz.to_excel(writer, sheet_name="NZ", index=False)

        doc = Intake(verbose=False).load_files([path]).get("open_po")
        assert set(doc.frame["source_sheet"]) == {"AUS", "NZ"}
        assert {"ABC-1", "NZ-1"} <= set(doc.frame["sku"])

    def test_skips_pivot_table_sheets(self, tmp_path, sap_open_po):
        """A pivot survives a naive read and would profile happily — reject it."""
        pivot = pd.DataFrame({
            "Sum of Qty": ["Column Labels", None, "Jul"],
            "Grand Total": [None, None, "120"],
        })
        path = tmp_path / "book.xlsx"
        with pd.ExcelWriter(path) as writer:
            pivot.to_excel(writer, sheet_name="Sheet1", index=False)
            sap_open_po.to_excel(writer, sheet_name="AUS", index=False)

        result = Intake(verbose=False).load_files([path])
        assert "open_po" in result.documents
        assert any("Sheet1" in name and "skipped" in reason
                   for name, reason in result.failures)

class TestAmbiguousRouting:
    """
    One ERP report exported twice under different filters produces two files with
    byte-identical headers. No header rule can separate them; only the rows can.
    """

    @staticmethod
    def _order_book(shipped: bool) -> pd.DataFrame:
        n = 6
        return pd.DataFrame({
            "Material #": [f"M-{i}" for i in range(n)],
            "Order #": [f"SO{100 + i}" for i in range(n)],
            "Customer": ["Acme"] * n,
            "Order Date": pd.to_datetime(["2026-05-01"] * n),
            "Requested Date": pd.to_datetime(["2026-06-01"] * n),
            "Order Qty": [10.0] * n,
            "Ship Qty": [10.0 if shipped else 0.0] * n,
            "Open Amt ($)": [0.0 if shipped else 500.0] * n,
            # Present in both exports; populated only once the line has shipped.
            "Ship Date": pd.to_datetime(["2026-05-28"] * n) if shipped else pd.NaT,
        })

    def test_content_separates_shipped_history_from_open_orders(self):
        registry = AdapterRegistry()
        assert registry.route(self._order_book(shipped=True), "a.xlsx").doc_type == "sales_history"
        assert registry.route(self._order_book(shipped=False), "b.xlsx").doc_type == "open_so"

    def test_the_decision_is_reported_as_content_based(self):
        result = AdapterRegistry().route(self._order_book(shipped=False), "b.xlsx")
        assert "content decided" in result.reason

    def test_an_empty_column_still_counts_as_evidence(self):
        """
        `Ship Date` present but wholly blank is the defining mark of an open order.
        The ordinary mapping skips empty columns, so without special handling the one
        signal that settles the question is invisible.
        """
        frame = self._order_book(shipped=False)
        assert frame["Ship Date"].isna().all()
        assert AdapterRegistry().route(frame, "b.xlsx").doc_type == "open_so"

    def test_a_missing_column_is_not_evidence(self):
        """
        A sales export has no receipt-date column because it is not a PO document.
        Reading that absence as "is_null(receive_date) holds" would let any unrelated
        file win an open-PO discriminator vacuously.
        """
        assert AdapterRegistry().route(
            self._order_book(shipped=True), "a.xlsx"
        ).doc_type != "open_po"


class TestIntakeMultiSheet:
    def test_reports_unreadable_files_without_aborting(self, tmp_path, planner_timeseries):
        planner_timeseries.to_excel(tmp_path / "demand.xlsx", index=False)
        (tmp_path / "notes.txt").write_text("not a data file")
        result = Intake(verbose=False).load_files(sorted(tmp_path.iterdir()))
        assert "demand_timeseries" in result.documents
        assert any("notes.txt" == name for name, _ in result.failures)
