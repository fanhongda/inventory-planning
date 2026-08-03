"""
Tests for service measurement, ordering diagnostics, forward risk and the KPI report.

The distinction these tests exist to protect: a line past its request date with stock
on the shelf is not a supply failure. Collapsing it into the OTD miss rate blames
planning for a collection problem and hides idle stock at the same time.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.policy.diagnostics import DiagnosticsAnalyzer
from inventory_planning.policy.service import (
    ON_TIME, OPEN_NOT_YET_DUE, OPEN_PAST_DUE_AVAILABLE, OPEN_PAST_DUE_SHORT,
    SHIPPED_LATE, ServiceAnalyzer,
)
from inventory_planning.reporting.kpi_report import KPIReport

AS_OF = date(2026, 8, 3)


@pytest.fixture
def shipped():
    """Two on time, one late."""
    return pd.DataFrame({
        "sku": ["A-1", "A-2", "A-3"],
        "customer": ["Acme", "Globex", "Acme"],
        "qty": [100.0, 200.0, 80.0],
        "amount": [4000.0, 1800.0, 12000.0],
        "order_date": pd.to_datetime(["2026-05-01", "2026-05-05", "2026-06-05"]),
        "customer_request_date": pd.to_datetime(["2026-06-01", "2026-06-10", "2026-07-05"]),
        "ship_date": pd.to_datetime(["2026-05-28", "2026-06-10", "2026-07-20"]),
    })


@pytest.fixture
def open_orders():
    """One past due with stock, one past due without, one not yet due."""
    return pd.DataFrame({
        "sku": ["STOCKED", "SHORT", "FUTURE"],
        "customer": ["Acme", "Globex", "Acme"],
        "open_qty": [50.0, 40.0, 30.0],
        "open_amount": [5000.0, 4000.0, 3000.0],
        "order_date": pd.to_datetime(["2026-06-01", "2026-06-05", "2026-07-01"]),
        "customer_request_date": pd.to_datetime(["2026-07-01", "2026-07-10", "2026-09-01"]),
    })


@pytest.fixture
def stock():
    return pd.DataFrame({
        "sku": ["STOCKED", "SHORT", "FUTURE"],
        "qty_on_hand": [500.0, 0.0, 100.0],
    })


# ── Service classification ───────────────────────────────────────────────────

class TestServiceClassification:
    def test_classifies_shipped_lines(self, shipped):
        result = ServiceAnalyzer().analyze(sales_history=shipped, as_of=AS_OF)
        states = dict(zip(result.lines["sku"], result.lines["service_state"]))
        assert states["A-1"] == ON_TIME
        assert states["A-2"] == ON_TIME       # shipped exactly on the request date
        assert states["A-3"] == SHIPPED_LATE

    def test_shipping_on_the_request_date_is_on_time(self, shipped):
        result = ServiceAnalyzer().analyze(sales_history=shipped, as_of=AS_OF)
        row = result.lines.set_index("sku").loc["A-2"]
        assert row["service_state"] == ON_TIME
        assert row["days_late"] == 0

    def test_separates_uncollected_stock_from_genuine_shortage(self, open_orders, stock):
        """The distinction this whole module exists for."""
        result = ServiceAnalyzer().analyze(open_so=open_orders, inventory=stock, as_of=AS_OF)
        states = dict(zip(result.lines["sku"], result.lines["service_state"]))
        assert states["STOCKED"] == OPEN_PAST_DUE_AVAILABLE
        assert states["SHORT"] == OPEN_PAST_DUE_SHORT
        assert states["FUTURE"] == OPEN_NOT_YET_DUE

    def test_uncollected_stock_is_not_an_otd_failure(self, open_orders, stock):
        result = ServiceAnalyzer().analyze(open_so=open_orders, inventory=stock, as_of=AS_OF)
        failures = result.failures_by_sku()
        assert "STOCKED" not in set(failures["sku"])
        assert "SHORT" in set(failures["sku"])

    def test_fair_reading_beats_harsh_reading(self, shipped, open_orders, stock):
        result = ServiceAnalyzer().analyze(shipped, open_orders, stock, as_of=AS_OF)
        assert result.otd_line_rate > result.otd_line_rate_harsh

    def test_on_hand_is_allocated_across_competing_lines(self):
        """
        Two lines for one SKU with only enough stock for one. Comparing each line
        against the full balance independently would report both as collectable and
        overstate the uncollected category.
        """
        open_so = pd.DataFrame({
            "sku": ["A-1", "A-1"],
            "open_qty": [100.0, 100.0],
            "order_date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "customer_request_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
        })
        inventory = pd.DataFrame({"sku": ["A-1"], "qty_on_hand": [100.0]})
        result = ServiceAnalyzer().analyze(open_so=open_so, inventory=inventory, as_of=AS_OF)
        states = list(result.lines["service_state"])
        assert states.count(OPEN_PAST_DUE_AVAILABLE) == 1
        assert states.count(OPEN_PAST_DUE_SHORT) == 1

    def test_no_stock_data_means_nothing_is_claimed_collectable(self, open_orders):
        result = ServiceAnalyzer().analyze(open_so=open_orders, as_of=AS_OF)
        assert OPEN_PAST_DUE_AVAILABLE not in set(result.lines["service_state"])

    def test_says_so_when_otd_cannot_be_measured(self):
        """
        Only uncollected and not-yet-due lines: nothing to judge supply performance
        against. A blank metric must say why, or it reads as "fine".
        """
        open_so = pd.DataFrame({
            "sku": ["STOCKED", "FUTURE"],
            "open_qty": [50.0, 30.0],
            "order_date": pd.to_datetime(["2026-06-01", "2026-07-01"]),
            "customer_request_date": pd.to_datetime(["2026-07-01", "2026-09-01"]),
        })
        inventory = pd.DataFrame({"sku": ["STOCKED", "FUTURE"], "qty_on_hand": [500.0, 100.0]})
        result = ServiceAnalyzer().analyze(open_so=open_so, inventory=inventory, as_of=AS_OF)

        assert np.isnan(result.otd_line_rate)
        assert "cannot be measured" in result.summary()

    def test_open_lines_alone_cannot_produce_an_otd_rate(self, open_orders, stock):
        """
        An open past-due line can never come out "on time" — had it been on time it
        would have shipped. Measuring OTD over open lines is therefore guaranteed to
        return 0%, which is a confident answer produced entirely by the method.
        Backlog is reported instead.
        """
        result = ServiceAnalyzer().analyze(open_so=open_orders, inventory=stock, as_of=AS_OF)
        assert np.isnan(result.otd_line_rate)
        assert result.past_due_backlog_lines == 1
        assert "backlog" in result.summary().lower()

    def test_otd_is_measured_over_completed_deliveries(self, shipped, open_orders, stock):
        """Two on time, one late, regardless of what the open book looks like."""
        result = ServiceAnalyzer().analyze(shipped, open_orders, stock, as_of=AS_OF)
        assert len(result.completed) == 3
        assert result.otd_line_rate == pytest.approx(2 / 3)


class TestRequestDateQuality:
    def test_flags_request_date_equal_to_order_date(self):
        sales = pd.DataFrame({
            "sku": ["A-1"], "qty": [10.0], "amount": [100.0],
            "order_date": pd.to_datetime(["2026-06-01"]),
            "customer_request_date": pd.to_datetime(["2026-06-01"]),
            "ship_date": pd.to_datetime(["2026-06-05"]),
        })
        quality = ServiceAnalyzer().analyze(sales_history=sales, as_of=AS_OF).quality
        assert quality.same_as_order_date == 1
        assert not quality.is_trustworthy

    def test_flags_orders_raised_already_past_due(self):
        sales = pd.DataFrame({
            "sku": ["A-1"], "qty": [10.0], "amount": [100.0],
            "order_date": pd.to_datetime(["2026-06-10"]),
            "customer_request_date": pd.to_datetime(["2026-06-01"]),
            "ship_date": pd.to_datetime(["2026-06-12"]),
        })
        quality = ServiceAnalyzer().analyze(sales_history=sales, as_of=AS_OF).quality
        assert quality.before_order_date == 1

    def test_clean_dates_are_trustworthy(self, shipped):
        quality = ServiceAnalyzer().analyze(sales_history=shipped, as_of=AS_OF).quality
        assert quality.clean_rate == 1.0
        assert quality.is_trustworthy

    def test_clean_only_otd_excludes_unwinnable_lines(self):
        sales = pd.DataFrame({
            "sku": ["GOOD", "ASAP"], "qty": [10.0, 10.0], "amount": [100.0, 100.0],
            "order_date": pd.to_datetime(["2026-05-01", "2026-06-01"]),
            "customer_request_date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "ship_date": pd.to_datetime(["2026-05-20", "2026-06-08"]),
        })
        result = ServiceAnalyzer().analyze(sales_history=sales, as_of=AS_OF)
        assert result.otd_line_rate == 0.5      # the ASAP line counts as late
        assert result.otd_line_rate_clean == 1.0  # excluded, it was unwinnable


# ── Ordering diagnostics ─────────────────────────────────────────────────────

class TestOrderingDiagnostics:
    @pytest.fixture
    def attributes(self):
        return pd.DataFrame({
            "sku": ["HOARD", "STEADY", "AIRY"],
            "demand_mean": [10.0, 100.0, 50.0],
            "unit_cost": [100.0, 20.0, 60.0],
        })

    @pytest.fixture
    def po_history(self):
        dates = pd.to_datetime(["2026-01-10", "2026-03-10", "2026-05-10", "2026-07-10"])
        return pd.DataFrame({
            "sku": ["HOARD"] * 4 + ["STEADY"] * 4 + ["AIRY"] * 4,
            "po_date": list(dates) * 3,
            "po_qty": [2000, 2000, 2000, 2000] + [300, 310, 295, 305] + [150, 900, 120, 1400],
            "transport_mode": ["SEA"] * 4 + ["SEA"] * 4 + ["AIR", "AIR", "AIR", "SEA"],
            "freight_cost": [500] * 4 + [200] * 4 + [1500, 1600, 1450, 300],
        })

    def test_detects_over_ordering(self, po_history, attributes):
        result = DiagnosticsAnalyzer().ordering(po_history, attributes, as_of=AS_OF)
        assert "HOARD" in set(result.over_ordered["sku"])
        assert "STEADY" not in set(result.over_ordered["sku"])

    def test_over_ordering_is_valued(self, po_history, attributes):
        result = DiagnosticsAnalyzer().ordering(po_history, attributes, as_of=AS_OF)
        assert result.total_over_order_value > 0

    def test_detects_erratic_lot_sizing(self, po_history, attributes):
        result = DiagnosticsAnalyzer().ordering(po_history, attributes, as_of=AS_OF)
        assert "AIRY" in set(result.erratic["sku"])
        assert "STEADY" not in set(result.erratic["sku"])

    def test_detects_chronic_air_freight(self, po_history, attributes):
        result = DiagnosticsAnalyzer().ordering(po_history, attributes, as_of=AS_OF)
        air = result.chronic_air.set_index("sku")
        assert "AIRY" in air.index
        assert air.loc["AIRY", "air_share"] == pytest.approx(0.75)
        assert "STEADY" not in air.index

    def test_no_transport_mode_means_no_air_finding(self, po_history, attributes):
        result = DiagnosticsAnalyzer().ordering(
            po_history.drop(columns=["transport_mode"]), attributes, as_of=AS_OF
        )
        assert not len(result.chronic_air)

    def test_handles_missing_po_history(self, attributes):
        result = DiagnosticsAnalyzer().ordering(None, attributes, as_of=AS_OF)
        assert not len(result.over_ordered)
        assert "No ordering anomalies" in result.summary()


# ── Forward risk ─────────────────────────────────────────────────────────────

class TestForwardRisk:
    @pytest.fixture
    def attributes(self):
        return pd.DataFrame({
            "sku": ["URGENT", "FINE", "SLOW", "DEAD"],
            "demand_mean": [300.0, 100.0, 1.0, 0.0],
            "unit_cost": [50.0, 20.0, 200.0, 150.0],
        })

    @pytest.fixture
    def stock(self):
        return pd.DataFrame({
            "sku": ["URGENT", "FINE", "SLOW", "DEAD"],
            "qty_on_hand": [100.0, 5000.0, 900.0, 400.0],
        })

    def test_flags_imminent_stockout(self, attributes, stock):
        risk = DiagnosticsAnalyzer().forward(attributes, stock, as_of=AS_OF)
        assert "URGENT" in set(risk.stockout["sku"])
        assert "FINE" not in set(risk.stockout["sku"])

    def test_backlog_consumes_stock_before_future_demand(self, attributes, stock):
        backlog = pd.DataFrame({"sku": ["FINE"], "open_qty": [4990.0]})
        risk = DiagnosticsAnalyzer().forward(attributes, stock, open_so=backlog, as_of=AS_OF)
        assert "FINE" in set(risk.stockout["sku"])

    def test_value_at_risk_is_unserved_demand_not_stock_value(self, attributes, stock):
        risk = DiagnosticsAnalyzer(horizon_days=180).forward(attributes, stock, as_of=AS_OF)
        row = risk.stockout.set_index("sku").loc["URGENT"]
        # ~180 days of demand at 10/day x $50, less the days it can still serve
        assert row["value_at_risk"] > float(stock.set_index("sku").loc["URGENT", "qty_on_hand"]) * 50

    def test_late_po_does_not_prevent_the_stockout(self, attributes, stock):
        late = pd.DataFrame({
            "sku": ["URGENT"], "open_qty": [5000.0],
            "committed_delivery": pd.to_datetime(["2026-11-01"]),
        })
        risk = DiagnosticsAnalyzer().forward(attributes, stock, open_po=late, as_of=AS_OF)
        row = risk.stockout.set_index("sku").loc["URGENT"]
        assert row["days_to_stockout"] <= 15
        assert row["inbound_day"] > row["days_to_stockout"]

    def test_timely_po_pushes_the_stockout_out(self, attributes, stock):
        soon = pd.DataFrame({
            "sku": ["URGENT"], "open_qty": [5000.0],
            "committed_delivery": pd.to_datetime(["2026-08-05"]),
        })
        base = DiagnosticsAnalyzer().forward(attributes, stock, as_of=AS_OF)
        covered = DiagnosticsAnalyzer().forward(attributes, stock, open_po=soon, as_of=AS_OF)
        base_days = base.stockout.set_index("sku").loc["URGENT", "days_to_stockout"]
        if "URGENT" in set(covered.stockout["sku"]):
            assert covered.stockout.set_index("sku").loc["URGENT", "days_to_stockout"] > base_days

    def test_identifies_slow_burn(self, attributes, stock):
        risk = DiagnosticsAnalyzer().forward(attributes, stock, as_of=AS_OF)
        slow = set(risk.slow_burn["sku"])
        assert {"SLOW", "DEAD"} <= slow
        assert "URGENT" not in slow

    def test_dead_stock_has_infinite_cover(self, attributes, stock):
        risk = DiagnosticsAnalyzer().forward(attributes, stock, as_of=AS_OF)
        row = risk.slow_burn.set_index("sku").loc["DEAD"]
        assert not np.isfinite(row["days_of_cover"])
        assert "never clear" in risk.summary()

    def test_slow_burn_with_inbound_po_is_called_out(self, attributes, stock):
        inbound = pd.DataFrame({"sku": ["SLOW"], "open_qty": [500.0]})
        risk = DiagnosticsAnalyzer().forward(attributes, stock, open_po=inbound, as_of=AS_OF)
        assert "push out or cancel" in risk.summary()


# ── Report ───────────────────────────────────────────────────────────────────

class TestKPIReport:
    def test_renders_with_no_inputs(self, tmp_path):
        html = KPIReport().render(output_path=tmp_path / "r.html")
        assert "Chapter 1" in html and "Chapter 2" in html
        assert (tmp_path / "r.html").exists()

    def test_renders_service_sections(self, tmp_path, shipped, open_orders, stock):
        service = ServiceAnalyzer().analyze(shipped, open_orders, stock, as_of=AS_OF)
        html = KPIReport().render(service=service, as_of=AS_OF,
                                  output_path=tmp_path / "r.html")
        assert "customer has not collected" in html
        assert "not a supply failure" in html.lower()

    def test_is_self_contained(self, tmp_path, shipped):
        """A strict-CSP page must not reach for a CDN, font, or remote image."""
        service = ServiceAnalyzer().analyze(sales_history=shipped, as_of=AS_OF)
        html = KPIReport().render(service=service, output_path=tmp_path / "r.html")
        for token in ("http://", "https://", "<script", "@import"):
            assert token not in html

    def test_authors_both_themes(self, tmp_path):
        html = KPIReport().render(output_path=tmp_path / "r.html")
        assert "prefers-color-scheme: dark" in html
        assert '[data-theme="dark"]' in html

    def test_escapes_sku_names(self, tmp_path, stock):
        hostile = pd.DataFrame({
            "sku": ["<img src=x onerror=alert(1)>"],
            "open_qty": [10.0],
            "order_date": pd.to_datetime(["2026-06-01"]),
            "customer_request_date": pd.to_datetime(["2026-07-01"]),
        })
        service = ServiceAnalyzer().analyze(open_so=hostile, as_of=AS_OF)
        html = KPIReport().render(service=service, output_path=tmp_path / "r.html")
        assert "<img src=x" not in html
        assert "&lt;img" in html
