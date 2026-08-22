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


# ── Charts and the sections built on them ────────────────────────────────────

class TestSupplyGapIsTheFirstThingInTheActions:
    """
    An expedite moves no money, and the work list ranks by the money an action moves —
    so all 53 of them sorted under everything else and fell off the end of a table
    capped at twelve. The most urgent thing in the run was the one thing invisible in
    the report.
    """

    @staticmethod
    def _recs(rows):
        return pd.DataFrame(rows)

    def test_the_gaps_get_their_own_block(self, tmp_path):
        recs = self._recs([{
            "sku": "A", "recommended_action": "EXPEDITE-INBOUND", "supply_gap": True,
            "supply_gap_days": 80.0, "on_hand_cover_days": 4.0,
            "days_to_next_arrival": 84.0, "inbound_past_due_qty": 600.0,
            "period_demand": 140.0, "suggested_po_qty": 0.0, "pushout_open_po_qty": 0.0,
        }])
        html = KPIReport().render(recommendations=recs, forward=None,
                                  output_path=tmp_path / "r.html")
        assert "Supply gap" in html
        assert "run dry before the next delivery" in html

    def test_it_survives_a_work_list_it_would_lose(self):
        """
        A gap worth nothing, against a push-out worth a fortune. Ranking by value puts
        the gap last; it still has to appear.
        """
        recs = self._recs([
            {"sku": "GAP", "recommended_action": "EXPEDITE-INBOUND", "supply_gap": True,
             "supply_gap_days": 40.0, "on_hand_cover_days": 0.0,
             "days_to_next_arrival": 40.0, "inbound_past_due_qty": 0.0,
             "period_demand": 10.0, "suggested_po_qty": 0.0, "pushout_open_po_qty": 0.0},
            *[{"sku": f"BIG{i}", "recommended_action": "PUSH-OUT-OPEN-PO",
               "supply_gap": False, "supply_gap_days": 0.0, "on_hand_cover_days": 400.0,
               "days_to_next_arrival": 5.0, "inbound_past_due_qty": 0.0,
               "period_demand": 1.0, "suggested_po_qty": 0.0,
               "pushout_open_po_qty": 99_999.0} for i in range(20)],
        ])
        html = KPIReport().render(recommendations=recs, forward=None)
        assert "GAP" in html

    def test_it_ranks_by_days_bare_not_by_money(self):
        """An empty shelf is a service failure whatever the part costs."""
        recs = self._recs([
            {"sku": "CHEAP-LONG", "recommended_action": "EXPEDITE-INBOUND",
             "supply_gap": True, "supply_gap_days": 100.0, "on_hand_cover_days": 1.0,
             "days_to_next_arrival": 101.0, "inbound_past_due_qty": 0.0,
             "period_demand": 1.0, "suggested_po_qty": 0.0, "pushout_open_po_qty": 0.0},
            {"sku": "DEAR-SHORT", "recommended_action": "EXPEDITE-INBOUND",
             "supply_gap": True, "supply_gap_days": 2.0, "on_hand_cover_days": 30.0,
             "days_to_next_arrival": 32.0, "inbound_past_due_qty": 0.0,
             "period_demand": 9999.0, "suggested_po_qty": 0.0, "pushout_open_po_qty": 0.0},
        ])
        html = KPIReport().render(recommendations=recs, forward=None)
        assert html.index("CHEAP-LONG") < html.index("DEAR-SHORT")

    def test_a_run_with_no_gaps_grows_no_heading(self):
        recs = self._recs([{
            "sku": "A", "recommended_action": "HOLD-OK", "supply_gap": False,
            "supply_gap_days": 0.0, "on_hand_cover_days": 90.0,
            "days_to_next_arrival": 5.0, "inbound_past_due_qty": 0.0,
            "period_demand": 10.0, "suggested_po_qty": 0.0, "pushout_open_po_qty": 0.0,
        }])
        assert "Supply gap" not in KPIReport().render(recommendations=recs, forward=None)


class TestStockingPolicyToReview:
    """
    The items whose ERP policy and whose demand point different ways, in the report a
    planner actually opens.

    Both directions, and the make-to-order half first: those have a customer waiting a
    full lead time for something the demand says should have been on the shelf. The
    other half is money sitting. Neither is applied — a policy is a decision someone
    made for reasons no extract records.
    """

    @staticmethod
    def _attributes(rows):
        return pd.DataFrame(rows)

    def test_both_directions_are_shown(self, tmp_path):
        attrs = self._attributes([
            {"sku": "A", "stocking_policy": "MTO", "suggested_stocking_policy": "MTS",
             "policy_basis": "demand in 11/12 months", "annual_value": 500_000.0},
            {"sku": "B", "stocking_policy": "MTS", "suggested_stocking_policy": "MTO",
             "policy_basis": "demand in 2/12 months", "annual_value": 9_000.0},
        ])
        html = KPIReport().render(attributes=attrs, output_path=tmp_path / "r.html")
        assert "Stocking policy to review" in html
        assert "Bought to order, but the demand recurs" in html
        assert "Held as stock, but the demand is sporadic" in html

    def test_it_ranks_by_what_is_at_stake(self, tmp_path):
        """A change has to be worth the argument, so the big number goes first."""
        attrs = self._attributes([
            {"sku": "SMALL", "stocking_policy": "MTO", "suggested_stocking_policy": "MTS",
             "policy_basis": "x", "annual_value": 1_000.0},
            {"sku": "BIG", "stocking_policy": "MTO", "suggested_stocking_policy": "MTS",
             "policy_basis": "x", "annual_value": 900_000.0},
        ])
        html = KPIReport().render(attributes=attrs, output_path=tmp_path / "r.html")
        assert html.index("BIG") < html.index("SMALL")

    def test_the_value_column_carries_the_value(self):
        """
        The first version recomputed `annual_value` from a demand column that is not on
        the attributes frame, overwriting the real one with NaN. Every row rendered as
        a dash and the ranking did nothing.
        """
        attrs = self._attributes([
            {"sku": "A", "stocking_policy": "MTO", "suggested_stocking_policy": "MTS",
             "policy_basis": "x", "annual_value": 1_351_069.0},
        ])
        html = KPIReport().render(attributes=attrs)
        assert "1,351,069" in html

    def test_agreement_is_said_rather_than_left_blank(self):
        attrs = self._attributes([
            {"sku": "A", "stocking_policy": "MTS", "suggested_stocking_policy": "MTS",
             "policy_basis": "x", "annual_value": 10.0},
        ])
        html = KPIReport().render(attributes=attrs)
        assert "agrees with what its demand did" in html

    def test_no_suggestion_column_means_no_section(self):
        """An extract with no planner worksheet must not grow an empty heading."""
        attrs = self._attributes([{"sku": "A", "annual_value": 10.0}])
        html = KPIReport().render(attributes=attrs)
        assert "Stocking policy to review" not in html


class TestMonthlyOTD:
    """
    A single OTD figure hides when it changed, which is the question that decides
    whether anything needs fixing. 82% flat and 82% after a collapse are the same
    number describing different situations.
    """

    @staticmethod
    def _year_of_shipments(on_time_by_month):
        rows = []
        for k, (n_on_time, n_late) in enumerate(on_time_by_month):
            month = pd.Timestamp("2025-01-01") + pd.DateOffset(months=k)
            for i in range(n_on_time + n_late):
                late = i >= n_on_time
                rows.append({
                    "sku": f"A-{i % 3}", "customer": "Acme", "qty": 10.0, "amount": 100.0,
                    "order_date": month - pd.Timedelta(days=30),
                    "customer_request_date": month + pd.Timedelta(days=10),
                    "ship_date": month + pd.Timedelta(days=20 if late else 5),
                })
        return pd.DataFrame(rows)

    def test_the_rate_is_computed_per_month(self):
        sales = self._year_of_shipments([(8, 2), (5, 5), (10, 0)])
        monthly = ServiceAnalyzer().analyze(sales_history=sales, as_of=date(2025, 5, 1)).monthly_otd()
        assert list(monthly["lines"]) == [10, 10, 10]
        assert list(monthly["otd_line_rate"].round(2)) == [0.8, 0.5, 1.0]

    def test_it_decomposes_the_headline(self):
        """The chart and the tile must not be two different definitions of OTD."""
        sales = self._year_of_shipments([(8, 2), (5, 5), (10, 0)])
        result = ServiceAnalyzer().analyze(sales_history=sales, as_of=date(2025, 5, 1))
        monthly = result.monthly_otd()
        assert monthly["on_time_lines"].sum() / monthly["lines"].sum() == pytest.approx(
            result.otd_line_rate
        )

    def test_a_silent_month_is_a_gap_not_a_zero(self):
        sales = pd.concat([
            self._year_of_shipments([(5, 0)]),
            self._year_of_shipments([(0, 0)] * 2 + [(5, 0)]),
        ], ignore_index=True)
        monthly = ServiceAnalyzer().analyze(sales_history=sales, as_of=date(2025, 6, 1)).monthly_otd()
        quiet = monthly[monthly["lines"] == 0]
        assert len(quiet), "the empty month must still appear"
        assert quiet["otd_line_rate"].isna().all(), "no deliveries is not a rate of zero"

    def test_a_thin_month_is_marked_rather_than_dropped(self):
        """A rate over three deliveries is not a measurement, but hiding it leaves a
        gap the reader misreads as zero."""
        sales = self._year_of_shipments([(20, 0), (2, 0)])
        monthly = ServiceAnalyzer().analyze(sales_history=sales, as_of=date(2025, 4, 1)).monthly_otd()
        assert list(monthly["thin"]) == [False, True]

    def test_the_month_is_the_one_the_customer_asked_for(self):
        """
        A line requested in March and shipped in July is a March miss, not a July one.
        Bucketing on the ship date put it in July, where it read as that month's
        failure — and the month the promise was actually broken showed nothing.
        """
        sales = pd.DataFrame([{
            "sku": "A", "customer": "Acme", "qty": 1.0, "amount": 100.0,
            "order_date": pd.Timestamp("2026-02-01"),
            "customer_request_date": pd.Timestamp("2026-03-15"),
            "ship_date": pd.Timestamp("2026-07-20"),
        }])
        monthly = ServiceAnalyzer().analyze(
            sales_history=sales, as_of=date(2026, 7, 31)).monthly_otd()
        march = monthly[monthly["period"].astype(str) == "2026-03"].iloc[0]
        assert march["lines"] == 1 and march["on_time_lines"] == 0
        # July carries no row at all: the miss belongs to March and is counted once.
        assert "2026-07" not in set(monthly["period"].astype(str))

    def test_a_line_still_sitting_there_counts_against_the_month_it_was_due(self):
        """
        The rule this overrules: the headline leaves open lines out as backlog. Against
        the request date they belong in the denominator — a promise not kept by the
        date is not kept, whatever becomes of the line later.
        """
        sales = pd.DataFrame([{
            "sku": "A", "customer": "Acme", "qty": 1.0, "amount": 100.0,
            "order_date": pd.Timestamp("2026-06-01"),
            "customer_request_date": pd.Timestamp("2026-07-10"),
            "ship_date": pd.Timestamp("2026-07-05"),
        }] * 400)
        open_so = pd.DataFrame([{
            "sku": "A", "customer": "Acme", "open_qty": 1.0, "open_amount": 100.0,
            "order_date": pd.Timestamp("2026-06-01"),
            "customer_request_date": pd.Timestamp("2026-07-20"),
        }] * 100)
        monthly = ServiceAnalyzer().analyze(
            sales_history=sales, open_so=open_so, as_of=date(2026, 7, 31)).monthly_otd()
        july = monthly[monthly["period"].astype(str) == "2026-07"].iloc[0]
        assert july["lines"] == 500
        assert july["on_time_lines"] == 400
        assert july["otd_line_rate"] == pytest.approx(0.80)

    def test_the_order_book_does_not_draw_months_that_have_not_happened(self):
        """
        Where the phantom future months came from. The book legitimately holds lines
        requested for October; a rate over deliveries still to come is not a
        measurement, it is a reading of how far ahead the book runs.
        """
        sales = pd.DataFrame([{
            "sku": "A", "customer": "Acme", "qty": 1.0, "amount": 100.0,
            "order_date": pd.Timestamp("2026-06-01"),
            "customer_request_date": pd.Timestamp("2026-07-10"),
            "ship_date": pd.Timestamp("2026-07-05"),
        }] * 20)
        open_so = pd.DataFrame([{
            "sku": "A", "customer": "Acme", "open_qty": 1.0, "open_amount": 100.0,
            "order_date": pd.Timestamp("2026-07-01"),
            "customer_request_date": d,
        } for d in [pd.Timestamp("2026-09-15")] * 25 + [pd.Timestamp("2026-11-01")] * 25])
        monthly = ServiceAnalyzer().analyze(
            sales_history=sales, open_so=open_so, as_of=date(2026, 7, 31)).monthly_otd()
        assert monthly["period"].max() == pd.Period("2026-07", "M")

    def test_a_month_the_extract_stops_inside_is_marked_partial(self):
        sales = pd.DataFrame([{
            "sku": "A", "customer": "Acme", "qty": 1.0, "amount": 100.0,
            "order_date": pd.Timestamp("2026-06-01"),
            "customer_request_date": pd.Timestamp("2026-07-05"),
            "ship_date": pd.Timestamp("2026-07-03"),
        }] * 10)
        monthly = ServiceAnalyzer().analyze(
            sales_history=sales, as_of=date(2026, 7, 14)).monthly_otd()
        assert bool(monthly.iloc[-1]["partial"])

    def test_the_measured_window_is_reported(self, shipped):
        result = ServiceAnalyzer().analyze(sales_history=shipped, as_of=AS_OF)
        first, last = result.measured_window
        assert str(first) == "2025-05" or str(first) == "2026-05"
        assert last >= first

    def test_an_unmeasurable_extract_says_so_rather_than_drawing_an_empty_axis(self, tmp_path):
        no_request_date = pd.DataFrame({
            "sku": ["A-1"], "qty": [10.0], "amount": [100.0],
            "order_date": pd.to_datetime(["2026-05-01"]),
            "ship_date": pd.to_datetime(["2026-05-10"]),
        })
        service = ServiceAnalyzer().analyze(sales_history=no_request_date, as_of=AS_OF)
        html = KPIReport().render(service=service, output_path=tmp_path / "r.html")
        assert "Not measurable from this extract" in html
        assert "no delivery has an outcome to score" in html


class TestChartsStayWithinTheHouseRules:

    def _report(self, tmp_path):
        sales = TestMonthlyOTD._year_of_shipments([(8, 2), (5, 5), (10, 0)])
        service = ServiceAnalyzer().analyze(sales_history=sales, as_of=date(2025, 5, 1))
        return KPIReport().render(service=service, service_target=0.95,
                                  output_path=tmp_path / "r.html")

    def test_charts_are_inline_svg_with_no_script(self, tmp_path):
        """The report opens from a share with no assets; a plotting library is not
        available and would not be appropriate if it were."""
        html = self._report(tmp_path)
        assert "<svg" in html
        for token in ("<script", "http://", "https://", "@import"):
            assert token not in html

    def test_gridlines_are_solid(self, tmp_path):
        """Dashing reads as 'projection' or 'threshold' when it is just a grid."""
        html = self._report(tmp_path)
        assert "stroke-dasharray" not in html

    def test_the_target_line_is_drawn_and_labelled(self, tmp_path):
        assert "target 95%" in self._report(tmp_path)

    def test_the_chart_does_not_label_every_point(self, tmp_path):
        """
        A value beside every dot is chaos and goes unread — only the newest month and
        the worst one are labelled. Counted as bare value labels, so the target line
        and the volume panel's own caption do not read as data labels.
        """
        import re

        html = self._report(tmp_path)
        chart = html.split("<svg")[1].split("</svg>")[0]
        labels = re.findall(r"<text class='lbl'[^>]*>(.*?)</text>", chart)
        values = [t for t in labels if re.fullmatch(r"\d+%", t)]
        markers = chart.count("<circle")
        assert markers == 3, "one marker per month"
        assert len(values) <= 2, f"expected at most two value labels, got {values}"


class TestCadencePatterns:

    def test_the_curve_is_kept_for_the_report_to_draw(self):
        from inventory_planning.policy.cadence import CadenceAnalyzer
        from test_cadence import FLAT, MONTHS, attributes, po_history, sales_history

        result = CadenceAnalyzer().analyze(
            po_history=po_history(A=[150] * 12), sales_history=sales_history(A=FLAT),
            sku_attributes=attributes("A"), as_of=date(2026, 7, 31),
        )
        curve = result.curve("A")
        assert curve is not None and len(curve) == 12
        # The cumulative is what the verdict was read off; it has to agree with it.
        assert curve["cumulative"].iloc[-1] == pytest.approx(
            float(result.frame.iloc[0]["closing_balance"])
        )
        assert list(curve.columns) == ["period", "po_qty", "demand_qty", "net", "cumulative"]

    def test_an_unknown_sku_has_no_curve(self):
        from inventory_planning.policy.cadence import CadenceAnalyzer
        from test_cadence import FLAT, attributes, po_history, sales_history

        result = CadenceAnalyzer().analyze(
            po_history=po_history(A=[150] * 12), sales_history=sales_history(A=FLAT),
            sku_attributes=attributes("A"), as_of=date(2026, 7, 31),
        )
        assert result.curve("NOT-A-SKU") is None


class TestCellRendering:

    def test_infinite_cover_reads_as_no_demand(self):
        """`inf` through the numeric formatter is an em dash, indistinguishable from
        a missing value — and 'never clears' is a finding, not an absence."""
        assert "no demand" in KPIReport._cell(np.inf, "cover")

    def test_an_unknown_stocking_policy_is_blank_not_guessed(self):
        """MTO and MTS carry opposite expectations about whether stock should have
        been on the shelf; defaulting one rewrites the finding."""
        cell = KPIReport._cell(np.nan, "policy")
        assert "MTO" not in cell and "MTS" not in cell

    def test_a_known_policy_is_shown_as_text_not_a_status_colour(self):
        cell = KPIReport._cell("MTS", "policy")
        assert "MTS" in cell and "chip" not in cell
