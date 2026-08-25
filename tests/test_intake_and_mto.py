"""
The checks that would have caught what a clean run hid.

Three failures reached a planner's report without raising anything: a value column read
as a quantity, a promise date read as a shipment, and an order-on-demand item whose
purchase order was already late when the run said there was nothing to do. Each is
cheap to detect and none of them was detected, so each gets a test that fails if the
detection goes away.
"""

import numpy as np
import pandas as pd
import pytest

from inventory_planning.analytics.purchase_recommender import PurchaseRecommender
from inventory_planning.fx import FxTable
from inventory_planning.readers.open_so_reader import OpenSOReader
from inventory_planning.reporting.intake_summary import summarise_intake

ANCHOR = pd.Timestamp("2026-08-11")


def _reader() -> OpenSOReader:
    return OpenSOReader.__new__(OpenSOReader)


@pytest.fixture
def healthy():
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame({
        "sku": [f"S{i % 30:03d}" for i in range(n)],
        "open_qty": rng.integers(1, 40, n).astype(float),
        "open_amount": rng.integers(1, 40, n) * 120.0,
        "po_date": pd.to_datetime("2026-06-01") + pd.to_timedelta(rng.integers(0, 60, n), "D"),
    })


class TestAValueColumnReadAsAQuantity:
    """`open_qty <- 'Still to be delivered (value)'` — $2.9bn of open POs, no error."""

    def test_a_fractional_quantity_column_is_flagged(self, healthy):
        healthy["open_qty"] = healthy["open_amount"] * 1.0001
        doc = summarise_intake({"open_po": healthy}, anchor=ANCHOR).documents[0]
        qty = next(r for r in doc.readings if r.column == "open_qty")
        assert qty.integer_share < 0.6
        assert any("whole numbers" in f for f in qty.flags)

    def test_a_unit_value_of_one_names_the_two_columns(self, healthy):
        healthy["open_qty"] = healthy["open_amount"]
        doc = summarise_intake({"open_po": healthy}, anchor=ANCHOR).documents[0]
        assert any("same number" in f for f in doc.flags)

    def test_a_healthy_document_raises_nothing(self, healthy):
        assert not summarise_intake({"open_po": healthy}, anchor=ANCHOR).suspect


class TestAPromiseDateReadAsAnEvent:
    """`ship_date <- 'Planned Ship Date'` — the run's today became 2027-12-30."""

    def test_a_future_date_in_a_backward_looking_column_is_flagged(self):
        so = pd.DataFrame({
            "sku": ["A"] * 20, "open_qty": np.arange(1.0, 21.0),
            "ship_date": pd.to_datetime("2026-08-01") + pd.to_timedelta(np.arange(20) * 30, "D"),
        })
        doc = summarise_intake({"open_so": so}, anchor=ANCHOR).documents[0]
        ship = next(r for r in doc.readings if r.column == "ship_date")
        assert any("days ahead" in f for f in ship.flags)

    def test_a_request_date_in_the_future_is_not_flagged_as_wrong(self):
        """A customer request date is *supposed* to be ahead. Flagging it is noise."""
        so = pd.DataFrame({
            "sku": ["A"] * 20, "open_qty": np.arange(1.0, 21.0),
            "customer_request_date": pd.to_datetime("2026-09-01")
            + pd.to_timedelta(np.arange(20) * 15, "D"),
        })
        doc = summarise_intake({"open_so": so}, anchor=ANCHOR).documents[0]
        request = next(r for r in doc.readings if r.column == "customer_request_date")
        assert not request.flags, "a forward-looking column in the future is normal"

    def test_a_request_date_beyond_any_commitment_is_still_flagged(self):
        so = pd.DataFrame({
            "sku": ["A"] * 5, "open_qty": [1.0] * 5,
            "customer_request_date": pd.to_datetime(["2030-01-01"] * 5),
        })
        doc = summarise_intake({"open_so": so}, anchor=ANCHOR).documents[0]
        request = next(r for r in doc.readings if r.column == "customer_request_date")
        assert any("date parsing" in f for f in request.flags)

    def test_a_future_anchor_is_reported_and_not_used_as_the_yardstick(self):
        so = pd.DataFrame({
            "sku": ["A"] * 5, "open_qty": [1.0] * 5,
            "ship_date": pd.to_datetime(["2027-01-01"] * 5),
        })
        summary = summarise_intake({"open_so": so}, anchor=pd.Timestamp("2027-12-30"))
        assert summary.anchor_flag and "in the future" in summary.anchor_flag
        # Judged against today, not against the corrupted anchor, so it still trips.
        ship = next(r for r in summary.documents[0].readings if r.column == "ship_date")
        assert ship.flags


class TestOrderOnDemandBuysOnTheOrderDate:
    """The buy date for an MTO item is the customer date minus the lead time."""

    @pytest.fixture
    def projection(self):
        return pd.DataFrame({
            "sku": ["MTO1"], "location_id": ["PL10"], "stocking_class": ["non-stocking"],
            "demand_mean_rolling": [5.9], "wma_lead_time_days": [51.7],
            "safety_stock": [0.0], "effective_position": [0.0],
            "inventory_status": ["non-stocking"], "pushout_candidate": [False],
            "pushout_open_po_qty": [0.0],
        })

    @pytest.fixture
    def book(self):
        # Wanted in 40 days. Outside a flat 30-day horizon, and already 12 days past
        # its order date once the 51.7-day lead time is applied.
        return pd.DataFrame({"sku": ["MTO1"], "open_qty": [6.0],
                             "customer_request_date": pd.to_datetime(["2026-09-20"])})

    @pytest.fixture
    def backlog(self):
        return pd.DataFrame({"sku": ["MTO1"], "backlog_qty": [6.0],
                             "backlog_due_qty": [0.0], "backlog_past_due_qty": [0.0]})

    def test_the_order_by_date_is_the_request_less_the_lead_time(self, book):
        sched = _reader().order_by_schedule(
            book, pd.DataFrame({"sku": ["MTO1"], "lead_time_days": [51.7]}),
            as_of=ANCHOR, review_period_days=30)
        row = sched.iloc[0]
        assert row["mto_order_by"] < ANCHOR
        assert row["mto_order_past_due_qty"] == pytest.approx(6.0)

    def test_a_flat_horizon_misses_it_entirely(self, projection, backlog):
        row = PurchaseRecommender().recommend(projection, None, backlog, None).iloc[0]
        assert row["recommended_action"] == "NO-ACTION"
        assert row["mto_order_status"] == "unscoped"

    def test_the_lead_time_aware_book_orders_it(self, projection, book, backlog):
        sched = _reader().order_by_schedule(
            book, pd.DataFrame({"sku": ["MTO1"], "lead_time_days": [51.7]}),
            as_of=ANCHOR, review_period_days=30)
        row = PurchaseRecommender().recommend(
            projection, None, backlog, None, mto_schedule=sched).iloc[0]
        assert row["recommended_action"] == "ORDER-FOR-BACKLOG"
        assert row["suggested_po_qty"] == pytest.approx(6.0)
        assert row["mto_order_status"] == "order-past-due"

    def test_a_line_whose_order_date_has_not_arrived_is_seen_but_not_bought(
        self, projection, backlog
    ):
        far = pd.DataFrame({"sku": ["MTO1"], "open_qty": [6.0],
                            "customer_request_date": pd.to_datetime(["2027-06-01"])})
        sched = _reader().order_by_schedule(
            far, pd.DataFrame({"sku": ["MTO1"], "lead_time_days": [51.7]}),
            as_of=ANCHOR, review_period_days=30)
        row = PurchaseRecommender().recommend(
            projection, None, backlog, None, mto_schedule=sched).iloc[0]
        assert row["recommended_action"] == "NO-ACTION"
        assert row["mto_order_status"] == "scheduled"

    def test_realization_discounts_the_mto_buy(self, projection, book, backlog):
        class _Half:
            global_rate = 0.5

            def apply(self, skus):
                return pd.Series(0.5, index=skus.index)

        sched = _reader().order_by_schedule(
            book, pd.DataFrame({"sku": ["MTO1"], "lead_time_days": [51.7]}),
            as_of=ANCHOR, review_period_days=30)
        row = PurchaseRecommender().recommend(
            projection, None, backlog, None, realization=_Half(), mto_schedule=sched).iloc[0]
        assert row["suggested_po_qty"] == pytest.approx(3.0)


class TestAPlaceholderRateIsNeverSilent:
    def test_seeded_currencies_are_marked_and_measured_ones_are_not(self):
        table = FxTable.load("config")
        assert table.is_placeholder("CNY")
        assert table.is_placeholder("SGD")
        assert not table.is_placeholder("GBP")

    def test_a_comment_key_is_not_read_as_a_currency(self):
        assert not any(c.startswith("_") for c in FxTable.load("config").currencies)
