"""
Rolling rows up to the contract grain adds what is additive and nothing else.

A stock report listed one part in eleven bins of one warehouse. The rollup to
`sku + location_id` summed every numeric measure, and `unit_cost` is a measure — so
the part arrived at eleven times its standard cost. Every quantity was right; the
inventory value was 3.3 times the report's, wrong by a whole-number factor that
differed part by part (the bin count).

Supersession already combined rows correctly, by unit. The rollup now uses the same
rule, `aggregate_on_key`, so there is one.
"""

import pandas as pd
import pytest

from inventory_planning.ingest.adapter import TransformStep
from inventory_planning.ingest.contract import default_registry
from inventory_planning.ingest.registry import AdapterRegistry


@pytest.fixture
def contracts():
    return default_registry()


def _rollup(frame, doc_type, contracts, tmp_path):
    """Force an adapter of this doc type through the rollup."""
    adapter = AdapterRegistry(adapters_dir=tmp_path).route(
        frame, source_name="bins.xlsx", doc_type_hint=doc_type).adapter
    adapter.grain, adapter.rollup_to = "bin", "sku_location"
    log: list[TransformStep] = []
    out = adapter._apply_rollup(frame.copy(), contracts.get(doc_type), log)
    assert any(s.action == "rolled_up" and "SKIPPED" not in s.detail for s in log)
    return out


def _bins(costs, qtys):
    n = len(qtys)
    return pd.DataFrame({
        "sku": ["4010-9701"] * n, "location_id": ["SG01"] * n,
        "line_number": [f"L{i:02d}" for i in range(n)],
        "qty_on_hand": qtys, "unit_cost": costs,
        "inventory_value": [round(q * c, 2) for q, c in zip(qtys, costs)],
        "currency": ["USD"] * n,
    })


class TestAPriceIsNotAnAmount:
    def test_eleven_bins_are_not_eleven_times_the_cost(self, contracts, tmp_path):
        """The case as it arrived: 108 units in 11 bins at 914.41."""
        frame = _bins([914.41] * 11, [10] * 10 + [8])
        out = _rollup(frame, "inventory", contracts, tmp_path)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["qty_on_hand"] == 108
        assert row["unit_cost"] == pytest.approx(914.41)
        assert row["inventory_value"] == pytest.approx(98_756.28)

    def test_quantity_times_cost_is_the_reports_own_value(self, contracts, tmp_path):
        """The check a planner does by hand, and what made the defect visible."""
        frame = _bins([914.41] * 11, [10] * 10 + [8])
        row = _rollup(frame, "inventory", contracts, tmp_path).iloc[0]
        assert row["qty_on_hand"] * row["unit_cost"] == pytest.approx(
            frame["inventory_value"].sum())

    def test_bins_at_different_costs_are_weighted_by_quantity(self, contracts, tmp_path):
        """
        A plain mean would say 55; the stock is worth 10x10 + 90x100, so 91 a unit.
        Weighting keeps quantity times cost equal to the value the bins hold.
        """
        row = _rollup(_bins([10.0, 100.0], [10, 90]), "inventory",
                      contracts, tmp_path).iloc[0]
        assert row["unit_cost"] == pytest.approx(91.0)
        assert row["qty_on_hand"] * row["unit_cost"] == pytest.approx(9_100.0)

    def test_a_part_with_nothing_on_the_shelf_keeps_its_cost(self, contracts, tmp_path):
        """Zero stock in every bin is ordinary; there is no weight, so the first stands."""
        row = _rollup(_bins([914.41, 914.41], [0, 0]), "inventory",
                      contracts, tmp_path).iloc[0]
        assert row["unit_cost"] == pytest.approx(914.41)


class TestOnlyAmountsAdd:
    def test_quantities_and_money_still_sum(self, contracts, tmp_path):
        row = _rollup(_bins([5.0, 5.0, 5.0], [1, 2, 3]), "inventory",
                      contracts, tmp_path).iloc[0]
        assert row["qty_on_hand"] == 6
        assert row["inventory_value"] == pytest.approx(30.0)

    def test_an_age_in_days_is_not_added_up(self, contracts, tmp_path):
        """Two bins aged 30 days are not 60 days old."""
        frame = _bins([5.0, 5.0], [1, 1]).assign(inventory_age_days=[30, 30])
        row = _rollup(frame, "inventory", contracts, tmp_path).iloc[0]
        assert row["inventory_age_days"] == 30


class TestAMasterRowIsADescription:
    def test_two_rows_of_one_part_do_not_double_its_moq(self, contracts, tmp_path):
        """
        A key of `[sku]` alone means each row describes the part. Two rows of MOQ 50
        are one MOQ of 50, and two 90-day lead times are not a 180-day one.
        """
        frame = pd.DataFrame({
            "sku": ["4010-9701", "4010-9701"], "min_order_qty": [50, 50],
            "lead_time_days": [90, 90], "unit_cost": [914.41, 914.41],
        })
        adapter = AdapterRegistry(adapters_dir=tmp_path).route(
            frame, source_name="master.xlsx", doc_type_hint="item_master").adapter
        adapter.grain, adapter.rollup_to = "sku_plant", "sku"
        out = adapter._apply_rollup(frame.copy(), contracts.get("item_master"), [])
        row = out.iloc[0]
        assert (row["min_order_qty"], row["lead_time_days"]) == (50, 90)
        assert row["unit_cost"] == pytest.approx(914.41)
