"""
Which items the run plans, and which receipts measure their lead time.

Two defects with the same shape: a global filter standing in for a per-item one, so
items drop out of a table without anything saying they did.

  lead-time window   `compute_supplier_lt` cut the history to the months before the
                     newest PO date in the *whole extract* before grouping. An item
                     bought quarterly, against a catalogue bought monthly, had every
                     one of its receipts cut away by other items' order dates and
                     vanished from the lead-time table — silently, and precisely for
                     the long-lead-time items where a measured lead time is worth most.
                     The window now selects and the pair's own history is the fallback.

  planning population Everything downstream of `classified` is a left join onto it, so
                     an item absent from the demand history was not planned, not
                     reported and not counted — including stock held against an item
                     that has never sold, which was invisible even to the slow-moving
                     analysis built on the same frame. The population is now the demand
                     history union the masters, and `demand_source` says which is which.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.demand_classifier import DemandClassifier
from inventory_planning.analytics.safety_stock import SafetyStockCalculator
from inventory_planning.orchestrator import InventoryPlanner
from inventory_planning.readers.po_history_reader import (
    DEFAULT_LT_ROLLING_MONTHS,
    POHistoryReader,
)

CONFIG_DIR = Path(__file__).parents[1] / "config"


def _config_with(tmp_path: Path, **overrides) -> Path:
    """A copy of the shipped config with `stocking_policy.json` edited."""
    out = tmp_path / "config"
    out.mkdir(exist_ok=True)
    for src in CONFIG_DIR.glob("*.json"):
        (out / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    path = out / "stocking_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy.update(overrides)
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return out


@pytest.fixture
def uneven_cadence() -> pd.DataFrame:
    """
    One item bought quarterly at 90 days, one bought monthly at 30 — and the newest
    receipt in the extract belongs to the monthly one.

    This is the shape that used to lose the quarterly item: the global cutoff is
    derived from the monthly item's latest order date.
    """
    rows = [{"sku": "QUARTERLY", "supplier": "S1", "po_date": d, "lead_time_days": 90}
            for d in ("2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15")]
    rows += [{"sku": "MONTHLY", "supplier": "S1", "po_date": f"2026-{m:02d}-10",
              "lead_time_days": 30} for m in range(1, 12)]
    df = pd.DataFrame(rows)
    df["po_date"] = pd.to_datetime(df["po_date"])
    return df


class TestTheLeadTimeWindowIsPerPairNotPerExtract:

    def test_an_infrequently_bought_item_keeps_its_measured_lead_time(self, uneven_cadence):
        """The defect itself: QUARTERLY used to disappear from the table entirely."""
        lt = POHistoryReader(CONFIG_DIR).compute_supplier_lt(uneven_cadence)
        assert set(lt["sku"]) == {"QUARTERLY", "MONTHLY"}
        row = lt[lt["sku"] == "QUARTERLY"].iloc[0]
        # Its own four receipts, not a stated value from master data with no sigma.
        assert row["wma_lead_time_days"] == 90.0
        assert row["sample_count"] == 4

    def test_a_lead_time_from_outside_the_window_says_so(self, uneven_cadence):
        """
        The fallback is labelled rather than passed off as a current measurement.

        A lead time measured from receipts a year old is evidence of a different grade,
        and `lt_source` is the column the rest of the run already reads for that.
        """
        lt = POHistoryReader(CONFIG_DIR).compute_supplier_lt(uneven_cadence)
        source = lt.set_index("sku")["lt_source"]
        assert source["QUARTERLY"] == "measured_outside_window"
        assert source["MONTHLY"] == "measured"

    def test_a_pair_inside_the_window_is_measured_on_the_window_alone(self, uneven_cadence):
        """
        The window still selects. MONTHLY has eleven receipts and seven of them fall in
        the six months before the newest — the fallback must not quietly widen it.
        """
        lt = POHistoryReader(CONFIG_DIR).compute_supplier_lt(uneven_cadence)
        assert lt.set_index("sku").loc["MONTHLY", "sample_count"] == 7

    def test_the_window_comes_from_config(self, tmp_path, uneven_cadence):
        """
        `lt_rolling_months` has sat in `stocking_policy.json` unread: the function took
        its own default and a planner who set 24 got 6, with nothing saying so.
        """
        reader = POHistoryReader(_config_with(tmp_path, lt_rolling_months=24))
        assert reader._lt_rolling_months() == 24
        lt = reader.compute_supplier_lt(uneven_cadence)
        # At 24 months both pairs are inside the window, so nothing falls back and
        # MONTHLY is measured on all eleven of its receipts rather than seven.
        assert set(lt["lt_source"]) == {"measured"}
        assert lt.set_index("sku").loc["MONTHLY", "sample_count"] == 11
        assert set(lt["lt_window_months"]) == {24}

    def test_an_absent_or_unusable_setting_keeps_the_shipped_window(self, tmp_path):
        """Six months is what this defaulted to for as long as it existed."""
        assert POHistoryReader(tmp_path)._lt_rolling_months() == DEFAULT_LT_ROLLING_MONTHS
        for bad in (0, -3, "soon", None):
            reader = POHistoryReader(_config_with(tmp_path, lt_rolling_months=bad))
            assert reader._lt_rolling_months() == DEFAULT_LT_ROLLING_MONTHS


@pytest.fixture
def item_master() -> pd.DataFrame:
    """Two items the demand history has never heard of, one of them stocked."""
    return pd.DataFrame({
        "sku": ["SOLD-01", "DEAD-01", "NEW-01"],
        "supplier": ["S1", "S1", "S2"],
        "lead_time_days": [30.0, 75.0, 40.0],
        "item_status": ["active", "active", "active"],
    })


@pytest.fixture
def sold_demand() -> tuple:
    """A one-item demand history: the series and the summary built from it."""
    periods = pd.period_range("2025-01", "2025-12", freq="M")
    ts = pd.DataFrame({"SOLD-01": [10.0] * 12}, index=periods)
    summary = pd.DataFrame([{
        "sku": "SOLD-01", "location_id": "DC-01", "demand_mean": 10.0,
        "demand_std": 0.0, "demand_cv": 0.0, "active_cycles": 12,
        "total_cycles": 12, "first_sale": pd.Timestamp("2025-01-01"),
        "last_sale": pd.Timestamp("2025-12-01"), "total_qty": 120.0,
        "total_amount": 1200.0,
    }])
    return ts, summary


class TestThePlanningPopulationIsDemandUnionMaster:

    def test_master_only_items_enter_the_population(self, tmp_path, sold_demand, item_master):
        ts, summary = sold_demand
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        out, population = planner._extend_population_from_masters(
            summary, ts, item_master, None)
        assert set(out["sku"]) == {"SOLD-01", "DEAD-01", "NEW-01"}
        assert population == {"SOLD-01", "DEAD-01", "NEW-01"}

    def test_each_row_says_where_it_came_from(self, tmp_path, sold_demand, item_master):
        """
        Without this the two are indistinguishable on the face of the report: both come
        out at zero should-be and NO-ACTION, and they want different answers — one is a
        dormant item the history measured, the other has no history at all.
        """
        ts, summary = sold_demand
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        out, _ = planner._extend_population_from_masters(summary, ts, item_master, None)
        source = out.set_index("sku")["demand_source"]
        assert source["SOLD-01"] == "sales_history"
        assert source["DEAD-01"] == "item_master"

    def test_they_arrive_at_zero_demand_rather_than_an_invented_one(
            self, tmp_path, sold_demand, item_master):
        """No history is not a forecast of nothing, and nothing here makes one up."""
        ts, summary = sold_demand
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        out, _ = planner._extend_population_from_masters(summary, ts, item_master, None)
        added = out[out["demand_source"] != "sales_history"]
        assert (added["demand_mean"] == 0.0).all()
        assert (added["active_cycles"] == 0).all()
        assert added["last_sale"].isna().all()

    def test_the_demand_series_is_not_padded(self, tmp_path, sold_demand, item_master):
        """
        `ts` is the forecaster's input and the basis of every demand-coverage figure.
        Zero columns there would collapse the median history depth and have the demand
        gate report a catalogue of dead series as a thin history.
        """
        ts, summary = sold_demand
        before = list(ts.columns)
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        planner._extend_population_from_masters(summary, ts, item_master, None)
        assert list(ts.columns) == before == ["SOLD-01"]

    def test_a_master_only_item_is_classified_and_carries_no_safety_stock(
            self, tmp_path, sold_demand, item_master):
        """
        End of the chain: it reaches safety stock, and gets none. There is no demand to
        size either safety stock or an order quantity against, and inventing one is the
        failure this is not allowed to become.
        """
        ts, summary = sold_demand
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        out, _ = planner._extend_population_from_masters(summary, ts, item_master, None)
        classified = DemandClassifier(CONFIG_DIR).classify(out, ts)
        assert classified.set_index("sku").loc["DEAD-01", "stocking_class"] == "non-stocking"

        lt = pd.DataFrame({"sku": ["SOLD-01"], "supplier": ["S1"],
                           "wma_lead_time_days": [30.0], "lt_std_days": [3.0],
                           "sample_count": [5]})
        ss = SafetyStockCalculator().calculate(classified, lt)
        assert "DEAD-01" in set(ss["sku"])
        assert ss.set_index("sku").loc["DEAD-01", "safety_stock"] == 0
        assert "demand_source" in ss.columns

    def test_the_master_lead_time_fill_is_scoped_to_the_population(
            self, tmp_path, sold_demand, item_master):
        """
        The fill used to be scoped to `ts.columns`, which is what made the union
        pointless on its own: a master-only item would have been planned with no lead
        time at all, so no reorder point and no coverage.
        """
        ts, summary = sold_demand
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        _, population = planner._extend_population_from_masters(
            summary, ts, item_master, None)
        measured = pd.DataFrame({"sku": ["SOLD-01"], "supplier": ["S1"],
                                 "wma_lead_time_days": [30.0], "lt_std_days": [3.0],
                                 "sample_count": [5]})
        filled = planner._fill_lead_time_from_masters(
            measured, population, item_master, None)
        rows = filled.set_index("sku")
        assert rows.loc["DEAD-01", "wma_lead_time_days"] == 75.0
        assert rows.loc["DEAD-01", "lt_source"] == "item_master"
        # Still labelled as stated rather than measured, and carrying no sigma.
        assert rows.loc["DEAD-01", "lt_std_days"] == 0.0

    def test_nothing_is_added_when_no_master_was_supplied(self, tmp_path, sold_demand):
        ts, summary = sold_demand
        planner = InventoryPlanner(output_dir=tmp_path, interactive=False)
        out, population = planner._extend_population_from_masters(summary, ts, None, None)
        assert set(out["sku"]) == {"SOLD-01"}
        assert population == {"SOLD-01"}
        assert list(out["demand_source"]) == ["sales_history"]
