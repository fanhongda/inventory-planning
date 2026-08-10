"""
Tests for lead-time persistence and drift detection.

The point of recording lead time in each snapshot is that a supplier can slide from
45 days to 62 over a few months without anything firing: every run measures it afresh
and builds an internally consistent plan around the new number. Only a comparison
across runs surfaces it.

So there are two halves to test — that the value the run planned on is what gets
written down, and that reading successive snapshots back reports the movement without
inventing movement that did not happen.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.feedback.drift import LeadTimeDriftTracker
from inventory_planning.feedback.snapshot import SnapshotSaver

CONFIG_DIR = Path(__file__).parents[1] / "config"


def _snapshot(tmp_path, month: str, entries: dict, run: str = "01_0900") -> Path:
    """One month's snapshot carrying only what the drift tracker reads."""
    folder = tmp_path / "history" / month
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "planning_month": month,
        "skus": {
            sku: {
                "lead_time_days": lt,
                "lt_source": source,
                "lt_samples": samples,
                "supplier": "ACME",
            }
            for sku, (lt, source, samples) in entries.items()
        },
    }
    path = folder / f"snapshot_{month.replace('-', '')}{run}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── Writing ──────────────────────────────────────────────────────────────────


class TestSnapshotRecordsLeadTime:

    @pytest.fixture
    def results(self):
        """A minimal run: recommendations drive the SKU list, safety_stock the LT."""
        return {
            "recommendations": pd.DataFrame({
                "sku": ["A-1", "B-1"],
                "recommended_action": ["HOLD-OK", "PURCHASE-REQUEST"],
                "suggested_po_qty": [0.0, 50.0],
                "forecast_next_period": [10.0, 20.0],
                "forecast_avg_monthly": [10.0, 20.0],
                "net_requirement": [0.0, 50.0],
                "available_supply": [100.0, 10.0],
                "safety_stock": [20.0, 30.0],
            }),
            "projection": pd.DataFrame({
                "sku": ["A-1", "B-1"],
                "days_of_supply": [60.0, 15.0],
                "inventory_status": ["OK", "SHORTAGE-RISK"],
            }),
            "classified_demand": pd.DataFrame({
                "sku": ["A-1", "B-1"],
                "demand_pattern": ["smooth", "smooth"],
                "demand_cv": [0.2, 0.4],
                "stocking_class": ["stocking-high", "stocking-high"],
            }),
            "safety_stock": pd.DataFrame({
                "sku": ["A-1", "B-1"],
                "wma_lead_time_days": [45.0, 90.0],
                "lt_std_days": [6.0, 0.0],
            }),
            "supplier_lt": pd.DataFrame({
                "sku": ["A-1", "B-1"],
                "supplier": ["ACME", "GLOBEX"],
                "sample_count": [11, 0],
                "lt_source": ["measured", "item_master"],
            }),
        }

    def _saved(self, results, tmp_path):
        path = SnapshotSaver().save(results, {}, tmp_path / "output")
        return json.loads(path.read_text(encoding="utf-8"))["skus"]

    def test_lead_time_is_recorded(self, results, tmp_path):
        assert self._saved(results, tmp_path)["A-1"]["lead_time_days"] == 45.0

    def test_sigma_and_sample_count_travel_with_it(self, results, tmp_path):
        row = self._saved(results, tmp_path)["A-1"]
        assert row["lt_sigma_days"] == 6.0
        assert row["lt_samples"] == 11

    def test_source_is_recorded(self, results, tmp_path):
        """A stated lead time must be distinguishable from a measured one later."""
        saved = self._saved(results, tmp_path)
        assert saved["A-1"]["lt_source"] == "measured"
        assert saved["B-1"]["lt_source"] == "item_master"

    def test_value_comes_from_what_the_plan_used(self, results, tmp_path):
        """
        B-1's 90 days came from an item master, so it is absent from any receipt
        history. Recomputing here instead of reading the safety-stock frame would lose
        it and the record would disagree with the plan.
        """
        assert self._saved(results, tmp_path)["B-1"]["lead_time_days"] == 90.0

    def test_missing_lead_time_frame_is_survivable(self, results, tmp_path):
        results = dict(results)
        results.pop("safety_stock")
        saved = self._saved(results, tmp_path)
        assert "lead_time_days" not in saved["A-1"]

    def test_supplier_with_most_history_is_the_one_recorded(self, results, tmp_path):
        """Same rule the planning layer uses — not the fastest supplier."""
        results = dict(results)
        results["supplier_lt"] = pd.DataFrame({
            "sku": ["A-1", "A-1"],
            "supplier": ["RARELY-USED", "WORKHORSE"],
            "sample_count": [1, 14],
            "lt_source": ["measured", "measured"],
        })
        assert self._saved(results, tmp_path)["A-1"]["supplier"] == "WORKHORSE"


# ── Reading ──────────────────────────────────────────────────────────────────


class TestDriftDetection:

    def test_material_lengthening_is_reported(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (62.0, "measured", 9)})

        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert len(result.moves) == 1
        move = result.moves[0]
        assert move.change_days == pytest.approx(17.0)
        assert move.direction == "lengthening"
        assert move.first_month == "2026-05" and move.latest_month == "2026-08"

    def test_shortening_is_reported_separately(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (60.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (30.0, "measured", 8)})

        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert len(result.shortening) == 1
        assert result.lengthening == []

    def test_small_wobble_is_not_drift(self, tmp_path):
        """40 to 42 days is noise; reporting it trains the reader to skip the section."""
        _snapshot(tmp_path, "2026-05", {"S1": (40.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (42.0, "measured", 8)})
        assert LeadTimeDriftTracker().track(tmp_path / "history").moves == []

    def test_both_thresholds_must_clear(self, tmp_path):
        """3 -> 5 days is +67% but only 2 days — an absolute floor keeps it out."""
        _snapshot(tmp_path, "2026-05", {"S1": (3.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (5.0, "measured", 8)})
        assert LeadTimeDriftTracker().track(tmp_path / "history").moves == []

    def test_large_absolute_move_on_a_long_lead_time_clears(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (90.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (115.0, "measured", 8)})
        assert len(LeadTimeDriftTracker().track(tmp_path / "history").moves) == 1

    def test_source_change_is_not_drift(self, tmp_path):
        """
        A stated 90 replacing a measured 44 means the pipeline learned something, not
        that the supplier changed. Calling it drift invents a supplier problem.
        """
        _snapshot(tmp_path, "2026-05", {"S1": (44.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (90.0, "item_master", 0)})

        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert result.moves == []
        assert len(result.source_changes) == 1
        assert result.source_changes[0].latest_source == "item_master"

    def test_two_stated_values_are_not_drift(self, tmp_path):
        """A master edit is a master-data change, already surfaced by the cross-check."""
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "item_master", 0)})
        _snapshot(tmp_path, "2026-08", {"S1": (90.0, "item_master", 0)})

        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert result.moves == []
        assert result.source_changes == []

    def test_a_single_month_cannot_show_drift(self, tmp_path):
        _snapshot(tmp_path, "2026-08", {"S1": (45.0, "measured", 8)})
        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert result.moves == []
        assert "second month" in result.reason

    def test_missing_history_is_not_an_error(self, tmp_path):
        result = LeadTimeDriftTracker().track(tmp_path / "nowhere")
        assert result.moves == []
        assert "no history" in result.reason

    def test_sku_present_in_only_one_month_is_skipped(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S2": (95.0, "measured", 8)})
        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert result.moves == []
        assert result.skus_tracked == 0

    def test_newest_run_in_a_month_wins(self, tmp_path):
        """A re-run replaces the earlier one rather than supplementing it."""
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (99.0, "measured", 8)}, run="01_0900")
        _snapshot(tmp_path, "2026-08", {"S1": (46.0, "measured", 9)}, run="28_1700")

        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert result.moves == [], "the corrected re-run, not the 99-day first attempt"

    def test_snapshots_without_lead_time_are_ignored(self, tmp_path):
        """Snapshots written before lead time was persisted must not break the read."""
        folder = tmp_path / "history" / "2026-04"
        folder.mkdir(parents=True)
        (folder / "snapshot_20260401_0900.json").write_text(
            json.dumps({"planning_month": "2026-04", "skus": {"S1": {"safety_stock": 10}}}),
            encoding="utf-8",
        )
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (62.0, "measured", 8)})

        result = LeadTimeDriftTracker().track(tmp_path / "history")
        assert result.months == ["2026-05", "2026-08"]
        assert len(result.moves) == 1

    def test_malformed_snapshot_does_not_abort_the_read(self, tmp_path):
        folder = tmp_path / "history" / "2026-04"
        folder.mkdir(parents=True)
        (folder / "snapshot_20260401_0900.json").write_text("{ not json", encoding="utf-8")
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (62.0, "measured", 8)})
        assert len(LeadTimeDriftTracker().track(tmp_path / "history").moves) == 1

    def test_thresholds_are_configurable(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (40.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (42.0, "measured", 8)})
        tracker = LeadTimeDriftTracker(relative_threshold=0.01, absolute_threshold=1.0)
        assert len(tracker.track(tmp_path / "history").moves) == 1


class TestDriftOutputs:

    @pytest.fixture
    def result(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8),
                                        "S2": (60.0, "measured", 5)})
        _snapshot(tmp_path, "2026-08", {"S1": (62.0, "measured", 9),
                                        "S2": (30.0, "measured", 6)})
        return LeadTimeDriftTracker().track(tmp_path / "history")

    def test_frame_is_writable(self, result, tmp_path):
        frame = result.frame()
        assert len(frame) == 2
        assert set(frame.columns) >= {"sku", "first_lt_days", "latest_lt_days",
                                      "change_days", "change_pct", "direction"}
        frame.to_csv(tmp_path / "drift.csv", index=False)

    def test_summary_separates_the_two_directions(self, result):
        text = result.summary()
        assert "Lengthening" in text and "Shortening" in text

    def test_summary_says_so_when_nothing_moved(self, tmp_path):
        _snapshot(tmp_path, "2026-05", {"S1": (45.0, "measured", 8)})
        _snapshot(tmp_path, "2026-08", {"S1": (45.0, "measured", 8)})
        summary = LeadTimeDriftTracker().track(tmp_path / "history").summary()
        assert "No lead time moved" in summary
