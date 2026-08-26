"""
Integration tests for the full inventory planning pipeline.
Uses generated sample data (non-interactive mode).
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.readers.sales_history_reader import SalesHistoryReader
from inventory_planning.readers.po_history_reader import POHistoryReader
from inventory_planning.readers.open_so_reader import OpenSOReader
from inventory_planning.readers.open_po_reader import OpenPOReader
from inventory_planning.readers.inventory_reader import InventoryReader
from inventory_planning.analytics.demand_classifier import DemandClassifier
from inventory_planning.analytics.safety_stock import SafetyStockCalculator
from inventory_planning.analytics.inventory_projector import InventoryProjector
from inventory_planning.analytics.forecaster import Forecaster
from inventory_planning.analytics.purchase_recommender import PurchaseRecommender
from inventory_planning.orchestrator import InventoryPlanner

CONFIG_DIR = Path(__file__).parents[1] / "config"
SAMPLE_DIR = Path(__file__).parents[1] / "sample_data"
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "test"


# ── Reader tests ──────────────────────────────────────────────────────────────

class TestSalesHistoryReader:
    def test_loads_and_cleans(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, report = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        assert "sku" in df.columns
        assert "qty" in df.columns
        assert "demand_date" in df.columns
        assert df["qty"].notna().all()
        assert (df["qty"] > 0).all(), "Returns should be filtered"

    def test_time_series_shape(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        ts = r.to_time_series(df)
        assert ts.shape[1] > 0, "Should have SKU columns"
        assert ts.shape[0] >= 6, "Should have at least 6 monthly periods"

    def test_summary_columns(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        summary = r.summarize(df)
        for col in ["sku", "demand_mean", "demand_std", "active_cycles", "total_cycles"]:
            assert col in summary.columns


class TestPOHistoryReader:
    def test_lead_time_computed(self):
        r = POHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "po_history.csv", interactive=False)
        assert "lead_time_days" in df.columns
        assert (df["lead_time_days"] > 0).all()
        assert (df["lead_time_days"] <= 730).all()

    def test_supplier_lt(self):
        r = POHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "po_history.csv", interactive=False)
        lt = r.compute_supplier_lt(df)
        assert "wma_lead_time_days" in lt.columns
        assert (lt["wma_lead_time_days"] > 0).all()


class TestInventoryReader:
    def test_git_adjusted_for_incoterm(self):
        r = InventoryReader(CONFIG_DIR)
        inv_df, _ = r.read(SAMPLE_DIR / "inventory.csv", interactive=False)

        po_r = OpenPOReader(CONFIG_DIR)
        open_po_df, _ = po_r.read(SAMPLE_DIR / "open_po.csv", interactive=False)
        open_po_summary = po_r.inbound_schedule(open_po_df)

        po_hist_r = POHistoryReader(CONFIG_DIR)
        po_hist_df, _ = po_hist_r.read(SAMPLE_DIR / "po_history.csv", interactive=False)
        supplier_lt = po_hist_r.compute_supplier_lt(po_hist_df)

        eff = r.effective_inventory(inv_df, open_po_summary, supplier_lt)
        assert "effective_position" in eff.columns
        # DDP SKUs should have qty_in_transit_adj == 0
        ddp_skus = supplier_lt[supplier_lt["incoterm"] == "DDP"]["sku"].unique()
        ddp_rows = eff[eff["sku"].isin(ddp_skus)]
        if len(ddp_rows):
            assert (ddp_rows["qty_in_transit_adj"] == 0).all(), "DDP GIT must not count"


# ── Analytics tests ───────────────────────────────────────────────────────────

class TestDemandClassifier:
    def test_stocking_classes_assigned(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        ts = r.to_time_series(df)
        summary = r.summarize(df)
        clf = DemandClassifier(CONFIG_DIR)
        result = clf.classify(summary, ts)
        assert "stocking_class" in result.columns
        valid = {"stocking-high", "stocking-med", "non-stocking"}
        assert set(result["stocking_class"]).issubset(valid)


class TestSafetyStock:
    def _get_classified(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        ts = r.to_time_series(df)
        summary = r.summarize(df)
        clf = DemandClassifier(CONFIG_DIR)
        return clf.classify(summary, ts)

    def _get_supplier_lt(self):
        r = POHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "po_history.csv", interactive=False)
        return r.compute_supplier_lt(df)

    def test_ss_non_negative(self):
        classified = self._get_classified()
        supplier_lt = self._get_supplier_lt()
        calc = SafetyStockCalculator()
        result = calc.calculate(classified, supplier_lt)
        assert (result["safety_stock"] >= 0).all()

    def test_ss_zero_for_non_stocking(self):
        classified = self._get_classified()
        supplier_lt = self._get_supplier_lt()
        calc = SafetyStockCalculator()
        result = calc.calculate(classified, supplier_lt)
        non_stocking = result[result["stocking_class"] == "non-stocking"]
        if len(non_stocking):
            assert (non_stocking["safety_stock"] == 0).all()


class TestForecaster:
    def test_forecast_produces_results(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        ts = r.to_time_series(df)
        f = Forecaster(horizon=6)
        result = f.forecast_all(ts)
        assert len(result) > 0
        assert "forecast_qty" in result.columns
        assert (result["forecast_qty"] >= 0).all()

    def test_forecast_horizon(self):
        r = SalesHistoryReader(CONFIG_DIR)
        df, _ = r.read(SAMPLE_DIR / "sales_history.csv", interactive=False)
        ts = r.to_time_series(df)
        f = Forecaster(horizon=6)
        result = f.forecast_all(ts)
        # Each SKU should have exactly 6 periods
        per_sku = result.groupby("sku").size()
        assert (per_sku == 6).all()


# ── End-to-end test ───────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        planner = InventoryPlanner(
            config_dir=CONFIG_DIR,
            output_dir=OUTPUT_DIR,
            interactive=False,
            store_root=tmp_path / "store",
        )
        sales_df,   _ = planner.load_sales_history(SAMPLE_DIR / "sales_history.csv")
        po_hist_df, _ = planner.load_po_history(SAMPLE_DIR / "po_history.csv")
        open_so_df, _ = planner.load_open_so(SAMPLE_DIR / "open_so.csv")
        open_po_df, _ = planner.load_open_po(SAMPLE_DIR / "open_po.csv")
        inv_df,     _ = planner.load_inventory(SAMPLE_DIR / "inventory.csv")

        results = planner.run_planning(sales_df, po_hist_df, open_so_df, open_po_df, inv_df)

        assert "recommendations" in results
        rec = results["recommendations"]
        assert len(rec) > 0
        valid_actions = {
            "PURCHASE-REQUEST", "PUSH-OUT-OPEN-PO", "HOLD-OK",
            "HOLD-EXCESS", "NO-ACTION", "ORDER-FOR-BACKLOG", "EXPEDITE-INBOUND"
        }
        assert set(rec["recommended_action"]).issubset(valid_actions)

        # Output files created
        assert (OUTPUT_DIR / "supplier_params.csv").exists()
        assert (OUTPUT_DIR / "sku_planning_params.csv").exists()

        # The run recorded what produced those files
        from inventory_planning.provenance import RunRegistry
        manifest = RunRegistry(OUTPUT_DIR).get(planner.run.run_id)
        assert manifest is not None
        assert {i["doc_type"] for i in manifest["inputs"]} == {
            "sales_history", "po_history", "open_so", "open_po", "inventory"
        }
        assert all(i["sha256"] for i in manifest["inputs"])
        assert manifest["outputs"]
        assert manifest["input_fingerprint"] and manifest["config_fingerprint"]

        # The run retained its facts, and the snapshot went to the store rather than
        # to a directory derived from --output.
        batches = planner.store.batches()
        assert {b["doc_type"] for b in batches} == {
            "sales_history", "po_history", "open_so", "open_po", "inventory"
        }
        assert all(b["valid_time"] for b in batches)
        assert all(b["run_id"] == planner.run.run_id for b in batches)
        # The parameter identity on a batch has to be the one the manifest records, or
        # the two cannot be joined — which is the only reason it is on the batch.
        assert all(b["config_fingerprint"] == manifest["config_fingerprint"]
                   for b in batches)
        assert list((planner.store.history_dir).glob("*/snapshot_*.json"))


class TestScenarioComparison:
    """
    Two runs over one dataset under two rule sets. This is the comparison run identity
    exists to make, and it is the one that was wrong: hashing only the config directory
    made the pair `identical`, whose description tells the reader that a real policy
    result is non-determinism.
    """

    def test_an_alternate_rule_set_compares_as_a_scenario(self, tmp_path):
        from inventory_planning.provenance import RunRegistry

        alt = tmp_path / "scenario_rules.md"
        alt.write_text(
            (CONFIG_DIR / "planning_parameters.md").read_text(encoding="utf-8")
            .replace("service_level: 0.95", "service_level: 0.99"),
            encoding="utf-8",
        )
        out = tmp_path / "out"

        run_ids = []
        for parameters_file in (None, alt):
            planner = InventoryPlanner(config_dir=CONFIG_DIR, output_dir=out,
                                       interactive=False, store_root=tmp_path / "store")
            inputs = planner.load_all(sorted(SAMPLE_DIR.glob("*.csv")))
            results = planner.run_planning(**inputs)
            planner.run_policy_analysis(
                results,
                inventory_df=inputs["inventory_df"], open_po_df=inputs["open_po_df"],
                parameters_file=parameters_file,
            )
            run_ids.append(planner.run.run_id)

        registry = RunRegistry(out)
        comparison = registry.compare(*run_ids)
        assert comparison.basis == "scenario"
        assert comparison.same_inputs
        assert not comparison.same_config
        # The facts are pinned, so the difference is attributable — and the batches are
        # still keyed on the config identity their manifest records.
        assert registry.get(run_ids[0])["config_fingerprint"] == \
            registry.get(run_ids[1])["config_fingerprint"]
