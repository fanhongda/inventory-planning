"""
Inventory Planning Orchestrator.
Coordinates all readers and analytics modules in the correct sequence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from .feedback.snapshot import SnapshotSaver
from .readers.sales_history_reader import SalesHistoryReader
from .readers.po_history_reader import POHistoryReader
from .readers.open_so_reader import OpenSOReader
from .readers.open_po_reader import OpenPOReader
from .readers.inventory_reader import InventoryReader
from .readers.timeseries_reader import TimeSeriesReader
from .analytics.demand_classifier import DemandClassifier
from .analytics.safety_stock import SafetyStockCalculator
from .analytics.inventory_projector import InventoryProjector
from .analytics.forecaster import Forecaster
from .analytics.purchase_recommender import PurchaseRecommender


class InventoryPlanner:
    """
    Single-stage DC inventory planning pipeline.
    multi-echelon ready: all outputs carry location_id.
    """

    def __init__(self, config_dir: Union[str, Path] = None, output_dir: Union[str, Path] = None,
                 interactive: bool = True):
        base = Path(__file__).parents[1]
        self.config_dir = Path(config_dir) if config_dir else base / "config"
        self.output_dir = Path(output_dir) if output_dir else base / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self._quality_log: list = []   # accumulates quality reports across all loads
        self._intake = None            # set by load_all(); carries adapter provenance
        self._intake_plan = None       # set by load_all(); what this run can answer

        # Readers
        self.sales_reader   = SalesHistoryReader(self.config_dir)
        self.po_reader      = POHistoryReader(self.config_dir)
        self.open_so_reader = OpenSOReader(self.config_dir)
        self.open_po_reader = OpenPOReader(self.config_dir)
        self.inv_reader     = InventoryReader(self.config_dir)
        self.ts_reader      = TimeSeriesReader(self.config_dir)

        # Analytics
        policy_cfg = json.loads((self.config_dir / "stocking_policy.json").read_text())
        self.classifier    = DemandClassifier(self.config_dir)
        self.ss_calc       = SafetyStockCalculator(self.config_dir)
        self.projector     = InventoryProjector(self.config_dir)
        self.forecaster    = Forecaster(horizon=policy_cfg["forecast_horizon_months"])
        self.recommender   = PurchaseRecommender()

    # ------------------------------------------------------------------
    # Phase 1: Data Ingestion (run in any order; returns raw clean dfs)
    # ------------------------------------------------------------------

    def load_sales_history(self, path: Union[str, Path]):
        df, report = self.sales_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_po_history(self, path: Union[str, Path]):
        df, report = self.po_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_open_so(self, path: Union[str, Path]):
        df, report = self.open_so_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_open_po(self, path: Union[str, Path]):
        df, report = self.open_po_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_inventory(self, path: Union[str, Path]):
        df, report = self.inv_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_timeseries(self, path: Union[str, Path], rolling_months: int = 36):
        """Load a pre-compiled wide-format time series file."""
        pivot, meta, report = self.ts_reader.read(
            path, rolling_months=rolling_months, interactive=self.interactive
        )
        return pivot, meta, report

    # ------------------------------------------------------------------
    # Phase 7: Policy analysis — should-be, levers, target
    # ------------------------------------------------------------------

    def run_policy_analysis(
        self,
        results: dict,
        inventory_df=None,
        open_po_df=None,
        target_value: float = None,
        deadline=None,
        parameters_file=None,
    ) -> dict:
        """
        Answer the planner's actual questions on top of a completed planning run.

        Produces, in order:
          should_be   what stock ought to be under the policy in force, split into
                      cycle / safety / buyer-owned pipeline so each part points at a
                      different lever
          levers      what each available action is worth, ranked on *net* annual
                      benefit — a lever that costs more in ordering than it frees in
                      stock ranks below one that does not
          frontier    when `target_value` is given, the ordered set of moves that
                      reaches it, what they cost, and what was deliberately excluded

        `results` is the dict returned by `run_planning`.
        """
        from .policy.assemble import build_sku_attributes
        from .policy.levers import LeverAnalyzer
        from .policy.parameters import PlanningParameters
        from .policy.should_be import ShouldBeCalculator
        from .policy.target import TargetPlanner

        print("\n[7/7] Policy analysis — should-be, levers, target...")

        params_file = parameters_file or (self.config_dir / "planning_parameters.md")
        planning_params = PlanningParameters(params_file)

        attributes = build_sku_attributes(
            classified_demand=results["classified_demand"],
            supplier_lt=results.get("supplier_lt"),
            inventory=inventory_df,
            forecast_summary=results.get("forecast_summary"),
            timeseries_meta=results.get("timeseries_meta"),
            params=planning_params,
        )
        resolved = planning_params.resolve(attributes)
        print(resolved.summary())

        calculator = ShouldBeCalculator(self.config_dir)
        should_be = calculator.calculate(resolved, actual=inventory_df)
        print()
        print(should_be.summary())

        levers = LeverAnalyzer(calculator).analyze(resolved, actual=inventory_df,
                                                   baseline=should_be)
        print()
        print(levers.summary())

        out = {
            "sku_attributes": attributes,
            "parameters": resolved,
            "should_be": should_be,
            "levers": levers,
            "frontier": None,
        }

        if target_value is not None:
            holding_rate = float(resolved.defaults.get("holding_cost_rate", 0.22))
            frontier = TargetPlanner(holding_rate).plan(
                should_be, target_value=target_value, deadline=deadline,
                levers=levers, open_po=open_po_df,
            )
            print()
            print(frontier.summary())
            out["frontier"] = frontier

        self._quality_log.append({
            "doc_type": "policy",
            "file": str(params_file.name),
            "rows_loaded": len(attributes),
            "issues": [str(h) for h in resolved.conflicts],
            "status": "WARNINGS" if resolved.conflicts else "OK",
        })
        return out

    # ------------------------------------------------------------------
    # Phase 8: KPI attribution — what happened, and what is coming
    # ------------------------------------------------------------------

    def run_kpi_review(
        self,
        policy: dict,
        sales_df=None,
        open_so_df=None,
        open_po_df=None,
        inventory_df=None,
        po_history_df=None,
        target_value: float = None,
        deadline=None,
        as_of=None,
        title: str = "Inventory & Service Review",
    ) -> dict:
        """
        Attribute the KPIs to the SKUs that moved them, then project forward.

        Chapter 1 answers "what happened and who caused it" — OTD misses, excess
        capital, and what the buying pattern cost. Chapter 2 answers "what is coming"
        — which SKUs run out, which will never move, and which inbound POs to change.

        The service split is the part worth reading carefully. A line past its request
        date with stock on the shelf is **not** an OTD failure — the goods were there.
        It is counted separately, because calling it a supply miss blames planning for
        a collection problem and simultaneously hides that the stock is sitting idle.
        """
        from datetime import date as _date

        from .policy.diagnostics import DiagnosticsAnalyzer
        from .policy.service import ServiceAnalyzer, latest_observed_date
        from .reporting.kpi_report import KPIReport

        print("\n[8/8] KPI attribution — service, ordering, forward risk...")

        # One anchor for the whole review, taken from the data. Mixing a data-derived
        # date in one section with today's date in another makes the two halves
        # disagree about what "past due" and "days to stockout" mean.
        if as_of is None:
            derived, stale = latest_observed_date(sales_df, open_so_df, po_history_df)
            as_of = derived or _date.today()
            if stale > 45:
                print(f"      Anchoring to the newest date in the data ({as_of}), "
                      f"{stale} days ago — not today. Scoring this extract against "
                      f"today would mark every open order past due.")

        attributes = policy["sku_attributes"]
        should_be = policy["should_be"]

        service = ServiceAnalyzer().analyze(
            sales_history=sales_df, open_so=open_so_df,
            inventory=inventory_df, as_of=as_of,
        )
        if len(service.lines):
            print()
            print(service.summary())

        diagnostics = DiagnosticsAnalyzer()
        ordering = diagnostics.ordering(po_history_df, attributes, as_of=as_of)
        print()
        print(ordering.summary())

        forward = diagnostics.forward(
            sku_attributes=attributes, inventory=inventory_df,
            open_po=open_po_df, open_so=open_so_df,
            safety_stock=should_be.frame.set_index("sku")["safety_qty"],
            as_of=as_of,
        )
        print()
        print(forward.summary())

        frontier = policy.get("frontier")
        if frontier is None and target_value is not None:
            from .policy.target import TargetPlanner
            holding = float(policy["parameters"].defaults.get("holding_cost_rate", 0.22))
            frontier = TargetPlanner(holding).plan(
                should_be, target_value=target_value, deadline=deadline,
                levers=policy.get("levers"), open_po=open_po_df, as_of=as_of,
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = self.output_dir / f"kpi_review_{stamp}.html"
        KPIReport(title).render(
            service=service, should_be=should_be, ordering=ordering,
            forward=forward, levers=policy.get("levers"), frontier=frontier,
            as_of=as_of, output_path=path,
        )
        print(f"\n  KPI review saved: {path}")
        print(f"  Open report: open {path}")

        return {
            "service": service,
            "ordering": ordering,
            "forward": forward,
            "frontier": frontier,
            "report_path": path,
        }

    # ------------------------------------------------------------------
    # Phase 1b: Contract-driven intake (preferred)
    # ------------------------------------------------------------------

    def load_all(
        self,
        paths: list,
        hints: dict = None,
        tenant: str = "default",
    ) -> dict:
        """
        Load every input file in one call, identifying each one automatically.

        Preferred over the individual `load_*` methods: the caller does not have to
        know which file is which, nor which loader a pre-aggregated demand series
        needs. Files are profiled, routed to a contract, transformed by an adapter and
        verified before any of them reach the analytics.

        Returns the keyword arguments for `run_planning`, so the whole pipeline is:

            planner = InventoryPlanner(output_dir=...)
            inputs  = planner.load_all(glob("exports/*"))
            results = planner.run_planning(**inputs)
        """
        from .ingest_bridge import IngestBridge

        bridge = IngestBridge(config_dir=self.config_dir, verbose=True)
        inputs = bridge.load(
            paths,
            hints=hints,
            tenant=tenant,
            baseline_path=self.output_dir.parent / "ingest_baselines.json",
        )

        plan = inputs["_intake_plan"]
        self._intake = inputs.pop("_intake", None)
        self._intake_plan = inputs.pop("_intake_plan", None)

        if not plan.can_run:
            missing = ", ".join(c.name for c in plan.missing_required)
            raise ValueError(
                f"Cannot run planning — missing required input(s): {missing}.\n"
                f"{plan.summary()}"
            )
        # Degradations are recorded rather than raised: the run is still meaningful,
        # but the report must be able to say which numbers rest on a fallback.
        self._quality_log.append({
            "doc_type": "intake",
            "file": "<multiple>",
            "rows_loaded": sum(len(f) for f in inputs.values() if isinstance(f, pd.DataFrame)),
            "issues": list(plan.degradations),
            "status": "OK" if not plan.degradations else "WARNINGS",
        })
        return inputs

    # ------------------------------------------------------------------
    # Phase 2: Analytics Pipeline (sequential)
    # ------------------------------------------------------------------

    def run_planning(
        self,
        sales_df: pd.DataFrame,
        po_history_df: pd.DataFrame,
        open_so_df: pd.DataFrame,
        open_po_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        timeseries_pivot: pd.DataFrame = None,   # pre-compiled TS overrides sales_df TS
        timeseries_meta: pd.DataFrame = None,
    ) -> dict:
        print("\n" + "="*60)
        print("  INVENTORY PLANNING PIPELINE")
        print("="*60)

        # Step 1: Demand time series + summary
        print("\n[1/6] Building demand time series...")
        if timeseries_pivot is not None:
            ts = timeseries_pivot
            # Build demand summary from the pre-compiled pivot
            demand_summary = self._summarize_from_pivot(ts, timeseries_meta)
            print(f"      Source: pre-compiled time series")
        else:
            ts = self.sales_reader.to_time_series(sales_df)
            demand_summary = self.sales_reader.summarize(sales_df)
        print(f"      {len(ts.columns)} SKUs × {len(ts)} periods ({ts.index[0]} → {ts.index[-1]})")

        # Step 2: Supplier lead times
        print("\n[2/6] Computing supplier lead times...")
        supplier_lt = self.po_reader.compute_supplier_lt(po_history_df)
        print(f"      {len(supplier_lt)} SKU×supplier LT records")

        # Step 3: Demand classification
        print("\n[3/6] Classifying demand (stocking policy + CV pattern)...")
        classified = self.classifier.classify(demand_summary, ts)
        counts = classified["stocking_class"].value_counts().to_dict()
        pattern_counts = classified["demand_pattern"].value_counts().to_dict() if "demand_pattern" in classified.columns else {}
        print(f"      Stocking classes: {counts}")
        if pattern_counts:
            print(f"      Demand patterns: {pattern_counts}")

        # Step 4: Forecast — must run before safety stock to supply forecast RMSE
        print("\n[4/6] Forecasting demand (6 months)...")
        forecast_detail = self.forecaster.forecast_all(ts, classified=classified)
        forecast_summary_df = self.forecaster.summary(forecast_detail)
        if not forecast_detail.empty:
            model_counts = forecast_detail.drop_duplicates("sku")["model_used"].value_counts().to_dict()
            print(f"      Models used: {model_counts}")

        # Step 5: Safety stock — uses forecast RMSE from step 4 as σDL
        print("\n[5/6] Calculating safety stock...")
        ss_df = self.ss_calc.calculate(classified, supplier_lt,
                                       forecast_summary=forecast_summary_df if not forecast_detail.empty else None)
        sigma_sources = ss_df["sigma_source"].value_counts().to_dict() if "sigma_source" in ss_df.columns else {}
        if sigma_sources:
            print(f"      σ source: {sigma_sources}")

        # Step 6: Effective inventory + projection + recommendations
        print("\n[6/6] Projecting inventory & generating recommendations...")
        if open_po_df is not None:
            open_po_df = self.open_po_reader.fill_estimated_delivery(open_po_df, supplier_lt)
        open_po_summary = self.open_po_reader.inbound_schedule(open_po_df) if open_po_df is not None else None
        open_so_summary = self.open_so_reader.backlog_summary(open_so_df) if open_so_df is not None else None
        eff_inv = self.inv_reader.effective_inventory(inventory_df, open_po_summary, supplier_lt)
        projection = self.projector.project(ss_df, eff_inv, open_po_summary)

        excess_count = (projection["inventory_status"] == "EXCESS").sum()
        shortage_count = (projection["inventory_status"] == "SHORTAGE-RISK").sum()
        print(f"      EXCESS: {excess_count} SKUs | SHORTAGE-RISK: {shortage_count} SKUs")

        # Purchase recommendations — uses forecast_next_period (t+1), not 6-month avg
        forecast_summary = forecast_summary_df
        recommendations = self.recommender.recommend(projection, forecast_summary, open_so_summary, open_po_summary)

        results = {
            "time_series": ts,
            "demand_summary": demand_summary,
            "supplier_lt": supplier_lt,
            "classified_demand": classified,
            "safety_stock": ss_df,
            "effective_inventory": eff_inv,
            "projection": projection,
            "forecast_detail": forecast_detail,
            "forecast_summary": forecast_summary,
            "recommendations": recommendations,
            "_quality_reports": self._quality_log,
        }

        self._save_outputs(results)
        self._print_summary(recommendations)
        return results

    def _save_outputs(self, results: dict) -> None:
        from .reporting.charts import ChartBuilder
        from .reporting.html_report import HTMLReportGenerator

        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        out = self.output_dir

        # ── CSV outputs ───────────────────────────────────────────────────────
        results["supplier_lt"].to_csv(out / "supplier_params.csv", index=False)
        results["classified_demand"].to_csv(out / "sku_planning_params.csv", index=False)
        results["projection"].to_csv(out / f"inventory_projection_{ts_str}.csv", index=False)
        results["forecast_detail"].to_csv(out / f"forecast_detail_{ts_str}.csv", index=False)
        results["recommendations"].to_csv(out / f"purchase_recommendations_{ts_str}.csv", index=False)

        # ── HTML Report with charts ───────────────────────────────────────────
        print("\n  Building charts & HTML report...")
        import json as _json
        node_cfg = self.config_dir / "node_config.json"
        location_id = _json.loads(node_cfg.read_text()).get("location_id", "DC-01") if node_cfg.exists() else "DC-01"
        try:
            cb = ChartBuilder()
            charts = cb.build_all(results)
            report_path = out / f"inventory_report_{ts_str}.html"
            HTMLReportGenerator().generate(
                results=results,
                charts=charts,
                quality_reports=results.get("_quality_reports", []),
                output_path=report_path,
                location_id=location_id,
            )
            print(f"  Open report: open {report_path}")
        except Exception as e:
            print(f"  Warning: HTML report failed ({e}) — CSV outputs still saved")

        # Save planning snapshot for next-month feedback comparison
        try:
            policy_cfg = json.loads((self.config_dir / "stocking_policy.json").read_text())
            snapshot_path = SnapshotSaver().save(results, policy_cfg, out)
            print(f"  Snapshot saved:   {snapshot_path.name}")
        except Exception as e:
            print(f"  Warning: snapshot save failed ({e})")

        print(f"\n  Outputs saved to: {out}")

    def _print_summary(self, recommendations: pd.DataFrame) -> None:
        print("\n" + "="*60)
        print("  PURCHASE RECOMMENDATIONS SUMMARY")
        print("="*60)
        action_counts = recommendations["recommended_action"].value_counts()
        for action, count in action_counts.items():
            print(f"  {action:<30} {count:>5} SKUs")

        purchase_skus = recommendations[recommendations["recommended_action"] == "PURCHASE-REQUEST"]
        if len(purchase_skus):
            total_qty = purchase_skus["suggested_po_qty"].sum()
            print(f"\n  Total suggested purchase qty : {total_qty:,.0f}")

        pushout_skus = recommendations[recommendations["recommended_action"] == "PUSH-OUT-OPEN-PO"]
        if len(pushout_skus):
            pushout_qty = pushout_skus["pushout_open_po_qty"].sum()
            print(f"  Total push-out candidate qty : {pushout_qty:,.0f}")

    def _summarize_from_pivot(self, ts: pd.DataFrame, meta: pd.DataFrame = None) -> pd.DataFrame:
        """Build a demand_summary DataFrame from a pre-compiled pivot."""
        import numpy as np
        location_id = self.inv_reader.location_id
        rows = []
        for sku in ts.columns:
            series = ts[sku]
            active = int((series > 0).sum())
            total = int(len(series))
            mean_d = float(series[series > 0].mean()) if active > 0 else 0.0
            std_d = float(series.std())
            rows.append({
                "sku": sku,
                "location_id": location_id,
                "demand_mean": round(mean_d, 2),
                "demand_std": round(std_d, 2),
                "demand_cv": round(std_d / mean_d, 3) if mean_d > 0 else np.nan,
                "active_cycles": active,
                "total_cycles": total,
                "total_qty": float(series.sum()),
                "total_amount": np.nan,
                "first_sale": ts.index[series > 0][0].to_timestamp() if active > 0 else pd.NaT,
                "last_sale":  ts.index[series > 0][-1].to_timestamp() if active > 0 else pd.NaT,
                # Enrich with metadata if available
                "description": meta.loc[sku, "description"] if (meta is not None and sku in meta.index and "description" in meta.columns) else "",
                "sopc_class":  meta.loc[sku, "sopc_classification"] if (meta is not None and sku in meta.index and "sopc_classification" in meta.columns) else "",
            })
        return pd.DataFrame(rows)

    def _check_quality(self, report: dict) -> None:
        self._quality_log.append(report)
        if report.get("issues"):
            if self.interactive:
                print(f"\n  ⚠  Quality issues found in {report['doc_type']}. Continue? [Y/n]: ", end="")
                if input().strip().lower() in ("n", "no"):
                    raise ValueError(f"User aborted after quality check on {report['doc_type']}")
