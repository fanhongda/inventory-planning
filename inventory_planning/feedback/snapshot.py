"""
Snapshot: saves a planning run's key outputs alongside the parameters used.
Called automatically at the end of each run_planning() call.

Snapshot is stored as:
  output/history/YYYY-MM/snapshot_YYYYMMDD_HHMM.json

The JSON contains:
  - run metadata (date, model versions, config params)
  - per-SKU: forecast_next_period, suggested_po_qty, safety_stock, days_of_supply, demand_pattern
  - path to the full CSV outputs for that run

Next month, the user fills in actuals by calling FeedbackCollector.record_actuals().
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd


class SnapshotSaver:

    def save(self, results: dict, config: dict, output_dir: Path) -> Path:
        """
        results:    output of InventoryPlanner.run_planning()
        config:     stocking_policy.json contents
        output_dir: the run's output directory (where CSVs were saved)

        Returns path to the saved snapshot JSON.
        """
        run_dt = datetime.now()
        month_label = run_dt.strftime("%Y-%m")
        ts_str = run_dt.strftime("%Y%m%d_%H%M")

        history_dir = output_dir.parent / "history" / month_label
        history_dir.mkdir(parents=True, exist_ok=True)

        rec = results.get("recommendations", pd.DataFrame())
        proj = results.get("projection", pd.DataFrame())
        classified = results.get("classified_demand", pd.DataFrame())

        skus = {}
        for _, row in rec.iterrows():
            sku = row["sku"]
            skus[sku] = {
                "recommended_action": row.get("recommended_action"),
                "suggested_po_qty": float(row.get("suggested_po_qty", 0)),
                "forecast_next_period": float(row.get("forecast_next_period", 0)),
                "forecast_avg_monthly": float(row.get("forecast_avg_monthly", 0)),
                "net_requirement": float(row.get("net_requirement", 0)),
                "available_supply": float(row.get("available_supply", 0)),
                "safety_stock": float(row.get("safety_stock", 0)),
            }

        # Enrich with projection fields
        for _, row in proj.iterrows():
            sku = row["sku"]
            if sku in skus:
                skus[sku].update({
                    "days_of_supply": float(row.get("days_of_supply", 0)) if pd.notna(row.get("days_of_supply")) else None,
                    "inventory_status": row.get("inventory_status"),
                })

        # Enrich with demand pattern
        for _, row in classified.iterrows():
            sku = row["sku"]
            if sku in skus:
                skus[sku].update({
                    "demand_pattern": row.get("demand_pattern"),
                    "demand_cv": float(row.get("demand_cv", 0)) if pd.notna(row.get("demand_cv")) else None,
                    "stocking_class": row.get("stocking_class"),
                })

        snapshot = {
            "run_at": run_dt.isoformat(),
            "planning_month": month_label,
            "config_snapshot": {
                "cv_intermittent_threshold": config.get("cv_intermittent_threshold"),
                "cv_erratic_threshold": config.get("cv_erratic_threshold"),
                "excess_dos_threshold_days": config.get("excess_dos_threshold_days"),
                "service_level_metric": config.get("service_level_metric"),
                "stocking_tiers": config.get("stocking_tiers"),
            },
            "output_dir": str(output_dir),
            "sku_count": len(skus),
            "skus": skus,
            # Actuals filled in next month by FeedbackCollector
            "actuals": {},
            "loss": {},
        }

        path = history_dir / f"snapshot_{ts_str}.json"
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return path
