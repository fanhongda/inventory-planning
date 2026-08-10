"""
Safety stock calculator.

Formula (MIT CTL §7 — combined demand × lead time variability):
  SS = Z * sqrt(LT_avg * σ_fc² + demand_avg² * σ_LT²)

Where:
  σ_fc = forecast RMSE (preferred) or demand std dev (fallback if RMSE unavailable)
  demand_avg = unconditional mean demand per month (including zero periods)

Service level note (see config/stocking_policy.json → service_level_metric):
  CSL: k = NORMSINV(CSL) — z_score from config, probability of no stockout per cycle
  IFR: requires solving G(k) = Q*(1-IFR)/σDL — not yet implemented, defaults to CSL

Supplier LT selection:
  Uses the ordering supplier (from config/supplier_incoterm.json) when available.
  Falls back to shortest WMA LT if no preferred supplier defined.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


class SafetyStockCalculator:

    def __init__(self, config_dir: Path = None):
        self._preferred_supplier: dict = {}
        if config_dir is not None:
            si_path = Path(config_dir) / "supplier_incoterm.json"
            if si_path.exists():
                data = json.loads(si_path.read_text(encoding="utf-8"))
                # supplier_incoterm.json maps supplier_id → incoterm; no preferred supplier list yet.
                # When a sku-level preferred supplier config is added, load it here.

    def calculate(self, classified_demand: pd.DataFrame, supplier_lt: pd.DataFrame,
                  forecast_summary: pd.DataFrame = None) -> pd.DataFrame:
        """
        classified_demand: output of DemandClassifier.classify()
        supplier_lt:       output of POHistoryReader.compute_supplier_lt()
        forecast_summary:  output of Forecaster.summary() — provides forecast_rmse per SKU.
                           When provided, forecast_rmse is used as σ_fc.
                           When absent, demand_std_rolling is used (overestimates SS).

        Returns per-SKU: safety_stock, reorder_point, ss_coverage_days, sigma_source
        """
        # ── Supplier LT selection ─────────────────────────────────────────────
        # Preferred: use the supplier with the most PO history (by order count) as proxy
        # for the "ordering supplier". Shortest LT can be an unreliable outlier supplier.
        # Fallback: shortest WMA LT (original behaviour).
        if "order_count" in supplier_lt.columns:
            ordering_lt = (
                supplier_lt.sort_values("order_count", ascending=False)
                .groupby("sku")
                .first()
                .reset_index()[["sku", "wma_lead_time_days", "lt_std_days"]]
            )
        else:
            ordering_lt = (
                supplier_lt.sort_values("wma_lead_time_days")
                .groupby("sku")
                .first()
                .reset_index()[["sku", "wma_lead_time_days", "lt_std_days"]]
            )

        df = classified_demand.merge(ordering_lt, on="sku", how="left")

        no_lt = df["wma_lead_time_days"].isna().sum()
        if no_lt:
            print(f"  Warning: {no_lt} SKUs have no supplier LT data — safety stock set to 0, review manually")
        df["wma_lead_time_days"] = df["wma_lead_time_days"].fillna(0)
        df["lt_std_days"] = df["lt_std_days"].fillna(0)

        # ── Merge forecast RMSE if available ─────────────────────────────────
        if forecast_summary is not None and "forecast_rmse" in forecast_summary.columns:
            rmse_map = forecast_summary.set_index("sku")["forecast_rmse"]
            df["forecast_rmse"] = df["sku"].map(rmse_map).fillna(np.nan)
            df["sigma_source"] = df["forecast_rmse"].apply(
                lambda v: "forecast_rmse" if pd.notna(v) and v > 0 else "demand_std"
            )
        else:
            df["forecast_rmse"] = np.nan
            df["sigma_source"] = "demand_std"

        # ── Safety stock calculation ──────────────────────────────────────────
        df["lt_months"] = df["wma_lead_time_days"] / 30.0
        df["lt_std_months"] = df["lt_std_days"] / 30.0

        def _ss(row):
            if row["stocking_class"] == "non-stocking" or pd.isna(row["z_score"]) or row["z_score"] is None:
                return 0.0
            z = float(row["z_score"])
            lt = float(row["lt_months"])
            lt_std = float(row["lt_std_months"])
            # demand_mean_rolling is the unconditional monthly mean (including zero months)
            d_mean = float(row["demand_mean_rolling"])
            # Use forecast RMSE when available; fall back to demand std dev
            if pd.notna(row["forecast_rmse"]) and row["forecast_rmse"] > 0:
                sigma_fc = float(row["forecast_rmse"])
            else:
                sigma_fc = float(row["demand_std_rolling"])
            # Combined variability formula (MIT CTL §7)
            variance = lt * (sigma_fc ** 2) + (d_mean ** 2) * (lt_std ** 2)
            return round(z * np.sqrt(max(variance, 0)), 1)

        df["safety_stock"] = df.apply(_ss, axis=1)

        # ROP uses unconditional mean (demand_mean_rolling already set to unconditional in classifier)
        df["demand_during_lt"] = (df["demand_mean_rolling"] * df["lt_months"]).round(1)
        df["reorder_point"] = (df["demand_during_lt"] + df["safety_stock"]).round(1)

        daily_demand = df["demand_mean_rolling"] / 30.0
        df["ss_coverage_days"] = (df["safety_stock"] / daily_demand.replace(0, np.nan)).round(0)

        return df
