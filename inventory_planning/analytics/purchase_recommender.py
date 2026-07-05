"""
Purchase recommender.

Net requirement for next planning cycle (1 month):

  gross_requirement = forecast_next_period + safety_stock + backlog_qty
  net_requirement   = max(0, gross_requirement − effective_position)

Key fix (C2 from AUDIT.md):
  Uses forecast_next_period (t+1 point forecast) — not forecast_avg_monthly.
  forecast_avg_monthly is the 6-month average and underestimates in uptrending demand.
  Fallback: demand_mean_rolling (unconditional monthly mean) when forecast unavailable.
"""

import pandas as pd
import numpy as np


class PurchaseRecommender:

    def recommend(self, projection: pd.DataFrame, forecast_summary: pd.DataFrame,
                  open_so: pd.DataFrame, open_po_df: pd.DataFrame) -> pd.DataFrame:
        df = projection.copy()

        # ── Next-period demand: use t+1 point forecast, not 6-month average ──
        if forecast_summary is not None and len(forecast_summary):
            if "forecast_next_period" in forecast_summary.columns:
                next_cycle = forecast_summary[["sku", "forecast_next_period",
                                               "forecast_avg_monthly"]].copy()
                df = df.merge(next_cycle, on="sku", how="left")
                # forecast_next_period is the authoritative demand estimate for net req
                df["next_period_demand"] = df["forecast_next_period"].fillna(
                    df.get("forecast_avg_monthly", df["demand_mean_rolling"])
                )
            else:
                # Legacy path: older forecast_summary without next_period column
                df = df.merge(
                    forecast_summary[["sku", "forecast_avg_monthly"]], on="sku", how="left"
                )
                df["next_period_demand"] = df["forecast_avg_monthly"]
                df["forecast_next_period"] = df["forecast_avg_monthly"]
        else:
            df["next_period_demand"] = df["demand_mean_rolling"]
            df["forecast_next_period"] = df["demand_mean_rolling"]
            df["forecast_avg_monthly"] = df["demand_mean_rolling"]

        df["next_period_demand"] = df["next_period_demand"].fillna(df["demand_mean_rolling"])

        # ── Backlog ───────────────────────────────────────────────────────────
        if open_so is not None and "backlog_qty" in open_so.columns:
            df = df.merge(open_so[["sku", "backlog_qty"]], on="sku", how="left")
        else:
            df["backlog_qty"] = 0
        df["backlog_qty"] = df["backlog_qty"].fillna(0)

        # ── Net requirement ───────────────────────────────────────────────────
        df["gross_requirement"] = df["next_period_demand"] + df["safety_stock"] + df["backlog_qty"]
        df["available_supply"] = df["effective_position"]
        df["net_requirement"] = (df["gross_requirement"] - df["available_supply"]).clip(lower=0).round(1)

        def _action(row):
            if row["stocking_class"] == "non-stocking":
                return "ORDER-FOR-BACKLOG" if row["backlog_qty"] > 0 else "NO-ACTION"
            if row["pushout_candidate"]:
                return "PUSH-OUT-OPEN-PO"
            if row["net_requirement"] > 0:
                return "PURCHASE-REQUEST"
            if row["inventory_status"] == "EXCESS":
                return "HOLD-EXCESS"
            return "HOLD-OK"

        df["recommended_action"] = df.apply(_action, axis=1)
        df["suggested_po_qty"] = df.apply(
            lambda r: r["net_requirement"] if r["recommended_action"] == "PURCHASE-REQUEST" else 0,
            axis=1
        )

        cols = [
            "sku", "location_id", "stocking_class", "recommended_action",
            "forecast_next_period", "forecast_avg_monthly", "backlog_qty",
            "gross_requirement", "available_supply", "net_requirement",
            "suggested_po_qty", "pushout_open_po_qty",
            "inventory_status", "days_of_supply", "safety_stock", "should_be_inventory",
        ]
        return df[[c for c in cols if c in df.columns]].sort_values("recommended_action")
