"""
Purchase recommender.
Computes net requirement for next planning cycle and generates buy/hold/push-out actions.
"""

import pandas as pd
import numpy as np


class PurchaseRecommender:

    def recommend(self, projection: pd.DataFrame, forecast_summary: pd.DataFrame,
                  open_so: pd.DataFrame, open_po_df: pd.DataFrame) -> pd.DataFrame:
        """
        Net requirement = forecast_next_cycle + safety_stock + backlog
                          - (qty_on_hand + GIT_adj + open_po_inbound)
        """
        df = projection.copy()

        # Join 1-month forecast
        if forecast_summary is not None and len(forecast_summary):
            # Use avg monthly forecast as next cycle demand
            next_cycle = forecast_summary[["sku", "forecast_avg_monthly"]].copy()
            df = df.merge(next_cycle, on="sku", how="left")
        else:
            df["forecast_avg_monthly"] = df["demand_mean_rolling"]

        df["forecast_avg_monthly"] = df["forecast_avg_monthly"].fillna(df["demand_mean_rolling"])

        # Join backlog
        if open_so is not None and "backlog_qty" in open_so.columns:
            df = df.merge(open_so[["sku", "backlog_qty"]], on="sku", how="left")
        else:
            df["backlog_qty"] = 0
        df["backlog_qty"] = df["backlog_qty"].fillna(0)

        # Net requirement
        df["gross_requirement"] = df["forecast_avg_monthly"] + df["safety_stock"] + df["backlog_qty"]
        df["available_supply"] = df["effective_position"]
        df["net_requirement"] = (df["gross_requirement"] - df["available_supply"]).clip(lower=0).round(1)

        def _action(row):
            if row["stocking_class"] == "non-stocking":
                if row["backlog_qty"] > 0:
                    return "ORDER-FOR-BACKLOG"
                return "NO-ACTION"
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
            "forecast_avg_monthly", "backlog_qty",
            "gross_requirement", "available_supply", "net_requirement",
            "suggested_po_qty", "pushout_open_po_qty",
            "inventory_status", "safety_stock", "should_be_inventory",
        ]
        return df[[c for c in cols if c in df.columns]].sort_values(
            "recommended_action"
        )
