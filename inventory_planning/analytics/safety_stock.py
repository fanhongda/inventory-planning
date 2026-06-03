"""
Safety stock calculator.
Uses the combined demand and lead time variability formula:
  SS = Z * sqrt(LT_avg * σ_demand² + demand_avg² * σ_LT²)

For non-stocking items: SS = 0 (order on demand).
"""

import pandas as pd
import numpy as np


class SafetyStockCalculator:

    def calculate(self, classified_demand: pd.DataFrame, supplier_lt: pd.DataFrame) -> pd.DataFrame:
        """
        classified_demand: output of DemandClassifier.classify()
        supplier_lt: output of POHistoryReader.compute_supplier_lt()

        Returns per-SKU: safety_stock, reorder_point, coverage_days
        """
        # Use best (shortest WMA LT) supplier per SKU
        best_lt = (
            supplier_lt.sort_values("wma_lead_time_days")
            .groupby("sku")
            .first()
            .reset_index()[["sku", "wma_lead_time_days", "lt_std_days"]]
        )

        df = classified_demand.merge(best_lt, on="sku", how="left")

        # For SKUs with no LT data, flag and use 0 (will be highlighted in output)
        no_lt = df["wma_lead_time_days"].isna().sum()
        if no_lt:
            print(f"  Warning: {no_lt} SKUs have no supplier LT data — safety stock set to 0, review manually")
        df["wma_lead_time_days"] = df["wma_lead_time_days"].fillna(0)
        df["lt_std_days"] = df["lt_std_days"].fillna(0)

        # Safety stock formula (demand in units/month → convert LT to months)
        df["lt_months"] = df["wma_lead_time_days"] / 30.0
        df["lt_std_months"] = df["lt_std_days"] / 30.0

        def _ss(row):
            if row["stocking_class"] == "non-stocking" or row["z_score"] is None:
                return 0.0
            z = row["z_score"]
            lt = row["lt_months"]
            lt_std = row["lt_std_months"]
            d_mean = row["demand_mean_rolling"]
            d_std = row["demand_std_rolling"]
            # Combined variability formula
            variance = lt * (d_std ** 2) + (d_mean ** 2) * (lt_std ** 2)
            return round(z * np.sqrt(variance), 1)

        df["safety_stock"] = df.apply(_ss, axis=1)

        # Reorder point = demand during lead time + safety stock
        df["demand_during_lt"] = (df["demand_mean_rolling"] * df["lt_months"]).round(1)
        df["reorder_point"] = (df["demand_during_lt"] + df["safety_stock"]).round(1)

        # Coverage: how many days of demand does the SS cover
        daily_demand = df["demand_mean_rolling"] / 30.0
        df["ss_coverage_days"] = (df["safety_stock"] / daily_demand.replace(0, np.nan)).round(0)

        return df
