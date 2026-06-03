"""
Sales history reader.
Outputs a monthly (or daily) time series of demand per SKU.
"""

import pandas as pd
import numpy as np
from .base_reader import BaseReader


class SalesHistoryReader(BaseReader):
    doc_type = "sales_history"

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with missing SKU or qty
        df = df.dropna(subset=["sku", "qty"])

        # Demand date priority: ship_date first, fallback to order_date
        # This ensures full history coverage even when ship_date is only recent
        has_ship = "ship_date" in df.columns
        has_order = "order_date" in df.columns

        if has_ship and has_order:
            df = df.copy()
            df["demand_date"] = df["ship_date"].combine_first(df["order_date"])
            n_ship = df["ship_date"].notna().sum()
            n_order_fallback = df["ship_date"].isna().sum()
            if n_order_fallback:
                print(f"  Demand date: {n_ship} rows use ship_date, {n_order_fallback} fall back to order_date")
        elif has_ship:
            df = df.copy()
            df["demand_date"] = df["ship_date"]
        elif has_order:
            df = df.copy()
            df["demand_date"] = df["order_date"]
        else:
            raise ValueError("Sales history: no date column found (ship_date or order_date required)")

        df = df.dropna(subset=["demand_date"])
        # Keep only positive qty (returns/credits may be negative — flag separately)
        returns = df[df["qty"] < 0].copy()
        df = df[df["qty"] > 0].copy()
        if len(returns):
            print(f"  Note: {len(returns)} return/credit rows (negative qty) excluded from demand series")
        return df

    def to_time_series(self, df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
        """
        Aggregate demand into a time series per SKU.
        freq: 'MS' = month-start (default), 'W' = weekly, 'D' = daily
        Returns a pivot table: index=period, columns=SKU, values=demand_qty.
        """
        df = df.copy()
        df["period"] = df["demand_date"].dt.to_period(
            "M" if freq == "MS" else freq.replace("S", "")
        )
        ts = (
            df.groupby(["period", "sku"])["qty"]
            .sum()
            .reset_index()
            .pivot(index="period", columns="sku", values="qty")
            .fillna(0)
            .sort_index()
        )
        return ts

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-SKU demand summary: mean, std, cv, active cycles, date range.
        """
        ts = self.to_time_series(df)
        stats = []
        for sku in ts.columns:
            series = ts[sku]
            active = (series > 0).sum()
            total_cycles = len(series)
            stats.append({
                "sku": sku,
                "location_id": df["location_id"].iloc[0] if "location_id" in df.columns else "DC-01",
                "demand_mean": series[series > 0].mean() if active else 0,
                "demand_std": series.std(),
                "demand_cv": series.std() / series.mean() if series.mean() > 0 else np.nan,
                "active_cycles": int(active),
                "total_cycles": int(total_cycles),
                "first_sale": df[df["sku"] == sku]["demand_date"].min(),
                "last_sale": df[df["sku"] == sku]["demand_date"].max(),
                "total_qty": series.sum(),
                "total_amount": df[df["sku"] == sku]["amount"].sum() if "amount" in df.columns else np.nan,
            })
        return pd.DataFrame(stats)
