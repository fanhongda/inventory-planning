"""
Inventory projector.
Compares should-be inventory vs current effective position.
Flags excess and suggests open PO push-out candidates.
"""

import pandas as pd
import numpy as np


class InventoryProjector:

    def project(self, safety_stock_df: pd.DataFrame, effective_inventory: pd.DataFrame,
                open_po_df: pd.DataFrame) -> pd.DataFrame:
        """
        should_be = demand_during_lt + safety_stock
        surplus   = effective_position - should_be
        If surplus > 0 and open POs exist: flag for push-out review.
        """
        df = safety_stock_df.merge(
            effective_inventory[["sku", "qty_on_hand", "qty_in_transit_adj",
                                  "total_open_po_qty", "effective_position"]],
            on="sku", how="left"
        )

        df["should_be_inventory"] = df["demand_during_lt"] + df["safety_stock"]
        df["surplus_deficit"] = df["effective_position"] - df["should_be_inventory"]

        def _status(row):
            if row["stocking_class"] == "non-stocking":
                return "non-stocking"
            s = row["surplus_deficit"]
            if pd.isna(s):
                return "no-data"
            if s > row["should_be_inventory"] * 0.5:  # >50% over target
                return "EXCESS"
            if s > 0:
                return "slightly-over"
            if s < -row["safety_stock"] * 0.5:
                return "SHORTAGE-RISK"
            return "OK"

        df["inventory_status"] = df.apply(_status, axis=1)

        # Push-out candidates: EXCESS status AND has open POs
        if open_po_df is not None and len(open_po_df):
            open_po_skus = set(open_po_df[open_po_df["total_open_po_qty"] > 0]["sku"])
            df["pushout_candidate"] = (
                (df["inventory_status"] == "EXCESS") &
                df["sku"].isin(open_po_skus)
            )
            df["pushout_open_po_qty"] = df.apply(
                lambda r: r["total_open_po_qty"] if r["pushout_candidate"] else 0, axis=1
            )
        else:
            df["pushout_candidate"] = False
            df["pushout_open_po_qty"] = 0

        cols = [
            "sku", "location_id", "stocking_class", "service_level",
            "demand_mean_rolling", "wma_lead_time_days",
            "safety_stock", "demand_during_lt", "should_be_inventory",
            "qty_on_hand", "qty_in_transit_adj", "total_open_po_qty", "effective_position",
            "surplus_deficit", "inventory_status", "pushout_candidate", "pushout_open_po_qty",
        ]
        return df[[c for c in cols if c in df.columns]]
