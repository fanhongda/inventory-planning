"""
Inventory projector.

Compares should-be inventory (ROP) vs current effective inventory position.

EXCESS definition (MIT CTL §9 — Days of Supply):
  DOS = effective_position / daily_demand
  EXCESS triggered when DOS > excess_dos_threshold_days (configurable in stocking_policy.json)
  This replaces the prior arbitrary "50% over ROP" heuristic.

IP definition (MIT CTL §10):
  IP = IOH + IOO − BO − CO
  effective_position from InventoryReader = IOH + GIT_adj + open_PO
  (Backorders handled separately via open_so backlog_qty in purchase_recommender)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


class InventoryProjector:

    def __init__(self, config_dir: Path = None):
        self.excess_dos_threshold = 90  # default: flag as EXCESS if > 90 days of supply
        if config_dir is not None:
            cfg_path = Path(config_dir) / "stocking_policy.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.excess_dos_threshold = cfg.get("excess_dos_threshold_days", 90)

    def project(self, safety_stock_df: pd.DataFrame, effective_inventory: pd.DataFrame,
                open_po_df: pd.DataFrame) -> pd.DataFrame:
        """
        should_be = demand_during_lt + safety_stock  (= ROP)
        surplus   = effective_position - should_be
        EXCESS    = DOS > excess_dos_threshold_days
        """
        # A left merge against a frame with repeated SKUs silently multiplies rows, and
        # the duplicates then carry the same open PO and the same backlog — which is how
        # one SKU ends up recommended for pull-in and push-out at the same time. The
        # consolidation belongs upstream; failing loudly here keeps it from being skipped.
        dupes = effective_inventory["sku"][effective_inventory["sku"].duplicated()].unique()
        if len(dupes):
            raise ValueError(
                f"effective_inventory has {len(dupes)} SKU(s) on more than one row "
                f"(e.g. {', '.join(map(str, dupes[:3]))}). Planning is single-node: run "
                f"readers.inventory_reader.consolidate_to_planning_grain first, or every "
                f"quantity joined on sku will be counted once per location."
            )

        df = safety_stock_df.merge(
            effective_inventory[["sku", "qty_on_hand", "qty_in_transit_adj",
                                  "total_open_po_qty", "effective_position"]],
            on="sku", how="left"
        )

        df["should_be_inventory"] = df["demand_during_lt"] + df["safety_stock"]
        df["surplus_deficit"] = df["effective_position"] - df["should_be_inventory"]

        # Days of Supply — based on unconditional mean demand
        daily_demand = df["demand_mean_rolling"] / 30.0
        df["days_of_supply"] = (
            df["effective_position"] / daily_demand.replace(0, np.nan)
        ).round(0)

        def _status(row):
            if row["stocking_class"] == "non-stocking":
                return "non-stocking"
            if pd.isna(row["effective_position"]):
                return "no-data"
            dos = row["days_of_supply"]
            surplus = row["surplus_deficit"]
            # EXCESS: DOS exceeds policy threshold (not arbitrary % over ROP)
            if pd.notna(dos) and dos > self.excess_dos_threshold:
                return "EXCESS"
            if surplus > 0:
                return "slightly-over"
            if surplus < -row["safety_stock"] * 0.5:
                return "SHORTAGE-RISK"
            return "OK"

        df["inventory_status"] = df.apply(_status, axis=1)

        # Push-out candidates: EXCESS AND open POs in flight
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
            "surplus_deficit", "days_of_supply", "inventory_status",
            "pushout_candidate", "pushout_open_po_qty",
        ]
        return df[[c for c in cols if c in df.columns]]
