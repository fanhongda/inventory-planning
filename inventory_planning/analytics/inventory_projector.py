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

    def __init__(self, config_dir: Path = None, horizon_days: int = 30):
        self.horizon_days = int(horizon_days)
        self.excess_dos_threshold = 90  # default: flag as EXCESS if > 90 days of supply
        if config_dir is not None:
            cfg_path = Path(config_dir) / "stocking_policy.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.excess_dos_threshold = cfg.get("excess_dos_threshold_days", 90)

    # Columns the phasing contributes, and what they mean when the open-PO extract
    # cannot supply them. A SKU with no open PO has nothing inbound and no wait for it;
    # `days_to_next_arrival` stays NaN, which reads as "no arrival to wait for" and
    # makes `runs_out_first` false rather than true.
    _PHASE_DEFAULTS = {
        "inbound_past_due_qty": 0.0,
        "inbound_due_qty": 0.0,
        "inbound_beyond_qty": 0.0,
        "days_to_next_arrival": np.nan,
    }

    @classmethod
    def _attach_phasing(cls, df: pd.DataFrame, open_po_df: pd.DataFrame) -> pd.DataFrame:
        """
        Bring the time-phased inbound quantities alongside the position.

        Defaulted rather than required, so an open-PO extract with no delivery dates —
        or no open-PO extract at all — degrades to "nothing is late, nothing is due"
        instead of raising. The run says elsewhere that it could not see arrival dates.
        """
        cols = [c for c in cls._PHASE_DEFAULTS if
                open_po_df is not None and c in getattr(open_po_df, "columns", [])]
        if cols:
            df = df.merge(open_po_df[["sku", *cols]], on="sku", how="left")
        for name, default in cls._PHASE_DEFAULTS.items():
            if name not in df.columns:
                df[name] = default
            elif not np.isnan(default):
                df[name] = df[name].fillna(default)
        return df

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
        df = self._attach_phasing(df, open_po_df)

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
            # Shortage is judged first, and excess is gated on there being a surplus at
            # all. The two measures answer different questions and used to be allowed to
            # contradict each other: `should_be` is the reorder point, so it carries the
            # safety stock and the lead time, while DOS divides the position by average
            # demand and knows about neither. On a lumpy SKU that gap is the whole point
            # — the variability that sizes the safety stock is exactly what holds the
            # average down, so DOS reads *high* on the items whose cover is most needed.
            #
            # Trusting DOS alone put 19 SKUs on this extract in EXCESS while they sat
            # below their reorder point, and sent 14 of them to push-out. One had 9,000
            # units inbound against a 9,180 reorder point and nothing at all on the
            # shelf, and the run advised delaying the delivery.
            if surplus < -row["safety_stock"] * 0.5:
                return "SHORTAGE-RISK"
            if surplus > 0 and pd.notna(dos) and dos > self.excess_dos_threshold:
                return "EXCESS"
            if surplus > 0:
                return "slightly-over"
            return "OK"

        df["inventory_status"] = df.apply(_status, axis=1)

        # ── Will the shelf last until the next delivery? ──────────────────────
        #
        # Everything above compares one total against another and never asks *when*
        # the supply lands. That is how a SKU with 37 units on the shelf against
        # 14,398 a month came back `excess, push out 79,500`: the position was over
        # the reorder point on paper, and every unit of it was still at the supplier.
        #
        # Two ways to run out, and both are the same emergency to a planner — the
        # shelf empties before anything arrives, or the supply that should already
        # have arrived has not and there is nothing left to sell in the meantime.
        # Neither is visible in the position, and the second is the more dangerous
        # because a late order still counts as inbound supply for as long as it is
        # late. Below, `days_to_next_arrival` is the wait for the next order still
        # *inside* its committed date; anything past due has no date left to use.
        daily = (df["demand_mean_rolling"] / 30.0).replace(0, np.nan)
        df["on_hand_cover_days"] = (
            (df["qty_on_hand"].fillna(0) + df["qty_in_transit_adj"].fillna(0)) / daily
        ).round(1)

        runs_out_first = df["on_hand_cover_days"] < df["days_to_next_arrival"]
        late_and_empty = (
            (df["inbound_past_due_qty"] > 0)
            & (df["on_hand_cover_days"] < self.horizon_days)
        )
        df["supply_gap"] = (
            (runs_out_first | late_and_empty).fillna(False)
            & (df["stocking_class"] != "non-stocking")
            & df["effective_position"].notna()
        )
        # How long the shelf is empty before relief arrives — the difference between
        # "runs out four hours early" and "runs out for six weeks", which the boolean
        # cannot express and a planner has to triage on. Where nothing is still inside
        # its committed date the wait is unknown, so the horizon stands in for it: that
        # is the period the run is answering for.
        wait = df["days_to_next_arrival"].fillna(self.horizon_days)
        df["supply_gap_days"] = np.where(
            df["supply_gap"], (wait - df["on_hand_cover_days"]).clip(lower=0).round(1), 0.0
        )

        # Push-out candidates: EXCESS AND open POs in flight
        if open_po_df is not None and len(open_po_df):
            open_po_skus = set(open_po_df[open_po_df["total_open_po_qty"] > 0]["sku"])
            # Nothing is deferred out of a shelf that will not last until the next
            # delivery. The position being over the reorder point does not survive
            # contact with the timing: the surplus is arriving, and the shortage is now.
            df["pushout_candidate"] = (
                (df["inventory_status"] == "EXCESS")
                & df["sku"].isin(open_po_skus)
                & ~df["supply_gap"]
            )
            # Only the part that is genuinely surplus, never the whole order. Pushing
            # out everything inbound is not a smaller version of the right advice, it
            # is the opposite one: a renumbered part here held 37 units against 14,398
            # a month with 79,500 on the way, and the run said to delay all 79,500 —
            # two hours of cover, and a stockout by the end of the day.
            #
            # `surplus_deficit` is exactly how far the position sits above the reorder
            # point, so deferring that much lands the position *on* the reorder point,
            # which is where it is supposed to be. Capped at what is actually inbound,
            # because stock already on the shelf cannot be pushed out.
            df["pushout_open_po_qty"] = np.where(
                df["pushout_candidate"],
                np.minimum(df["surplus_deficit"], df["total_open_po_qty"])
                  .clip(lower=0).round(1),
                0.0,
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
            "on_hand_cover_days", "inbound_past_due_qty", "inbound_due_qty",
            "inbound_beyond_qty", "days_to_next_arrival", "supply_gap", "supply_gap_days",
            "pushout_candidate", "pushout_open_po_qty",
        ]
        return df[[c for c in cols if c in df.columns]]
