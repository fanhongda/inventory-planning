"""
Open purchase order reader — future inbound supply.
Handles:
- Closed Status filtering (only open POs)
- Computing open_qty = order_qty - delivered_qty when direct open_qty absent
- Missing committed_delivery: estimated from order_date + WMA LT (set at run time)
"""

import pandas as pd
import numpy as np
from .base_reader import BaseReader


class OpenPOReader(BaseReader):
    doc_type = "open_po"

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter out closed POs
        if "closed_status" in df.columns:
            before = len(df)
            df = df[df["closed_status"].str.strip().str.lower() != "yes"].copy()
            filtered = before - len(df)
            if filtered:
                print(f"  Filtered {filtered} closed PO rows (Closed Status = Yes); {len(df)} open rows remain")

        # Compute open_qty if not directly available
        if "open_qty" not in df.columns or df["open_qty"].isna().all():
            if "order_qty" in df.columns and "delivered_qty" in df.columns:
                df["order_qty"] = pd.to_numeric(df["order_qty"], errors="coerce").fillna(0)
                df["delivered_qty"] = pd.to_numeric(df["delivered_qty"], errors="coerce").fillna(0)
                df["open_qty"] = (df["order_qty"] - df["delivered_qty"]).clip(lower=0)
                print(f"  Computed open_qty = order_qty - delivered_qty")
            elif "order_qty" in df.columns:
                df["open_qty"] = pd.to_numeric(df["order_qty"], errors="coerce").fillna(0)
                print(f"  open_qty set to order_qty (no delivered_qty column)")

        df = df.dropna(subset=["sku"])
        if "open_qty" in df.columns:
            df["open_qty"] = pd.to_numeric(df["open_qty"], errors="coerce").fillna(0)
            df = df[df["open_qty"] > 0]

        # Flag missing committed_delivery
        if "committed_delivery" not in df.columns or df["committed_delivery"].isna().all():
            print(f"  Note: No committed delivery date found in open PO — will estimate from order_date + WMA LT")
            df["committed_delivery"] = pd.NaT
            df["delivery_estimated"] = True
        else:
            df["delivery_estimated"] = df["committed_delivery"].isna()

        return df

    def fill_estimated_delivery(self, df: pd.DataFrame, supplier_lt: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing committed_delivery with order_date + WMA LT.
        Call this after supplier LT has been computed.
        """
        if "delivery_estimated" not in df.columns:
            return df
        missing = df["delivery_estimated"] == True

        if missing.sum() == 0:
            return df

        # Build SKU→LT lookup
        lt_lookup = (
            supplier_lt.sort_values("wma_lead_time_days")
            .groupby("sku")["wma_lead_time_days"]
            .first()
            .to_dict()
        )
        default_lt = supplier_lt["wma_lead_time_days"].median() if len(supplier_lt) else 45

        def _estimate(row):
            if not row["delivery_estimated"]:
                return row["committed_delivery"]
            if pd.isna(row.get("order_date")):
                return pd.NaT
            lt = lt_lookup.get(row["sku"], default_lt)
            return row["order_date"] + pd.Timedelta(days=lt)

        df = df.copy()
        df["committed_delivery"] = df.apply(_estimate, axis=1)
        n_filled = missing.sum()
        print(f"  Estimated committed delivery for {n_filled} open PO rows using WMA LT")
        return df

    def inbound_schedule(self, df: pd.DataFrame, as_of=None,
                         horizon_days: int = 30) -> pd.DataFrame:
        """
        Open PO qty grouped by SKU — as a total, and phased by when it is due.

        The total on its own is what let a SKU with an empty shelf be called excess:
        summed against the reorder point, an order committed for four months' time
        counts exactly as much as stock already on the rack. It is not the same supply,
        and next month's demand cannot be met out of it.

        So the quantity is split at two boundaries — already past its committed date,
        due inside the horizon, due after it — and the date of the next arrival still
        to come is carried alongside. Past due is kept apart from the rest rather than
        folded into "arriving soon": supply that was owed three weeks ago has no date
        on it any more, and the planner's job there is to chase it, not to plan on it.
        """
        grp = df.groupby("sku").agg(
            total_open_po_qty=("open_qty", "sum"),
            open_po_lines=("open_qty", "count"),
        ).reset_index()

        if "committed_delivery" in df.columns:
            dates = df.groupby("sku")["committed_delivery"].agg(["min", "max"])
            grp = grp.merge(
                dates.rename(columns={"min": "earliest_delivery", "max": "latest_delivery"}),
                on="sku", how="left"
            )
            grp = grp.merge(self._phase(df, as_of, horizon_days), on="sku", how="left")

        if "open_amount" in df.columns:
            amounts = df.groupby("sku")["open_amount"].sum().rename("total_open_po_amount")
            grp = grp.merge(amounts, on="sku", how="left")

        if "incoterm" in df.columns:
            inco = df.groupby("sku")["incoterm"].agg(
                lambda x: x.mode()[0] if x.notna().any() else None
            )
            grp = grp.merge(inco.rename("incoterm"), on="sku", how="left")

        grp["location_id"] = (df["location_id"].iloc[0] if "location_id" in df.columns
                              else self.location_id)
        return grp

    @staticmethod
    def _phase(df: pd.DataFrame, as_of, horizon_days: int) -> pd.DataFrame:
        """
        Split each SKU's open quantity by when it is due, and name the next arrival.

        With no `as_of` there is no boundary to split on, so everything is reported as
        due inside the horizon — the same reading the pipeline had before, rather than
        a silent zero that would make every SKU look starved.
        """
        due = pd.to_datetime(df["committed_delivery"], errors="coerce")
        qty = pd.to_numeric(df["open_qty"], errors="coerce").fillna(0.0)

        if as_of is None:
            phased = df.assign(_past=0.0, _due=qty, _beyond=0.0, _next=due)
        else:
            now = pd.Timestamp(as_of)
            cut = now + pd.Timedelta(days=horizon_days)
            phased = df.assign(
                _past=qty.where(due < now, 0.0),
                _due=qty.where((due >= now) & (due <= cut), 0.0),
                # An unknown date is not near-term supply. Estimated dates are filled in
                # upstream, so what is left here is genuinely unschedulable.
                _beyond=qty.where(due.isna() | (due > cut), 0.0),
                _next=due.where(due >= now),
            )

        out = phased.groupby("sku").agg(
            inbound_past_due_qty=("_past", "sum"),
            inbound_due_qty=("_due", "sum"),
            inbound_beyond_qty=("_beyond", "sum"),
            next_arrival=("_next", "min"),
        ).reset_index()

        out["days_to_next_arrival"] = (
            (out["next_arrival"] - pd.Timestamp(as_of)).dt.days if as_of is not None
            else np.nan
        )
        return out
