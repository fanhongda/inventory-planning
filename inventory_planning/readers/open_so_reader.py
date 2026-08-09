"""Open sales order (backlog) reader."""

import pandas as pd
from .base_reader import BaseReader


class OpenSOReader(BaseReader):
    doc_type = "open_so"

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["sku", "open_qty"])
        df = df[df["open_qty"] > 0]
        return df

    def backlog_summary(self, df: pd.DataFrame, as_of=None,
                        horizon_days: int = 30) -> pd.DataFrame:
        """
        Total open backlog qty and amount per SKU, with earliest request date.

        Also splits the book by *when it is wanted*. The total order book is not next
        period's requirement — a line requested five months out is not something to buy
        stock for this cycle. `backlog_due_qty` is the part due inside the planning
        horizon, which is what the purchase recommender compares against the forecast.

        Lines with no request date are counted as due: an undated commitment is more
        likely to be wanted now than to be wanted in six months.
        """
        grp = df.groupby("sku").agg(
            backlog_qty=("open_qty", "sum"),
            backlog_amount=("open_amount", "sum") if "open_amount" in df.columns else ("open_qty", "count"),
            earliest_request=("customer_request_date", "min"),
            latest_request=("customer_request_date", "max"),
            open_so_lines=("open_qty", "count"),
        ).reset_index()

        grp = grp.merge(self._due_split(df, as_of, horizon_days), on="sku", how="left")
        for col in ("backlog_due_qty", "backlog_past_due_qty"):
            grp[col] = grp[col].fillna(0.0)

        grp["location_id"] = df["location_id"].iloc[0] if "location_id" in df.columns else "DC-01"
        return grp

    @staticmethod
    def _due_split(df: pd.DataFrame, as_of, horizon_days: int) -> pd.DataFrame:
        """Backlog inside the planning horizon, and the part already past due."""
        request = (
            pd.to_datetime(df["customer_request_date"], errors="coerce")
            if "customer_request_date" in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        anchor = pd.Timestamp(as_of) if as_of is not None else request.max()
        if pd.isna(anchor):
            # No dates anywhere: the whole book is the horizon's requirement.
            due, past_due = df["open_qty"], pd.Series(0.0, index=df.index)
        else:
            cutoff = anchor + pd.Timedelta(days=horizon_days)
            due = df["open_qty"].where(request.isna() | (request <= cutoff), 0.0)
            past_due = df["open_qty"].where(request.notna() & (request < anchor), 0.0)

        return (
            df.assign(_due=due, _past_due=past_due)
            .groupby("sku")
            .agg(backlog_due_qty=("_due", "sum"), backlog_past_due_qty=("_past_due", "sum"))
            .reset_index()
        )
