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

        grp["location_id"] = (df["location_id"].iloc[0] if "location_id" in df.columns
                              else self.location_id)
        return grp

    def order_by_schedule(self, df: pd.DataFrame, lead_times=None, as_of=None,
                          review_period_days=30) -> pd.DataFrame:
        """
        For an order-on-demand item: when does each SKU's order have to be *placed*.

        A stocking item is bought against a rate, so a flat thirty-day window on the
        order book is a reasonable scope for it. An order-on-demand item is not — it
        holds no policy stock, so the only thing standing between a customer date and a
        miss is the supplier's lead time. The line that matters is not the one wanted
        soonest; it is the one whose *order date* has arrived:

            order_by = customer_request_date − lead_time

        Scoping that book to thirty days is what makes a 51-day lead time invisible. A
        line wanted in 45 days needed its purchase order two weeks ago, and a flat
        horizon reports the SKU as needing nothing right up until it is unrecoverable.

        `lead_times` may be a frame with `sku` and a lead-time column, a Series indexed
        by SKU, or one number. Missing lead time means the order-by date is the request
        date itself — the honest reading when nothing says how long supply takes, and
        it errs towards acting early rather than late.

        Returns per SKU: `mto_actionable_qty` (must be ordered by the end of this review
        period), `mto_order_past_due_qty` (should already have been ordered),
        `mto_order_by` (the earliest such date) and `mto_next_request`.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=[
                "sku", "mto_actionable_qty", "mto_order_past_due_qty",
                "mto_order_by", "mto_next_request"])

        work = df.copy()
        request = (
            pd.to_datetime(work["customer_request_date"], errors="coerce")
            if "customer_request_date" in work.columns
            else pd.Series(pd.NaT, index=work.index)
        )
        anchor_ts = pd.Timestamp(as_of) if as_of is not None else request.max()

        lt_days = self._lead_time_days(work["sku"], lead_times)
        work["_order_by"] = request - pd.to_timedelta(lt_days.fillna(0.0), unit="D")
        work["_request"] = request

        if pd.isna(anchor_ts):
            # No dates anywhere. Everything on the book is actionable, which is the same
            # reading `_due_split` takes and for the same reason.
            actionable = work["open_qty"]
            past_due = pd.Series(0.0, index=work.index)
        else:
            cutoff = anchor_ts + pd.Timedelta(days=float(review_period_days))
            # An undated line counts as actionable: an open commitment with no date is
            # more likely to be wanted now than to be wanted after the next review.
            actionable = work["open_qty"].where(
                work["_order_by"].isna() | (work["_order_by"] <= cutoff), 0.0)
            past_due = work["open_qty"].where(
                work["_order_by"].notna() & (work["_order_by"] < anchor_ts), 0.0)

        work["_actionable"] = actionable
        work["_order_past_due"] = past_due
        # The deadline reported is the earliest one still open, because that is the one
        # that expires first.
        return (
            work.groupby("sku")
            .agg(mto_actionable_qty=("_actionable", "sum"),
                 mto_order_past_due_qty=("_order_past_due", "sum"),
                 mto_order_by=("_order_by", "min"),
                 mto_next_request=("_request", "min"))
            .reset_index()
        )

    @staticmethod
    def _lead_time_days(skus: pd.Series, lead_times) -> pd.Series:
        """Per-line lead time, from a frame, a Series, a number, or nothing."""
        if lead_times is None:
            return pd.Series(float("nan"), index=skus.index, dtype="float64")
        if isinstance(lead_times, pd.DataFrame):
            col = next((c for c in ("lead_time_days", "wma_lead_time_days")
                        if c in lead_times.columns), None)
            if col is None or "sku" not in lead_times.columns:
                return pd.Series(float("nan"), index=skus.index, dtype="float64")
            lookup = lead_times.drop_duplicates("sku").set_index("sku")[col]
            return pd.to_numeric(skus.map(lookup), errors="coerce")
        if isinstance(lead_times, pd.Series):
            return pd.to_numeric(skus.map(lead_times), errors="coerce")
        return pd.Series(float(lead_times), index=skus.index, dtype="float64")

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
