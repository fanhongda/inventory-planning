"""
Service performance — on-time delivery, measured rather than modelled.

OTD is simple arithmetic: a line is on time when it shipped on or before the customer's
request date. What makes it worth a module is that the *failures* are not one thing,
and treating them as one thing hides both problems at once.

    on_time                    shipped on or before request date
    shipped_late               shipped, but after the request date
    open_past_due_short        past request date, still open, and stock is not there
                               — a genuine supply failure
    open_past_due_available    past request date, still open, but the stock is sitting
                               in the warehouse — the customer has not collected

That last category is the one that matters most for being believed. It is **not** a
planning failure — the goods were there — but counting it as one makes OTD look worse
than the supply chain deserves. It is instead an inventory-efficiency problem: stock
that is committed, paid for, aged, and not moving. Two different owners, two different
fixes, and a combined number tells neither of them anything.

Request-date quality is measured, not caveated. A request date equal to the order date
is not a requirement, it is "as soon as possible"; a request date already in the past
when the order was raised is not a date anyone planned to. Both make OTD look bad for
reasons that have nothing to do with the supply chain, so their share is reported
alongside the metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ON_TIME = "on_time"
SHIPPED_LATE = "shipped_late"
OPEN_PAST_DUE_SHORT = "open_past_due_short"
OPEN_PAST_DUE_AVAILABLE = "open_past_due_available"
OPEN_NOT_YET_DUE = "open_not_yet_due"
UNJUDGEABLE = "unjudgeable"

# Categories that count against the supply chain. `open_past_due_available` is
# deliberately absent: the stock was there.
FAILURE_STATES = {SHIPPED_LATE, OPEN_PAST_DUE_SHORT}
JUDGEABLE_STATES = {ON_TIME, SHIPPED_LATE, OPEN_PAST_DUE_SHORT, OPEN_PAST_DUE_AVAILABLE}

STATE_LABELS = {
    ON_TIME: "On time",
    SHIPPED_LATE: "Shipped late",
    OPEN_PAST_DUE_SHORT: "Past due — no stock",
    OPEN_PAST_DUE_AVAILABLE: "Past due — stock available, not collected",
    OPEN_NOT_YET_DUE: "Open, not yet due",
    UNJUDGEABLE: "Cannot judge (no request date)",
}


@dataclass
class RequestDateQuality:
    """How trustworthy the request dates are — the denominator of every OTD figure."""

    total_lines: int = 0
    missing: int = 0
    same_as_order_date: int = 0
    before_order_date: int = 0

    @property
    def usable(self) -> int:
        return self.total_lines - self.missing

    @property
    def clean(self) -> int:
        return self.usable - self.same_as_order_date - self.before_order_date

    @property
    def clean_rate(self) -> float:
        return self.clean / self.total_lines if self.total_lines else 0.0

    @property
    def is_trustworthy(self) -> bool:
        return self.clean_rate >= 0.70

    def summary(self) -> str:
        if not self.total_lines:
            return "    No lines to assess."
        lines = [
            f"    {self.clean:,}/{self.total_lines:,} lines ({self.clean_rate:.0%}) "
            f"have a request date that reflects a real requirement",
        ]
        if self.missing:
            lines.append(f"      {self.missing:,} have no request date — excluded from OTD")
        if self.same_as_order_date:
            lines.append(
                f"      {self.same_as_order_date:,} have request date = order date — "
                f"that is 'as soon as possible', not a requirement; OTD against it is "
                f"unwinnable by construction"
            )
        if self.before_order_date:
            lines.append(
                f"      {self.before_order_date:,} were already past due when raised — "
                f"no supply chain could have met these"
            )
        if not self.is_trustworthy:
            lines.append(
                f"    ⚠ Under 70% clean. Treat the OTD figure as indicative and fix the "
                f"request-date discipline before setting a target against it."
            )
        return "\n".join(lines)


@dataclass
class ServiceResult:
    """Line-level service classification with per-SKU attribution."""

    lines: pd.DataFrame
    quality: RequestDateQuality
    as_of: date
    data_latest: Optional[date] = None
    stale_days: int = 0

    # ── Headline metrics ─────────────────────────────────────────────────────

    @property
    def judgeable(self) -> pd.DataFrame:
        return self.lines[self.lines["service_state"].isin(JUDGEABLE_STATES)]

    @property
    def completed(self) -> pd.DataFrame:
        """
        Lines whose delivery outcome is settled.

        OTD is measured over these only. An open line that is past due can never come
        out "on time" — had it been on time it would have shipped — so including open
        lines in the denominator drives the rate toward zero by construction, and an
        extract with no delivery history would report a confident 0%. Past-due open
        lines are a real failure, but they are *backlog*, reported separately below.
        """
        return self.lines[self.lines["service_state"].isin({ON_TIME, SHIPPED_LATE})]

    @property
    def otd_line_rate(self) -> float:
        """Share of completed deliveries that met the request date."""
        settled = self.completed
        if settled.empty:
            return float("nan")
        return float((settled["service_state"] == ON_TIME).sum() / len(settled))

    @property
    def otd_line_rate_harsh(self) -> float:
        """
        OTD if uncollected stock were also counted against supply — what a naive
        report prints when it folds every past-due line into the miss rate.
        """
        settled = self.completed
        uncollected = self.lines[self.lines["service_state"] == OPEN_PAST_DUE_AVAILABLE]
        denominator = len(settled) + len(uncollected)
        if not denominator:
            return float("nan")
        return float((settled["service_state"] == ON_TIME).sum() / denominator)

    @property
    def past_due_backlog_lines(self) -> int:
        """Open lines past the request date with no stock to fill them."""
        return int((self.lines["service_state"] == OPEN_PAST_DUE_SHORT).sum())

    @property
    def past_due_backlog_value(self) -> float:
        stuck = self.lines[self.lines["service_state"] == OPEN_PAST_DUE_SHORT]
        return float(stuck["line_value"].sum())

    @property
    def otd_line_rate_clean(self) -> float:
        """
        OTD over lines whose request date represents a real requirement.

        Lines raised with request date = order date are unwinnable by construction, and
        lines already past due when raised were never achievable. If this differs
        materially from the headline, the metric is measuring order-entry habits rather
        than supply performance — and the fix is upstream of planning.
        """
        settled = self.completed
        fair = settled[
            settled["request_date"].notna()
            & settled["order_date"].notna()
            & ((settled["request_date"] - settled["order_date"]).dt.days > 0)
        ]
        if fair.empty:
            return float("nan")
        return float((fair["service_state"] == ON_TIME).sum() / len(fair))

    @property
    def otd_value_rate(self) -> float:
        settled = self.completed
        total = settled["line_value"].sum()
        if not total:
            return float("nan")
        return float(settled.loc[settled["service_state"] == ON_TIME, "line_value"].sum() / total)

    def state_counts(self) -> pd.DataFrame:
        counts = (
            self.lines.groupby("service_state")
            .agg(lines=("service_state", "size"), value=("line_value", "sum"))
            .reindex(list(STATE_LABELS), fill_value=0)
            .reset_index()
        )
        counts["label"] = counts["service_state"].map(STATE_LABELS)
        counts["share"] = counts["lines"] / max(len(self.lines), 1)
        return counts

    @property
    def measured_window(self) -> Optional[tuple]:
        """
        First and last month a delivery outcome was actually observed.

        Every attribution below is scored over this window and nothing else, so a
        heading that omits it invites the reader to take "SKU-006 caused five misses"
        as a statement about now rather than about a period that may have ended months
        before the extract was taken.
        """
        settled = self.completed
        if settled.empty or "actual_date" not in settled.columns:
            return None
        stamps = pd.to_datetime(settled["actual_date"], errors="coerce").dropna()
        if stamps.empty:
            return None
        return stamps.min().to_period("M"), stamps.max().to_period("M")

    # ── Over time ────────────────────────────────────────────────────────────

    def monthly_otd(self, min_lines: int = 5) -> pd.DataFrame:
        """
        On-time delivery month by month, over exactly the lines the headline uses.

        A single OTD figure answers "how did we do" and hides "when did it change",
        which is the question that decides whether anything needs fixing. 82% flat for
        two years and 82% because the last four months collapsed are the same number
        and different situations.

        Bucketed on the **ship** date, not the request date: the month a delivery
        succeeded or failed is the month it happened, and bucketing on the promise
        would credit a late shipment to the month it was due rather than the month it
        eventually moved.

        `thin` marks months whose line count is too small for the rate to mean
        anything. They are returned rather than dropped — a month with three lines is
        itself a finding, and silently removing it leaves a gap the reader will
        misread as zero.
        """
        cols = ["period", "lines", "on_time_lines", "otd_line_rate",
                "value", "on_time_value", "otd_value_rate", "thin"]
        settled = self.completed
        if settled.empty or "actual_date" not in settled.columns:
            return pd.DataFrame(columns=cols)

        work = settled.assign(_stamp=pd.to_datetime(settled["actual_date"], errors="coerce"))
        work = work.dropna(subset=["_stamp"])
        if work.empty:
            return pd.DataFrame(columns=cols)

        work["period"] = work["_stamp"].dt.to_period("M")
        work["_on_time"] = work["service_state"] == ON_TIME
        work["_value"] = pd.to_numeric(work["line_value"], errors="coerce").fillna(0.0)

        out = (
            work.groupby("period")
            .agg(lines=("_on_time", "size"),
                 on_time_lines=("_on_time", "sum"),
                 value=("_value", "sum"),
                 on_time_value=("_value", lambda s: float(s[work.loc[s.index, "_on_time"]].sum())))
            .reset_index()
        )
        # Every month between the first and last, so a month with no deliveries at all
        # shows as a gap in the volume rather than as an absent point on the rate.
        full = pd.period_range(out["period"].min(), out["period"].max(), freq="M")
        out = out.set_index("period").reindex(full).rename_axis("period").reset_index()
        for col in ("lines", "on_time_lines", "value", "on_time_value"):
            out[col] = out[col].fillna(0.0)

        out["otd_line_rate"] = np.where(out["lines"] > 0, out["on_time_lines"] / out["lines"], np.nan)
        out["otd_value_rate"] = np.where(out["value"] > 0, out["on_time_value"] / out["value"], np.nan)
        out["thin"] = out["lines"] < min_lines
        return out[cols]

    # ── Attribution ──────────────────────────────────────────────────────────

    def failures_by_sku(self, top: int = None) -> pd.DataFrame:
        """
        Which SKUs caused the OTD misses. Ranked by value at stake, because a Pareto by
        line count over-weights cheap high-frequency items.
        """
        failures = self.lines[self.lines["service_state"].isin(FAILURE_STATES)]
        if failures.empty:
            return pd.DataFrame(columns=["sku", "failed_lines", "failed_value",
                                         "avg_days_late", "customers"])
        out = (
            failures.groupby("sku")
            .agg(failed_lines=("service_state", "size"),
                 failed_value=("line_value", "sum"),
                 avg_days_late=("days_late", "mean"),
                 max_days_late=("days_late", "max"),
                 customers=("customer", lambda s: s.nunique()))
            .reset_index()
            .sort_values("failed_value", ascending=False)
        )
        total = self.lines.groupby("sku").size().rename("total_lines")
        out = out.merge(total, on="sku", how="left")
        out["failure_rate"] = out["failed_lines"] / out["total_lines"]
        out["cum_share_of_failures"] = out["failed_value"].cumsum() / out["failed_value"].sum()
        return out.head(top) if top else out

    def uncollected_by_sku(self, top: int = None) -> pd.DataFrame:
        """
        Past request date, stock on the shelf, customer has not taken it.

        Not a supply failure — but the stock is committed, aged and immobile, so it
        belongs in the inventory-efficiency conversation instead.
        """
        stuck = self.lines[self.lines["service_state"] == OPEN_PAST_DUE_AVAILABLE]
        if stuck.empty:
            return pd.DataFrame(columns=["sku", "stuck_lines", "stuck_qty",
                                         "stuck_value", "avg_days_overdue"])
        out = (
            stuck.groupby("sku")
            .agg(stuck_lines=("service_state", "size"),
                 stuck_qty=("qty", "sum"),
                 stuck_value=("line_value", "sum"),
                 avg_days_overdue=("days_late", "mean"),
                 max_days_overdue=("days_late", "max"),
                 customers=("customer", lambda s: s.nunique()))
            .reset_index()
            .sort_values("stuck_value", ascending=False)
        )
        return out.head(top) if top else out

    def failures_by_customer(self, top: int = 10) -> pd.DataFrame:
        failures = self.lines[self.lines["service_state"].isin(FAILURE_STATES)]
        if failures.empty or "customer" not in failures.columns:
            return pd.DataFrame()
        return (
            failures.groupby("customer")
            .agg(failed_lines=("service_state", "size"),
                 failed_value=("line_value", "sum"))
            .reset_index()
            .sort_values("failed_value", ascending=False)
            .head(top)
        )

    def summary(self) -> str:
        counts = self.state_counts()
        lines = [
            "  Service performance (measured)",
            "  " + "-" * 58,
            f"    as of {self.as_of}   {len(self.lines):,} order lines",
            "",
        ]
        if self.stale_days > 45:
            lines.append(
                f"    ⓘ Anchored to the newest date in the data ({self.data_latest}), "
                f"{self.stale_days} days before today. Scoring this extract against "
                f"today's date would mark every open line past due."
            )
            lines.append("")
        for _, row in counts.iterrows():
            if not row["lines"]:
                continue
            lines.append(f"    {row['label']:<44}{int(row['lines']):>7,}  "
                         f"${row['value']:>12,.0f}")

        lines.append("")
        fair, harsh = self.otd_line_rate, self.otd_line_rate_harsh
        if np.isnan(fair):
            # Saying nothing here reads as "OTD is fine". Name the missing input.
            shipped = int((self.lines["source"] == "shipped").sum()) if "source" in self.lines else 0
            reason = (
                f"none of the {shipped:,} delivered lines carry a customer request date"
                if shipped else
                "no delivery history was supplied — only open orders"
            )
            lines.append(
                f"    OTD cannot be measured: {reason}. It is measured over *completed*"
                f" deliveries; open past-due lines are backlog, counted separately below."
            )
        else:
            lines.append(f"    OTD (lines)  {fair:.1%}"
                         f"    by value  {self.otd_value_rate:.1%}")
            if not np.isnan(harsh) and abs(fair - harsh) > 0.005:  # noqa: E501
                lines.append(
                    f"    Counting uncollected stock as a miss would read {harsh:.1%} — "
                    f"{fair - harsh:+.1%} of the gap is stock the customer did not take, "
                    f"not stock we did not have."
                )

            clean = self.otd_line_rate_clean
            if not np.isnan(clean) and abs(clean - fair) > 0.005:
                lines.append(
                    f"    On lines with a genuine request date only: {clean:.1%} "
                    f"({clean - fair:+.1%}) — the difference is order-entry practice, "
                    f"not supply performance."
                )

        if self.past_due_backlog_lines:
            lines.append(
                f"    Past-due backlog: {self.past_due_backlog_lines:,} lines, "
                f"${self.past_due_backlog_value:,.0f} — open past the request date with "
                f"no stock. A separate metric from OTD, not folded into it."
            )

        lines.append("")
        lines.append("  Request date quality")
        lines.append(self.quality.summary())
        return "\n".join(lines)


def latest_observed_date(*frames: Optional[pd.DataFrame]) -> tuple:
    """
    The newest real date anywhere in the supplied frames, and how stale it is.

    Used to anchor "now" to the extract rather than the clock. Future-dated columns
    are excluded from the maximum — a request or promise date is routinely months
    ahead, and taking it as "now" would flip the error the other way.
    """
    backward_looking = ("ship_date", "demand_date", "order_date", "invoice_date")
    newest: Optional[pd.Timestamp] = None
    for frame in frames:
        if frame is None or len(frame) == 0:
            continue
        for col in backward_looking:
            if col not in frame.columns:
                continue
            series = pd.to_datetime(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            candidate = series.max()
            if newest is None or candidate > newest:
                newest = candidate

    if newest is None:
        return None, 0
    observed = newest.date()
    return observed, max((date.today() - observed).days, 0)


class ServiceAnalyzer:
    """Classifies order lines against the customer's request date."""

    def __init__(self, availability_tolerance: float = 0.95):
        # A line counts as fulfillable when on-hand covers this share of it. Slightly
        # under 1.0 so a rounding difference does not reclassify a genuine miss.
        self.availability_tolerance = availability_tolerance

    def analyze(
        self,
        sales_history: pd.DataFrame = None,
        open_so: pd.DataFrame = None,
        inventory: pd.DataFrame = None,
        as_of: date = None,
    ) -> ServiceResult:
        # Anchor on the data, not the wall clock. An extract taken months ago scored
        # against today marks every open line past due and drives OTD to zero — a
        # confident, entirely artificial answer. The caller can still override.
        derived, stale_days = latest_observed_date(sales_history, open_so)
        if as_of is None:
            as_of = derived or date.today()
        as_of_ts = pd.Timestamp(as_of)

        frames = []
        if sales_history is not None and len(sales_history):
            frames.append(self._classify_shipped(sales_history))
        if open_so is not None and len(open_so):
            frames.append(self._classify_open(open_so, inventory, as_of_ts))

        if not frames:
            return ServiceResult(
                lines=pd.DataFrame(columns=["sku", "service_state", "line_value",
                                            "days_late", "qty", "customer"]),
                quality=RequestDateQuality(),
                as_of=as_of,
            )

        lines = pd.concat(frames, ignore_index=True)
        return ServiceResult(
            lines=lines,
            quality=self._assess_request_dates(lines),
            as_of=as_of,
            data_latest=derived,
            stale_days=stale_days,
        )

    # ── Shipped lines ────────────────────────────────────────────────────────

    def _classify_shipped(self, sales: pd.DataFrame) -> pd.DataFrame:
        df = sales.copy()
        request = self._dates(df, "customer_request_date")
        shipped = self._dates(df, "ship_date")
        if shipped.isna().all():
            shipped = self._dates(df, "demand_date")

        out = pd.DataFrame({
            "sku": df.get("sku"),
            "customer": df.get("customer", pd.Series(None, index=df.index)),
            "so_number": df.get("so_number", pd.Series(None, index=df.index)),
            "qty": pd.to_numeric(df.get("qty", 0), errors="coerce").fillna(0.0),
            "line_value": self._line_value(df),
            "request_date": request,
            "order_date": self._dates(df, "order_date"),
            "actual_date": shipped,
            "source": "shipped",
        })
        out["days_late"] = (out["actual_date"] - out["request_date"]).dt.days

        judgeable = out["request_date"].notna() & out["actual_date"].notna()
        out["service_state"] = np.where(
            ~judgeable, UNJUDGEABLE,
            np.where(out["days_late"] <= 0, ON_TIME, SHIPPED_LATE),
        )
        # Only lateness is meaningful; an early shipment is not "negative days late".
        out["days_late"] = out["days_late"].clip(lower=0).where(judgeable)
        return out

    # ── Open lines ───────────────────────────────────────────────────────────

    def _classify_open(
        self, open_so: pd.DataFrame, inventory: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        df = open_so.copy()
        request = self._dates(df, "customer_request_date")
        qty = pd.to_numeric(df.get("open_qty", 0), errors="coerce").fillna(0.0)

        out = pd.DataFrame({
            "sku": df.get("sku"),
            "customer": df.get("customer", pd.Series(None, index=df.index)),
            "so_number": df.get("so_number", pd.Series(None, index=df.index)),
            "qty": qty,
            "line_value": self._line_value(df, qty_col="open_qty"),
            "request_date": request,
            "order_date": self._dates(df, "order_date"),
            "actual_date": pd.NaT,
            "source": "open",
        })
        out["days_late"] = (as_of - out["request_date"]).dt.days

        available = self._availability(out, inventory)
        past_due = out["request_date"].notna() & (out["request_date"] < as_of)

        out["service_state"] = np.where(
            out["request_date"].isna(), UNJUDGEABLE,
            np.where(
                ~past_due, OPEN_NOT_YET_DUE,
                np.where(available, OPEN_PAST_DUE_AVAILABLE, OPEN_PAST_DUE_SHORT),
            ),
        )
        out["days_late"] = out["days_late"].clip(lower=0).where(past_due)
        out["stock_available"] = available
        return out

    def _availability(self, lines: pd.DataFrame, inventory: pd.DataFrame) -> pd.Series:
        """
        Whether stock exists to fill each open line.

        On-hand is allocated across a SKU's open lines oldest-request-first, mirroring
        how a warehouse actually picks. Comparing every line against the full on-hand
        balance independently would report the same units as available many times over
        and overstate the "customer has not collected" category.
        """
        if inventory is None or len(inventory) == 0 or "qty_on_hand" not in inventory.columns:
            return pd.Series(False, index=lines.index)

        on_hand = (
            inventory.groupby("sku")["qty_on_hand"]
            .sum().clip(lower=0).to_dict()
        )
        available = pd.Series(False, index=lines.index)

        order = lines.sort_values("request_date", na_position="last").index
        remaining = dict(on_hand)
        for idx in order:
            sku = lines.at[idx, "sku"]
            need = float(lines.at[idx, "qty"] or 0)
            have = float(remaining.get(sku, 0.0))
            if need > 0 and have >= need * self.availability_tolerance:
                available.at[idx] = True
                remaining[sku] = have - need
        return available

    # ── Request date quality ─────────────────────────────────────────────────

    @staticmethod
    def _assess_request_dates(lines: pd.DataFrame) -> RequestDateQuality:
        quality = RequestDateQuality(total_lines=len(lines))
        quality.missing = int(lines["request_date"].isna().sum())

        both = lines["request_date"].notna() & lines["order_date"].notna()
        if both.any():
            delta = (lines.loc[both, "request_date"] - lines.loc[both, "order_date"]).dt.days
            quality.same_as_order_date = int((delta == 0).sum())
            quality.before_order_date = int((delta < 0).sum())
        return quality

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dates(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(pd.NaT, index=df.index)
        return pd.to_datetime(df[col], errors="coerce")

    @staticmethod
    def _line_value(df: pd.DataFrame, qty_col: str = "qty") -> pd.Series:
        for col in ("amount", "open_amount", "line_amount"):
            if col in df.columns:
                value = pd.to_numeric(df[col], errors="coerce")
                if value.notna().any():
                    return value.fillna(0.0)
        if "unit_price" in df.columns and qty_col in df.columns:
            return (pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
                    * pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0))
        return pd.Series(0.0, index=df.index)
