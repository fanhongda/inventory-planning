"""
Days on hand, ageing and slow movement — by product line, not by SKU.

## Why the aggregate cannot be an average

`should_be.py` computes `actual_dioh` per SKU, and the obvious way to report a product
line is to average those. It is wrong, and wrong by an amount that inverts the answer.

DIOH is a ratio, and averaging ratios weights every SKU equally regardless of how much
stock it represents. One dead item with 40 units and demand of 0.1 a month scores
12,000 days; a hundred healthy items score 60. The mean is 180 and the line looks
diseased. The correct figure divides the line's total stock value by the line's daily
cost of sales — which is 62, and is the number the business actually holds.

So every aggregate here is value-weighted: total value over total daily COGS. A per-SKU
DIOH still appears in the detail, because that is where the outliers live and finding
them is the point of the detail.

## Ageing is measured where it can be, and named where it cannot

True ageing needs a date: a goods-receipt date per batch, or a last-movement date per
SKU. Most inventory extracts carry neither — one real export gives opening stock,
receipts, issues and closing stock for a window, and no ages at all.

Where the extract has one, age is reported as age. Where it does not, the fallback is
**days of cover** — stock divided by daily demand — and it is reported under its own
name, never as ageing. The two answer different questions and are wrong about each
other in both directions: a fast-moving item restocked yesterday can carry 400 days of
cover and no age at all, and a genuinely ancient part with a sudden order shows almost
no cover while still being five years old. Presenting cover as ageing would put the
wrong items on a write-off list.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import numpy as np
import pandas as pd

# No demand for this many months is the working definition of slow-moving. Six rather
# than three because a quarterly-ordered item is not slow, it is quarterly, and the
# extract that prompted this has a median SKU selling in three months out of twenty-two.
_SLOW_MONTHS = 6
# Cover beyond this is the threshold for the ageing proxy. A year is chosen because it
# is the point at which stock has survived a full seasonal cycle without being needed.
_LONG_COVER_DAYS = 365.0


@dataclass
class InventoryHealth:
    """DIOH, ageing and slow movement, rolled up and in detail."""

    by_family: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    slow_moving: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    long_aging: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    detail: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    # Whether `long_aging` is measured from a date or inferred from cover. The two are
    # not interchangeable and the report has to say which it is showing.
    aging_measured: bool = False
    aging_basis: str = ""
    total_value: float = 0.0
    currency: str = "USD"

    @property
    def slow_value(self) -> float:
        return float(self.slow_moving["stock_value"].sum()) if len(self.slow_moving) else 0.0

    @property
    def aging_value(self) -> float:
        return float(self.long_aging["stock_value"].sum()) if len(self.long_aging) else 0.0

    def summary(self) -> str:
        if not len(self.detail):
            return "  Inventory health — not built (no stock position, or no cost)"
        lines = ["  Inventory health — days on hand by product line",
                 "  " + "-" * 58]
        if len(self.by_family):
            lines.append(f"    {'Product line':<28}{'Value':>13}{'DIOH':>8}{'Slow':>13}")
            for _, row in self.by_family.head(12).iterrows():
                dioh = ("—" if not np.isfinite(row["dioh"]) else f"{row['dioh']:,.0f}")
                lines.append(
                    f"    {str(row['product_family'])[:27]:<28}"
                    f"{row['stock_value']:>13,.0f}{dioh:>8}"
                    f"{row['slow_value']:>13,.0f}"
                )
            lines.append("")
            lines.append("    DIOH is value-weighted — the line's stock over the line's "
                         "daily cost of sales. An average of per-SKU DIOH would let one "
                         "dead part speak for a hundred healthy ones.")
        share = (self.slow_value / self.total_value) if self.total_value else 0.0
        lines.append("")
        lines.append(
            f"    Slow-moving: {self.slow_value:,.0f} {self.currency} "
            f"({share:.0%} of stock) across {len(self.slow_moving):,} "
            f"SKU{'s' if len(self.slow_moving) != 1 else ''} with no demand in "
            f"{_SLOW_MONTHS} months."
        )
        if len(self.long_aging):
            share = (self.aging_value / self.total_value) if self.total_value else 0.0
            lines.append(
                f"    {self.aging_basis}: {self.aging_value:,.0f} {self.currency} "
                f"({share:.0%}) across {len(self.long_aging):,} SKUs."
            )
            if not self.aging_measured:
                lines.append(
                    "      No ageing date in the stock extract, so this is days of "
                    "cover, not age. They are wrong about each other in both "
                    "directions — a fast item restocked yesterday can show a year of "
                    "cover, and a five-year-old part with one order shows almost none. "
                    "Supply a stock-ageing report or a last-movement date to measure it."
                )
        return "\n".join(lines)


def build_inventory_health(
    should_be=None,
    time_series: pd.DataFrame = None,
    attributes: pd.DataFrame = None,
    inventory: pd.DataFrame = None,
    currency: str = "USD",
) -> InventoryHealth:
    """
    Roll the per-SKU position up to product line, and pull out the two tails.

    `should_be` supplies the position and its valuation, `time_series` says when each
    item last moved, `attributes` carries the family it rolls up to, and `inventory`
    is consulted only for an ageing column if the extract happened to have one.
    """
    if should_be is None or not len(getattr(should_be, "frame", [])):
        return InventoryHealth(currency=currency)

    detail = should_be.frame.copy()
    detail["sku"] = detail["sku"].astype(str)
    for column in ("actual_qty", "actual_value", "demand_mean", "actual_dioh"):
        if column not in detail.columns:
            detail[column] = np.nan
    detail = detail.rename(columns={"actual_value": "stock_value",
                                    "actual_qty": "stock_qty"})

    detail = _attach_dimensions(detail, attributes)
    detail = _attach_last_movement(detail, time_series)
    detail, measured, basis = _attach_age(detail, inventory)

    detail["daily_cogs"] = (pd.to_numeric(detail["demand_mean"], errors="coerce")
                            .fillna(0.0) * 12 / 365.0
                            * pd.to_numeric(detail.get("unit_cost"), errors="coerce")
                            .fillna(0.0))
    detail["is_slow"] = detail["months_since_demand"].fillna(np.inf) >= _SLOW_MONTHS
    detail["slow_value"] = np.where(detail["is_slow"],
                                    detail["stock_value"].fillna(0.0), 0.0)

    total = float(pd.to_numeric(detail["stock_value"], errors="coerce").fillna(0.0).sum())
    return InventoryHealth(
        by_family=_roll_up(detail),
        slow_moving=_slow_frame(detail),
        long_aging=_aging_frame(detail, measured),
        detail=detail,
        aging_measured=measured,
        aging_basis=basis,
        total_value=total,
        currency=currency,
    )


# ── Inputs ───────────────────────────────────────────────────────────────────


def _attach_dimensions(detail: pd.DataFrame, attributes) -> pd.DataFrame:
    wanted = ["product_family", "business_unit", "product_family_source", "unit_cost",
              "description", "abc_class"]
    if attributes is None or not len(attributes):
        for column in wanted:
            detail.setdefault(column, np.nan)
        detail["product_family"] = detail.get("product_family", "unclassified")
        return detail
    dims = attributes[["sku"] + [c for c in wanted if c in attributes.columns]].copy()
    dims["sku"] = dims["sku"].astype(str)
    keep = [c for c in dims.columns if c == "sku" or c not in detail.columns]
    detail = detail.merge(dims[keep].drop_duplicates("sku"), on="sku", how="left")
    if "product_family" not in detail.columns:
        detail["product_family"] = "unclassified"
    detail["product_family"] = detail["product_family"].fillna("unclassified")
    return detail


def _attach_last_movement(detail: pd.DataFrame, time_series) -> pd.DataFrame:
    """
    Months since the item last sold, from the demand series.

    Demand rather than goods movement, because that is what the pipeline has. A
    transfer or a stock correction moves the material without anybody wanting it, and
    an item that has not been *ordered* in six months is the one worth surfacing
    whether or not the warehouse touched it.
    """
    detail["months_since_demand"] = np.nan
    if time_series is None or not len(time_series):
        return detail
    last = {}
    total = len(time_series)
    for sku in time_series.columns:
        series = pd.to_numeric(time_series[sku], errors="coerce").fillna(0.0)
        active = np.flatnonzero(series.values > 0)
        last[str(sku)] = (total - 1 - int(active[-1])) if len(active) else np.inf
    detail["months_since_demand"] = detail["sku"].map(last)
    return detail


def _attach_age(detail: pd.DataFrame, inventory) -> tuple:
    """
    A real age where the extract carries one; days of cover, named as such, where not.
    """
    detail["stock_age_days"] = np.nan
    if inventory is not None and len(inventory):
        frame = inventory.copy()
        frame["sku"] = frame["sku"].astype(str)
        if "inventory_age_days" in frame.columns:
            ages = pd.to_numeric(frame["inventory_age_days"], errors="coerce")
            if ages.notna().any():
                detail["stock_age_days"] = detail["sku"].map(
                    frame.assign(_a=ages).groupby("sku")["_a"].max())
                return detail, True, "Aged over a year"
        if "last_movement_date" in frame.columns:
            moved = pd.to_datetime(frame["last_movement_date"], errors="coerce")
            if moved.notna().any():
                anchor = moved.max()
                days = (anchor - moved).dt.days
                detail["stock_age_days"] = detail["sku"].map(
                    frame.assign(_a=days).groupby("sku")["_a"].max())
                return detail, True, "Not moved in over a year"

    daily = (pd.to_numeric(detail["demand_mean"], errors="coerce").fillna(0.0) * 12 / 365.0)
    cover = np.where(daily > 0,
                     pd.to_numeric(detail["stock_qty"], errors="coerce").fillna(0.0) / daily,
                     np.inf)
    detail["days_of_cover"] = cover
    return detail, False, "Over a year of cover"


# ── Rollups ──────────────────────────────────────────────────────────────────


def _roll_up(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Value-weighted DIOH per product line: total stock value over total daily COGS.

    Not the mean of `actual_dioh`. That statistic weights a dead SKU holding forty
    dollars the same as a live one holding forty thousand, and on any real catalogue
    the dead ones are the majority — so the average reports a business drowning in
    stock while the value-weighted figure, which is what the money actually does, reads
    normal.
    """
    if not len(detail):
        return pd.DataFrame()
    keys = ["product_family"]
    if "business_unit" in detail.columns and detail["business_unit"].notna().any():
        keys = ["business_unit", "product_family"]
    grouped = (detail.groupby(keys, as_index=False, dropna=False)
               .agg(skus=("sku", "nunique"),
                    stock_qty=("stock_qty", "sum"),
                    stock_value=("stock_value", "sum"),
                    daily_cogs=("daily_cogs", "sum"),
                    slow_skus=("is_slow", "sum"),
                    slow_value=("slow_value", "sum")))
    grouped["dioh"] = np.where(grouped["daily_cogs"] > 0,
                               grouped["stock_value"] / grouped["daily_cogs"], np.inf)
    grouped["slow_share"] = np.where(grouped["stock_value"] > 0,
                                     grouped["slow_value"] / grouped["stock_value"], 0.0)
    return grouped.sort_values("stock_value", ascending=False).reset_index(drop=True)


def _slow_frame(detail: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ("sku", "description", "business_unit", "product_family",
                           "stock_qty", "stock_value", "months_since_demand",
                           "abc_class") if c in detail.columns]
    slow = detail[detail["is_slow"] & (detail["stock_value"].fillna(0.0) > 0)]
    return slow[columns].sort_values("stock_value", ascending=False).reset_index(drop=True)


def _aging_frame(detail: pd.DataFrame, measured: bool) -> pd.DataFrame:
    if measured:
        mask = pd.to_numeric(detail["stock_age_days"], errors="coerce") > 365
        metric = "stock_age_days"
    else:
        mask = pd.to_numeric(detail.get("days_of_cover"), errors="coerce") > _LONG_COVER_DAYS
        metric = "days_of_cover"
    columns = [c for c in ("sku", "description", "business_unit", "product_family",
                           "stock_qty", "stock_value", metric, "abc_class")
               if c in detail.columns]
    aged = detail[mask.fillna(False) & (detail["stock_value"].fillna(0.0) > 0)]
    return aged[columns].sort_values("stock_value", ascending=False).reset_index(drop=True)
