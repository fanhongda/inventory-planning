"""
The supply and demand plan, period by period, in money.

## What this answers that the rest of the pipeline does not

Everything else here is per SKU and mostly point-in-time: this item needs that much
stock, this order should be pushed out. That is the buyer's view and it is the right
one for buying. It is not the view an S&IOP meeting is held to take.

That meeting asks three questions a month at a time, and the answer to each is a single
number in currency:

    what is the business planning to consume     demand at cost
    what is already committed to arrive          supply at cost
    where do the two fail to meet                the gap, and what it is worth

None of the three existed. The forecast was in units and per SKU, the open order book
was a per-SKU total with no time phasing beyond "inside 30 days or not", and the gap
was a per-SKU flag rather than an amount anyone could take to a meeting.

## Why demand is valued at cost and not at revenue

The S&OP worksheet values demand at selling price, because sales are reviewing their
own number and revenue is the language they own. This is the same demand valued at
**cost**, because the question here is about inventory and supply — what the business
has to buy and hold to serve that plan. The two figures are both correct and they are
not comparable; keeping them in separate outputs is deliberate, and each says which it
is on the face of it.

## What the gap is, precisely

Not "demand exceeds supply". A period is short when the projected closing position
falls below the safety stock the policy calls for — which is the point at which service
is at risk, not the point at which the shelf is empty. Running the projection through
safety stock rather than through zero is what makes the number a *planning* gap rather
than a stockout count, and it is why a period can show a gap while every SKU in it
still has stock on the rack.

The projection is run per SKU and summed, never on the aggregate. Aggregating first
nets one item's surplus against another's shortage — different materials, not
interchangeable — and reports a business in balance that cannot ship a single order.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SIOPPlan:
    """The period balance, its per-SKU detail, and what it rests on."""

    by_period: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    per_sku: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    by_family: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    costed_skus: int = 0
    uncosted_skus: int = 0
    currency: str = "USD"

    @property
    def periods(self) -> List[str]:
        return list(self.by_period["period"]) if len(self.by_period) else []

    def summary(self) -> str:
        if not len(self.by_period):
            return "  S&IOP plan — not built (no forecast, or no cost to value it at)"
        frame = self.by_period
        lines = [
            "  S&IOP — supply and demand by period, at cost",
            "  " + "-" * 58,
            f"    {'Period':<10}{'Demand':>14}{'Supply':>14}{'Closing':>14}{'Gap':>14}",
        ]
        for _, row in frame.iterrows():
            gap = row["gap_value"]
            mark = " *" if gap > 0 else "  "
            lines.append(
                f"    {row['period']:<10}{row['demand_cogs']:>14,.0f}"
                f"{row['supply_value']:>14,.0f}{row['closing_value']:>14,.0f}"
                f"{gap:>12,.0f}{mark}"
            )
        short = frame[frame["gap_value"] > 0]
        if len(short):
            first = short.iloc[0]
            lines.append("")
            lines.append(
                f"    First shortfall in {first['period']}: {first['gap_value']:,.0f} "
                f"{self.currency} across {int(first['gap_skus']):,} SKUs. A period is "
                f"short when the projected close falls below the safety stock the "
                f"policy calls for — service is at risk there, not only where the "
                f"shelf is bare."
            )
        else:
            lines.append("")
            lines.append("    No period falls below its safety stock on committed "
                         "supply. The gap is measured per SKU and summed, never on the "
                         "aggregate — one item's surplus cannot cover another's short.")
        if self.uncosted_skus:
            lines.append(
                f"    {self.uncosted_skus:,} of "
                f"{self.costed_skus + self.uncosted_skus:,} SKUs have no unit cost and "
                f"are excluded from every amount above — their quantities are real and "
                f"their value is unknown, so counting them at zero would understate the "
                f"plan without saying so."
            )
        return "\n".join(lines)


def build_siop_plan(
    forecast_detail: pd.DataFrame,
    inventory: pd.DataFrame = None,
    open_po: pd.DataFrame = None,
    attributes: pd.DataFrame = None,
    safety_stock: pd.DataFrame = None,
    currency: str = "USD",
) -> SIOPPlan:
    """
    Time-phase demand and committed supply over the forecast horizon and value both.

    `forecast_detail` is already bucketed by period, which is what makes this cheap:
    the demand side needs no re-derivation, only costing. The supply side does — an
    open PO carries a delivery date and nothing groups it — so it is bucketed here
    against the same period index, and anything already past due is placed in the
    first period rather than dropped. Past-due supply has no date any more, but it is
    not gone, and excluding it would report a shortfall the warehouse is about to
    resolve.
    """
    if forecast_detail is None or not len(forecast_detail):
        return SIOPPlan(currency=currency)

    demand = forecast_detail[["sku", "period", "forecast_qty"]].copy()
    demand["sku"] = demand["sku"].astype(str)
    demand["period"] = demand["period"].astype(str)
    periods = sorted(demand["period"].unique())

    cost = _unit_cost(attributes, inventory)
    demand["unit_cost"] = demand["sku"].map(cost)
    costed = set(demand.loc[demand["unit_cost"].notna(), "sku"])
    uncosted = set(demand["sku"]) - costed

    demand["demand_cogs"] = (pd.to_numeric(demand["forecast_qty"], errors="coerce")
                             .fillna(0.0) * demand["unit_cost"])

    supply = _phase_supply(open_po, periods)
    supply["supply_value"] = supply["supply_qty"] * supply["sku"].map(cost)

    opening = _opening_position(inventory)
    target = _safety_target(safety_stock)

    per_sku = _project(demand, supply, opening, target, periods, cost)
    by_period = (per_sku.groupby("period", as_index=False)
                 .agg(demand_qty=("demand_qty", "sum"),
                      demand_cogs=("demand_cogs", "sum"),
                      supply_qty=("supply_qty", "sum"),
                      supply_value=("supply_value", "sum"),
                      closing_qty=("closing_qty", "sum"),
                      closing_value=("closing_value", "sum"),
                      gap_qty=("gap_qty", "sum"),
                      gap_value=("gap_value", "sum"))
                 .assign(gap_skus=per_sku[per_sku["gap_qty"] > 0]
                         .groupby("period")["sku"].nunique()
                         .reindex(periods).fillna(0).values)
                 .sort_values("period"))
    # Opening is the position the period starts from, so it is the previous period's
    # close rather than an aggregate of anything — reading it off the per-SKU frame
    # would double-count SKUs that carry no demand in a period and therefore no row.
    by_period["opening_value"] = (by_period["closing_value"].shift(1)
                                  .fillna(float(sum(opening.get(s, 0.0) * cost.get(s, np.nan)
                                                    for s in costed
                                                    if pd.notna(cost.get(s, np.nan))))))

    return SIOPPlan(
        by_period=by_period.reset_index(drop=True),
        per_sku=per_sku,
        by_family=_by_family(per_sku, attributes),
        costed_skus=len(costed), uncosted_skus=len(uncosted), currency=currency,
    )


# ── Inputs ───────────────────────────────────────────────────────────────────


def _unit_cost(attributes, inventory) -> dict:
    """
    Cost per SKU, from the attribute frame where the resolver has already ranked the
    sources, otherwise straight off the stock report.
    """
    for frame in (attributes, inventory):
        if frame is None or not len(frame) or "unit_cost" not in frame.columns:
            continue
        costs = (frame[["sku", "unit_cost"]].dropna()
                 .drop_duplicates("sku").set_index("sku")["unit_cost"])
        costs = pd.to_numeric(costs, errors="coerce")
        costs = costs[costs > 0]
        if len(costs):
            return {str(k): float(v) for k, v in costs.items()}
    return {}


def _phase_supply(open_po: pd.DataFrame, periods: List[str]) -> pd.DataFrame:
    """
    Open purchase orders bucketed into the forecast's periods.

    Anything due before the first period — past due, or landing this month — goes into
    the first bucket. It has no forward date any more, but the goods are still coming,
    and leaving it out would report a shortfall that the receiving dock is about to
    close. Anything due beyond the horizon is dropped: it cannot serve demand inside it.
    """
    empty = pd.DataFrame(columns=["sku", "period", "supply_qty"])
    if open_po is None or not len(open_po) or "open_qty" not in open_po.columns:
        return empty
    date_col = next((c for c in ("committed_delivery", "estimated_delivery",
                                 "delivery_date") if c in open_po.columns), None)
    if date_col is None or not periods:
        return empty

    frame = open_po[["sku", "open_qty", date_col]].copy()
    frame["sku"] = frame["sku"].astype(str)
    frame["open_qty"] = pd.to_numeric(frame["open_qty"], errors="coerce").fillna(0.0)
    frame = frame[frame["open_qty"] > 0]
    if not len(frame):
        return empty

    due = pd.to_datetime(frame[date_col], errors="coerce")
    bucket = due.dt.to_period("M").astype(str)
    # Undated open supply is treated as arriving in the first period for the same
    # reason as past-due supply: it exists, and calling it zero is the larger error.
    bucket = bucket.where(due.notna(), periods[0])
    bucket = bucket.where(bucket >= periods[0], periods[0])
    frame["period"] = bucket
    frame = frame[frame["period"].isin(periods)]
    return (frame.groupby(["sku", "period"], as_index=False)["open_qty"].sum()
            .rename(columns={"open_qty": "supply_qty"}))


def _opening_position(inventory: pd.DataFrame) -> dict:
    if inventory is None or not len(inventory):
        return {}
    columns = [c for c in ("qty_on_hand", "qty_in_transit") if c in inventory.columns]
    if not columns:
        return {}
    frame = inventory.copy()
    frame["sku"] = frame["sku"].astype(str)
    total = sum(pd.to_numeric(frame[c], errors="coerce").fillna(0.0) for c in columns)
    return frame.assign(_qty=total).groupby("sku")["_qty"].sum().to_dict()


def _safety_target(safety_stock: pd.DataFrame) -> dict:
    if safety_stock is None or not len(safety_stock) or "safety_stock" not in safety_stock:
        return {}
    frame = safety_stock[["sku", "safety_stock"]].dropna().copy()
    frame["sku"] = frame["sku"].astype(str)
    return frame.drop_duplicates("sku").set_index("sku")["safety_stock"].to_dict()


# ── The projection ───────────────────────────────────────────────────────────


def _project(demand, supply, opening, target, periods, cost) -> pd.DataFrame:
    """
    Roll each SKU forward through the horizon, one period at a time.

    Per SKU and then summed, never on the aggregate: netting first would let one
    item's surplus cover another's shortage and report a business in balance that
    cannot ship an order.
    """
    grid = (demand.groupby(["sku", "period"], as_index=False)
            .agg(demand_qty=("forecast_qty", "sum"),
                 demand_cogs=("demand_cogs", "sum")))
    grid = grid.merge(supply, on=["sku", "period"], how="outer")
    for column in ("demand_qty", "demand_cogs", "supply_qty"):
        grid[column] = pd.to_numeric(grid.get(column), errors="coerce").fillna(0.0)
    grid["period"] = grid["period"].astype(str)
    grid = grid[grid["period"].isin(periods)]

    rows = []
    for sku, part in grid.groupby("sku", sort=False):
        unit = cost.get(sku, np.nan)
        position = float(opening.get(sku, 0.0))
        floor = float(target.get(sku, 0.0))
        # Only the three numeric columns. `part` still carries `sku`, and filling an
        # object column with a float is the downcast pandas 2 warns about and pandas 3
        # changes the behaviour of — a silent dtype shift in the middle of a projection
        # is exactly the kind of thing that would be noticed a version too late.
        part = (part[["period", "demand_qty", "demand_cogs", "supply_qty"]]
                .set_index("period").reindex(periods).astype(float).fillna(0.0))
        for period in periods:
            demand_qty = float(part.at[period, "demand_qty"])
            supply_qty = float(part.at[period, "supply_qty"])
            position = position + supply_qty - demand_qty
            # The shortfall against the policy position, not against zero. A period is
            # short where service is at risk, which is above an empty shelf.
            gap_qty = max(0.0, floor - position)
            rows.append({
                "sku": sku, "period": period,
                "demand_qty": demand_qty,
                "demand_cogs": float(part.at[period, "demand_cogs"]),
                "supply_qty": supply_qty,
                "supply_value": supply_qty * unit if pd.notna(unit) else 0.0,
                # Closing is floored at zero for valuation: a negative position is a
                # backorder, not negative money on the balance sheet, and letting it
                # subtract would net one SKU's shortage off another's stock.
                "closing_qty": position,
                "closing_value": max(0.0, position) * unit if pd.notna(unit) else 0.0,
                "gap_qty": gap_qty,
                "gap_value": gap_qty * unit if pd.notna(unit) else 0.0,
                "safety_stock": floor,
            })
    return pd.DataFrame(rows)


def _by_family(per_sku: pd.DataFrame, attributes) -> pd.DataFrame:
    """The same balance one level up, for the meeting that is held by product line."""
    if not len(per_sku) or attributes is None or not len(attributes):
        return pd.DataFrame()
    keys = [c for c in ("business_unit", "product_family") if c in attributes.columns]
    if not keys:
        return pd.DataFrame()
    dims = attributes[["sku"] + keys].copy()
    dims["sku"] = dims["sku"].astype(str)
    merged = per_sku.merge(dims.drop_duplicates("sku"), on="sku", how="left")
    return (merged.groupby(keys + ["period"], as_index=False)
            .agg(demand_cogs=("demand_cogs", "sum"),
                 supply_value=("supply_value", "sum"),
                 closing_value=("closing_value", "sum"),
                 gap_value=("gap_value", "sum"))
            .sort_values(keys + ["period"]))
