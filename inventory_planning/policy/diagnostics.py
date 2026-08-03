"""
Ordering-behaviour diagnostics and forward risk.

Two halves of the same question — what did the ordering pattern already cost, and what
is it about to cost.

Looking backwards:

  over_ordering      Bought far more than demand can absorb. Shows up twice: as stock
                     that will sit for years, and as freight paid to move it early.
  erratic_lot_size   Order quantities with no discernible policy. Every reorder is a
                     fresh decision, which is where cycle stock and expediting come from.
  chronic_air        Air freight used routinely rather than exceptionally. Air on a
                     genuine emergency is insurance; air every month is a planning
                     failure being paid for in freight, and the fix is upstream.

Looking forwards:

  stockout_risk      When each SKU runs out, given current position, inbound POs and
                     forecast demand. Ranked by value at risk, not by days — running out
                     of a cheap item is not the same event as running out of an
                     expensive one.
  slow_burn          How long current stock takes to consume. The counterpart to
                     stockout risk, and the input to deciding which inbound POs to
                     push out or cancel.

Every finding names the SKUs and the money. A diagnostic that says "ordering is erratic"
without saying which items and what it costs cannot be acted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# A SKU is chronically air-freighted above this share of its recent orders.
CHRONIC_AIR_THRESHOLD = 0.30
# Coefficient of variation of order size above which lot sizing looks unmanaged.
ERRATIC_LOT_CV = 0.60
# Days of supply beyond which a purchase is hard to justify against any policy.
OVER_ORDER_DOS = 365


@dataclass
class OrderingDiagnostics:
    """Backward-looking findings about how ordering has actually been done."""

    over_ordered: pd.DataFrame
    erratic: pd.DataFrame
    chronic_air: pd.DataFrame
    window_months: int

    @property
    def total_over_order_value(self) -> float:
        return float(self.over_ordered["excess_value"].sum()) if len(self.over_ordered) else 0.0

    @property
    def total_air_premium(self) -> float:
        return float(self.chronic_air["air_spend"].sum()) if len(self.chronic_air) else 0.0

    def summary(self) -> str:
        lines = [
            "  Ordering behaviour — last "
            f"{self.window_months} months",
            "  " + "-" * 58,
        ]
        if len(self.over_ordered):
            lines.append(
                f"    Over-ordered            {len(self.over_ordered):>4} SKUs   "
                f"${self.total_over_order_value:>12,.0f} of stock beyond any policy"
            )
            for _, row in self.over_ordered.head(5).iterrows():
                dos = row["dos_purchased"]
                # "inf days of supply" is a division artefact, not a finding. Say what
                # it actually means: bought against demand that never materialised.
                cover = (f"{dos:,.0f} days of supply" if np.isfinite(dos)
                         else "no demand at all in the window")
                lines.append(
                    f"      {row['sku']:<14} bought {row['ordered_qty']:,.0f} vs "
                    f"{row['demand_in_window']:,.0f} demand — {cover} "
                    f"(${row['excess_value']:,.0f})"
                )
        if len(self.erratic):
            lines.append(
                f"    Erratic lot sizing      {len(self.erratic):>4} SKUs   "
                f"order size has no consistent policy"
            )
            for _, row in self.erratic.head(3).iterrows():
                lines.append(
                    f"      {row['sku']:<14} {row['order_count']:.0f} orders, "
                    f"{row['min_qty']:,.0f}–{row['max_qty']:,.0f} units (CV {row['lot_cv']:.1f})"
                )
        if len(self.chronic_air):
            lines.append(
                f"    Chronic air freight     {len(self.chronic_air):>4} SKUs   "
                f"${self.total_air_premium:>12,.0f} spent on air"
            )
            for _, row in self.chronic_air.head(5).iterrows():
                lines.append(
                    f"      {row['sku']:<14} {row['air_share']:.0%} of orders by air "
                    f"({row['air_orders']:.0f}/{row['total_orders']:.0f}) — "
                    f"${row['air_spend']:,.0f}"
                )
            lines.append("      Air used routinely is a planning gap paid for in freight; "
                         "the fix is upstream of the freight decision.")
        if not (len(self.over_ordered) or len(self.erratic) or len(self.chronic_air)):
            lines.append("    No ordering anomalies found.")
        return "\n".join(lines)


@dataclass
class ForwardRisk:
    """Forward-looking position: what runs out, what will not move."""

    stockout: pd.DataFrame
    slow_burn: pd.DataFrame
    horizon_days: int
    as_of: date

    @property
    def value_at_risk(self) -> float:
        return float(self.stockout["value_at_risk"].sum()) if len(self.stockout) else 0.0

    def urgent(self, within_days: int = 30) -> pd.DataFrame:
        if not len(self.stockout):
            return self.stockout
        return self.stockout[self.stockout["days_to_stockout"] <= within_days]

    def summary(self) -> str:
        lines = [
            f"  Forward risk — next {self.horizon_days} days (from {self.as_of})",
            "  " + "-" * 58,
        ]
        if len(self.stockout):
            urgent = self.urgent(30)
            lines.append(
                f"    Stockout risk           {len(self.stockout):>4} SKUs   "
                f"${self.value_at_risk:>12,.0f} of demand at risk"
            )
            if len(urgent):
                lines.append(f"      {len(urgent)} within 30 days:")
                for _, row in urgent.head(6).iterrows():
                    inbound_day = row.get("inbound_day")
                    if pd.isna(inbound_day):
                        cover = "no inbound PO"
                    elif inbound_day > row["days_to_stockout"]:
                        # The PO exists but lands after the shelf is empty. Saying only
                        # "PO arrives day 17" reads as covered, which is the opposite.
                        cover = (f"PO arrives day {inbound_day:.0f} — "
                                 f"{inbound_day - row['days_to_stockout']:.0f}d too late")
                    else:
                        cover = f"covered by PO on day {inbound_day:.0f}"
                    lines.append(
                        f"      {row['sku']:<14} out in {row['days_to_stockout']:>3.0f}d, "
                        f"${row['value_at_risk']:>10,.0f} at risk — {cover}"
                    )
        else:
            lines.append("    Stockout risk           none within the horizon")

        if len(self.slow_burn):
            lines.append("")
            lines.append(
                f"    Slow burn               {len(self.slow_burn):>4} SKUs   "
                f"${self.slow_burn['stock_value'].sum():>12,.0f} tied up"
            )
            for _, row in self.slow_burn.head(6).iterrows():
                years = row["days_of_cover"] / 365.0
                cover = (f"{years:.1f} years of cover" if np.isfinite(years)
                         else "will never clear — no demand at all")
                inbound = ""
                if row.get("inbound_qty", 0) > 0:
                    inbound = f", {row['inbound_qty']:,.0f} more inbound"
                lines.append(
                    f"      {row['sku']:<14} {cover} "
                    f"(${row['stock_value']:,.0f}){inbound}"
                )
            still_buying = self.slow_burn[self.slow_burn.get("inbound_qty", 0) > 0]
            if len(still_buying):
                lines.append(
                    f"      ⚠ {len(still_buying)} of these still have inbound POs — "
                    f"first candidates to push out or cancel."
                )
        return "\n".join(lines)


class DiagnosticsAnalyzer:
    """Computes ordering diagnostics and forward risk."""

    def __init__(self, window_months: int = 12, horizon_days: int = 180):
        self.window_months = window_months
        self.horizon_days = horizon_days

    # ── Backward ─────────────────────────────────────────────────────────────

    def ordering(
        self,
        po_history: pd.DataFrame,
        sku_attributes: pd.DataFrame,
        as_of: date = None,
    ) -> OrderingDiagnostics:
        as_of = as_of or date.today()
        empty = OrderingDiagnostics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                    self.window_months)
        if po_history is None or len(po_history) == 0 or "sku" not in po_history.columns:
            return empty

        df = po_history.copy()
        df["po_date"] = pd.to_datetime(df.get("po_date"), errors="coerce")
        cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=self.window_months)
        recent = df[df["po_date"] >= cutoff].copy()
        if recent.empty:
            return empty

        qty_col = next((c for c in ("po_qty", "received_qty", "order_qty")
                        if c in recent.columns), None)
        if qty_col is None:
            return empty
        recent["_qty"] = pd.to_numeric(recent[qty_col], errors="coerce").fillna(0.0)

        attrs = sku_attributes.set_index("sku") if "sku" in sku_attributes.columns else sku_attributes

        return OrderingDiagnostics(
            over_ordered=self._over_ordered(recent, attrs),
            erratic=self._erratic_lots(recent),
            chronic_air=self._chronic_air(recent),
            window_months=self.window_months,
        )

    def _over_ordered(self, recent: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
        """
        Purchases that bought more days of supply than any policy could justify.

        Compared against demand over the same window, so a genuinely growing SKU is not
        flagged for buying ahead of a rising run rate.
        """
        by_sku = recent.groupby("sku")["_qty"].sum().rename("ordered_qty").reset_index()
        by_sku = by_sku.merge(
            attrs.reset_index()[["sku", "demand_mean", "unit_cost"]]
            if "demand_mean" in attrs.columns else pd.DataFrame(columns=["sku"]),
            on="sku", how="left",
        )
        if "demand_mean" not in by_sku.columns:
            return pd.DataFrame()

        by_sku["demand_in_window"] = (
            pd.to_numeric(by_sku["demand_mean"], errors="coerce").fillna(0.0) * self.window_months
        )
        daily = pd.to_numeric(by_sku["demand_mean"], errors="coerce").fillna(0.0) / 30.0
        by_sku["dos_purchased"] = np.where(
            daily > 0, by_sku["ordered_qty"] / daily, np.inf
        )
        over = by_sku[by_sku["dos_purchased"] > OVER_ORDER_DOS].copy()
        if over.empty:
            return pd.DataFrame()

        over["excess_qty"] = (over["ordered_qty"] - over["demand_in_window"]).clip(lower=0)
        over["excess_value"] = over["excess_qty"] * pd.to_numeric(
            over["unit_cost"], errors="coerce"
        ).fillna(0.0)
        return over.sort_values("excess_value", ascending=False).reset_index(drop=True)

    @staticmethod
    def _erratic_lots(recent: pd.DataFrame) -> pd.DataFrame:
        """
        Order sizes with no consistent policy.

        Needs at least four orders — with fewer, variation is not evidence of anything.
        """
        grouped = recent.groupby("sku")["_qty"].agg(
            order_count="size", mean_qty="mean", std_qty="std",
            min_qty="min", max_qty="max",
        ).reset_index()
        grouped = grouped[(grouped["order_count"] >= 4) & (grouped["mean_qty"] > 0)]
        if grouped.empty:
            return pd.DataFrame()

        grouped["lot_cv"] = grouped["std_qty"] / grouped["mean_qty"]
        erratic = grouped[grouped["lot_cv"] > ERRATIC_LOT_CV].copy()
        return erratic.sort_values("lot_cv", ascending=False).reset_index(drop=True)

    @staticmethod
    def _chronic_air(recent: pd.DataFrame) -> pd.DataFrame:
        """Air freight used as routine rather than as exception."""
        if "transport_mode" not in recent.columns:
            return pd.DataFrame()

        mode = recent["transport_mode"].astype(str).str.upper().str.strip()
        recent = recent.assign(_is_air=mode.isin(["AIR", "AIR FREIGHT", "AF", "EXPRESS"]))
        if not recent["_is_air"].any():
            return pd.DataFrame()

        freight = pd.to_numeric(recent.get("freight_cost", 0), errors="coerce").fillna(0.0)
        recent = recent.assign(_freight=freight)

        grouped = recent.groupby("sku").agg(
            total_orders=("_is_air", "size"),
            air_orders=("_is_air", "sum"),
            air_spend=("_freight", lambda s: float(s[recent.loc[s.index, "_is_air"]].sum())),
            total_freight=("_freight", "sum"),
        ).reset_index()
        grouped = grouped[grouped["total_orders"] >= 3]
        if grouped.empty:
            return pd.DataFrame()

        grouped["air_share"] = grouped["air_orders"] / grouped["total_orders"]
        chronic = grouped[grouped["air_share"] >= CHRONIC_AIR_THRESHOLD].copy()
        return chronic.sort_values("air_spend", ascending=False).reset_index(drop=True)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        sku_attributes: pd.DataFrame,
        inventory: pd.DataFrame = None,
        open_po: pd.DataFrame = None,
        open_so: pd.DataFrame = None,
        safety_stock: pd.Series = None,
        as_of: date = None,
    ) -> ForwardRisk:
        as_of = as_of or date.today()
        df = sku_attributes.copy()

        on_hand = self._sum_by_sku(inventory, "qty_on_hand")
        df["on_hand"] = df["sku"].map(on_hand).fillna(0.0)

        backlog = self._sum_by_sku(open_so, "open_qty")
        df["backlog"] = df["sku"].map(backlog).fillna(0.0)

        inbound_qty = self._sum_by_sku(open_po, "open_qty")
        df["inbound_qty"] = df["sku"].map(inbound_qty).fillna(0.0)
        df["inbound_day"] = df["sku"].map(self._first_arrival(open_po, as_of))

        df["daily_demand"] = pd.to_numeric(df["demand_mean"], errors="coerce").fillna(0.0) / 30.0
        df["unit_cost"] = pd.to_numeric(df.get("unit_cost"), errors="coerce").fillna(0.0)

        ss = df["sku"].map(safety_stock).fillna(0.0) if safety_stock is not None else 0.0
        df["safety_stock"] = ss

        # Backlog consumes stock before any future demand does.
        df["available_now"] = (df["on_hand"] - df["backlog"]).clip(lower=0)

        return ForwardRisk(
            stockout=self._stockout(df),
            slow_burn=self._slow_burn(df),
            horizon_days=self.horizon_days,
            as_of=as_of,
        )

    # Empty results keep the full schema so callers can index them unconditionally.
    # A bare pd.DataFrame() raises KeyError on `df["sku"]`, which turns "nothing to
    # report" into a crash at exactly the moment the pipeline is working correctly.
    _STOCKOUT_COLS = ["sku", "days_to_stockout", "on_hand", "backlog", "available_now",
                      "inbound_qty", "inbound_day", "daily_demand", "qty_at_risk",
                      "value_at_risk"]
    _SLOW_BURN_COLS = ["sku", "days_of_cover", "available_now", "stock_value",
                       "inbound_qty", "daily_demand"]

    def _stockout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Days until stock hits the safety level, and the demand value exposed if it does.

        Inbound POs postpone the date only if they arrive before it; a PO landing after
        the stockout does not prevent the stockout, it just shortens it.
        """
        work = df[df["daily_demand"] > 0].copy()
        if work.empty:
            return pd.DataFrame(columns=self._STOCKOUT_COLS)

        usable = (work["available_now"] - work["safety_stock"]).clip(lower=0)
        work["days_to_stockout"] = np.floor(usable / work["daily_demand"])

        arrives_in_time = (
            work["inbound_day"].notna() & (work["inbound_day"] <= work["days_to_stockout"])
        )
        with_inbound = (
            (usable + work["inbound_qty"] - work["safety_stock"] * 0) / work["daily_demand"]
        )
        work["days_to_stockout"] = np.where(
            arrives_in_time, np.floor(with_inbound), work["days_to_stockout"]
        )

        at_risk = work[work["days_to_stockout"] <= self.horizon_days].copy()
        if at_risk.empty:
            return pd.DataFrame(columns=self._STOCKOUT_COLS)

        # Value exposed is the demand that would go unserved over the rest of the
        # horizon, not the value of the stock — a stockout costs sales, not inventory.
        unserved_days = (self.horizon_days - at_risk["days_to_stockout"]).clip(lower=0)
        at_risk["qty_at_risk"] = unserved_days * at_risk["daily_demand"]
        at_risk["value_at_risk"] = at_risk["qty_at_risk"] * at_risk["unit_cost"]

        cols = ["sku", "days_to_stockout", "on_hand", "backlog", "available_now",
                "inbound_qty", "inbound_day", "daily_demand", "qty_at_risk", "value_at_risk"]
        if "abc_class" in at_risk.columns:
            cols.append("abc_class")
        return at_risk[cols].sort_values("value_at_risk", ascending=False).reset_index(drop=True)

    def _slow_burn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stock that will take an implausibly long time to consume."""
        work = df.copy()
        work["days_of_cover"] = np.where(
            work["daily_demand"] > 0,
            work["available_now"] / work["daily_demand"],
            np.inf,
        )
        work["stock_value"] = work["available_now"] * work["unit_cost"]

        slow = work[
            (work["days_of_cover"] > OVER_ORDER_DOS) & (work["available_now"] > 0)
        ].copy()
        if slow.empty:
            return pd.DataFrame(columns=self._SLOW_BURN_COLS)

        cols = ["sku", "days_of_cover", "available_now", "stock_value",
                "inbound_qty", "daily_demand"]
        if "abc_class" in slow.columns:
            cols.append("abc_class")
        return slow[cols].sort_values("stock_value", ascending=False).reset_index(drop=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sum_by_sku(frame: Optional[pd.DataFrame], col: str) -> Dict[str, float]:
        if frame is None or len(frame) == 0 or col not in frame.columns or "sku" not in frame.columns:
            return {}
        return (
            frame.assign(**{col: pd.to_numeric(frame[col], errors="coerce").fillna(0.0)})
            .groupby("sku")[col].sum().to_dict()
        )

    @staticmethod
    def _first_arrival(open_po: Optional[pd.DataFrame], as_of: date) -> Dict[str, float]:
        """Days from now until each SKU's earliest inbound receipt."""
        if open_po is None or len(open_po) == 0 or "sku" not in open_po.columns:
            return {}
        date_col = next((c for c in ("committed_delivery", "estimated_delivery", "eta")
                         if c in open_po.columns), None)
        if date_col is None:
            return {}
        dates = pd.to_datetime(open_po[date_col], errors="coerce")
        days = (dates - pd.Timestamp(as_of)).dt.days
        return (
            pd.DataFrame({"sku": open_po["sku"], "days": days})
            .dropna().groupby("sku")["days"].min().clip(lower=0).to_dict()
        )
