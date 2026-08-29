"""
The sales & operations planning worksheet.

## What this is for

The forecast the pipeline produces is a planning input. It is not a document a sales
organisation will look at, and until somebody in sales has looked at it the plan rests
on extrapolated history alone — which knows nothing about the contract signed last
week, the competitor who just left the market, or the project that slipped a quarter.

So the forecast is written out a second time, in a shape a sales manager reads rather
than a shape a join consumes: one row per line of business, history and forecast side
by side, **in units and in money**. Money is the point. A planner argues about units
and a sales manager argues about revenue, and a forecast presented only in units gets
reviewed by nobody.

## Why the amount is computed rather than forecast

The forward amount is `forecast_qty × ASP`, not a separately modelled revenue series.
Two series modelled independently disagree — the units say 400 and the money says 380
units' worth — and nothing reconciles them. One series and one price is a number a
planner and a sales manager can both hold, and where the price is the questionable
part, the column says which price it is.

## The split, and what it costs

Where the extract carries a country, the sheet is split by it: one distribution centre
serves several markets, and a country growing while another shrinks reads as flat
demand at the aggregate. The split is **top-down** — the SKU's forecast apportioned by
each country's share of its recent history — rather than a separate model per country
cell.

That is a real choice with a real cost. Top-down cannot see a country diverging from
the SKU's overall trend; only a model fitted to that cell can, and this does not fit
one. What it buys is coherence for free: the parts sum to the total exactly, so the
country sheets and the SKU plan can never disagree, and the replenishment that runs on
the SKU total is planning the same demand the sales review adjusted. A bottom-up
alternative has to be reconciled back, and an unreconciled hierarchy is how a business
ends up buying to one number and selling to another.

`split_basis` on every row says which it was, so the day a cell earns its own model the
sheet can say that instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Trailing window the average selling price is measured over. Twelve months rather
# than three: a price is being applied to a six-month horizon, and three months of a
# seasonal item is a promotional price extrapolated into a full year.
_PRICE_MONTHS = 12
# ...but three is the fallback when twelve has nothing in it, which happens on an item
# that only started selling recently — exactly the item whose forward revenue matters
# most to whoever is reviewing.
_PRICE_MONTHS_FALLBACK = 3
# A country holding less than this share of a SKU's history is not given its own row:
# apportioning 0.4% of a forecast produces a line of noise that a reviewer has to read
# past, and the quantity rounds to nothing anyway. It is folded into the SKU's largest
# country rather than dropped, so the parts still sum to the total.
_MIN_COUNTRY_SHARE = 0.02


@dataclass
class PriceBasis:
    """Per-SKU selling price and the honest label for where it came from."""

    frame: pd.DataFrame
    counts: Dict[str, int] = dc_field(default_factory=dict)

    def summary(self) -> str:
        if not self.counts:
            return "  Selling price: no revenue column in the demand history"
        lines = ["  Selling price — how the forward amount was valued", "  " + "-" * 58]
        for basis, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {n:>6,} SKUs   {basis}")
        if self.counts.get("none", 0):
            lines.append(
                "    SKUs with no price carry a quantity forecast and no amount. They "
                "are left blank rather than valued at zero — a zero would sum into the "
                "total and understate it silently."
            )
        return "\n".join(lines)


def average_selling_price(sales_df: pd.DataFrame,
                          unit_cost: pd.DataFrame = None,
                          months: int = _PRICE_MONTHS) -> PriceBasis:
    """
    What each SKU actually sold for, per unit, over the trailing window.

    Realised revenue over realised units, not a list price and not a mean of line
    prices. A mean of line prices weights a one-unit sample the same as a thousand-unit
    one; the ratio of the two totals is what the business actually achieved.

    The fallback chain descends by how much the number is worth, and every step is
    recorded rather than blended away. A price from the item master's standard cost is
    a *cost*, and the label says so — a reviewer who sees `standard cost (not a price)`
    against a line knows the margin in that row is zero by construction.
    """
    empty = pd.DataFrame(columns=["sku", "asp", "price_basis"])
    if sales_df is None or not len(sales_df) or "amount" not in sales_df.columns:
        return PriceBasis(frame=empty)

    df = sales_df.dropna(subset=["sku"]).copy()
    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[(df["qty"] > 0) & df["amount"].notna()]
    if not len(df) or "demand_date" not in df.columns:
        return PriceBasis(frame=empty)

    latest = pd.to_datetime(df["demand_date"]).max()

    def _window(n_months: int) -> pd.DataFrame:
        cutoff = latest - pd.DateOffset(months=n_months)
        window = df[pd.to_datetime(df["demand_date"]) > cutoff]
        if not len(window):
            return pd.DataFrame(columns=["sku", "asp"])
        agg = window.groupby("sku")[["amount", "qty"]].sum()
        agg = agg[agg["qty"] > 0]
        return pd.DataFrame({"sku": agg.index, "asp": agg["amount"] / agg["qty"]})

    layers = [
        (f"realised, trailing {months}m", _window(months)),
        (f"realised, trailing {_PRICE_MONTHS_FALLBACK}m", _window(_PRICE_MONTHS_FALLBACK)),
    ]
    whole = df.groupby("sku")[["amount", "qty"]].sum()
    whole = whole[whole["qty"] > 0]
    layers.append(("realised, whole history",
                   pd.DataFrame({"sku": whole.index,
                                 "asp": whole["amount"] / whole["qty"]})))
    if unit_cost is not None and len(unit_cost) and "unit_cost" in unit_cost.columns:
        cost = unit_cost.dropna(subset=["sku", "unit_cost"]).drop_duplicates("sku")
        layers.append(("standard cost (not a price)",
                       pd.DataFrame({"sku": cost["sku"].astype(str),
                                     "asp": pd.to_numeric(cost["unit_cost"],
                                                          errors="coerce")})))

    resolved: Dict[str, tuple] = {}
    for basis, frame in layers:
        if frame is None or not len(frame):
            continue
        for sku, asp in zip(frame["sku"].astype(str), frame["asp"]):
            if sku in resolved or not np.isfinite(asp) or asp <= 0:
                continue
            resolved[sku] = (float(asp), basis)

    every_sku = set(df["sku"].astype(str))
    rows = [{"sku": sku,
             "asp": resolved.get(sku, (np.nan, "none"))[0],
             "price_basis": resolved.get(sku, (np.nan, "none"))[1]}
            for sku in sorted(every_sku)]
    frame = pd.DataFrame(rows)
    return PriceBasis(frame=frame,
                      counts=frame["price_basis"].value_counts().to_dict())


def value_series(sales_df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Revenue per SKU per period — the money twin of the demand time series.

    Built from what was invoiced rather than from quantity × today's price, so the
    history column shows the price the business was getting then. A history valued at
    the current price hides exactly the thing a sales review is looking for: that the
    units held up and the money did not.
    """
    if (sales_df is None or not len(sales_df) or "amount" not in sales_df.columns
            or "demand_date" not in sales_df.columns):
        return pd.DataFrame()
    df = sales_df.dropna(subset=["sku", "demand_date"]).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"].notna()]
    if not len(df):
        return pd.DataFrame()
    df["period"] = pd.to_datetime(df["demand_date"]).dt.to_period(freq)
    return (df.groupby(["period", "sku"])["amount"].sum().reset_index()
            .pivot(index="period", columns="sku", values="amount")
            .fillna(0.0).sort_index())


def country_shares(sales_df: pd.DataFrame, months: int = _PRICE_MONTHS) -> pd.DataFrame:
    """
    Each SKU's recent demand split by country, as shares summing to one.

    Measured on the trailing window rather than on all history, because a market
    entered two years ago and a market exited last year should not weigh the same. The
    shares are what the top-down split apportions by, so a share that is stale is a
    forecast pointed at the wrong country.
    """
    if (sales_df is None or not len(sales_df) or "country" not in sales_df.columns
            or "demand_date" not in sales_df.columns):
        return pd.DataFrame(columns=["sku", "country", "share"])

    df = sales_df.dropna(subset=["sku", "country", "demand_date"]).copy()
    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    df = df[df["qty"] > 0]
    if not len(df):
        return pd.DataFrame(columns=["sku", "country", "share"])

    latest = pd.to_datetime(df["demand_date"]).max()
    cutoff = latest - pd.DateOffset(months=months)
    recent = df[pd.to_datetime(df["demand_date"]) > cutoff]

    # Per SKU, not globally. A first pass fell back to the whole history only when the
    # *entire* window was empty, so an item that last sold fourteen months ago got no
    # share at all and landed on the sheet as "(not stated)" — 259 of them on one real
    # extract, every one a SKU with a perfectly good country history just outside the
    # window. Slow movers are exactly the items whose demand is old, and dropping their
    # geography is dropping it for the items a review is least sure about.
    fresh = set(recent["sku"].astype(str))
    stale = df[~df["sku"].astype(str).isin(fresh)]
    window = pd.concat([recent, stale], ignore_index=True) if len(stale) else recent
    if not len(window):
        window = df

    grouped = window.groupby(["sku", "country"])["qty"].sum().reset_index()
    totals = grouped.groupby("sku")["qty"].transform("sum")
    grouped["share"] = grouped["qty"] / totals.replace(0, np.nan)
    grouped = grouped.dropna(subset=["share"])

    # Fold the slivers into the SKU's largest market rather than dropping them, so the
    # shares still sum to one and the country rows still add back to the SKU total.
    out = []
    for sku, part in grouped.groupby("sku"):
        part = part.sort_values("share", ascending=False)
        keep = part[part["share"] >= _MIN_COUNTRY_SHARE]
        if not len(keep):
            keep = part.head(1)
        residual = 1.0 - float(keep["share"].sum())
        keep = keep.copy()
        keep.iloc[0, keep.columns.get_loc("share")] += residual
        out.append(keep[["sku", "country", "share"]])
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["sku", "country", "share"]))


# ── The sheet ────────────────────────────────────────────────────────────────


@dataclass
class SOPResult:
    """The worksheet, and everything needed to say what is in it."""

    sheet: pd.DataFrame
    price: PriceBasis
    horizon_periods: List[str] = dc_field(default_factory=list)
    history_periods: List[str] = dc_field(default_factory=list)
    split_by_country: bool = False

    def summary(self) -> str:
        if not len(self.sheet):
            return "  S&OP worksheet: nothing to review"
        rows = len(self.sheet)
        skus = self.sheet["sku"].nunique()
        lines = [
            "  S&OP worksheet — history, forecast, and what both are worth",
            "  " + "-" * 58,
            f"    {rows:,} review rows over {skus:,} SKUs, "
            f"{len(self.history_periods)} months of history and "
            f"{len(self.horizon_periods)} ahead",
        ]
        if self.split_by_country:
            countries = self.sheet["country"].nunique()
            lines.append(
                f"    Split across {countries} countries, top-down from each SKU's "
                f"forecast by its share of recent demand — the parts sum to the total "
                f"by construction, and `split_basis` on the row says so."
            )
        value = pd.to_numeric(self.sheet.get("forecast_amount_total"),
                              errors="coerce").sum()
        if value:
            lines.append(f"    Forecast value over the horizon: {value:,.0f}")
        # The classification gap, counted on the sheet itself rather than only at the
        # gate. A number a reader cannot reconcile against the report it describes is a
        # number they will eventually stop believing.
        if "product_family_source" in self.sheet.columns:
            guessed = self.sheet.loc[self.sheet["product_family_source"] == "inferred",
                                     "sku"].nunique()
            if guessed:
                lines.append(
                    f"    {guessed:,} of {skus:,} SKUs have no product family in the "
                    f"master — theirs is guessed from the part number. Filter on "
                    f"`product_family_source == \"master\"` for a rollup containing "
                    f"only what is known."
                )

        unpriced = int((self.sheet.get("price_basis") == "none").sum()) \
            if "price_basis" in self.sheet.columns else 0
        if unpriced:
            lines.append(
                f"    {unpriced:,} rows carry no amount — no revenue history and no "
                f"cost to fall back on. Left blank, never zero: a zero would sum into "
                f"the total and understate it without saying so."
            )
        return "\n".join(lines)


class SOPWorksheet:
    """Builds the sheet the sales organisation reviews."""

    def __init__(self, horizon: int = 6, history_months: int = 12):
        self.horizon = horizon
        # Twelve months of history against six of forecast. Enough to see last year's
        # same season beside the forecast of this one, which is the comparison a
        # reviewer makes first and cannot make on a trailing six.
        self.history_months = history_months

    def build(
        self,
        time_series: pd.DataFrame,
        forecast_detail: pd.DataFrame,
        sales_df: pd.DataFrame = None,
        attributes: pd.DataFrame = None,
        unit_cost: pd.DataFrame = None,
    ) -> SOPResult:
        """
        One row per line of business, reading left to right as time.

        `attributes` supplies the segmentation — business unit, product family,
        description — from master data. It is joined, never derived: a family invented
        here from the part number is the failure the `product_dimension` gate exists to
        stop, and reintroducing it in the review sheet would put it in front of the one
        audience least able to spot it.
        """
        price = average_selling_price(sales_df, unit_cost=unit_cost)
        if forecast_detail is None or not len(forecast_detail):
            return SOPResult(sheet=pd.DataFrame(), price=price)

        values = value_series(sales_df)
        shares = country_shares(sales_df)
        split = bool(len(shares))

        base = self._per_sku(time_series, forecast_detail, values, price)
        sheet = self._apply_split(base, shares) if split else base
        sheet = self._attach_dimensions(sheet, attributes)

        history_periods = [str(p) for p in (time_series.index[-self.history_months:]
                                            if time_series is not None else [])]
        horizon_periods = sorted(forecast_detail["period"].astype(str).unique())
        return SOPResult(
            sheet=self._order_columns(sheet, history_periods, horizon_periods),
            price=price,
            history_periods=history_periods,
            horizon_periods=horizon_periods,
            split_by_country=split,
        )

    # ── Pieces ───────────────────────────────────────────────────────────────

    def _per_sku(self, time_series, forecast_detail, values, price) -> pd.DataFrame:
        """History and forecast, in units and money, one row per SKU."""
        future_qty = (forecast_detail.pivot_table(index="sku", columns="period",
                                                  values="forecast_qty", aggfunc="first")
                      .rename(columns=lambda c: f"fcst qty {c}"))

        meta_cols = [c for c in ("model_used", "selected_by", "vs_naive", "backtest_mape")
                     if c in forecast_detail.columns]
        meta = (forecast_detail[forecast_detail.get("is_next_period", True)]
                .drop_duplicates("sku").set_index("sku")[meta_cols])

        frame = meta.join(future_qty, how="outer")

        if time_series is not None and len(time_series):
            history = time_series.tail(self.history_months).T
            history.index.name = "sku"
            frame = frame.join(history.rename(columns=lambda c: f"hist qty {c}"),
                               how="left")
        if values is not None and len(values):
            money = values.tail(self.history_months).T
            money.index.name = "sku"
            frame = frame.join(money.rename(columns=lambda c: f"hist amt {c}"),
                               how="left")

        frame = frame.reset_index().rename(columns={"index": "sku"})
        frame["sku"] = frame["sku"].astype(str)
        asp = price.frame.set_index("sku") if len(price.frame) else pd.DataFrame()
        frame["asp"] = frame["sku"].map(asp["asp"]) if len(asp) else np.nan
        frame["price_basis"] = (frame["sku"].map(asp["price_basis"]) if len(asp)
                                else "none")
        frame["price_basis"] = frame["price_basis"].fillna("none")
        return frame

    def _apply_split(self, frame: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
        """
        Fan each SKU row out to one row per country, apportioning every quantity.

        Only quantities are apportioned. The average selling price is a rate, not a
        total, so it carries across unchanged — multiplying a price by a share is the
        kind of arithmetic that produces a plausible number and no error anywhere.
        """
        qty_cols = [c for c in frame.columns
                    if c.startswith("hist qty ") or c.startswith("fcst qty ")]
        amt_cols = [c for c in frame.columns if c.startswith("hist amt ")]

        merged = frame.merge(shares, on="sku", how="left")
        # A SKU with no country at all keeps one row and says so, rather than being
        # dropped out of a sheet that is supposed to cover the business.
        merged["country"] = merged["country"].fillna("(not stated)")
        merged["share"] = pd.to_numeric(merged["share"], errors="coerce").fillna(1.0)

        for col in qty_cols + amt_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce") * merged["share"]

        merged["split_basis"] = np.where(
            merged["share"] < 1.0,
            "top-down: SKU forecast × country share of trailing 12m demand",
            "single country",
        )
        return merged.drop(columns=["share"])

    @staticmethod
    def _attach_dimensions(frame: pd.DataFrame, attributes) -> pd.DataFrame:
        """Join the segmentation from master data. Never derive it."""
        wanted = ["business_unit", "product_family", "product_family_source",
                  "description", "abc_class", "stocking_policy"]
        if attributes is None or not len(attributes):
            for col in ("business_unit", "product_family"):
                frame[col] = np.nan
            return frame
        cols = ["sku"] + [c for c in wanted if c in attributes.columns]
        merged = frame.merge(attributes[cols].drop_duplicates("sku").assign(
            sku=lambda d: d["sku"].astype(str)), on="sku", how="left")
        for col in ("business_unit", "product_family"):
            if col not in merged.columns:
                merged[col] = np.nan
        return merged

    def _order_columns(self, frame: pd.DataFrame, history_periods,
                       horizon_periods) -> pd.DataFrame:
        """
        Value the forecast, then lay the row out as time reads.

        The amount columns are computed here rather than in `_per_sku` because the
        split has to happen first: apportioning the quantity and then valuing it gives
        each country its own amount, while valuing first and apportioning after would
        give the same answer only as long as every country pays the same price — which
        is the assumption being made, and is better made once, visibly, than twice.
        """
        for period in horizon_periods:
            qty_col, amt_col = f"fcst qty {period}", f"fcst amt {period}"
            if qty_col in frame.columns:
                frame[amt_col] = (pd.to_numeric(frame[qty_col], errors="coerce")
                                  * pd.to_numeric(frame["asp"], errors="coerce"))

        hist_qty = [f"hist qty {p}" for p in history_periods if f"hist qty {p}" in frame]
        hist_amt = [f"hist amt {p}" for p in history_periods if f"hist amt {p}" in frame]
        fcst_qty = [f"fcst qty {p}" for p in horizon_periods if f"fcst qty {p}" in frame]
        fcst_amt = [f"fcst amt {p}" for p in horizon_periods if f"fcst amt {p}" in frame]

        # The totals a reviewer looks at before reading a single month, and the pair
        # they argue about: what the last year did, and what the next six are claimed
        # to do.
        # `min_count=1` so a row with nothing to add totals to blank rather than to
        # zero. The distinction is the same one the unpriced row is about: a zero is a
        # claim that the value is nil, and it sums into the sheet's grand total.
        def _total(columns):
            return (frame[columns].apply(pd.to_numeric, errors="coerce")
                    .sum(axis=1, min_count=1) if columns else np.nan)

        frame["history_qty_total"] = _total(hist_qty)
        frame["history_amount_total"] = _total(hist_amt)
        frame["forecast_qty_total"] = _total(fcst_qty)
        frame["forecast_amount_total"] = _total(fcst_amt)

        lead = [c for c in ("sku", "description", "business_unit", "product_family",
                            "product_family_source", "country", "abc_class",
                            "stocking_policy") if c in frame]
        totals = ["history_qty_total", "history_amount_total",
                  "forecast_qty_total", "forecast_amount_total"]
        provenance = [c for c in ("model_used", "selected_by", "vs_naive",
                                  "backtest_mape", "asp", "price_basis", "split_basis")
                      if c in frame]

        ordered = lead + totals + provenance + hist_qty + hist_amt + fcst_qty + fcst_amt
        rest = [c for c in frame.columns if c not in ordered]
        out = frame[ordered + rest].copy()
        sort_cols = [c for c in ("business_unit", "product_family", "country") if c in out]
        return (out.sort_values(sort_cols + ["forecast_amount_total"],
                                ascending=[True] * len(sort_cols) + [False])
                .reset_index(drop=True))
