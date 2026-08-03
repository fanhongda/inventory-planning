"""
Should-be inventory — what stock *ought* to be, given the policy in force.

The planner's core operation. Compute it, compare to actual, and the gap is the
analysis. Everything else in this module exists to make that gap attributable.

The decomposition matters more than the total, because each part answers to a
different lever:

    cycle    = D · R          review frequency, lot size
    safety   = z·√((R+LT)·σ_D² + D̄²·σ_LT²)   service level, forecast accuracy, LT reliability
    pipeline = D · transit    incoterm, transport mode, supplier location

Two conventions are deliberately *not* hard-coded, because the planner's practice and
the textbook disagree and both are defensible:

  cycle_stock_basis      `peak` (D·R) is the level right after receipt and matches how
                         a target ceiling is managed. `average` (D·R/2) is the
                         time-average and is what a DIOH snapshot should be compared
                         to. The difference is a factor of two on cycle stock, so it
                         is a stated convention, not an assumption.

  pipeline_basis         Goods the supplier has not shipped are on nobody's finished-
                         goods books, so they never count. Goods in transit count only
                         when the incoterm passes title at origin (EXW/FOB/CIF) — the
                         same rule the inventory projector already applies to actual
                         GIT, reused here so should-be and actual are measured on the
                         same boundary. Comparing them otherwise is meaningless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .parameters import ParameterSet

# Columns the engine needs on the SKU frame, and what each is for.
REQUIRED = {
    "sku": "identity",
    "demand_mean": "monthly unconditional mean demand",
    "demand_sigma": "monthly forecast error (RMSE preferred) or demand std",
    "lead_time_days": "mean replenishment lead time",
    "review_period_days": "from planning parameters",
    "service_level": "from planning parameters",
}

DAYS_PER_MONTH = 30.0


@dataclass
class ShouldBeResult:
    """Per-SKU should-be inventory with its decomposition and the actual gap."""

    frame: pd.DataFrame
    conventions: Dict[str, Any]
    unpriced_skus: int = 0
    missing_lt_skus: int = 0
    non_stocking_skus: int = 0

    @property
    def total_should_be_value(self) -> float:
        return float(self.frame["should_be_value"].sum())

    @property
    def total_actual_value(self) -> float:
        return float(self.frame["actual_value"].sum())

    @property
    def gap_value(self) -> float:
        """Net position against policy. Positive = holding more than required."""
        return self.total_actual_value - self.total_should_be_value

    @property
    def excess_value(self) -> float:
        """
        Capital tied up in SKUs that are over policy — the sum of positive gaps only.

        Never the net. Overstock on one SKU does not offset a shortage on another;
        the units are not interchangeable and neither is the cash, since the excess is
        already spent and the shortage still has to be bought. Netting them reports
        "no excess" for a warehouse holding six figures of it.
        """
        gaps = self.frame["gap_value"]
        return float(gaps[gaps > 0].sum())

    @property
    def shortfall_value(self) -> float:
        """Value needed to bring under-stocked SKUs up to policy (positive number)."""
        gaps = self.frame["gap_value"]
        return float(-gaps[gaps < 0].sum())

    def summary(self) -> str:
        f = self.frame
        parts = [
            ("cycle", f["cycle_value"].sum()),
            ("safety", f["safety_value"].sum()),
            ("pipeline (buyer-owned)", f["pipeline_value"].sum()),
        ]
        total = self.total_should_be_value
        lines = [
            "  Should-be inventory",
            "  " + "-" * 58,
            f"    basis: cycle={self.conventions.get('cycle_stock_basis')}, "
            f"pipeline={self.conventions.get('pipeline_basis')}, "
            f"SS exposure={self.conventions.get('safety_stock_exposure')}",
            "",
        ]
        for name, value in parts:
            share = value / total if total else 0
            lines.append(f"    {name:<24} ${value:>13,.0f}   {share:>5.0%}")
        lines.append(f"    {'':<24} {'':>14}   {'':>5}")
        lines.append(f"    {'SHOULD-BE':<24} ${total:>13,.0f}")
        lines.append(f"    {'ACTUAL':<24} ${self.total_actual_value:>13,.0f}")

        gap = self.gap_value
        verdict = "above policy" if gap > 0 else "below policy"
        lines.append(f"    {'NET':<24} ${gap:>13,.0f}   {verdict}")

        over = f[f["gap_value"] > 0]
        under = f[f["gap_value"] < 0]
        lines.append("")
        # The gross figures are the actionable ones. The net is a single number that
        # describes no SKU: excess is cash already spent, shortfall is cash still to
        # spend, and neither cancels the other.
        lines.append(f"    {'EXCESS':<24} ${self.excess_value:>13,.0f}   "
                     f"tied up across {len(over)} SKUs")
        lines.append(f"    {'SHORTFALL':<24} ${self.shortfall_value:>13,.0f}   "
                     f"needed across {len(under)} SKUs")

        if self.non_stocking_skus:
            lines.append(f"    · {self.non_stocking_skus} SKUs are non-stocking "
                         f"(order on demand) — no cycle or safety stock is planned for them")
        if self.unpriced_skus:
            lines.append(f"    ⚠ {self.unpriced_skus} SKUs have no unit cost — "
                         f"counted in units only, excluded from all value totals")
        if self.missing_lt_skus:
            lines.append(f"    ⚠ {self.missing_lt_skus} SKUs have no lead time — "
                         f"pipeline and safety stock understated")
        return "\n".join(lines)


class ShouldBeCalculator:
    """Computes should-be inventory from resolved parameters and demand statistics."""

    def __init__(self, config_dir: Path = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parents[2] / "config"
        rules_path = self.config_dir / "incoterm_rules.json"
        data = json.loads(rules_path.read_text()) if rules_path.exists() else {}
        self.incoterm_rules: Dict[str, Any] = data.get("rules", {})
        self.default_incoterm: str = data.get("default_incoterm", "FOB")

    # ── Public API ───────────────────────────────────────────────────────────

    def calculate(self, params: ParameterSet, actual: pd.DataFrame = None) -> ShouldBeResult:
        """
        `params.frame` must carry demand_mean, demand_sigma, lead_time_days and the
        resolved policy parameters. `actual` supplies on-hand and in-transit so the gap
        can be measured on the same ownership boundary as should-be.
        """
        df = params.frame.copy()
        conventions = params.conventions
        self._check_required(df)

        df["demand_mean"] = pd.to_numeric(df["demand_mean"], errors="coerce").fillna(0.0)
        df["demand_sigma"] = pd.to_numeric(df["demand_sigma"], errors="coerce").fillna(0.0)
        df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce")
        df["lt_sigma_days"] = _column(df, "lt_sigma_days", 0.0)

        missing_lt = int(df["lead_time_days"].isna().sum())
        df["lead_time_days"] = df["lead_time_days"].fillna(0.0)

        df["daily_demand"] = df["demand_mean"] / DAYS_PER_MONTH

        df["cycle_qty"] = self._cycle(df, conventions)
        df["safety_qty"] = self._safety(df, conventions)
        df["pipeline_qty"] = self._pipeline(df, conventions)

        # A SKU classified non-stocking is ordered on demand — the business has already
        # decided to hold none of it. Computing a policy stock for it anyway inflates
        # should-be by the whole long tail and turns a healthy position into a spurious
        # "below policy" gap. Goods already in transit are still owned, so pipeline stays.
        non_stocking = self._non_stocking_mask(df)
        if non_stocking.any():
            df.loc[non_stocking, ["cycle_qty", "safety_qty"]] = 0.0

        df["should_be_qty"] = df["cycle_qty"] + df["safety_qty"] + df["pipeline_qty"]

        df, unpriced = self._to_value(df)
        df = self._attach_actual(df, actual)

        df["gap_qty"] = df["actual_qty"] - df["should_be_qty"]
        df["gap_value"] = df["actual_value"] - df["should_be_value"]
        # Coverage ratio reads more naturally than a raw gap when comparing SKUs of
        # very different sizes: 1.0 is exactly at policy, 2.0 is double what is needed.
        df["coverage_ratio"] = np.where(
            df["should_be_qty"] > 0, df["actual_qty"] / df["should_be_qty"], np.nan
        )

        annual_demand = df["demand_mean"] * 12
        df["should_be_dioh"] = np.where(
            annual_demand > 0,
            df["should_be_qty"] / (annual_demand / conventions.get("days_per_year", 365)),
            np.nan,
        )
        df["actual_dioh"] = np.where(
            annual_demand > 0,
            df["actual_qty"] / (annual_demand / conventions.get("days_per_year", 365)),
            np.nan,
        )

        return ShouldBeResult(
            frame=df,
            conventions=dict(conventions),
            unpriced_skus=unpriced,
            missing_lt_skus=missing_lt,
            non_stocking_skus=int(non_stocking.sum()),
        )

    @staticmethod
    def _non_stocking_mask(df: pd.DataFrame) -> pd.Series:
        """SKUs the stocking policy has already excluded from being held."""
        if "stocking_class" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["stocking_class"].astype(str).str.lower().eq("non-stocking")

    # ── Components ───────────────────────────────────────────────────────────

    @staticmethod
    def _cycle(df: pd.DataFrame, conventions: Dict[str, Any]) -> pd.Series:
        """
        Stock held purely because orders arrive in batches rather than continuously.

        For a min-max / (s,S) policy the batch is the order quantity, not the review
        period, so the lot size drives cycle stock instead. Where an EOQ or MOQ has
        been resolved it is used; otherwise the review period stands in.
        """
        basis = conventions.get("cycle_stock_basis", "peak")
        divisor = 1.0 if basis == "peak" else 2.0

        cycle = df["daily_demand"] * df["review_period_days"] / divisor

        if "replenishment_method" in df.columns and "order_qty" in df.columns:
            lot_based = df["replenishment_method"].isin(["min_max", "reorder_point"])
            lot = pd.to_numeric(df["order_qty"], errors="coerce")
            cycle = cycle.where(~(lot_based & lot.notna()), lot / divisor)

        return cycle.clip(lower=0).round(1)

    @staticmethod
    def _safety(df: pd.DataFrame, conventions: Dict[str, Any]) -> pd.Series:
        """
        Combined demand and lead-time variability (MIT CTL §7):

            SS = z · √(L · σ_D² + D̄² · σ_LT²)

        The exposure period L is R + LT under periodic review, not LT. Using LT alone
        understates safety stock by √((R+LT)/LT) — 41% for a monthly review with a
        30-day lead time — because it ignores that a shortfall discovered just after a
        review cannot be acted on until the next one.
        """
        exposure_days = df["lead_time_days"].copy()
        if conventions.get("safety_stock_exposure", "review_plus_lt") == "review_plus_lt":
            periodic = df.get("replenishment_method", pd.Series("periodic", index=df.index))
            # Continuous-review policies react immediately, so their exposure really is
            # the lead time alone.
            exposure_days = exposure_days + df["review_period_days"].where(
                periodic.isin(["periodic", "min_max"]), 0
            )

        exposure_months = exposure_days / DAYS_PER_MONTH
        lt_sigma_months = df["lt_sigma_days"] / DAYS_PER_MONTH

        z = df["service_level"].map(_z_from_service_level).astype(float)

        variance = (
            exposure_months * df["demand_sigma"] ** 2
            + df["demand_mean"] ** 2 * lt_sigma_months ** 2
        )
        return (z * np.sqrt(variance.clip(lower=0))).fillna(0.0).round(1)

    def _pipeline(self, df: pd.DataFrame, conventions: Dict[str, Any]) -> pd.Series:
        """
        Buyer-owned goods in transit.

        Only the transit leg counts, and only when the incoterm passes title at origin.
        What the supplier has not yet shipped is on nobody's finished-goods books, so
        the common shortcut of D·LT overstates the balance sheet — often by a lot, since
        production usually dominates the lead time.
        """
        if conventions.get("pipeline_basis") == "none":
            return pd.Series(0.0, index=df.index)

        transit_days = self._transit_days(df, conventions)

        if conventions.get("pipeline_basis") == "incoterm_aware":
            incoterm = (
                df.get("incoterm", pd.Series(None, index=df.index))
                .fillna(self.default_incoterm)
                .astype(str).str.upper().str.strip()
            )
            buyer_owned = incoterm.map(
                lambda t: bool(
                    self.incoterm_rules.get(t, self.incoterm_rules.get(self.default_incoterm, {}))
                    .get("include_git_in_inventory", True)
                )
            )
            transit_days = transit_days.where(buyer_owned, 0.0)

        return (df["daily_demand"] * transit_days).clip(lower=0).round(1)

    @staticmethod
    def _transit_days(df: pd.DataFrame, conventions: Dict[str, Any]) -> pd.Series:
        """
        Split the lead time into the supplier's leg and the transit leg.

        Measured transit time is used when the data supports it; otherwise the
        configured share of lead time applies. The share is a stated convention rather
        than a silent constant because it moves the pipeline figure directly.
        """
        if "transit_days" in df.columns:
            measured = pd.to_numeric(df["transit_days"], errors="coerce")
            if measured.notna().any():
                share = float(conventions.get("transit_share_of_lt", 0.45))
                return measured.fillna(df["lead_time_days"] * share)

        share = float(conventions.get("transit_share_of_lt", 0.45))
        return df["lead_time_days"] * share

    # ── Valuation and actuals ────────────────────────────────────────────────

    @staticmethod
    def _to_value(df: pd.DataFrame) -> tuple:
        """
        Convert quantities to money.

        SKUs without a unit cost are left at zero value rather than imputed. An
        invented cost would flow into the inventory target and quietly change what the
        pipeline recommends buying, so the gap is reported instead of filled.
        """
        cost = _column(df, "unit_cost", np.nan, fill=False)
        unpriced = int(cost.isna().sum())
        cost_filled = cost.fillna(0.0)

        for part in ("cycle", "safety", "pipeline", "should_be"):
            df[f"{part}_value"] = (df[f"{part}_qty"] * cost_filled).round(2)
        df["unit_cost"] = cost
        df["has_cost"] = cost.notna()
        return df, unpriced

    def _attach_actual(self, df: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        """
        Bring in real stock on the *same ownership boundary* as should-be.

        On-hand always counts. In-transit counts only where the incoterm says the buyer
        owns it — the identical test applied to the pipeline component. Measuring the
        two sides differently is the easiest way to manufacture a gap that is not there.
        """
        if actual is None or len(actual) == 0:
            df["actual_qty"] = 0.0
            df["actual_value"] = 0.0
            df["actual_on_hand"] = 0.0
            df["actual_in_transit_owned"] = 0.0
            return df

        cols = ["sku", "qty_on_hand"]
        if "qty_in_transit" in actual.columns:
            cols.append("qty_in_transit")
        if "incoterm" in actual.columns and "incoterm" not in df.columns:
            cols.append("incoterm")

        agg = actual[cols].groupby("sku", as_index=False).sum(numeric_only=True)
        df = df.merge(agg, on="sku", how="left", suffixes=("", "_actual"))

        df["actual_on_hand"] = _column(df, "qty_on_hand", 0.0)

        git = _column(df, "qty_in_transit", 0.0)
        incoterm = (
            df.get("incoterm", pd.Series(None, index=df.index))
            .fillna(self.default_incoterm).astype(str).str.upper().str.strip()
        )
        buyer_owned = incoterm.map(
            lambda t: bool(
                self.incoterm_rules.get(t, self.incoterm_rules.get(self.default_incoterm, {}))
                .get("include_git_in_inventory", True)
            )
        )
        df["actual_in_transit_owned"] = git.where(buyer_owned, 0.0)

        df["actual_qty"] = df["actual_on_hand"] + df["actual_in_transit_owned"]
        df["actual_value"] = (df["actual_qty"] * df["unit_cost"].fillna(0.0)).round(2)
        return df

    @staticmethod
    def _check_required(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            detail = ", ".join(f"{c} ({REQUIRED[c]})" for c in missing)
            raise ValueError(f"Should-be calculation needs column(s): {detail}")


# ── Service level → z ────────────────────────────────────────────────────────

# Cached so the calculation does not depend on scipy being present. Values are the
# standard normal inverse CDF at the service levels a planner actually uses.
_Z_TABLE = {
    0.50: 0.000, 0.75: 0.674, 0.80: 0.842, 0.85: 1.036, 0.90: 1.282,
    0.91: 1.341, 0.92: 1.405, 0.93: 1.476, 0.94: 1.555, 0.95: 1.645,
    0.96: 1.751, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326, 0.995: 2.576,
    0.999: 3.090,
}


def _column(df: pd.DataFrame, name: str, default: Any, fill: bool = True) -> pd.Series:
    """
    A numeric column, or a full-length Series of `default` when it is absent.

    `df.get(name, 0.0)` returns the bare scalar when the column is missing, and every
    Series operation after it then fails — which happens the moment a real export omits
    an optional column such as in-transit quantity.
    """
    if name in df.columns:
        series = pd.to_numeric(df[name], errors="coerce")
    else:
        series = pd.Series(default, index=df.index, dtype="float64")
    return series.fillna(default) if fill else series


def _z_from_service_level(sl: Any) -> float:
    """Standard-normal z for a cycle service level, interpolating between table points."""
    if sl is None or (isinstance(sl, float) and np.isnan(sl)):
        return 0.0
    try:
        sl = float(sl)
    except (TypeError, ValueError):
        return 0.0
    if sl <= 0.5:
        return 0.0
    if sl >= 0.999:
        return _Z_TABLE[0.999]
    if sl in _Z_TABLE:
        return _Z_TABLE[sl]

    keys = sorted(_Z_TABLE)
    lower = max(k for k in keys if k <= sl)
    upper = min(k for k in keys if k >= sl)
    if lower == upper:
        return _Z_TABLE[lower]
    weight = (sl - lower) / (upper - lower)
    return _Z_TABLE[lower] + weight * (_Z_TABLE[upper] - _Z_TABLE[lower])
