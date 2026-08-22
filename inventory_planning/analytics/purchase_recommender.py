"""
Purchase recommender.

Net requirement for the next planning cycle:

  period_demand     = per `demand_basis` (below)
  gross_requirement = period_demand + safety_stock
  net_requirement   = max(0, gross_requirement − effective_position)

## Why the forecast and the backlog are not added together

The old formula was `forecast + safety_stock + backlog`. That over-orders twice over.

The forecast is fitted on *shipment* history. Those shipments came from orders that
were open backlog before they shipped, so the forecast already anticipates the demand
the order book represents. Adding the two counts the same units in both terms — the
error grows with how much backlog is carried, which is exactly when the buy matters
most.

Second, it assumes the whole book converts. Where customers do not collect on
schedule, the book is permanently larger than what will move, so the inflation is not
noise that averages out — it is a standing bias in one direction.

The fix is the standard MRP treatment, forecast consumption: within a period, the
forecast and the firm orders are two estimates of *one* demand, so the requirement is
the larger of them, not the sum. The backlog side is discounted first by a measured
realization rate (see `analytics.backlog_realization`) and scoped to the lines
actually due inside the horizon (`backlog_due_qty`), because a line requested five
months out is not this cycle's requirement.

`demand_basis` in `config/stocking_policy.json` selects the treatment:

  forecast_consumption    max(forecast, backlog_due × realization)   default
  max_forecast_backlog    max(forecast, backlog_due)                 no discount
  forecast_plus_backlog   forecast + backlog                         legacy, additive

Demand estimate (C2 from AUDIT.md): `forecast_next_period` (t+1 point forecast), not
`forecast_avg_monthly` — the 6-month average underestimates in uptrending demand.
Fallback: `demand_mean_rolling` when no forecast is available.
"""

import pandas as pd
import numpy as np

DEMAND_BASIS_CONSUMPTION = "forecast_consumption"
DEMAND_BASIS_MAX = "max_forecast_backlog"
DEMAND_BASIS_ADDITIVE = "forecast_plus_backlog"

VALID_DEMAND_BASIS = (DEMAND_BASIS_CONSUMPTION, DEMAND_BASIS_MAX, DEMAND_BASIS_ADDITIVE)

DAYS_PER_MONTH = 30.0


class PurchaseRecommender:

    def __init__(self, demand_basis: str = DEMAND_BASIS_CONSUMPTION,
                 horizon_days: int = 30):
        if demand_basis not in VALID_DEMAND_BASIS:
            raise ValueError(
                f"demand_basis must be one of {VALID_DEMAND_BASIS}, got {demand_basis!r}"
            )
        self.demand_basis = demand_basis
        self.horizon_days = int(horizon_days)

    def recommend(self, projection: pd.DataFrame, forecast_summary: pd.DataFrame,
                  open_so: pd.DataFrame, open_po_df: pd.DataFrame,
                  realization=None) -> pd.DataFrame:
        """
        `realization` is a `RealizationResult` from `BacklogRealizationEstimator`.
        When absent, the backlog is taken at face value — an unmeasured discount is
        never applied silently.
        """
        df = projection.copy()

        # ── Next-period demand: use t+1 point forecast, not 6-month average ──
        if forecast_summary is not None and len(forecast_summary):
            if "forecast_next_period" in forecast_summary.columns:
                next_cycle = forecast_summary[["sku", "forecast_next_period",
                                               "forecast_avg_monthly"]].copy()
                df = df.merge(next_cycle, on="sku", how="left")
                # forecast_next_period is the authoritative demand estimate for net req
                df["next_period_demand"] = df["forecast_next_period"].fillna(
                    df.get("forecast_avg_monthly", df["demand_mean_rolling"])
                )
            else:
                # Legacy path: older forecast_summary without next_period column
                df = df.merge(
                    forecast_summary[["sku", "forecast_avg_monthly"]], on="sku", how="left"
                )
                df["next_period_demand"] = df["forecast_avg_monthly"]
                df["forecast_next_period"] = df["forecast_avg_monthly"]
        else:
            df["next_period_demand"] = df["demand_mean_rolling"]
            df["forecast_next_period"] = df["demand_mean_rolling"]
            df["forecast_avg_monthly"] = df["demand_mean_rolling"]

        df["next_period_demand"] = df["next_period_demand"].fillna(df["demand_mean_rolling"])
        # The forecast is monthly; the horizon may not be.
        df["forecast_horizon_qty"] = (
            df["next_period_demand"] * self.horizon_days / DAYS_PER_MONTH
        )

        # ── Backlog ───────────────────────────────────────────────────────────
        df = self._attach_backlog(df, open_so)
        df["backlog_realization_rate"] = (
            realization.apply(df["sku"]) if realization is not None else 1.0
        )
        df["firm_demand_qty"] = (
            df["backlog_due_qty"] * df["backlog_realization_rate"]
        ).round(1)

        # ── Period demand: consumption, not addition ──────────────────────────
        if self.demand_basis == DEMAND_BASIS_ADDITIVE:
            df["period_demand"] = df["forecast_horizon_qty"] + df["backlog_qty"]
            df["demand_driver"] = "forecast+backlog"
        else:
            firm = (df["backlog_due_qty"] if self.demand_basis == DEMAND_BASIS_MAX
                    else df["firm_demand_qty"])
            df["period_demand"] = np.maximum(df["forecast_horizon_qty"], firm)
            df["demand_driver"] = np.where(
                firm > df["forecast_horizon_qty"], "backlog", "forecast"
            )
        df["period_demand"] = df["period_demand"].round(1)
        df["demand_basis"] = self.demand_basis

        # ── Net requirement ───────────────────────────────────────────────────
        df["gross_requirement"] = (df["period_demand"] + df["safety_stock"]).round(1)
        df["available_supply"] = df["effective_position"]
        df["net_requirement"] = (df["gross_requirement"] - df["available_supply"]).clip(lower=0).round(1)

        # For an order-on-demand SKU there is no policy stock to plan, so the buy is
        # whatever the realizable order book needs beyond what is already on the way.
        df["backlog_shortfall"] = (
            df["firm_demand_qty"] - df["available_supply"].fillna(0)
        ).clip(lower=0).round(1)

        def _action(row):
            if row["stocking_class"] == "non-stocking":
                return "ORDER-FOR-BACKLOG" if row["backlog_shortfall"] > 0 else "NO-ACTION"
            # Covering the period in front of you comes before every other verdict. A
            # shelf that empties before the next delivery lands is not a position to be
            # rebalanced, and the order already placed is not the answer to it — the
            # answer is to get that order moving. Ranked first so it can overrule
            # push-out, which is what it used to lose to on exactly these items.
            if row.get("supply_gap"):
                return ("EXPEDITE-INBOUND" if row["total_open_po_qty"] > 0
                        else "PURCHASE-REQUEST")
            if row["pushout_candidate"]:
                return "PUSH-OUT-OPEN-PO"
            if row["net_requirement"] > 0:
                return "PURCHASE-REQUEST"
            if row["inventory_status"] == "EXCESS":
                return "HOLD-EXCESS"
            return "HOLD-OK"

        df["recommended_action"] = df.apply(_action, axis=1)
        # ORDER-FOR-BACKLOG used to carry a quantity of zero, which is not an order.
        df["suggested_po_qty"] = np.select(
            [df["recommended_action"] == "PURCHASE-REQUEST",
             df["recommended_action"] == "ORDER-FOR-BACKLOG"],
            [df["net_requirement"], df["backlog_shortfall"]],
            default=0.0,
        )

        cols = [
            "sku", "location_id", "stocking_class", "recommended_action",
            "forecast_next_period", "forecast_avg_monthly", "forecast_horizon_qty",
            "backlog_qty", "backlog_due_qty", "backlog_past_due_qty",
            "backlog_realization_rate", "firm_demand_qty",
            "demand_basis", "demand_driver", "period_demand",
            "gross_requirement", "available_supply", "net_requirement",
            "suggested_po_qty", "pushout_open_po_qty",
            "inventory_status", "days_of_supply", "safety_stock", "should_be_inventory",
            "on_hand_cover_days", "inbound_past_due_qty", "inbound_due_qty",
            "days_to_next_arrival", "supply_gap", "supply_gap_days",
        ]
        return df[[c for c in cols if c in df.columns]].sort_values("recommended_action")

    @staticmethod
    def _attach_backlog(df: pd.DataFrame, open_so: pd.DataFrame) -> pd.DataFrame:
        """Merge the backlog summary, tolerating a summary built before the due split."""
        wanted = ["backlog_qty", "backlog_due_qty", "backlog_past_due_qty"]
        has_summary = open_so is not None and "backlog_qty" in open_so.columns
        has_due_split = has_summary and "backlog_due_qty" in open_so.columns

        if has_summary:
            df = df.merge(
                open_so[["sku"] + [c for c in wanted if c in open_so.columns]],
                on="sku", how="left",
            )

        for col in wanted:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        if not has_due_split:
            # A summary built before the due split says nothing about when the book is
            # wanted. Treating all of it as due this cycle is the conservative reading.
            df["backlog_due_qty"] = df["backlog_qty"]
        return df
