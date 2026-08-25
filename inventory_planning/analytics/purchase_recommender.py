"""
Purchase recommender.

The order for this cycle, under the policy the SKU is actually on:

  coverage_days     = review period + lead time      the exposure the order must survive
  period_demand     = demand over coverage_days, per `demand_basis` (below)
  reorder_point  s  = period_demand + safety_stock
  order_lot      Q  = EOQ, raised to the MOQ and the order multiple
  order_up_to    S  = s + Q                          S = s where no lot applies
  order             = S − IP  (periodic)  |  ceil((s − IP)/Q) × Q  ((s, Q))  if IP ≤ s

## Why the lead time is in there

It was not. The requirement used to be one horizon of demand — thirty days, and the
thirty was a constructor default rather than anything the SKU said — plus safety stock.
Nothing in it covered the time the supplier takes to deliver. On a SKU with a 51.7-day
lead time selling 14,398 a month, that sized the order at ~14,398 where the review
period plus the lead time asks for ~39,200: under-ordered 2.7x, silently, on exactly the
items where being short costs the most.

The other half of the pipeline already knew better. `policy/should_be.py` has always
sized safety stock on R + LT exposure, and `analytics/safety_stock.py` now does too. The
recommender reads the same `ss_exposure_days` those produced rather than deriving its
own, because two definitions of one exposure period in one pipeline is how the order
that gets raised and the stock level it is aiming at come to disagree by construction.

## Why the policy matters to the quantity

Which arithmetic is right is decided by the axes in `policy/profile.py`, and the answer
is not the same for every SKU. A periodic-review item orders the gap up to S. An item on
(s, Q) orders whole lots, because the lot is the point — its ordering cost is what put it
on that policy. An order-on-demand item holds no policy stock at all and buys against
firm backlog only. `replenishment_method` in `config/planning_parameters.md` decides
which, per SKU, by rules a planner writes and can defend.

## Where the order book sits, and why it is not in the position

`effective_position` is on hand + in transit + open PO. It deliberately does **not**
net out backorders, which is a departure from the textbook inventory position
(IP = IOH + IOO − BO − CO, MIT CTL §10). The order book enters on the demand side
instead, through `period_demand = max(forecast, backlog_due × realization)` — it raises
the level being aimed at rather than lowering the position aiming at it.

Doing both would buy the same demand twice, and the error would grow with how much
backlog is carried, which is exactly when the buy matters most. One side or the other,
and this pipeline chose the demand side because the forecast is fitted on shipments and
those shipments came from backlog — so the two are already two estimates of one demand
and belong in the same `max()`. This is a decision, not an oversight; changing it means
changing forecast consumption at the same time.

The one thing it leaves on the table: past-due backlog is demand already missed, and it
reaches the requirement only through that `max()` rather than as a debt subtracted from
the position. Where a planner wants it treated as an outstanding obligation in its own
right, that is a separate change with `backlog_past_due_qty` as its input.

`reorder_point` here is not `should_be_inventory` from the projection, and the two are
meant to differ. `should_be_inventory` is the policy target implied by demand *history*;
`reorder_point` is this cycle's trigger, built on the forecast and the order book in
front of you. They share an exposure window and a safety stock, and nothing else.

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

from ..lot_sizing import economic_order_quantity, round_to_lot
from ..policy.profile import (
    POLICY_MAKE_TO_ORDER,
    POLICY_PERIODIC,
    POLICY_REORDER_POINT,
    canonical_policy,
)

DEMAND_BASIS_CONSUMPTION = "forecast_consumption"
DEMAND_BASIS_MAX = "max_forecast_backlog"
DEMAND_BASIS_ADDITIVE = "forecast_plus_backlog"

VALID_DEMAND_BASIS = (DEMAND_BASIS_CONSUMPTION, DEMAND_BASIS_MAX, DEMAND_BASIS_ADDITIVE)

DAYS_PER_MONTH = 30.0


class PurchaseRecommender:

    def __init__(self, demand_basis: str = DEMAND_BASIS_CONSUMPTION,
                 horizon_days: int = 30, order_cost: float = 350.0,
                 holding_rate: float = 0.22):
        if demand_basis not in VALID_DEMAND_BASIS:
            raise ValueError(
                f"demand_basis must be one of {VALID_DEMAND_BASIS}, got {demand_basis!r}"
            )
        self.demand_basis = demand_basis
        # The fallback review period, used only for a SKU whose parameters never
        # reached here. It is not the planning cycle — that comes per SKU from the
        # rule engine.
        self.horizon_days = int(horizon_days)
        self.order_cost = float(order_cost)
        self.holding_rate = float(holding_rate)

    def recommend(self, projection: pd.DataFrame, forecast_summary: pd.DataFrame,
                  open_so: pd.DataFrame, open_po_df: pd.DataFrame,
                  realization=None, parameters: pd.DataFrame = None,
                  mto_schedule: pd.DataFrame = None) -> pd.DataFrame:
        """
        `realization` is a `RealizationResult` from `BacklogRealizationEstimator`.
        When absent, the backlog is taken at face value — an unmeasured discount is
        never applied silently.

        `mto_schedule` is `OpenSOReader.order_by_schedule` — when each SKU's order has
        to be *placed* to meet the dates on the book, given its lead time. It is what an
        order-on-demand item is bought against; without it the order book is scoped to a
        flat horizon and a long lead time goes unseen until it is too late to act on.

        `parameters` is the resolved per-SKU frame from `PlanningParameters.resolve`:
        review period, replenishment method, MOQ and order multiple, as the planner's
        rules decided them. Without it every SKU falls back to the constructor's review
        period on periodic review, which is the old behaviour and is stated on each row
        in `policy_source` rather than left to be guessed.
        """
        frame, order_cost, holding_rate = self._unpack_parameters(parameters)
        df = projection.copy()
        df = self._attach_policy(df, frame)

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
        # The forecast is monthly; the window the order has to survive is R + LT.
        df["forecast_horizon_qty"] = (
            df["next_period_demand"] * df["coverage_days"] / DAYS_PER_MONTH
        ).round(1)

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

        # ── Reorder point, lot size, order-up-to ──────────────────────────────
        df["gross_requirement"] = (df["period_demand"] + df["safety_stock"]).round(1)
        # The same number under the name the policy calls it. `gross_requirement` is
        # kept because every downstream report and test speaks it.
        df["reorder_point"] = df["gross_requirement"]
        df["available_supply"] = df["effective_position"]
        df["net_requirement"] = (
            df["gross_requirement"] - df["available_supply"]
        ).clip(lower=0).round(1)

        df["eoq_qty"] = economic_order_quantity(
            annual_demand=df["next_period_demand"].fillna(0.0) * 12,
            unit_cost=df["unit_cost"],
            order_cost=order_cost,
            holding_rate=holding_rate,
        ).round(0)
        # No EOQ means no priced trade-off to make, not a lot size of nothing — the
        # supplier's minimum is still a real constraint and still applies.
        df["order_lot"] = round_to_lot(
            df["eoq_qty"].fillna(df["min_order_qty"]),
            min_order_qty=df["min_order_qty"],
            order_multiple=df["order_multiple"],
        ).round(1)
        df["order_up_to"] = (df["reorder_point"] + df["order_lot"]).round(1)

        # Order-up-to and reorder point are what a min/max setting in the ERP is, so
        # they are emitted under that name too — for a (s, Q) item they are the output
        # that matters, and for the rest they are the comparison against what is set.
        df["suggested_min_qty"] = df["reorder_point"]
        df["suggested_max_qty"] = df["order_up_to"]

        df["order_quantity"] = self._order_quantity(df)

        # ── What an order-on-demand item has to buy ───────────────────────────
        #
        # There is no policy stock to plan, so the buy is whatever the realizable order
        # book needs beyond what is already on the way. The question is which part of
        # the book counts, and the answer is not "the part wanted soon" — it is the part
        # whose *order date* has arrived. Those differ by exactly the lead time, and on
        # a long-lead-time item they differ by more than the horizon, which is how a
        # line that needed ordering a fortnight ago reads as nothing to do.
        df = self._attach_mto_schedule(df, mto_schedule)
        df["mto_demand_qty"] = (
            df["mto_actionable_qty"].where(df["mto_actionable_qty"].notna(),
                                           df["backlog_due_qty"])
            * df["backlog_realization_rate"]
        ).round(1)
        df["backlog_shortfall"] = (
            df["mto_demand_qty"] - df["available_supply"].fillna(0)
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
            [df["order_quantity"], df["backlog_shortfall"]],
            default=0.0,
        )

        cols = [
            "sku", "location_id", "stocking_class", "recommended_action",
            "forecast_next_period", "forecast_avg_monthly", "forecast_horizon_qty",
            "backlog_qty", "backlog_due_qty", "backlog_past_due_qty",
            "backlog_realization_rate", "firm_demand_qty",
            "mto_order_by", "mto_order_status", "mto_actionable_qty",
            "mto_order_past_due_qty", "mto_demand_qty", "backlog_shortfall",
            "demand_basis", "demand_driver", "period_demand",
            "replenishment_method", "policy_source", "review_period_days",
            "wma_lead_time_days", "coverage_days", "coverage_basis",
            "gross_requirement", "reorder_point", "eoq_qty", "order_lot",
            "order_up_to", "suggested_min_qty", "suggested_max_qty",
            "available_supply", "net_requirement", "order_quantity",
            "suggested_po_qty", "pushout_open_po_qty",
            "inventory_status", "days_of_supply", "safety_stock", "should_be_inventory",
            "on_hand_cover_days", "inbound_past_due_qty", "inbound_due_qty",
            "days_to_next_arrival", "supply_gap", "supply_gap_days",
        ]
        return df[[c for c in cols if c in df.columns]].sort_values("recommended_action")

    # ── Policy ───────────────────────────────────────────────────────────────

    def _unpack_parameters(self, parameters):
        """
        Accept either the resolved `ParameterSet` or a plain per-SKU frame.

        Given the ParameterSet, the ordering and holding costs come from the same
        parameter file the review periods did, so the lot size a planner sees is sized
        on the costs they set rather than on a constant compiled into this module.
        """
        if parameters is None:
            return None, self.order_cost, self.holding_rate
        frame = getattr(parameters, "frame", parameters)
        defaults = getattr(parameters, "defaults", {}) or {}
        return (
            frame,
            float(defaults.get("order_cost_usd", self.order_cost)),
            float(defaults.get("holding_cost_rate", self.holding_rate)),
        )

    _POLICY_DEFAULTS = {
        "replenishment_method": POLICY_PERIODIC,
        "min_order_qty": 0.0,
        "order_multiple": 1.0,
        "unit_cost": np.nan,
    }

    def _attach_policy(self, df: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Put the review period, the method and the lot constraints on every row.

        Precedence is deliberate: the resolved parameters win, because they are the
        rules a planner wrote and can point at. What the projection already carries
        comes second — it came from the same rules one stage earlier. The constructor
        default is last, and a SKU that lands on it is marked `default` so a run can be
        read for how much of it was actually parameterised.
        """
        wanted = ["review_period_days", *self._POLICY_DEFAULTS]
        if parameters is not None and "sku" in getattr(parameters, "columns", []):
            available = [c for c in wanted if c in parameters.columns]
            incoming = parameters.drop_duplicates("sku").set_index("sku")
            for col in available:
                supplied = df["sku"].map(incoming[col])
                df[col] = supplied if col not in df.columns else supplied.where(
                    supplied.notna(), df[col])

        df["policy_source"] = np.where(
            df["review_period_days"].notna() if "review_period_days" in df.columns
            else False,
            "rules", "default",
        )
        if "review_period_days" not in df.columns:
            df["review_period_days"] = np.nan
        df["review_period_days"] = pd.to_numeric(
            df["review_period_days"], errors="coerce").fillna(self.horizon_days)

        for col, default in self._POLICY_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        df["replenishment_method"] = df["replenishment_method"].map(canonical_policy)
        for col in ("min_order_qty", "order_multiple", "unit_cost"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["min_order_qty"] = df["min_order_qty"].fillna(0.0)
        df["order_multiple"] = df["order_multiple"].fillna(1.0)

        # The exposure the order has to survive. Reading it off the safety stock rather
        # than recomputing R + LT is what keeps the two from drifting apart: whatever
        # window the safety stock was sized for is the window the order covers.
        lt = (pd.to_numeric(df["wma_lead_time_days"], errors="coerce")
              if "wma_lead_time_days" in df.columns
              else pd.Series(np.nan, index=df.index, dtype="float64"))
        derived = df["review_period_days"] + lt.fillna(0.0)
        if "ss_exposure_days" in df.columns:
            exposure = pd.to_numeric(df["ss_exposure_days"], errors="coerce")
            df["coverage_days"] = exposure.where(exposure > 0, derived)
            df["coverage_basis"] = np.where(exposure > 0, "safety_stock_exposure",
                                            "review_plus_lt")
        else:
            df["coverage_days"] = derived
            df["coverage_basis"] = np.where(lt.notna(), "review_plus_lt",
                                            "review_period_only")
        df["coverage_days"] = df["coverage_days"].fillna(float(self.horizon_days))
        return df

    _MTO_COLUMNS = ("mto_actionable_qty", "mto_order_past_due_qty",
                    "mto_order_by", "mto_next_request")

    @staticmethod
    def _attach_mto_schedule(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the order-by schedule and state, per SKU, what it says about the deadline.

        `mto_order_status` is the reading a buyer acts on:

          order-past-due   the order date has gone. The customer date is already at
                           risk and no amount of ordering today recovers the lead time.
          order-now        the date falls inside this review period. Wait for the next
                           review and it joins the row above.
          scheduled        there is a commitment, but its order date is beyond this
                           review. Nothing to do yet, and worth seeing that it exists.
          none             no open book.

        A frame with no schedule reports `unscoped` rather than `none`: the difference
        between "no backlog" and "no visibility of when the backlog must be ordered" is
        exactly what this column exists to keep.
        """
        if schedule is not None and "sku" in getattr(schedule, "columns", []):
            keep = ["sku"] + [c for c in PurchaseRecommender._MTO_COLUMNS
                              if c in schedule.columns]
            df = df.merge(schedule[keep].drop_duplicates("sku"), on="sku", how="left")
        for col in PurchaseRecommender._MTO_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NaT if col in ("mto_order_by", "mto_next_request") else np.nan

        has_schedule = df["mto_actionable_qty"].notna()
        past_due = df["mto_order_past_due_qty"].fillna(0.0) > 0
        actionable = df["mto_actionable_qty"].fillna(0.0) > 0
        booked = df["backlog_qty"].fillna(0.0) > 0

        df["mto_order_status"] = np.select(
            [~has_schedule, past_due, actionable, booked],
            ["unscoped", "order-past-due", "order-now", "scheduled"],
            default="none",
        )
        return df

    @staticmethod
    def _order_quantity(df: pd.DataFrame) -> pd.Series:
        """
        How much to buy, given that the position has fallen to the reorder point.

        Periodic review closes the gap to S. A (s, Q) item buys whole lots — the lot is
        the reason it is on that policy — and buys as many as it takes to clear the
        reorder point, because one lot on an item that has fallen well below s leaves it
        below s and orders again next cycle at another order cost.
        """
        below = df["net_requirement"] > 0
        lot = df["order_lot"].fillna(0.0)
        gap = df["net_requirement"].clip(lower=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            lots_needed = np.ceil(gap / lot.replace(0, np.nan))
        whole_lots = (lots_needed.fillna(0.0) * lot).round(1)

        is_sq = df["replenishment_method"] == POLICY_REORDER_POINT
        # An (s, Q) item with no computable lot has nothing to round to, so it falls
        # back to closing the gap — the alternative is recommending zero on an item that
        # is genuinely short.
        quantity = np.where(
            is_sq & (lot > 0), whole_lots, (gap + lot).round(1)
        )
        quantity = np.where(df["replenishment_method"] == POLICY_MAKE_TO_ORDER,
                            0.0, quantity)
        return pd.Series(np.where(below, quantity, 0.0), index=df.index).round(1)

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
