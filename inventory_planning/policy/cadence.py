"""
Replenishment cadence — did the buying keep pace with the selling, and at what cost.

The order-size CV in `diagnostics.erratic` answers a narrow question: were the lot
sizes consistent. A planner can score perfectly on it and still be failing, because
consistent lots ordered at the wrong moments produce exactly the same stockouts as
erratic ones. CV is blind to timing, blind to direction, and blind to whether the
orders were needed at all.

What this module measures instead is the **cumulative flow balance**: month by month,
what was ordered minus what was sold, accumulated across the window.

    net_m  = po_qty_m − sales_qty_m
    cum_m  = Σ net_1..m

The shape of that curve is the diagnosis, and it separates failures the CV cannot:

  cum returns to ≈ 0            The run rates match. Over twelve months the planner
                                bought what the business sold. This is the target, and
                                it is a *stronger* statement than any single-month
                                accuracy figure — errors that cancel are errors the
                                inventory never had to carry.

  cum sits negative for months  Buying is behind selling and staying behind. Each month
                                below the line is a month of cover being consumed
                                rather than replaced. Replenishment is late, not wrong.

  cum climbs and stays up       Persistent positive bias. Nothing is pulling the orders
                                back down to the run rate, which means the control
                                mechanism is not closing the loop — the definition of
                                a broken review, not of a bad forecast.

  cum swings both ways          Over-correction. Chasing a shortage into a surplus and
                                back. This is the bullwhip signature at single-SKU
                                scale, and it is the most expensive of the four because
                                it pays for both failures in turn.

Against that curve sits the second question the CV never asks: **how many orders were
spent getting there**. A review period of one month entitles the planner to twelve
orders a year. Twelve orders that land the cumulative on zero is control. Forty orders
that land on the same zero is the same result bought at three times the ordering cost,
and it usually means the cadence is being overridden by expediting. Fewer orders than
the cadence allows is not automatically good either — it is often how a deficit run
gets started.

The three readings combine, per the planner's own framing: **order frequency, the money
standing behind the imbalance, and the number of periods spent out of balance**. They
are reported together and ranked by the money, because a wildly mistimed cheap item is
not the same finding as a well-behaved expensive one.

One principle constrains the recommendation rather than the measurement: **ordering
frequency above the review cadence has to be earned, and only critical items earn it**.
A high-value, service-critical part is worth watching weekly. A C-class consumable
ordered weekly is paying an ordering cost per unit that dwarfs anything the tighter
cycle stock saves, and the fix is a longer review period, not a better forecast.

What this deliberately does not model: the lag between raising a PO and receiving it.
The flow balance is keyed on the **order** date, so it measures the planner's decisions
against the demand those decisions were meant to cover. Supplier delivery against those
decisions is a different question, and `diagnostics.forward` and `service` already
answer it. Mixing the two here would let a reliable planner buying from a late supplier
read as a control failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Months of average demand the cumulative may drift by before it counts as out of
# balance. Below this the wobble is lumpy ordering, not a control problem.
DEFAULT_TOLERANCE_MONTHS = 0.5

# A deficit or surplus has to persist to mean anything. One month either side of the
# line is the ordinary sawtooth of any reorder policy.
PERSISTENT_RUN_MONTHS = 2
BIAS_RUN_MONTHS = 3

# Closing imbalance, as a share of the window's demand, above which the run rates have
# genuinely diverged rather than merely wobbled.
BIAS_SHARE = 0.25

# Order counts within this factor of the cadence are conforming; the cadence was never
# meant to be hit exactly.
CADENCE_TOLERANCE = 1.25
# Beyond this the ordering pattern is not a cadence at all.
CADENCE_EXCESS = 1.5

# Classes for which a review period shorter than the cadence is worth paying for.
CRITICAL_CLASSES = ("A",)

DAYS_PER_MONTH = 30.44

VERDICTS = {
    "controlled": "Run rates match and the cadence was kept",
    "oscillating": "Over-correcting — chased a shortage into a surplus and back",
    "chasing": "Buying persistently behind selling — replenishment is late",
    "losing_control": "Persistent positive bias — nothing pulls orders back to the run rate",
    "unearned_frequency": "Balanced, but bought far more often than the cadence needs",
    "under_ordered": "Fewer orders than the cadence allows, and behind on volume",
}


@dataclass
class CadenceResult:
    """Per-SKU cadence findings, ranked by the money standing behind them."""

    frame: pd.DataFrame
    window_months: int
    as_of: date
    tolerance_months: float = DEFAULT_TOLERANCE_MONTHS
    notes: List[str] = dc_field(default_factory=list)
    # The curves the verdicts were read off, indexed by SKU: one row per SKU, one
    # column per month. Kept rather than discarded because the diagnosis *is* a shape —
    # "six months behind then a correction" is a sentence about a picture, and three
    # columns of summary statistics make the reader rebuild that picture in their head.
    months: Optional[pd.PeriodIndex] = None
    cumulative: Optional[pd.DataFrame] = None
    monthly_po: Optional[pd.DataFrame] = None
    monthly_demand: Optional[pd.DataFrame] = None

    def curve(self, sku: str) -> Optional[pd.DataFrame]:
        """Per-month ordered, sold, net and cumulative balance for one SKU."""
        if self.cumulative is None or sku not in self.cumulative.index:
            return None
        po = self.monthly_po.loc[sku]
        so = self.monthly_demand.loc[sku]
        return pd.DataFrame({
            "period": list(self.months),
            "po_qty": po.to_numpy(dtype="float64"),
            "demand_qty": so.to_numpy(dtype="float64"),
            "net": po.to_numpy(dtype="float64") - so.to_numpy(dtype="float64"),
            "cumulative": self.cumulative.loc[sku].to_numpy(dtype="float64"),
        })

    # ── Slices ───────────────────────────────────────────────────────────────

    def verdict(self, name: str) -> pd.DataFrame:
        if not len(self.frame):
            return self.frame
        return self.frame[self.frame["verdict"] == name]

    @property
    def controlled(self) -> pd.DataFrame:
        return self.verdict("controlled")

    @property
    def out_of_control(self) -> pd.DataFrame:
        """Everything that is not `controlled`, ranked by exposure."""
        if not len(self.frame):
            return self.frame
        return self.frame[self.frame["verdict"] != "controlled"]

    @property
    def unearned_frequency(self) -> pd.DataFrame:
        """
        Ordered above the cadence without being critical enough to justify it.

        Cuts across the verdicts on purpose: an item can be chasing demand *and* be
        over-ordered on a part that never warranted the attention, and the second
        finding is the one that frees up planner time.
        """
        if not len(self.frame):
            return self.frame
        return self.frame[
            (self.frame["orders_vs_expected"] > CADENCE_EXCESS)
            & ~self.frame["frequency_earned"]
        ].sort_values("avoidable_order_cost", ascending=False)

    # ── Totals ───────────────────────────────────────────────────────────────

    @property
    def total_exposure(self) -> float:
        return float(self.frame["exposure_value"].sum()) if len(self.frame) else 0.0

    @property
    def total_avoidable_order_cost(self) -> float:
        return (float(self.unearned_frequency["avoidable_order_cost"].sum())
                if len(self.frame) else 0.0)

    @property
    def control_rate(self) -> float:
        """Share of measured SKUs whose buying tracked their selling."""
        if not len(self.frame):
            return 0.0
        return float(len(self.controlled) / len(self.frame))

    def counts(self) -> Dict[str, int]:
        if not len(self.frame):
            return {}
        return self.frame["verdict"].value_counts().to_dict()

    # ── Narrative ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"  Replenishment cadence — {self.window_months} months to {self.as_of}",
            "  " + "-" * 58,
        ]
        if not len(self.frame):
            lines.append("    Not enough overlapping purchase and demand history to judge cadence.")
            lines.extend(f"    {note}" for note in self.notes)
            return "\n".join(lines)

        counts = self.counts()
        lines.append(
            f"    In control              {counts.get('controlled', 0):>4} of "
            f"{len(self.frame):>4} SKUs   ({self.control_rate:.0%}) — "
            f"cumulative PO minus sales back to ~zero on the cadence's own order count"
        )

        for name in ("chasing", "losing_control", "oscillating", "under_ordered"):
            rows = self.verdict(name)
            if not len(rows):
                continue
            lines.append(
                f"    {VERDICTS[name][:52]:<52} {len(rows):>4} SKUs   "
                f"${rows['exposure_value'].sum():>12,.0f}"
            )
            for _, row in rows.head(4).iterrows():
                lines.append(f"      {self._line(row)}")

        unearned = self.unearned_frequency
        if len(unearned):
            lines.append("")
            lines.append(
                f"    Frequency not earned    {len(unearned):>4} SKUs   "
                f"${self.total_avoidable_order_cost:>12,.0f}/yr of ordering cost"
            )
            for _, row in unearned.head(5).iterrows():
                lines.append(
                    f"      {row['sku']:<14} {row['order_count']:.0f} orders vs "
                    f"{row['expected_orders']:.0f} the cadence needs, class "
                    f"{row['abc_class'] or '?'} — ${row['avoidable_order_cost']:,.0f}/yr"
                )
            lines.append("      High ordering frequency is for critical items. On these it is "
                         "being paid for without buying anything back; the fix is a longer "
                         "review period, not a better forecast.")

        lines.extend(f"    {note}" for note in self.notes)
        return "\n".join(lines)

    @staticmethod
    def _line(row: pd.Series) -> str:
        if row["verdict"] == "chasing":
            shape = (f"{row['deficit_months']:.0f}/{row['months']:.0f} months behind, "
                     f"worst {row['max_deficit']:,.0f} units short")
        elif row["verdict"] == "losing_control":
            shape = (f"{row['surplus_months']:.0f}/{row['months']:.0f} months long, "
                     f"closing {row['closing_balance']:+,.0f} units "
                     f"({row['imbalance_pct']:+.0%} of demand)")
        elif row["verdict"] == "oscillating":
            shape = (f"swung {row['max_deficit']:,.0f} short to "
                     f"{row['max_surplus']:,.0f} long")
        else:
            shape = (f"closing {row['closing_balance']:+,.0f} units on only "
                     f"{row['order_count']:.0f} orders")
        return (f"{row['sku']:<14} {shape} — {row['order_count']:.0f} orders "
                f"(cadence: {row['expected_orders']:.0f}), ${row['exposure_value']:,.0f}")


class CadenceAnalyzer:
    """
    Builds the monthly PO-versus-sales balance per SKU and classifies its shape.

    `order_cost` prices the ordering effort so that "too many orders" arrives as a
    number rather than as an opinion; it is the same `order_cost_usd` the EOQ and lever
    economics use, so the three cannot disagree about what an order costs.
    """

    def __init__(
        self,
        window_months: int = 12,
        order_cost: float = 350.0,
        tolerance_months: float = DEFAULT_TOLERANCE_MONTHS,
        critical_classes: Sequence[str] = CRITICAL_CLASSES,
        min_orders: int = 2,
    ):
        self.window_months = window_months
        self.order_cost = float(order_cost)
        self.tolerance_months = float(tolerance_months)
        self.critical_classes = tuple(critical_classes)
        # One order in a year is a shape with no cadence to conform to.
        self.min_orders = int(min_orders)

    # ── Entry point ──────────────────────────────────────────────────────────

    def analyze(
        self,
        po_history: pd.DataFrame,
        sales_history: pd.DataFrame = None,
        demand_pivot: pd.DataFrame = None,
        sku_attributes: pd.DataFrame = None,
        parameters: pd.DataFrame = None,
        as_of: date = None,
        default_review_days: float = 30.0,
    ) -> CadenceResult:
        as_of = as_of or date.today()
        notes: List[str] = []
        empty = CadenceResult(pd.DataFrame(columns=_FRAME_COLS), self.window_months,
                              as_of, self.tolerance_months, notes)

        months = self._window(as_of)
        po_qty, orders = self._po_monthly(po_history, months, notes)
        if po_qty is None:
            notes.append("No purchase history with dated quantities — cadence needs both sides.")
            return empty

        demand = self._demand_monthly(sales_history, demand_pivot, months, notes)
        if demand is None:
            notes.append("No dated demand history — the PO series has nothing to be measured against.")
            return empty

        # Only SKUs seen on both sides can have a balance. One-sided items are already
        # findings elsewhere: bought and never sold is `over_ordered`, sold and never
        # bought is a stockout, and neither is a cadence problem.
        skus = po_qty.index.intersection(demand.index)
        if not len(skus):
            notes.append("Purchase history and demand history share no SKUs in the window.")
            return empty

        po_m = po_qty.loc[skus].to_numpy(dtype="float64")
        so_m = demand.loc[skus].to_numpy(dtype="float64")
        cum = np.cumsum(po_m - so_m, axis=1)

        frame = self._metrics(skus, po_m, so_m, cum, orders)
        frame = self._attach_attributes(frame, sku_attributes, parameters, default_review_days)
        frame = self._classify(frame)
        frame = self._price(frame)

        thin = int((frame["order_count"] < self.min_orders).sum())
        frame = frame[frame["order_count"] >= self.min_orders].copy()
        if thin:
            notes.append(f"{thin:,} SKUs bought fewer than {self.min_orders} times in the "
                         f"window — too few orders to have a cadence, so not judged.")

        # Ranking is by money, so an item with no unit cost sinks to the bottom of the
        # list whatever its imbalance. Saying so is the difference between "no exposure"
        # and "exposure not known".
        unpriced = int(frame["unpriced"].sum())
        if unpriced:
            notes.append(f"{unpriced:,} of {len(frame):,} SKUs have no unit cost — their "
                         f"exposure counts only the ordering cost, so they rank below "
                         f"items whose imbalance could be valued.")

        frame = frame.sort_values("exposure_value", ascending=False).reset_index(drop=True)

        # Carry the curves for the SKUs that survived the filters, so the report can
        # draw the shape each verdict was read from.
        kept = pd.Index(frame["sku"])
        cum_frame = pd.DataFrame(cum, index=pd.Index(skus, name="sku"), columns=months)
        return CadenceResult(
            frame, self.window_months, as_of, self.tolerance_months, notes,
            months=months,
            cumulative=cum_frame.reindex(kept),
            monthly_po=po_qty.loc[skus].reindex(kept),
            monthly_demand=demand.loc[skus].reindex(kept),
        )

    # ── Monthly series ───────────────────────────────────────────────────────

    def _window(self, as_of: date) -> pd.PeriodIndex:
        end = pd.Period(as_of, freq="M")
        return pd.period_range(end=end, periods=self.window_months, freq="M")

    def _po_monthly(self, po_history: pd.DataFrame, months: pd.PeriodIndex, notes: List[str]):
        """Ordered quantity by month, plus the distinct order count behind it."""
        if po_history is None or not len(po_history) or "sku" not in po_history.columns:
            return None, None
        qty_col = next((c for c in ("po_qty", "order_qty", "received_qty")
                        if c in po_history.columns), None)
        if qty_col is None or "po_date" not in po_history.columns:
            return None, None

        df = pd.DataFrame({
            "sku": po_history["sku"],
            "qty": pd.to_numeric(po_history[qty_col], errors="coerce").fillna(0.0),
            "stamp": pd.to_datetime(po_history["po_date"], errors="coerce"),
        })
        # A PO number groups the lines a planner raised in one act. Without one, each
        # line counts as an order, which over-states the effort on multi-line POs.
        df["order_id"] = (po_history["po_number"].astype("string")
                          if "po_number" in po_history.columns
                          else pd.Series(np.arange(len(po_history)), index=po_history.index).astype("string"))

        _note_undated(df, notes, "purchase", "po_date")
        df = df.dropna(subset=["stamp"])
        df["month"] = df["stamp"].dt.to_period("M")
        df = df[df["month"].isin(months)]
        if df.empty:
            return None, None

        qty = self._pivot(df, "qty", months)
        orders = df.groupby("sku")["order_id"].nunique()
        return qty, orders

    def _demand_monthly(self, sales_history, demand_pivot, months: pd.PeriodIndex, notes: List[str]):
        """
        Sold quantity by month, from transactional sales or a pre-compiled series.

        Sales history is preferred: the compiled series has usually been through its
        own aggregation, and a cadence measured against a smoothed demand curve is a
        cadence measured against something no planner ever had to cover.
        """
        if sales_history is not None and len(sales_history) and "sku" in sales_history.columns:
            qty_col = next((c for c in ("qty", "shipped_qty", "demand_qty")
                            if c in sales_history.columns), None)
            date_col = next((c for c in ("demand_date", "ship_date", "order_date")
                             if c in sales_history.columns), None)
            if qty_col and date_col:
                df = pd.DataFrame({
                    "sku": sales_history["sku"],
                    "qty": pd.to_numeric(sales_history[qty_col], errors="coerce").fillna(0.0),
                    "stamp": pd.to_datetime(sales_history[date_col], errors="coerce"),
                })
                _note_undated(df, notes, "demand", date_col)
                df = df.dropna(subset=["stamp"])
                df["month"] = df["stamp"].dt.to_period("M")
                df = df[df["month"].isin(months)]
                if not df.empty:
                    return self._pivot(df, "qty", months)

        if demand_pivot is not None and len(demand_pivot):
            piv = demand_pivot.copy()
            if not isinstance(piv.index, pd.PeriodIndex):
                piv.index = pd.PeriodIndex(piv.index, freq="M")
            piv = piv.reindex(months).fillna(0.0)
            return piv.T          # periods x sku -> sku x periods

        return None

    @staticmethod
    def _pivot(df: pd.DataFrame, value: str, months: pd.PeriodIndex) -> pd.DataFrame:
        """SKU x month matrix over the full window, with silent months as zero."""
        wide = df.pivot_table(index="sku", columns="month", values=value,
                              aggfunc="sum", fill_value=0.0)
        return wide.reindex(columns=months, fill_value=0.0).fillna(0.0)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def _metrics(self, skus, po_m, so_m, cum, orders) -> pd.DataFrame:
        n_months = po_m.shape[1]
        demand_total = so_m.sum(axis=1)
        po_total = po_m.sum(axis=1)
        mean_demand = demand_total / n_months

        # The band scales with the item's own run rate, so a 50-unit drift is a crisis
        # on a part selling 20 a month and noise on one selling 5,000.
        #
        # It also cannot go below one typical lot, because that is the swing a periodic
        # review produces even when it is working perfectly: the cumulative falls by up
        # to a full lot between orders and is restored when the next one lands. A part
        # bought quarterly in lots of 300 will sit 300 units down for most of every
        # quarter, and that is the policy operating, not failing. Judged against half a
        # month of demand instead, correct quarterly ordering reads as chronic
        # under-buying, and low-volume intermittent items — the bulk of any real
        # catalogue — fill the report with findings of "+1 unit long".
        order_count = pd.Series(list(skus)).map(orders).fillna(0).to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            typical_lot = np.where(order_count > 0, po_total / order_count, 0.0)
        tol = np.maximum(mean_demand * self.tolerance_months, typical_lot)
        band = tol[:, None]

        below, above = cum < -band, cum > band
        closing = cum[:, -1]

        with np.errstate(divide="ignore", invalid="ignore"):
            imbalance_pct = np.where(demand_total > 0, closing / demand_total, np.nan)

        return pd.DataFrame({
            "sku": list(skus),
            "months": n_months,
            "po_qty": po_total,
            "demand_qty": demand_total,
            "mean_monthly_demand": mean_demand,
            "order_count": order_count,
            "order_months": (po_m > 0).sum(axis=1),
            "closing_balance": closing,
            "imbalance_pct": imbalance_pct,
            "deficit_months": below.sum(axis=1),
            "surplus_months": above.sum(axis=1),
            "longest_deficit_run": _longest_run(below),
            "longest_surplus_run": _longest_run(above),
            "max_deficit": np.maximum(-cum.min(axis=1), 0.0),
            "max_surplus": np.maximum(cum.max(axis=1), 0.0),
            "tolerance_qty": tol,
        })

    def _attach_attributes(self, frame, sku_attributes, parameters, default_review_days) -> pd.DataFrame:
        """Unit cost and ABC to price the finding; review period to size the cadence."""
        frame["unit_cost"] = np.nan
        frame["abc_class"] = pd.Series([None] * len(frame), dtype="object")
        if sku_attributes is not None and len(sku_attributes) and "sku" in sku_attributes.columns:
            attrs = sku_attributes.drop_duplicates("sku").set_index("sku")
            for col in ("unit_cost", "abc_class"):
                if col in attrs.columns:
                    frame[col] = frame["sku"].map(attrs[col])

        review = pd.Series(default_review_days, index=frame.index, dtype="float64")
        if (parameters is not None and len(parameters)
                and {"sku", "review_period_days"} <= set(parameters.columns)):
            mapped = frame["sku"].map(
                parameters.drop_duplicates("sku").set_index("sku")["review_period_days"]
            )
            review = pd.to_numeric(mapped, errors="coerce").fillna(default_review_days)
        frame["review_period_days"] = review.replace(0, np.nan).fillna(default_review_days)

        # The cadence's own entitlement: a monthly review over twelve months is twelve
        # orders. This is the count the planner's parameters promised, not a target the
        # analysis invented.
        review_months = frame["review_period_days"] / DAYS_PER_MONTH
        frame["expected_orders"] = np.maximum(
            np.round(frame["months"] / review_months.replace(0, np.nan)), 1.0
        ).fillna(1.0)
        frame["orders_vs_expected"] = frame["order_count"] / frame["expected_orders"]
        frame["frequency_earned"] = frame["abc_class"].isin(self.critical_classes)
        return frame

    # ── Classification ───────────────────────────────────────────────────────

    def _classify(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        One verdict per SKU, most expensive failure first.

        Order matters. Oscillation is checked before either one-sided failure because a
        SKU that swung both ways would otherwise be filed as whichever direction it
        happened to end the window in — and the end of the window is an accident of
        when the extract was taken.
        """
        deficit_run = frame["longest_deficit_run"]
        surplus_run = frame["longest_surplus_run"]
        share = frame["imbalance_pct"].fillna(0.0)

        persistent_deficit = (
            (deficit_run >= PERSISTENT_RUN_MONTHS)
            | (frame["deficit_months"] >= frame["months"] / 3)
        )
        # A closing balance outside the band is what makes a share meaningful. Left
        # ungated, "+1 unit on a part that sold 2 all year" reads as a 50% bias, and
        # the report fills with arithmetic rather than findings.
        over_band = frame["closing_balance"] > frame["tolerance_qty"]
        under_band = frame["closing_balance"] < -frame["tolerance_qty"]
        persistent_surplus = (
            (surplus_run >= BIAS_RUN_MONTHS) | ((share > BIAS_SHARE) & over_band)
        )
        balanced = (
            frame["closing_balance"].abs() <= frame["tolerance_qty"]
        ) & (frame["deficit_months"] == 0)

        verdict = pd.Series("controlled", index=frame.index, dtype="object")
        verdict[balanced & (frame["orders_vs_expected"] > CADENCE_EXCESS)] = "unearned_frequency"
        verdict[persistent_surplus] = "losing_control"
        verdict[persistent_deficit] = "chasing"
        # Behind on volume while under-using the cadence is a distinct failure: the
        # orders the review period entitled the planner to were never placed. It has to
        # still be behind at the close — a SKU that ran short mid-window and then caught
        # up was late, not under-ordered, and the two have different fixes.
        verdict[
            persistent_deficit & under_band
            & (frame["orders_vs_expected"] < 1 / CADENCE_TOLERANCE)
        ] = "under_ordered"
        verdict[
            (deficit_run >= PERSISTENT_RUN_MONTHS) & (surplus_run >= PERSISTENT_RUN_MONTHS)
        ] = "oscillating"

        frame["verdict"] = verdict
        frame["verdict_note"] = verdict.map(VERDICTS)
        return frame

    def _price(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Money behind each finding, so the list ranks by consequence rather than by ratio.

        Three components, because the planner named three: what the surplus tied up,
        what the deficit left uncovered, and what the extra orders cost to place.
        """
        cost = pd.to_numeric(frame["unit_cost"], errors="coerce").fillna(0.0)
        frame["excess_value"] = (frame["max_surplus"] * cost).round(2)
        frame["shortfall_value"] = (frame["max_deficit"] * cost).round(2)

        # Only orders above the cadence are avoidable, and only where the cadence was
        # not deliberately tightened for a critical item.
        surplus_orders = (frame["order_count"] - frame["expected_orders"]).clip(lower=0)
        frame["avoidable_order_cost"] = np.where(
            frame["frequency_earned"], 0.0, surplus_orders * self.order_cost
        ).round(2)

        frame["exposure_value"] = (
            frame["excess_value"] + frame["shortfall_value"] + frame["avoidable_order_cost"]
        ).round(2)
        frame["unpriced"] = pd.to_numeric(frame["unit_cost"], errors="coerce").isna()
        return frame[_FRAME_COLS]


_FRAME_COLS = [
    "sku", "verdict", "verdict_note", "months",
    "po_qty", "demand_qty", "mean_monthly_demand",
    "closing_balance", "imbalance_pct",
    "deficit_months", "surplus_months", "longest_deficit_run", "longest_surplus_run",
    "max_deficit", "max_surplus", "tolerance_qty",
    "order_count", "order_months", "expected_orders", "orders_vs_expected",
    "review_period_days", "frequency_earned", "abc_class", "unit_cost",
    "excess_value", "shortfall_value", "avoidable_order_cost", "exposure_value",
    "unpriced",
]


def _note_undated(df: pd.DataFrame, notes: List[str], side: str, column: str) -> None:
    """
    Flag quantity that carries no date, because it silently biases the balance.

    A row with no date cannot be placed in a month, so it drops out of one side of the
    subtraction while the other side keeps everything. On the real extract 22.5% of
    demand lines lost their date to a rollup, and every SKU consequently read as
    over-bought by roughly that much. The imbalance was in the ingest, not the buying.
    """
    undated = df["stamp"].isna()
    if not undated.any():
        return
    lost_qty = float(df.loc[undated, "qty"].sum())
    total_qty = float(df["qty"].sum())
    share = lost_qty / total_qty if total_qty else 0.0
    direction = "toward surplus" if side == "demand" else "toward deficit"
    notes.append(
        f"⚠ {int(undated.sum()):,} {side} rows ({share:.0%} of the quantity) have no "
        f"usable {column} and cannot be placed in a month, biasing the balance "
        f"{direction}. The loss is not spread evenly, so it moves individual verdicts "
        f"more than the portfolio picture — treat a single SKU's reading as provisional "
        f"until the date mapping is fixed."
    )


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest consecutive True run per row. Twelve columns, so a loop is the clear form."""
    if mask.size == 0:
        return np.zeros(mask.shape[0], dtype="int64")
    best = np.zeros(mask.shape[0], dtype="int64")
    current = np.zeros(mask.shape[0], dtype="int64")
    for col in range(mask.shape[1]):
        current = np.where(mask[:, col], current + 1, 0)
        best = np.maximum(best, current)
    return best
