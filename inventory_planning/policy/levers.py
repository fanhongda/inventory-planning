"""
Lever analysis — what each available action is actually worth, per SKU.

The point of this module is that **the ranking is computed, not assumed**. Textbook
advice ("shorten lead times", "review more often") is unconditional; the value of each
lever depends on where a SKU sits in the R / LT / σ_D / σ_LT space, and the ordering
routinely reverses between two SKUs in the same warehouse.

The single most important consequence, and the reason this module exists:

    Once pipeline stock is measured on the ownership boundary — only buyer-owned goods
    in transit — reducing the *mean* lead time barely moves the balance sheet, because
    it acts only through the √(R+LT) term in safety stock. Reducing lead-time
    *variability* moves it several times more. The negotiating ask is therefore
    reliability, not speed, and those are different conversations with a supplier.

Each lever is evaluated by perturbing one input and re-running the same should-be
calculation, so the arithmetic can never drift from the baseline it is compared to.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .parameters import ParameterSet
from .should_be import ShouldBeCalculator, ShouldBeResult
from ..lot_sizing import economic_order_quantity

# Levers are defined declaratively so adding one is a table entry, not a code path.
# `apply` mutates a copy of the SKU frame; `cost` describes what it takes to get it.


@dataclass
class LeverSpec:
    """One available action, and how to model it."""

    key: str
    label: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]
    cost_kind: str          # what the planner spends: "ordering", "service", "freight", "supplier", "forecast"
    description: str
    service_impact: bool = False    # True when the saving is paid for in customer service
    # Days before the change shows up in stock. A policy parameter takes effect at the
    # next review; a supplier reliability programme takes quarters. Against a dated
    # target this is decisive — a lever that cannot land before the deadline is not
    # part of the plan, however large its steady-state saving.
    time_to_effect_days: int = 0


def _scale(col: str, factor: float) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df[col] = pd.to_numeric(df[col], errors="coerce") * factor
        return df
    return _apply


def _set_floor(col: str, value: float) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Set a column to `value`, but never make it worse than it already is."""
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df[col] = np.minimum(pd.to_numeric(df[col], errors="coerce"), value)
        return df
    return _apply


def default_levers() -> List[LeverSpec]:
    """
    The standard set. Magnitudes are deliberately modest and uniform so the *ranking*
    reflects structural sensitivity rather than differing assumptions about how far
    each lever can be pushed.
    """
    return [
        LeverSpec(
            key="review_weekly",
            label="Review weekly instead of current cadence",
            apply=_set_floor("review_period_days", 7.0),
            cost_kind="ordering",
            description="More frequent ordering: higher admin cost, weaker freight consolidation",
            time_to_effect_days=30,
        ),
        LeverSpec(
            key="lt_reliability",
            label="Cut lead-time variability by half",
            apply=_scale("lt_sigma_days", 0.5),
            cost_kind="supplier",
            description="Supplier on-time performance programme; no unit price impact",
            time_to_effect_days=180,
        ),
        LeverSpec(
            key="forecast_accuracy",
            label="Cut forecast error by a quarter",
            apply=_scale("demand_sigma", 0.75),
            cost_kind="forecast",
            description="Better demand planning; effort, not spend",
            time_to_effect_days=90,
        ),
        LeverSpec(
            key="lt_speed",
            label="Cut mean lead time by a third",
            apply=_scale("lead_time_days", 2 / 3),
            cost_kind="freight",
            description="Faster mode or closer source; higher freight cost per unit",
            time_to_effect_days=60,
        ),
        LeverSpec(
            key="service_level_down",
            label="Lower service level by 5 points",
            apply=lambda df: df.assign(
                service_level=(pd.to_numeric(df["service_level"], errors="coerce") - 0.05).clip(lower=0.50)
            ),
            cost_kind="service",
            description="Accept more stockouts",
            service_impact=True,
            time_to_effect_days=30,
        ),
    ]


@dataclass
class LeverResult:
    """One lever's effect, per SKU and in total."""

    spec: LeverSpec
    frame: pd.DataFrame          # sku, saving_qty, saving_value, saving_pct, net_annual
    total_saving: float          # one-off balance-sheet reduction
    baseline_total: float
    annual_holding_saving: float = 0.0   # recurring benefit of holding less
    annual_cost: float = 0.0             # recurring cost of running the lever
    cost_known: bool = True              # False when the cost cannot be priced from data

    @property
    def saving_pct(self) -> float:
        return self.total_saving / self.baseline_total if self.baseline_total else 0.0

    @property
    def net_annual(self) -> float:
        """
        Recurring benefit minus recurring cost.

        The balance-sheet reduction is a one-off; the ordering cost it incurs repeats
        every year. Ranking on the one-off alone systematically over-recommends
        frequent ordering, which is exactly the failure the EOQ guard exists to stop.
        """
        return self.annual_holding_saving - self.annual_cost

    @property
    def is_value_destroying(self) -> bool:
        return self.cost_known and self.net_annual < 0

    def top(self, n: int = 5) -> pd.DataFrame:
        return self.frame.nlargest(n, "saving_value")


@dataclass
class LeverAnalysis:
    """All levers evaluated against one baseline."""

    baseline: ShouldBeResult
    results: Dict[str, LeverResult] = dc_field(default_factory=dict)
    guards: pd.DataFrame = None          # per-SKU constraint findings

    @property
    def ranked(self) -> List[LeverResult]:
        """
        Ranked by net annual benefit where the cost is known, otherwise by the
        balance-sheet reduction. Unpriced levers sort after priced ones of similar
        size so an un-costed option never outranks one that has been paid for.
        """
        def key(r: LeverResult):
            return (0 if r.cost_known else 1, -(r.net_annual if r.cost_known else r.total_saving))
        return sorted(self.results.values(), key=key)

    def per_sku_ranking(self) -> pd.DataFrame:
        """
        For each SKU, which lever is worth most. This is the output that makes the
        analysis actionable: a single global ranking hides that the best move for a
        long-lead-time item is not the best move for a fast-moving one.
        """
        frames = []
        for key, result in self.results.items():
            f = result.frame[["sku", "saving_value", "saving_qty", "net_annual",
                              "annual_cost"]].copy()
            f["lever"] = key
            f["lever_label"] = result.spec.label
            f["service_impact"] = result.spec.service_impact
            f["cost_known"] = result.cost_known
            frames.append(f)
        if not frames:
            return pd.DataFrame()

        stacked = pd.concat(frames, ignore_index=True)
        # Rank on net where it is known, so a lever that costs more in ordering than it
        # frees in stock cannot come out on top for that SKU.
        stacked["rank_basis"] = np.where(
            stacked["cost_known"], stacked["net_annual"], stacked["saving_value"] * 0.22
        )
        stacked["rank"] = stacked.groupby("sku")["rank_basis"].rank(ascending=False, method="first")
        return stacked.sort_values(["sku", "rank"])

    def best_lever_per_sku(self) -> pd.DataFrame:
        ranking = self.per_sku_ranking()
        if ranking.empty:
            return pd.DataFrame()
        return ranking[ranking["rank"] == 1].drop(columns=["rank"])

    def summary(self) -> str:
        base = self.baseline.total_should_be_value
        lines = [
            "  Lever analysis — value of each action",
            "  " + "-" * 58,
            f"    baseline should-be: ${base:,.0f}",
            "",
            f"    {'lever':<38}{'stock freed':>13}{'net/yr':>15}",
        ]
        for result in self.ranked:
            label = result.spec.label[:37]
            if not result.cost_known:
                net = "unpriced"
            elif result.is_value_destroying:
                net = f"-${abs(result.net_annual):,.0f}"
            else:
                net = f"+${result.net_annual:,.0f}"

            flag = ""
            if result.is_value_destroying:
                flag = "  ✗ costs more than it frees"
            elif result.spec.service_impact:
                flag = "  ⚠ paid for in service"
            elif not result.cost_known:
                flag = f"  ? {result.spec.cost_kind} cost not in data"

            lines.append(
                f"    {label:<38}${result.total_saving:>11,.0f}{net:>15}{flag}"
            )

        best = self.best_lever_per_sku()
        if not best.empty:
            spread = best["lever_label"].value_counts()
            lines.append("")
            lines.append("    Best lever varies by SKU:")
            for label, count in spread.items():
                lines.append(f"      {count:>4} SKUs  {label}")

        if self.guards is not None and len(self.guards):
            lines.append("")
            lines.append("    Constraints found:")
            for _, row in self.guards.head(8).iterrows():
                lines.append(f"      {row['sku']:<14} {row['finding']}")
            if len(self.guards) > 8:
                lines.append(f"      … and {len(self.guards) - 8} more")
        return "\n".join(lines)


class LeverAnalyzer:
    """Evaluates levers by re-running the should-be calculation under perturbation."""

    def __init__(self, calculator: ShouldBeCalculator = None, levers: List[LeverSpec] = None):
        self.calculator = calculator or ShouldBeCalculator()
        self.levers = levers if levers is not None else default_levers()

    def analyze(
        self,
        params: ParameterSet,
        actual: pd.DataFrame = None,
        baseline: ShouldBeResult = None,
    ) -> LeverAnalysis:
        baseline = baseline or self.calculator.calculate(params, actual)
        base_frame = baseline.frame[["sku", "should_be_qty", "should_be_value"]].rename(
            columns={"should_be_qty": "base_qty", "should_be_value": "base_value"}
        )

        analysis = LeverAnalysis(baseline=baseline)

        for spec in self.levers:
            perturbed = ParameterSet(
                frame=spec.apply(params.frame.copy()),
                hits=params.hits,
                conventions=params.conventions,
                defaults=params.defaults,
                segmentation=params.segmentation,
            )
            result = self.calculator.calculate(perturbed, actual)

            merged = result.frame[["sku", "should_be_qty", "should_be_value"]].merge(
                base_frame, on="sku", how="inner"
            )
            merged["saving_qty"] = merged["base_qty"] - merged["should_be_qty"]
            merged["saving_value"] = merged["base_value"] - merged["should_be_value"]
            merged["saving_pct"] = np.where(
                merged["base_value"] > 0, merged["saving_value"] / merged["base_value"], 0.0
            )

            # Price the lever on the state it *proposes*, not the state it starts from.
            # Checking only the current cadence would let a lever recommend weekly
            # ordering without ever noticing that weekly breaks the economic lot size.
            holding_rate = float(params.defaults.get("holding_cost_rate", 0.22))
            inventory_saving = float(merged["saving_value"].sum())
            annual_cost, cost_known = self._annual_cost(
                spec, params.frame, perturbed.frame, params
            )
            merged = self._attach_net(merged, spec, params.frame, perturbed.frame,
                                      params, holding_rate)

            analysis.results[spec.key] = LeverResult(
                spec=spec,
                frame=merged[["sku", "saving_qty", "saving_value", "saving_pct",
                              "net_annual", "annual_cost"]],
                total_saving=inventory_saving,
                baseline_total=baseline.total_should_be_value,
                annual_holding_saving=inventory_saving * holding_rate,
                annual_cost=annual_cost,
                cost_known=cost_known,
            )

        analysis.guards = self.check_constraints(params, baseline)
        return analysis

    # ── Lever economics ──────────────────────────────────────────────────────

    @staticmethod
    def _orders_per_year(frame: pd.DataFrame) -> pd.Series:
        """Order frequency implied by a review cadence."""
        review = pd.to_numeric(frame["review_period_days"], errors="coerce")
        return (365.0 / review.replace(0, np.nan)).fillna(0.0)

    def _annual_cost(
        self, spec: LeverSpec, before: pd.DataFrame, after: pd.DataFrame, params: ParameterSet
    ) -> tuple:
        """
        Recurring cost of running a lever, and whether it can be priced at all.

        Only the ordering-cost lever is priceable from data already in hand. Freight
        premiums need a rate table and service loss needs a shortage cost; rather than
        invent either, those return `cost_known=False` so the report can say the
        trade-off is unpriced instead of implying it is free.
        """
        if spec.cost_kind == "ordering":
            order_cost = float(params.defaults.get("order_cost_usd", 350))
            delta_orders = (self._orders_per_year(after) - self._orders_per_year(before))
            return float((delta_orders.clip(lower=0) * order_cost).sum()), True

        if spec.cost_kind in ("freight", "service"):
            return 0.0, False        # needs a freight rate table / shortage cost

        return 0.0, True             # effort, not spend

    def _attach_net(
        self, merged: pd.DataFrame, spec: LeverSpec, before: pd.DataFrame,
        after: pd.DataFrame, params: ParameterSet, holding_rate: float,
    ) -> pd.DataFrame:
        """Per-SKU net annual benefit, so a value-destroying lever is visible per item."""
        merged["annual_cost"] = 0.0
        if spec.cost_kind == "ordering":
            order_cost = float(params.defaults.get("order_cost_usd", 350))
            delta = (self._orders_per_year(after) - self._orders_per_year(before)).clip(lower=0)
            cost_by_sku = pd.DataFrame({"sku": before["sku"], "annual_cost": delta * order_cost})
            merged = merged.drop(columns=["annual_cost"]).merge(cost_by_sku, on="sku", how="left")
            merged["annual_cost"] = merged["annual_cost"].fillna(0.0)

        merged["net_annual"] = merged["saving_value"] * holding_rate - merged["annual_cost"]
        return merged

    # ── Constraint checks ────────────────────────────────────────────────────

    def check_constraints(self, params: ParameterSet, baseline: ShouldBeResult) -> pd.DataFrame:
        """
        Findings that invalidate or bound a lever.

        These are the statements that keep the analysis honest. A lever with a large
        computed saving that violates a constraint is worse than useless — it is a
        recommendation the planner will follow and then regret.
        """
        df = baseline.frame
        findings: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            sku = row["sku"]
            lt = float(row.get("lead_time_days") or 0)
            review = float(row.get("review_period_days") or 0)

            # 1. Lead time dominating the review period. The classic wasted effort:
            #    shortening R when SS moves as √(R+LT) and LT is most of the exposure.
            if lt > 0 and review > 0 and lt > 4 * review:
                findings.append({
                    "sku": sku, "lever": "review_weekly", "severity": "info",
                    "finding": (
                        f"LT {lt:.0f}d is {lt / review:.0f}x the review period — "
                        f"shortening review barely moves safety stock; the lever here is LT reliability"
                    ),
                })

            # 2. EOQ as a floor on order frequency. Per planner practice EOQ is not a
            #    target to hit; it is the point past which ordering more often costs
            #    more than the cycle stock it saves.
            eoq_finding = self._eoq_guard(row, params)
            if eoq_finding:
                findings.append({"sku": sku, **eoq_finding})

            # 3. MOQ / order multiple making the cycle-stock lever unavailable.
            moq = float(row.get("min_order_qty") or 0)
            if moq > 0:
                cycle_qty = float(row.get("cycle_qty") or 0)
                if cycle_qty > 0 and moq > cycle_qty * 1.5:
                    findings.append({
                        "sku": sku, "lever": "review_weekly", "severity": "blocking",
                        "finding": (
                            f"MOQ {moq:,.0f} exceeds the policy cycle stock {cycle_qty:,.0f} — "
                            f"order frequency is set by the supplier, not by review cadence"
                        ),
                    })

        return pd.DataFrame(findings) if findings else pd.DataFrame(
            columns=["sku", "lever", "severity", "finding"]
        )

    @staticmethod
    def _eoq_guard(row: pd.Series, params: ParameterSet) -> Optional[Dict[str, Any]]:
        """
        Compare the order size implied by the current review cadence against EOQ.

        EOQ appears only when it is violated. Reporting an EOQ for every SKU would be
        noise — planners do not order to EOQ, they order to a review cadence, and EOQ's
        job is to say when that cadence has become uneconomic.
        """
        demand_mean = float(row.get("demand_mean") or 0)
        unit_cost = row.get("unit_cost")
        review = float(row.get("review_period_days") or 0)
        if demand_mean <= 0 or review <= 0 or pd.isna(unit_cost) or not unit_cost:
            return None

        order_cost = float(params.defaults.get("order_cost_usd", 350))
        holding_rate = float(params.defaults.get("holding_cost_rate", 0.22))
        annual_demand = demand_mean * 12
        if float(unit_cost) * holding_rate <= 0:
            return None

        eoq = float(economic_order_quantity(
            [annual_demand], [unit_cost], order_cost, holding_rate).iloc[0])
        implied_order = demand_mean * review / 30.0
        if implied_order <= 0 or not np.isfinite(eoq) or eoq <= 0:
            return None

        ratio = eoq / implied_order
        if ratio < 2:
            return None      # within a factor of two is close enough to be uninteresting

        orders_per_year = annual_demand / implied_order
        eoq_orders_per_year = annual_demand / eoq
        excess_order_cost = (orders_per_year - eoq_orders_per_year) * order_cost

        return {
            "lever": "review_weekly", "severity": "warning",
            "finding": (
                f"ordering {orders_per_year:.0f}x/yr vs EOQ's {eoq_orders_per_year:.0f}x/yr "
                f"(lot {implied_order:,.0f} vs EOQ {eoq:,.0f}) — "
                f"~${excess_order_cost:,.0f}/yr of avoidable ordering cost; "
                f"review cadence is already past the economic point"
            ),
        }

