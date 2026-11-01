"""
Hard inventory target — the action frontier.

Answers "cut inventory to $5M by year end" with a set of ordered moves rather than a
single number, because the decision the planner has to defend is a trade-off and a
point recommendation hides exactly the part they will be asked about.

Three things separate this from a sorted list of excess:

  Burn-down is time-bounded.  Excess stock only converts to cash as fast as demand
                              consumes it. A SKU with 300 days of cover cannot
                              contribute its full excess by December — only the
                              portion that will actually be consumed. Ignoring this
                              is how inventory plans miss by a wide margin while
                              every line item looks defensible.

  Actions are ordered by pain, not by size.  Free and reversible first (stop buying,
                              push out inbound), then recurring cost, then service.
                              A larger action that costs service should never appear
                              above a smaller one that costs nothing.

  Exclusions are stated.      What was deliberately *not* recommended, and why, is
                              part of the output. The planner needs it to answer
                              "did you consider…" and it prevents a later run from
                              silently reintroducing a rejected idea.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .levers import LeverAnalysis
from .should_be import ShouldBeResult

# Ordering of pain. Actions are ranked by this first, value second.
PAIN_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

TARGETS_FILE = "targets.json"


class StatedTargetError(ValueError):
    """`targets.json` says something the planner cannot act on."""


@dataclass(frozen=True)
class StatedTarget:
    """The standing target, as the config file holds it."""

    value: Optional[float] = None
    deadline: Optional[date] = None
    source: str = "not stated"
    # Days of inventory on hand. The *primary* way the balance target is stated, with
    # `value` derived from it per run — see `dioh_days` below.
    dioh_days: Optional[float] = None
    # Customer-request-date on-time delivery. Not a balance target and it enters no
    # frontier: it is what the service figure is scored against.
    otd_target: Optional[float] = None

    def __bool__(self) -> bool:
        return self.value is not None or self.dioh_days is not None

    @property
    def undated(self) -> bool:
        """A target with no deadline: planned, but without the burn-down limit."""
        return bool(self) and self.deadline is None

    @property
    def stated_in(self) -> Optional[str]:
        """Which unit the planner stated the balance target in, if they stated one."""
        if self.dioh_days is not None:
            return "dioh_days"
        if self.value is not None:
            return "value"
        return None


def stated_target(config_dir: Union[str, Path, None]) -> StatedTarget:
    """
    Read the standing inventory target, or report that none is stated.

    Null is not zero, and the difference matters more here than anywhere else in the
    config: null means nobody has said what the balance should be, so no frontier is
    computed and the run offers no opinion about what stock must be. Zero is a target,
    and a reachable one for a catalogue being discontinued.

    A malformed target refuses rather than defaults. Every other reading in this package
    degrades — a missing rate blanks a column, a missing adapter is drafted — because
    the degraded answer is still an answer about the data. There is no degraded reading
    of "cut inventory to $5M": a target nobody can parse, silently dropped, produces a
    run that looks exactly like a run nobody set a target on.
    """
    root = Path(config_dir) if config_dir else Path(__file__).parents[2] / "config"
    path = root / TARGETS_FILE
    if not path.exists():
        return StatedTarget(source=f"no {TARGETS_FILE}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StatedTargetError(f"{path} could not be read: {exc}") from exc

    value, deadline = raw.get("inventory_target_value"), raw.get("inventory_target_date")
    dioh = _as_number(raw.get("dioh_target_days"), "dioh_target_days", path, floor=0.0)
    otd = _as_number(raw.get("otd_target_crsd"), "otd_target_crsd", path,
                     floor=0.0, ceiling=1.0)

    # One balance target, in one unit. DIOH days and a currency amount are the same
    # target expressed two ways — days x daily cost of sales — so accepting both as
    # inputs lets them disagree, and the run would then plan down to one while the
    # report scored against the other. Neither is more correct, so the file is not
    # allowed to say both and the planner is told which one to clear.
    if dioh is not None and value is not None:
        raise StatedTargetError(
            f"{path} states the balance target twice: dioh_target_days ({dioh}) and "
            f"inventory_target_value ({value}). They are one target in two units — "
            f"days x daily cost of sales — so holding both lets them contradict, with "
            f"nothing to say which the run planned to. Keep dioh_target_days, which is "
            f"the unit a planner holds steady while cost and demand move underneath "
            f"it, and clear inventory_target_value; the currency figure is then derived "
            f"per run and reported beside it.")

    if value is None and dioh is None:
        if deadline is not None:
            raise StatedTargetError(
                f"{path} gives inventory_target_date ({deadline}) but no balance "
                f"target. A date on its own plans nothing, and ignoring it would hide "
                f"a target somebody meant to set. Give dioh_target_days, or clear the "
                f"date too — to withdraw a target, clear the date first and the target "
                f"second, since the state in between is this one.")
        return StatedTarget(source=str(path), otd_target=otd)
    if value is not None and (isinstance(value, bool)
                              or not isinstance(value, (int, float))):
        raise StatedTargetError(
            f"{path}: inventory_target_value must be a number in the reporting "
            f"currency — {value!r} is not.")
    return StatedTarget(value=float(value) if value is not None else None,
                        deadline=_as_date(deadline, path), source=str(path),
                        dioh_days=dioh, otd_target=otd)


def resolve_stated_target(stated: StatedTarget, should_be) -> Tuple[Optional[float],
                                                                      Optional[str]]:
    """
    The currency figure the frontier plans to, from however the planner stated it.

    A days-of-inventory target is converted here and nowhere else, because the
    conversion needs the run: days x daily cost of sales, and the daily cost of sales
    is a property of this catalogue at these costs. That is also the reason DIOH is the
    unit the planner states and the currency figure is derived rather than the other
    way round — a balance-sheet number held fixed while cost and demand move under it
    silently becomes a different amount of cover every month, which is the opposite of
    what anyone setting it meant.

    Returns `(value, note)`. The note is for the planner and says which way round the
    two figures were worked out, so the report never shows a derived number in a place
    a stated one would sit without saying so.
    """
    if stated.dioh_days is None:
        return stated.value, None
    value = should_be.value_at_dioh(stated.dioh_days) if should_be is not None else None
    if value is None:
        return None, (
            f"    dioh_target_days is {stated.dioh_days:g} and no frontier was "
            f"computed: the catalogue's daily cost of sales is zero, so there is "
            f"nothing to convert the days through. Prices or demand are missing, not "
            f"the target.")
    return value, (
        f"    target stated as {stated.dioh_days:g} days of inventory — "
        f"${value:,.0f} at this run's daily cost of sales "
        f"(${should_be.daily_cost_of_sales:,.0f}/day). The currency figure is derived; "
        f"it moves with cost and demand while the days do not.")


def _as_number(raw: Any, key: str, path: Path, floor: float = None,
               ceiling: float = None) -> Optional[float]:
    """
    One numeric target, or None where none is stated.

    Refused rather than coerced, and bounded. `otd_target_crsd: 95` is the shape this
    catches: a percentage typed where a fraction is read would set a target of 9,500%,
    which no run could miss, so the figure would report success for ever.
    """
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise StatedTargetError(f"{path}: {key} must be a number — {raw!r} is not.")
    number = float(raw)
    if floor is not None and number < floor:
        raise StatedTargetError(f"{path}: {key} must be at least {floor} — "
                                f"{number} is not.")
    if ceiling is not None and number > ceiling:
        raise StatedTargetError(
            f"{path}: {key} must be between {floor} and {ceiling} — {number} is not. "
            f"A service level is a fraction, so 95% is 0.95; {number} would be a "
            f"target no run could miss and the figure would report success for ever.")
    return number


def _as_date(raw: Any, path: Path) -> Optional[date]:
    if raw is None:
        return None
    try:
        return datetime.strptime(str(raw).strip(), "%Y-%m-%d").date()
    except ValueError:
        raise StatedTargetError(
            f"{path}: inventory_target_date must be YYYY-MM-DD — {raw!r} is not. The "
            f"deadline is what bounds the burn-down, so a date read loosely would move "
            f"how much of the excess counts as reachable.") from None


@dataclass
class Action:
    """One move toward the target."""

    kind: str
    label: str
    skus: List[str]
    value_released: float          # balance-sheet reduction achievable by the deadline
    value_theoretical: float       # reduction if time were unlimited
    annual_cost: float = 0.0
    cost_known: bool = True
    service_risk: str = "none"     # none | low | medium | high
    reversible: bool = True
    rationale: str = ""
    time_to_effect_days: int = 0
    lands_in_time: bool = True

    @property
    def pain_rank(self) -> int:
        return PAIN_ORDER.get(self.service_risk, 2)

    @property
    def time_limited(self) -> bool:
        return self.value_released < self.value_theoretical * 0.99

    def __str__(self) -> str:
        cost = ""
        if not self.cost_known:
            cost = "  (cost unpriced)"
        elif self.annual_cost > 0:
            cost = f"  costs ${self.annual_cost:,.0f}/yr"
        limit = ""
        if self.time_limited:
            limit = f"  [${self.value_theoretical:,.0f} in full, limited by burn-down]"
        return (f"{self.label:<44} ${self.value_released:>11,.0f}"
                f"  {len(self.skus):>4} SKUs{cost}{limit}")


@dataclass
class ActionFrontier:
    """The ordered set of moves, with cumulative progress against the target."""

    current_value: float
    target_value: float
    deadline: Optional[date]
    actions: List[Action] = dc_field(default_factory=list)
    excluded: List[Action] = dc_field(default_factory=list)
    too_slow: List[Action] = dc_field(default_factory=list)

    @property
    def gap(self) -> float:
        return self.current_value - self.target_value

    @property
    def total_available(self) -> float:
        return sum(a.value_released for a in self.actions)

    @property
    def reachable(self) -> bool:
        return self.total_available >= self.gap

    def cumulative(self) -> pd.DataFrame:
        """Actions in order, with the running balance after each."""
        rows = []
        remaining = self.current_value
        for action in self.actions:
            remaining -= action.value_released
            rows.append({
                "action": action.label,
                "skus": len(action.skus),
                "released": action.value_released,
                "annual_cost": action.annual_cost if action.cost_known else np.nan,
                "service_risk": action.service_risk,
                "running_balance": remaining,
                "hits_target": remaining <= self.target_value,
            })
        return pd.DataFrame(rows)

    def sufficient_set(self) -> List[Action]:
        """The shortest prefix of the frontier that reaches the target."""
        out, remaining = [], self.current_value
        for action in self.actions:
            if remaining <= self.target_value:
                break
            out.append(action)
            remaining -= action.value_released
        return out

    def summary(self) -> str:
        lines = [
            "  Inventory target — action frontier",
            "  " + "-" * 62,
            f"    current   ${self.current_value:>13,.0f}",
            f"    target    ${self.target_value:>13,.0f}"
            + (f"   by {self.deadline}" if self.deadline else ""),
            f"    gap       ${self.gap:>13,.0f}",
            "",
        ]
        if self.gap <= 0:
            lines.append("    Already at or below target — no action needed.")
            return "\n".join(lines)

        lines.append("    Ordered by pain, not by size:")
        lines.append("")
        remaining = self.current_value
        reached = False
        for i, action in enumerate(self.actions, 1):
            remaining -= action.value_released
            marker = " "
            if not reached and remaining <= self.target_value:
                marker, reached = "◀", True
            lines.append(f"    {i}. {action}")
            lines.append(f"       └ balance ${remaining:,.0f} {marker}")
            if action.rationale:
                lines.append(f"         {action.rationale}")

        lines.append("")
        if self.reachable:
            needed = self.sufficient_set()
            worst = max((a.service_risk for a in needed), key=lambda r: PAIN_ORDER.get(r, 0))
            lines.append(f"    ✓ Target reachable with the first {len(needed)} action(s); "
                         f"worst service risk incurred: {worst}")
        else:
            short = self.gap - self.total_available
            lines.append(f"    ✗ Target NOT reachable — ${short:,.0f} short even after every "
                         f"action above. The target needs renegotiating, or demand has to fall.")

        if self.too_slow:
            lines.append("")
            lines.append("    Right move, wrong horizon — start now, counts next year:")
            for action in self.too_slow:
                lines.append(f"      ⏳ {action.label} (${action.value_theoretical:,.0f}, "
                             f"~{action.time_to_effect_days}d to take effect)")

        if self.excluded:
            lines.append("")
            lines.append("    Deliberately not recommended:")
            for action in self.excluded:
                lines.append(f"      ✗ {action.label} (${action.value_theoretical:,.0f})")
                lines.append(f"         {action.rationale}")
        return "\n".join(lines)


class TargetPlanner:
    """Builds the action frontier for a hard inventory target."""

    def __init__(self, holding_rate: float = 0.22):
        self.holding_rate = holding_rate

    def plan(
        self,
        should_be: ShouldBeResult,
        target_value: float,
        deadline: date = None,
        levers: LeverAnalysis = None,
        open_po: pd.DataFrame = None,
        as_of: date = None,
    ) -> ActionFrontier:
        as_of = as_of or date.today()
        days_left = max((deadline - as_of).days, 0) if deadline else None

        frontier = ActionFrontier(
            current_value=should_be.total_actual_value,
            target_value=float(target_value),
            deadline=deadline,
        )
        if frontier.gap <= 0:
            return frontier

        df = should_be.frame

        candidates = [self._stop_buying(df, days_left)]
        if open_po is not None and len(open_po):
            candidates.append(self._push_out(df, open_po, days_left))
        if levers is not None:
            candidates.extend(self._from_levers(levers, df, days_left))

        candidates = [a for a in candidates if a and a.value_released > 0]

        # An action that cannot take effect before the deadline is real work but not
        # part of *this* plan. Keeping it in the running balance would produce a plan
        # that reconciles on paper and misses in reality.
        frontier.actions = [a for a in candidates if a.lands_in_time]
        frontier.too_slow = [a for a in candidates if not a.lands_in_time]
        frontier.excluded = self._exclusions(df, levers)

        # Free and reversible first, then cost, then service. Value only breaks ties
        # inside a pain tier — a bigger but more painful action must never outrank a
        # smaller free one.
        frontier.actions.sort(key=lambda a: (a.pain_rank, not a.reversible, -a.value_released))
        return frontier

    # ── Individual actions ───────────────────────────────────────────────────

    def _stop_buying(self, df: pd.DataFrame, days_left: Optional[int]) -> Optional[Action]:
        """
        Let demand consume what is already above policy.

        The largest and cheapest source of reduction, and the one most often overstated
        — which is why the burn-down limit matters. Excess only becomes cash at the
        rate demand consumes it.
        """
        over = df[df["gap_value"] > 0].copy()
        if over.empty:
            return None

        theoretical = float(over["gap_value"].sum())

        if days_left is not None:
            daily = over["demand_mean"] / 30.0
            consumable_qty = np.minimum(over["gap_qty"], daily * days_left)
            releasable = float((consumable_qty * over["unit_cost"].fillna(0.0)).sum())
        else:
            releasable = theoretical

        slow = over[over["demand_mean"] <= 0]
        note = (
            f"Stop replenishing while stock is above policy; demand draws it down. "
            f"{len(over)} SKUs above should-be."
        )
        if len(slow):
            note += (f" {len(slow)} of them have no demand and will not burn down at all — "
                     f"those need liquidation, not patience.")

        return Action(
            kind="stop_buying",
            label="Stop buying where above policy (burn down)",
            skus=over["sku"].astype(str).tolist(),
            value_released=releasable,
            value_theoretical=theoretical,
            service_risk="none",
            reversible=True,
            rationale=note,
        )

    def _push_out(
        self, df: pd.DataFrame, open_po: pd.DataFrame, days_left: Optional[int]
    ) -> Optional[Action]:
        """
        Delay inbound POs for SKUs already above policy.

        Free and fully reversible, but only for SKUs whose position stays above
        should-be without the receipt — pushing out a PO that is covering a genuine
        shortage converts a balance-sheet win into a stockout.
        """
        if "sku" not in open_po.columns or "open_qty" not in open_po.columns:
            return None

        over = df[df["gap_value"] > 0][["sku", "gap_qty", "unit_cost"]]
        # Take only the two columns needed from the PO frame. Merging it whole collides
        # on `unit_cost`, which both sides legitimately carry, and the suffixed result
        # silently loses the valuation.
        candidates = open_po[["sku", "open_qty"]].merge(over, on="sku", how="inner")
        if candidates.empty:
            return None

        # Never push out more than the SKU is actually over by.
        pushable = candidates.groupby("sku", as_index=False).agg(
            open_qty=("open_qty", "sum"), gap_qty=("gap_qty", "first"),
            unit_cost=("unit_cost", "first"),
        )
        pushable["push_qty"] = np.minimum(pushable["open_qty"], pushable["gap_qty"]).clip(lower=0)
        pushable["push_value"] = pushable["push_qty"] * pushable["unit_cost"].fillna(0.0)
        pushable = pushable[pushable["push_value"] > 0]
        if pushable.empty:
            return None

        value = float(pushable["push_value"].sum())
        return Action(
            kind="push_out_po",
            label="Push out inbound POs on over-stocked SKUs",
            skus=pushable["sku"].astype(str).tolist(),
            value_released=value,
            value_theoretical=value,
            service_risk="low",
            reversible=True,
            rationale=("Capped at each SKU's excess, so no PO covering a genuine "
                       "shortage is delayed. Supplier may charge for rescheduling."),
        )

    def _from_levers(
        self, levers: LeverAnalysis, df: pd.DataFrame, days_left: Optional[int] = None
    ) -> List[Action]:
        """Turn policy levers into target actions, carrying their economics through."""
        out: List[Action] = []
        risk_by_kind = {"service": "high", "ordering": "none",
                        "freight": "low", "supplier": "none", "forecast": "none"}

        for result in levers.results.values():
            if result.total_saving <= 0:
                continue
            # A lever that costs more than it frees does not belong in a plan to free
            # cash — it appears in the exclusions instead.
            if result.is_value_destroying:
                continue

            lead = result.spec.time_to_effect_days
            lands = days_left is None or lead <= days_left
            note = result.spec.description
            if not lands:
                note += (f" — takes ~{lead}d to affect stock, past the deadline; "
                         f"worth starting, but it will not help this target")

            affected = result.frame[result.frame["saving_value"] > 0]["sku"].astype(str).tolist()
            out.append(Action(
                kind=f"lever:{result.spec.key}",
                label=result.spec.label,
                skus=affected,
                value_released=result.total_saving,
                value_theoretical=result.total_saving,
                annual_cost=result.annual_cost,
                cost_known=result.cost_known,
                service_risk=risk_by_kind.get(result.spec.cost_kind, "medium"),
                reversible=True,
                rationale=note,
                time_to_effect_days=lead,
                lands_in_time=lands,
            ))
        return out

    def _exclusions(self, df: pd.DataFrame, levers: Optional[LeverAnalysis]) -> List[Action]:
        """
        Moves that would reduce inventory but should not be taken.

        Stating these is what lets the planner answer "did you consider…" without
        re-deriving the analysis, and stops a later run from quietly proposing
        something already rejected on principle.
        """
        out: List[Action] = []

        under = df[df["gap_value"] < 0]
        if len(under):
            out.append(Action(
                kind="excluded:cut_below_policy",
                label=f"Cut the {len(under)} SKUs already below policy",
                skus=under["sku"].astype(str).tolist(),
                value_released=0.0,
                value_theoretical=float(-under["gap_value"].sum()),
                service_risk="high",
                reversible=False,
                rationale=("These are already under should-be. Cutting them buys balance "
                           "sheet with stockouts on items that are short today."),
            ))

        if levers is not None:
            for result in levers.results.values():
                if result.is_value_destroying:
                    out.append(Action(
                        kind=f"excluded:{result.spec.key}",
                        label=result.spec.label,
                        skus=[],
                        value_released=0.0,
                        value_theoretical=result.total_saving,
                        annual_cost=result.annual_cost,
                        service_risk="none",
                        rationale=(
                            f"Frees ${result.total_saving:,.0f} of stock but costs "
                            f"${result.annual_cost:,.0f}/yr to run — a net loss of "
                            f"${abs(result.net_annual):,.0f}/yr. Buying balance sheet with P&L."
                        ),
                    ))
        return out
