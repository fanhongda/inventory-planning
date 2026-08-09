"""
How much of the open order book actually turns into shipments.

The purchase recommender needs a number for next period's demand. Two sources claim
to be that number and they do not agree:

  the forecast    fitted on shipment history — what the customer has actually taken
  the backlog     the open order book — what the customer has said they will take

Adding them together, which is what this pipeline used to do, counts the same demand
twice: the forecast is fitted on shipments, and those shipments came from orders that
were once backlog. On top of that it treats every open line as certain to ship. Where
customers do not collect on schedule, the order book is systematically larger than
what will move, and buying against it over-orders every single cycle.

This module measures the discount instead of assuming one. The signal is already in
the data and `policy.service` already computes it: an open line **past its request
date with the stock sitting on the shelf** is a line the customer has demonstrably not
pulled. Its share of the order book is the non-realization rate.

    realization_rate = 1 − uncollected_qty / judgeable_open_qty

Two deliberate choices:

  shrinkage   A SKU with one open line would otherwise get a rate of exactly 0% or
              exactly 100%. Per-SKU rates are pulled toward the portfolio rate in
              proportion to how few lines support them, so a thin sample says little
              and a thick one says a lot.

  a floor     The rate never drops below a configured minimum. A measured 5% would
              zero out purchasing for that SKU on the strength of a collection problem
              that may resolve next week, and being unable to ship at all is a worse
              failure than holding stock. The floor is a policy choice, so it is
              configuration rather than a constant.

Where availability cannot be judged — no inventory snapshot — nothing is measured and
the rate is 1.0 for everything. An unmeasured discount is not applied silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ..policy.service import (
    OPEN_NOT_YET_DUE,
    OPEN_PAST_DUE_AVAILABLE,
    OPEN_PAST_DUE_SHORT,
    ServiceAnalyzer,
)

# Open-line states that carry a usable request date, and so can be judged at all.
_JUDGEABLE = {OPEN_PAST_DUE_AVAILABLE, OPEN_PAST_DUE_SHORT, OPEN_NOT_YET_DUE}

# Prior strength, in order lines, for pulling a per-SKU rate toward the portfolio rate.
# Three lines is roughly where a per-SKU rate starts carrying more information than the
# portfolio average.
_SHRINKAGE_LINES = 3.0


@dataclass
class RealizationResult:
    """Per-SKU backlog realization rates and the evidence behind them."""

    per_sku: pd.DataFrame            # sku, open_qty, uncollected_qty, raw_rate, realization_rate, open_lines
    global_rate: float = 1.0
    measured: bool = False
    reason: str = ""
    floor: float = 0.25
    as_of: Optional[date] = None
    _lookup: dict = dc_field(default_factory=dict, repr=False)

    def __post_init__(self):
        if len(self.per_sku):
            self._lookup = dict(zip(self.per_sku["sku"], self.per_sku["realization_rate"]))

    def rate_for(self, sku: str) -> float:
        """Realization rate for one SKU, falling back to the portfolio rate."""
        return float(self._lookup.get(sku, self.global_rate))

    def apply(self, skus: pd.Series) -> pd.Series:
        return skus.map(self._lookup).fillna(self.global_rate).astype(float)

    def summary(self) -> str:
        if not self.measured:
            applied = (
                "Open backlog is taken at face value."
                if self.global_rate >= 0.9995
                else f"Every SKU is discounted to {self.global_rate:.0%} regardless of "
                     f"its own order book."
            )
            return f"  Backlog realization: not measured — {self.reason}. {applied}"

        f = self.per_sku
        discounted = int((f["realization_rate"] < 0.995).sum())
        at_floor = int((f["raw_rate"] < self.floor).sum())
        lines = [
            "  Backlog realization (measured)",
            "  " + "-" * 58,
            f"    Portfolio rate      : {self.global_rate:.0%} of open backlog qty is "
            f"expected to ship",
            f"    Basis               : open lines past the request date with stock on "
            f"the shelf are counted as not realizing",
            f"    SKUs discounted     : {discounted:,} of {len(f):,}",
        ]
        if at_floor:
            lines.append(
                f"    ⚠ {at_floor:,} SKUs measured below the {self.floor:.0%} floor and were "
                f"held at it — a collection problem, not a demand signal. Buying nothing "
                f"for them would turn one failure into two."
            )
        worst = f.nsmallest(5, "realization_rate")
        worst = worst[worst["realization_rate"] < 0.995]
        if len(worst):
            lines.append("")
            lines.append("    Lowest realization:")
            for _, row in worst.iterrows():
                lines.append(
                    f"      {str(row['sku']):<18} {row['realization_rate']:>5.0%}   "
                    f"{row['uncollected_qty']:>10,.0f} of {row['open_qty']:>10,.0f} open "
                    f"qty sitting uncollected ({int(row['open_lines'])} lines)"
                )
        return "\n".join(lines)


class BacklogRealizationEstimator:
    """Estimates what share of open backlog converts to shipments."""

    def __init__(self, floor: float = 0.25, shrinkage_lines: float = _SHRINKAGE_LINES):
        self.floor = float(floor)
        self.shrinkage_lines = float(shrinkage_lines)

    def estimate(
        self,
        open_so: pd.DataFrame = None,
        inventory: pd.DataFrame = None,
        as_of: date = None,
    ) -> RealizationResult:
        empty = pd.DataFrame(columns=["sku", "open_qty", "uncollected_qty", "raw_rate",
                                      "realization_rate", "open_lines"])

        if open_so is None or len(open_so) == 0:
            return RealizationResult(empty, reason="no open sales orders supplied")
        if inventory is None or len(inventory) == 0 or "qty_on_hand" not in inventory.columns:
            return RealizationResult(
                empty,
                reason="no inventory snapshot — whether a past-due line was fillable "
                       "cannot be judged without knowing what was on the shelf",
            )

        # Reuse the service classifier rather than re-deriving availability. It
        # allocates on-hand across a SKU's lines oldest-first, so the same units are
        # not reported as available to several lines at once.
        service = ServiceAnalyzer().analyze(open_so=open_so, inventory=inventory, as_of=as_of)
        lines = service.lines[service.lines["source"] == "open"]
        lines = lines[lines["service_state"].isin(_JUDGEABLE)]

        if lines.empty:
            return RealizationResult(
                empty,
                reason="no open line carries a customer request date, so none can be "
                       "judged past due",
                as_of=service.as_of,
            )

        lines = lines.assign(
            _uncollected=np.where(
                lines["service_state"] == OPEN_PAST_DUE_AVAILABLE, lines["qty"], 0.0
            )
        )

        total_open = float(lines["qty"].sum())
        total_uncollected = float(lines["_uncollected"].sum())
        global_rate = 1.0 - (total_uncollected / total_open) if total_open > 0 else 1.0
        global_rate = float(np.clip(global_rate, self.floor, 1.0))

        per_sku = (
            lines.groupby("sku")
            .agg(open_qty=("qty", "sum"),
                 uncollected_qty=("_uncollected", "sum"),
                 open_lines=("qty", "size"))
            .reset_index()
        )
        per_sku["raw_rate"] = np.where(
            per_sku["open_qty"] > 0,
            1.0 - per_sku["uncollected_qty"] / per_sku["open_qty"],
            1.0,
        )

        # Empirical-Bayes shrinkage toward the portfolio rate, weighted by line count.
        weight = per_sku["open_lines"] / (per_sku["open_lines"] + self.shrinkage_lines)
        per_sku["realization_rate"] = (
            weight * per_sku["raw_rate"] + (1 - weight) * global_rate
        ).clip(lower=self.floor, upper=1.0).round(3)

        return RealizationResult(
            per_sku=per_sku,
            global_rate=round(global_rate, 3),
            measured=True,
            floor=self.floor,
            as_of=service.as_of,
        )
