"""
Lot sizing — how much to buy once you know you are buying.

EOQ balances the cost of ordering too often against the cost of holding too much:

    Q* = sqrt(2 · ct · D / ce)        ct = cost per order, ce = c · h per unit per year

It answers a different question from *when* to order, which is why it survives
unchanged when a lead time is added — lead time moves the reorder point and leaves Q*
alone (MIT CTL §5).

Three callers need it and used to each carry their own copy: the recommender sizes the
buy with it, the suggestion engine reads the ordering *frequency* it implies, and the
lever analysis asks whether the observed order sizes are anywhere near it. Three copies
of one formula is three chances for them to disagree in front of a planner, so it lives
here once.

The real order is rarely Q* exactly. A supplier has a minimum, a pallet has a count, a
container has a capacity — so `round_to_lot` applies those constraints afterwards, and
always upward. EOQ is famously flat near its optimum (a 50% error in Q costs about 8%),
and where the curve is that shallow the asymmetry decides: rounding down risks a second
order and a second order cost, rounding up costs a little carrying. Order slightly more,
not slightly less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def economic_order_quantity(annual_demand, unit_cost, order_cost: float,
                            holding_rate: float):
    """
    Q* per SKU. Returns NaN where it is undefined rather than a number.

    An unpriced item has no holding cost to trade off, and an item with no demand has
    no ordering frequency to optimise. Both are common in a real extract and neither is
    an error — but substituting a zero would produce a confident lot size built on
    nothing, so they come back as NaN and the caller decides what to say about it.
    """
    demand = pd.to_numeric(pd.Series(annual_demand), errors="coerce").clip(lower=0)
    cost = pd.to_numeric(pd.Series(unit_cost), errors="coerce")
    holding_per_unit_year = cost * float(holding_rate)

    with np.errstate(divide="ignore", invalid="ignore"):
        eoq = np.sqrt(2 * demand * float(order_cost) / holding_per_unit_year)
    eoq = pd.Series(eoq, index=demand.index).replace([np.inf, -np.inf], np.nan)
    return eoq.where(demand > 0)


def round_to_lot(qty, min_order_qty=0, order_multiple=1):
    """
    Raise a quantity to what the supplier will actually accept.

    Always upward, and the minimum is applied before the multiple so an MOQ that is not
    itself a whole number of pallets still comes out orderable.
    """
    q = pd.to_numeric(pd.Series(qty), errors="coerce").fillna(0.0).clip(lower=0)
    moq = pd.to_numeric(pd.Series(min_order_qty), errors="coerce").fillna(0.0)
    if not isinstance(min_order_qty, pd.Series):
        moq = pd.Series(float(moq.iloc[0]) if len(moq) else 0.0, index=q.index)
    multiple = pd.to_numeric(pd.Series(order_multiple), errors="coerce").fillna(1.0)
    if not isinstance(order_multiple, pd.Series):
        multiple = pd.Series(float(multiple.iloc[0]) if len(multiple) else 1.0, index=q.index)

    moq = moq.reindex(q.index).fillna(0.0)
    multiple = multiple.reindex(q.index).fillna(1.0).replace(0.0, 1.0)

    raised = np.maximum(q, moq)
    # Only a multiple above 1 constrains anything; below that the ceiling would round
    # every fractional quantity to a whole unit, which is a different decision.
    stepped = np.where(multiple > 1, np.ceil(raised / multiple) * multiple, raised)
    return pd.Series(stepped, index=q.index).where(q > 0, 0.0)
