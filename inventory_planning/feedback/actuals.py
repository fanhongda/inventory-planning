"""
What actually happened, read from the store instead of hand-fed.

INTERFACE.md §7 item 5 says decisions belong in the store, and that feedback learning is
the thing to build that for. Looking at it turned up the reason to do something else
first: **feedback learning has never run.** 2,430 decision snapshots, 63 MB, and not one
of them has ever had an actual recorded against it. `FeedbackCollector`, `loss.py` and
`drift.py` have no caller anywhere in the package.

The reason is visible in the signature it was never called with:

    collector.record_actuals(actual_sales_df, actual_inventory_df)

It asks a person, a month later, to assemble two frames by hand and point them at a
snapshot path. Nobody does that. And the store already holds both — 575 batches of
sales history, 576 of inventory — so the loop was never closed because it was built
before the store existed and never rewired.

This module is the rewiring. Two things fall out of it:

**Nothing is written back.** `record_actuals` wrote the actuals into the snapshot file,
which makes a decision record mutable — against the discipline DATA_LAYER sets for the
decision table, and against the reason facts are append-only. Reading the actuals at
scoring time means there is nothing to write back, so the snapshot is immutable because
nothing needs it to change.

**The scoring period comes from `as_of`, not from `planning_month`.** A snapshot carries
both and they are different things: `planning_month` is the wall clock when the run
executed, `as_of` is the newest date in the data the run anchored to. `forecast_next_period`
is the forecast for the period *after `as_of`*. Scoring against `planning_month` would
compare a forecast for 2024-08 against demand in 2026-09 — a confident, wrong number, of
exactly the kind the rest of this pipeline exists to refuse.

## Why a named batch, and not a query

`FactQuery.current("sales_history")` refuses: the stored batches carry no
`so_line_number`, so the natural key is incomplete and reducing on a partial key would
collapse rows that are genuinely different. That refusal is correct and is not this
module's business to get round. `history()` would multiply-count instead — 283,985 row
observations across 569 batches, 558 of which are the same 540-row sample file
re-imported by development runs.

So scoring names the batch it scores against, the way a run manifest names the facts it
planned on. One extract is internally consistent, needs no cross-batch key, and the
batch id travels into the score — which makes the score reproducible rather than a
number produced against whatever the store happened to hold that day.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ActualsUnavailable(ValueError):
    """Scoring cannot proceed, with what is missing and what would supply it."""


def scoring_period(snapshot: Dict[str, Any]) -> Tuple[date, date]:
    """
    The period a snapshot's `forecast_next_period` was a forecast *for*.

    The month after `as_of`, half-open: `[start, end)`. Derived rather than read off the
    snapshot because `planning_month` is the wall clock and would score the wrong
    months — see the module docstring.
    """
    import pandas as pd

    raw = snapshot.get("as_of")
    if not raw:
        raise ActualsUnavailable(
            "this snapshot records no `as_of`, so there is no way to know which period "
            "its forecast was for. It predates the field; score a newer run.")
    anchor = pd.Timestamp(raw)
    start = (anchor + pd.offsets.MonthBegin(1)).normalize()
    end = (start + pd.offsets.MonthBegin(1)).normalize()
    return start.date(), end.date()


@dataclass
class ActualsSource:
    """Which batches a score was computed against — carried into the score itself."""

    sales_batch: str
    inventory_batch: Optional[str] = None
    period_start: str = ""
    period_end: str = ""
    sales_rows: int = 0
    inventory_rows: int = 0
    notes: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"sales_batch": self.sales_batch,
                "inventory_batch": self.inventory_batch,
                "period_start": self.period_start, "period_end": self.period_end,
                "sales_rows": self.sales_rows, "inventory_rows": self.inventory_rows,
                "notes": list(self.notes)}


def candidates(store, doc_type: str, covering: date = None,
               limit: int = 8) -> List[Dict[str, Any]]:
    """
    Batches that could supply the actuals, largest first.

    Largest first because the useful ones are the real extracts: a store that has been
    developed against is mostly the same sample file re-imported by every run, and a
    list in load order buries the extract under five hundred of them.
    """
    entries = [b for b in store.ledger.batches(doc_type=doc_type)]
    if covering:
        entries = [b for b in entries
                   if str(b.get("valid_time") or "") >= covering.isoformat()] or entries
    entries.sort(key=lambda b: (-(b.get("rows") or 0), str(b.get("valid_time") or "")))
    return [{"batch_id": b["batch_id"], "rows": b.get("rows"),
             "valid_time": b.get("valid_time"), "source_name": b.get("source_name")}
            for b in entries[:limit]]


def actuals_for(snapshot: Dict[str, Any], store, sales_batch: str,
                inventory_batch: str = None) -> Tuple[Dict[str, Any], ActualsSource]:
    """
    The actuals for one snapshot's scoring period, in memory.

    Returns what `loss.py` expects under `snapshot["actuals"]`, and the source that
    produced it. Nothing is written: the snapshot stays exactly as the run left it.
    """
    import pandas as pd

    start, end = scoring_period(snapshot)
    planned = snapshot.get("skus") or {}
    if not planned:
        raise ActualsUnavailable("this snapshot recommends nothing, so there is "
                                 "nothing to score.")

    sales = store.read_batch(sales_batch, doc_type="sales_history")
    if sales is None:
        raise ActualsUnavailable(
            f"no sales_history batch {sales_batch!r} in this store.")

    source = ActualsSource(sales_batch=sales_batch, inventory_batch=inventory_batch,
                           period_start=start.isoformat(), period_end=end.isoformat())

    date_col = next((c for c in ("demand_date", "ship_date", "order_date")
                     if c in sales.columns), None)
    if date_col is None:
        raise ActualsUnavailable(
            f"batch {sales_batch} carries none of demand_date / ship_date / "
            f"order_date, so its rows cannot be placed in a period.")
    if date_col != "demand_date":
        source.notes.append(
            f"placed by `{date_col}`; this batch carries no `demand_date`")

    dates = pd.to_datetime(sales[date_col], errors="coerce")
    window = sales[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))]
    source.sales_rows = int(len(window))
    if not len(window):
        raise ActualsUnavailable(
            f"batch {sales_batch} has no rows dated in {start:%Y-%m} — the period this "
            f"plan was made for. Scoring it against an extract that does not cover the "
            f"period would report every SKU as having sold nothing.")

    demand = (window.groupby(window["sku"].astype(str))["qty"].sum().to_dict()
              if "qty" in window.columns else {})

    on_hand: Dict[str, float] = {}
    if inventory_batch:
        inventory = store.read_batch(inventory_batch, doc_type="inventory")
        if inventory is None:
            raise ActualsUnavailable(
                f"no inventory batch {inventory_batch!r} in this store.")
        source.inventory_rows = int(len(inventory))
        if "qty_on_hand" in inventory.columns:
            on_hand = (inventory.groupby(inventory["sku"].astype(str))["qty_on_hand"]
                       .sum().to_dict())
    else:
        source.notes.append(
            "no inventory batch given — realised days of supply cannot be computed, "
            "and the forecast half of the score is unaffected")

    # A SKU the plan covered but the extract did not mention sold nothing *that the
    # extract knows about*, which is not the same as having sold nothing. Left as None
    # rather than zeroed: `loss.py` skips a SKU with no actual demand, and a zero would
    # instead score it as a total forecast miss.
    actuals = {}
    for sku in planned:
        key = str(sku)
        actuals[key] = {
            "actual_demand": float(demand[key]) if key in demand else None,
            "actual_eom_inv": float(on_hand[key]) if key in on_hand else None,
            "actual_receipt_qty": None,
        }
    scored = sum(1 for a in actuals.values() if a["actual_demand"] is not None)
    if not scored:
        raise ActualsUnavailable(
            f"none of the {len(planned)} SKUs this plan covers appear in batch "
            f"{sales_batch} for {start:%Y-%m}. That is usually two extracts keyed on "
            f"different numbering systems rather than a month with no sales.")
    source.notes.append(f"{scored} of {len(planned)} planned SKUs had actual demand")
    return actuals, source


def scored_snapshot(snapshot: Dict[str, Any], actuals: Dict[str, Any]) -> Dict[str, Any]:
    """
    A copy of the snapshot with the actuals attached, for the scorer to read.

    A copy, and never the file. The snapshot is what the run decided; attaching what
    happened to it in place would make a decision record mutable, and a record that can
    be edited after the fact cannot answer what was decided at the time — which is the
    only question it exists for.
    """
    return dict(snapshot, actuals=actuals)
