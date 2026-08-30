"""
Did the plan we published turn out to be right?

## Why this is not the backtest

The forecaster already scores itself: every model is refitted on a rolling origin and
ranked on periods it had not seen. That number answers "is this the best model
available for this series", which is the modeller's question, and it is available on
the first run because it never leaves the history.

It is not the question an S&IOP meeting asks. That meeting published a number — after
sales reviewed it and, often, after they changed it — and the business bought stock,
booked capacity and set targets against it. What it needs to know next quarter is
whether **that** number held up. A model can win its backtest and still be wrong about
next March, and a reviewed forecast that overrode the model is not in the backtest at
all.

So accuracy here is the published plan against what actually sold. It requires history:
on a first run there is nothing to compare and this says so rather than substituting a
statistic that would look like an answer.

## Why there is no weighted average

A single accuracy number is a scoreboard, and a scoreboard closes a discussion. "MAPE
was 34%" tells nobody what to do differently; every attendee nods and the meeting moves
on. The same data ranked by **where the plan missed by the most money** opens one: this
line was forty per cent under all half-year, whose assumption was that, and what do we
change.

So the output is a ranking, not a mean — biggest absolute value deviation first, by
product line and by item, with the direction attached. Ranking by **value** rather than
by percentage is the same argument once more: a 300% miss on a part worth eight hundred
dollars is a rounding error, and a 6% miss on the largest line in the business is the
conversation worth having.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastAccuracy:
    """The published plan measured against what happened, ranked by the size of the miss."""

    by_family: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    by_sku: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    detail: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    adjustments: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    scored_periods: int = 0
    scored_skus: int = 0
    plans_read: int = 0
    # Net and gross, both in currency. Net says which way the business is biased —
    # they cancel, and a plan that is wildly wrong in both directions can net to zero.
    # Gross says how much was wrong at all. Neither is an average.
    bias_value: float = 0.0
    abs_error_value: float = 0.0
    planned_value: float = 0.0
    reviewed_skus: int = 0
    reason: str = ""
    currency: str = "USD"

    @property
    def measured(self) -> bool:
        return self.scored_periods > 0

    @property
    def bias_direction(self) -> str:
        if not self.measured or self.planned_value <= 0:
            return "unknown"
        share = self.bias_value / self.planned_value
        if share > 0.05:
            return "over-forecast"
        if share < -0.05:
            return "under-forecast"
        return "balanced"

    def summary(self, top: int = 5) -> str:
        if not self.measured:
            return ("  Forecast accuracy — not measurable yet\n"
                    "  " + "-" * 58 + f"\n    {self.reason}")

        lines = ["  Forecast accuracy — published plan against actual",
                 "  " + "-" * 58]
        lines.append(
            f"    {self.scored_periods:,} SKU-periods from {self.plans_read} earlier "
            f"plan{'s' if self.plans_read != 1 else ''} have closed and can be scored."
        )
        net = self.bias_value
        lines.append(
            f"    Net {abs(net):,.0f} {self.currency} "
            f"{'over' if net >= 0 else 'under'}-planned on "
            f"{self.planned_value:,.0f} planned — {self.bias_direction}. "
            f"{self.abs_error_value:,.0f} was wrong in one direction or the other."
        )
        lines.append(
            "    Net and gross, not an average. A mean closes the discussion; the "
            "ranking below opens one."
        )

        if len(self.by_family):
            lines.append("")
            lines.append("    Biggest misses by product line")
            lines.append(f"      {'Line':<26}{'Planned':>12}{'Actual':>12}{'Miss':>12}  ")
            for _, row in self.by_family.head(top).iterrows():
                arrow = "over" if row["bias_value"] >= 0 else "under"
                lines.append(
                    f"      {str(row['label'])[:25]:<26}{row['planned_value']:>12,.0f}"
                    f"{row['actual_value']:>12,.0f}{row['bias_value']:>11,.0f} {arrow}"
                )

        if len(self.by_sku):
            lines.append("")
            lines.append("    Biggest misses by item")
            for _, row in self.by_sku.head(top).iterrows():
                arrow = "over" if row["bias_value"] >= 0 else "under"
                pct = (f"{row['bias_pct']:+.0%}" if np.isfinite(row["bias_pct"]) else "—")
                lines.append(
                    f"      {str(row['sku'])[:16]:<18}{row['bias_value']:>12,.0f} "
                    f"{arrow:<6}{pct:>7}   {str(row.get('product_family', ''))[:22]}"
                )
        lines.append("")
        lines.append("    Ranked by the value of the miss, not the percentage. A 300% "
                     "miss on an eight-hundred-dollar part is a rounding error; 6% on "
                     "the largest line is the conversation.")
        return "\n".join(lines)


def build_forecast_accuracy(
    time_series: pd.DataFrame = None,
    history_root=None,
    forecast_detail: pd.DataFrame = None,
    attributes: pd.DataFrame = None,
    price: pd.Series = None,
    currency: str = "USD",
    exclude_run: str = "",
) -> ForecastAccuracy:
    """
    Score every earlier published plan against the demand that has since been observed.

    `time_series` is this run's actuals — period × SKU — and is the only source of
    truth about what happened. `history_root` is where snapshots accumulate; each holds
    the plan its run published.
    """
    plans, read = _load_plans(history_root, exclude_run)
    review = _review_frame(forecast_detail, attributes)

    if time_series is None or not len(time_series):
        return ForecastAccuracy(
            adjustments=review, plans_read=read, currency=currency,
            reason="No demand history in this run to compare a plan against.")
    if not len(plans):
        return ForecastAccuracy(
            adjustments=review, plans_read=read, currency=currency,
            reason=("No earlier plan has been published yet, so there is nothing to "
                    "score. Accuracy here is the plan the business committed to against "
                    "what then sold — it needs one closed period to exist, and appears "
                    "on the next run. The forecaster's own backtest is on every row of "
                    "`forecast_<run>.csv` in the meantime; it answers a different "
                    "question."))

    actual = _actual_frame(time_series)
    scored = plans.merge(actual, on=["sku", "period"], how="inner")
    if not len(scored):
        return ForecastAccuracy(
            adjustments=review, plans_read=read, currency=currency,
            reason=("Earlier plans exist but none of their periods has closed yet — "
                    "every period they cover is still in the future or absent from the "
                    "demand history."))

    scored["unit_price"] = (scored["sku"].map(price) if price is not None
                            else np.nan)
    scored["unit_price"] = pd.to_numeric(scored["unit_price"], errors="coerce")
    # Both sides valued at the same price, so the difference is a forecasting miss and
    # not a price movement. Valuing actuals at their own realised price would mix the
    # two and make a discount look like a demand error.
    scored["planned_value"] = scored["forecast_qty"] * scored["unit_price"]
    scored["actual_value"] = scored["actual_qty"] * scored["unit_price"]
    scored["error_qty"] = scored["forecast_qty"] - scored["actual_qty"]
    scored["error_value"] = scored["planned_value"] - scored["actual_value"]
    scored = _attach_dimensions(scored, attributes)

    by_sku = _rank_skus(scored)
    return ForecastAccuracy(
        by_family=_rank_families(scored),
        by_sku=by_sku,
        detail=scored,
        adjustments=review,
        scored_periods=len(scored),
        scored_skus=int(scored["sku"].nunique()),
        plans_read=read,
        bias_value=float(scored["error_value"].fillna(0.0).sum()),
        abs_error_value=float(scored["error_value"].abs().fillna(0.0).sum()),
        planned_value=float(scored["planned_value"].fillna(0.0).sum()),
        reviewed_skus=int(review["sku"].nunique()) if len(review) else 0,
        currency=currency,
    )


# ── Inputs ───────────────────────────────────────────────────────────────────


def _load_plans(history_root, exclude_run: str) -> tuple:
    """
    Every plan an earlier run published, as one frame.

    Where two runs planned the same SKU and period, the **earlier** one is kept. That
    is the number the business acted on: a plan republished a fortnight later with more
    information is not the commitment anyone bought stock against, and scoring the
    latest revision would flatter every forecast by measuring it after the fact.
    """
    if history_root is None:
        return pd.DataFrame(), 0
    root = Path(history_root)
    if not root.exists():
        return pd.DataFrame(), 0

    rows, read = [], 0
    for path in sorted(root.rglob("snapshot_*.json")):
        if exclude_run and exclude_run in path.name:
            continue
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        plan = snapshot.get("plan") or []
        if not plan:
            continue
        read += 1
        run_at = str(snapshot.get("run_at") or "")
        for entry in plan:
            rows.append({
                "sku": str(entry.get("sku")),
                "period": str(entry.get("period")),
                "forecast_qty": float(entry.get("forecast_qty") or 0.0),
                "forecast_source": entry.get("forecast_source", "statistical"),
                "planned_at": run_at,
            })
    if not rows:
        return pd.DataFrame(), read
    frame = pd.DataFrame(rows).sort_values("planned_at")
    return frame.drop_duplicates(["sku", "period"], keep="first"), read


def _actual_frame(time_series: pd.DataFrame) -> pd.DataFrame:
    """This run's demand history, long, as the record of what actually happened."""
    frame = time_series.copy()
    frame.index = [str(p) for p in frame.index]
    long = (frame.reset_index().melt(id_vars="index", var_name="sku",
                                     value_name="actual_qty")
            .rename(columns={"index": "period"}))
    long["sku"] = long["sku"].astype(str)
    long["actual_qty"] = pd.to_numeric(long["actual_qty"], errors="coerce").fillna(0.0)
    return long


def _attach_dimensions(frame: pd.DataFrame, attributes) -> pd.DataFrame:
    wanted = ["business_unit", "product_family", "description"]
    if attributes is None or not len(attributes):
        frame["product_family"] = "unclassified"
        return frame
    dims = attributes[["sku"] + [c for c in wanted if c in attributes.columns]].copy()
    dims["sku"] = dims["sku"].astype(str)
    frame = frame.merge(dims.drop_duplicates("sku"), on="sku", how="left")
    if "product_family" not in frame.columns:
        frame["product_family"] = "unclassified"
    frame["product_family"] = frame["product_family"].fillna("unclassified")
    return frame


# ── Rankings ─────────────────────────────────────────────────────────────────


def _rank_families(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Product lines ordered by how much money the plan was wrong by.

    Sorted on the absolute miss and reported with the signed one, because the two say
    different things and a meeting needs both: the size decides whether it is worth
    discussing, the direction decides what the discussion is about.
    """
    keys = ["product_family"]
    if "business_unit" in scored.columns and scored["business_unit"].notna().any():
        keys = ["business_unit", "product_family"]
    grouped = (scored.groupby(keys, as_index=False, dropna=False)
               .agg(skus=("sku", "nunique"),
                    periods=("period", "nunique"),
                    planned_qty=("forecast_qty", "sum"),
                    actual_qty=("actual_qty", "sum"),
                    planned_value=("planned_value", "sum"),
                    actual_value=("actual_value", "sum"),
                    bias_value=("error_value", "sum"),
                    abs_error_value=("error_value", lambda s: float(s.abs().sum()))))
    grouped["label"] = grouped[keys].astype(str).agg(" · ".join, axis=1)
    grouped["bias_pct"] = np.where(grouped["actual_value"] > 0,
                                   grouped["bias_value"] / grouped["actual_value"],
                                   np.nan)
    return (grouped.reindex(grouped["bias_value"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def _rank_skus(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = (scored.groupby("sku", as_index=False)
               .agg(periods=("period", "nunique"),
                    planned_qty=("forecast_qty", "sum"),
                    actual_qty=("actual_qty", "sum"),
                    planned_value=("planned_value", "sum"),
                    actual_value=("actual_value", "sum"),
                    bias_value=("error_value", "sum"),
                    abs_error_value=("error_value", lambda s: float(s.abs().sum()))))
    for column in ("product_family", "business_unit", "description"):
        if column in scored.columns:
            grouped[column] = grouped["sku"].map(
                scored.drop_duplicates("sku").set_index("sku")[column])
    grouped["bias_pct"] = np.where(grouped["actual_value"] > 0,
                                   grouped["bias_value"] / grouped["actual_value"],
                                   np.nan)
    return (grouped.reindex(grouped["bias_value"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def _review_frame(forecast_detail: pd.DataFrame, attributes) -> pd.DataFrame:
    """
    Every period a person changed in this run's plan, with both numbers.

    Kept here rather than in the scoring above because it is about the plan being
    published now, not the ones already scored. It becomes scoreable next period —
    which is the point of storing both numbers.
    """
    if forecast_detail is None or not len(forecast_detail):
        return pd.DataFrame()
    if "forecast_source" not in forecast_detail.columns:
        return pd.DataFrame()
    changed = forecast_detail[forecast_detail["forecast_source"] == "sales_review"]
    if not len(changed):
        return pd.DataFrame()
    columns = [c for c in ("sku", "period", "statistical_qty", "forecast_qty",
                           "reviewed_by", "review_reason", "model_used")
               if c in changed.columns]
    out = changed[columns].copy()
    out["sku"] = out["sku"].astype(str)
    out["delta_qty"] = out["forecast_qty"] - out["statistical_qty"]
    out["delta_pct"] = np.where(out["statistical_qty"] > 0,
                                out["delta_qty"] / out["statistical_qty"], np.nan)
    out = _attach_dimensions(out, attributes)
    return out.reindex(out["delta_qty"].abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)
