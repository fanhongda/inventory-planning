"""
Reading the reviewed forecast back in.

## Why this is not part of the automatic intake

Every other document the pipeline reads is an *extract*: a record of something that
happened, produced by a system, identified by profiling its shape. This one is a
*decision*: a person looked at the forecast and said it is wrong, and here is the
number to plan on instead.

That difference is why it is loaded by name rather than routed by fingerprint. A stray
spreadsheet in an input folder should never be able to overwrite the demand plan
silently, and a reviewed worksheet is wide with period columns — the same shape as a
pre-compiled demand series, which the profiler routes on shape. Naming it removes both
problems at once, and it costs one explicit call.

## What an override does and does not change

It replaces the quantity. It does **not** replace the forecast error.

That distinction decides how much stock gets held, so it is worth being plain about.
`forecast_rmse` becomes σDL in the safety stock calculation — the measure of how wrong
the forecast has been, which is what safety stock exists to absorb. A sales adjustment
is not evidence that the demand became more predictable; if anything a number set by
judgement is less predictable than one fitted to history. Letting the override reset
the error to zero, or recompute it against the new number, would cut safety stock at
exactly the moment the plan started resting on an opinion.

So the statistical model's error is kept, and the statistical quantity is kept beside
the reviewed one. The second is not bookkeeping: without it there is no way, next
quarter, to answer whether the review made the forecast better or worse. A review
process that cannot be scored is a review process that never improves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import numpy as np
import pandas as pd

# The column the worksheet asks a reviewer to type into, as written on the sheet.
_REVIEW_PREFIX = "reviewed qty"
_PERIOD_RE = re.compile(r"(\d{4})[-/]?(\d{2})$")


@dataclass
class SalesPlan:
    """A reviewed forecast: what sales changed, and what they left alone."""

    adjustments: pd.DataFrame          # sku, period, reviewed_qty, reviewed_by, reason
    source: str = ""
    skipped: List[str] = dc_field(default_factory=list)

    def __len__(self) -> int:
        return len(self.adjustments)

    @property
    def skus(self) -> set:
        return set(self.adjustments["sku"].astype(str)) if len(self.adjustments) else set()

    def summary(self) -> str:
        if not len(self.adjustments):
            return (f"  Sales plan ({self.source}): no adjustments — every month was "
                    f"left at the statistical forecast")
        periods = sorted(self.adjustments["period"].astype(str).unique())
        lines = [
            f"  Sales plan — reviewed forecast from {self.source}",
            "  " + "-" * 58,
            f"    {len(self.adjustments):,} adjustments over {len(self.skus):,} SKUs, "
            f"periods {periods[0]} → {periods[-1]}",
        ]
        # Counted by masking rather than by replacing "" with NaN: `replace` on an
        # object column downcasts, which pandas 2 warns about and pandas 3 does
        # differently. A blank reviewer is an absent one either way.
        reviewers = self.adjustments["reviewed_by"].fillna("").astype(str).str.strip()
        named = int(reviewers[reviewers != ""].nunique())
        if named:
            lines.append(f"    {named} reviewer(s) named on the sheet")
        unexplained = int((self.adjustments["reason"].fillna("") == "").sum())
        if unexplained:
            lines.append(
                f"    {unexplained:,} carry no reason. They are applied — a number "
                f"without a rationale is still the number sales committed to — but "
                f"nothing will be able to say later why the plan moved."
            )
        if self.skipped:
            lines.append(f"    Ignored columns: {', '.join(self.skipped[:6])}")
        return "\n".join(lines)


def read_sales_plan(path, sheet_name: str = "Review by SKU") -> SalesPlan:
    """
    Read the worksheet back, taking only the cells a reviewer actually filled in.

    A blank cell means "no change", not "zero". The difference is the whole file: a
    reviewer types into two months out of six and leaves the rest alone, and reading
    the blanks as zeroes would silently plan four months of no demand for every SKU on
    the sheet.
    """
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() in (".csv", ".txt"):
        frame = pd.read_csv(path)
    else:
        book = pd.ExcelFile(path)
        target = sheet_name if sheet_name in book.sheet_names else book.sheet_names[0]
        frame = pd.read_excel(path, sheet_name=target)

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    sku_col = next((lowered[k] for k in ("sku", "material", "part number", "item code")
                    if k in lowered), None)
    if sku_col is None:
        raise ValueError(
            f"{path.name} has no item column. The reviewed worksheet must keep its "
            f"`sku` column — it is the only thing that connects a reviewed number back "
            f"to the item it is about."
        )

    by_col = next((lowered[k] for k in ("reviewed by", "reviewer", "owner")
                   if k in lowered), None)
    why_col = next((lowered[k] for k in ("reviewed reason", "reason", "comment",
                                         "rationale") if k in lowered), None)
    country_col = next((lowered[k] for k in ("country", "region") if k in lowered), None)

    review_cols, skipped = [], []
    for column in frame.columns:
        name = str(column).strip().lower()
        if not name.startswith(_REVIEW_PREFIX):
            continue
        match = _PERIOD_RE.search(name.replace(" ", ""))
        if not match:
            skipped.append(str(column))
            continue
        review_cols.append((column, f"{match.group(1)}-{match.group(2)}"))

    rows = []
    for column, period in review_cols:
        values = pd.to_numeric(frame[column], errors="coerce")
        filled = values.notna()
        if not filled.any():
            continue
        part = frame.loc[filled, [sku_col]].copy()
        part.columns = ["sku"]
        part["period"] = period
        part["reviewed_qty"] = values[filled].astype(float).values
        part["reviewed_by"] = (frame.loc[filled, by_col].astype(str).values
                               if by_col else "")
        part["reason"] = (frame.loc[filled, why_col].fillna("").astype(str).values
                          if why_col else "")
        if country_col:
            part["country"] = frame.loc[filled, country_col].astype(str).values
        rows.append(part)

    if not rows:
        return SalesPlan(adjustments=pd.DataFrame(
            columns=["sku", "period", "reviewed_qty", "reviewed_by", "reason"]),
            source=path.name, skipped=skipped)

    adjustments = pd.concat(rows, ignore_index=True)
    adjustments["sku"] = adjustments["sku"].astype(str).str.strip()
    adjustments = adjustments[adjustments["sku"] != ""]

    # A split sheet has several country rows per SKU and period. The plan the pipeline
    # runs on is at SKU level — the stock is one pool — so the reviewed country numbers
    # are summed back. Summing is the right operation precisely because the split was
    # top-down: the parts were apportioned from one total and belong added together.
    grouped = (adjustments.groupby(["sku", "period"], as_index=False)
               .agg(reviewed_qty=("reviewed_qty", "sum"),
                    reviewed_by=("reviewed_by", _join_text),
                    reason=("reason", _join_text)))
    return SalesPlan(adjustments=grouped, source=path.name, skipped=skipped)


def _join_text(values) -> str:
    """
    Collapse the text cells of a group into one, ignoring the empties.

    Written out rather than inlined because an unfilled Excel cell comes back as a
    float `nan`, not as a blank string, and joining those raises. On a split sheet
    every SKU has several country rows and most of them are unfilled, so this is the
    normal case rather than the edge one.
    """
    seen = {str(v).strip() for v in values}
    return "; ".join(sorted(v for v in seen if v and v.lower() != "nan"))


@dataclass
class OverrideResult:
    """The forecast after review, and an account of what moved."""

    forecast_detail: pd.DataFrame
    applied: int = 0
    unmatched: List[str] = dc_field(default_factory=list)
    qty_before: float = 0.0
    qty_after: float = 0.0

    def summary(self) -> str:
        if not self.applied:
            head = "  Sales review: nothing applied to the forecast"
            if not self.unmatched:
                return head
            # Reported even when nothing matched — especially then. A sheet where every
            # reviewed SKU missed is a key problem, and returning "nothing applied"
            # alone would present the most alarming possible outcome as a quiet no-op.
            return (f"{head}\n"
                    f"    {len(self.unmatched):,} reviewed SKUs are not in the forecast "
                    f"and were ignored: {', '.join(self.unmatched[:5])}")
        delta = self.qty_after - self.qty_before
        pct = (delta / self.qty_before * 100) if self.qty_before else float("nan")
        lines = [
            "  Sales review applied to the forecast",
            "  " + "-" * 58,
            f"    {self.applied:,} SKU-months replaced",
            f"    Horizon quantity {self.qty_before:,.0f} → {self.qty_after:,.0f} "
            f"({delta:+,.0f}, {pct:+.1f}%)",
            "    The statistical forecast is kept beside it in `statistical_qty`, and "
            "safety stock still uses the statistical model's error — a number set by "
            "judgement is not evidence that demand became more predictable.",
        ]
        if self.unmatched:
            lines.append(
                f"    {len(self.unmatched):,} reviewed SKUs are not in the forecast and "
                f"were ignored: {', '.join(self.unmatched[:5])}"
            )
            lines.append(
                "      Usually an item that has been reviewed for years and has no "
                "demand history in this extract. Nothing was invented for them — a "
                "forecast conjured from a review alone has no error to size stock on."
            )
        return "\n".join(lines)


def apply_sales_plan(forecast_detail: pd.DataFrame,
                     plan: Optional[SalesPlan]) -> OverrideResult:
    """
    Replace the reviewed quantities, keep everything else.

    `forecast_rmse` is deliberately untouched — see the module docstring. So is
    `model_used`: the row still records which model produced the number that was
    argued with, and losing that would make the review unscoreable.
    """
    if forecast_detail is None or not len(forecast_detail):
        return OverrideResult(forecast_detail=forecast_detail)

    out = forecast_detail.copy()
    out["statistical_qty"] = out["forecast_qty"]
    out["forecast_source"] = "statistical"
    out["reviewed_by"] = ""
    out["review_reason"] = ""

    if plan is None or not len(plan):
        return OverrideResult(forecast_detail=out,
                              qty_before=float(out["forecast_qty"].sum()),
                              qty_after=float(out["forecast_qty"].sum()))

    before = float(out["forecast_qty"].sum())
    key = out["sku"].astype(str) + "||" + out["period"].astype(str)
    adjustments = plan.adjustments.copy()
    adjustments["key"] = (adjustments["sku"].astype(str) + "||"
                          + adjustments["period"].astype(str))
    lookup = adjustments.set_index("key")

    hit = key.isin(lookup.index)
    if hit.any():
        matched = key[hit]
        out.loc[hit, "forecast_qty"] = matched.map(lookup["reviewed_qty"]).values
        out.loc[hit, "forecast_source"] = "sales_review"
        out.loc[hit, "reviewed_by"] = matched.map(lookup["reviewed_by"]).fillna("").values
        out.loc[hit, "review_reason"] = matched.map(lookup["reason"]).fillna("").values

    unmatched = sorted(plan.skus - set(out["sku"].astype(str)))
    return OverrideResult(
        forecast_detail=out,
        applied=int(hit.sum()),
        unmatched=unmatched,
        qty_before=before,
        qty_after=float(out["forecast_qty"].sum()),
    )
