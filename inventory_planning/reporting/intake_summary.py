"""
Intake summary — the shape of what was loaded, before anything is computed on it.

Every silent failure this pipeline has actually produced was visible in the totals long
before it was visible in the recommendations, and invisible in the mapping log that
preceded them. Two real examples:

  A value column mapped to a quantity.  `open_qty <- 'Still to be delivered (value)'`
  is a legal mapping of a real column to a real field. Nothing was missing, nothing
  failed to parse, and the run completed. It surfaced as $2.9bn of open purchase orders
  and 181,435 days of supply — 497 years — which nobody was looking at until the report
  was read line by line against the source.

  A promise date mapped to a shipment date.  `ship_date <- 'Planned Ship Date'` pushed
  the run's "today" to 2027-12-30 and quietly restated on-time delivery, past-due and
  stockout risk against a date more than a year in the future.

Neither is detectable from column names. Both are obvious from one look at the values:
a quantity column whose values are 87% fractional is not a quantity, and a backward
looking date column whose maximum is sixteen months ahead is not backward looking.

So this prints the totals a person would check by hand if they thought to, in the order
they would check them, and flags the readings that cannot be true. It computes nothing
the planning uses; it exists to be read before the planning is believed.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# A quantity this fractional is not a count of things. Deliberately generous — pack
# sizes, weights and part-units are real — so that only a column of money or of a
# computed ratio trips it.
INTEGER_SHARE_FLOOR = 0.60

# A unit value outside this range is not a price on any product this pipeline plans.
IMPLIED_UNIT_VALUE_RANGE = (0.01, 1_000_000.0)

# Columns that are supposed to hold future dates. A commitment beyond this is not a
# commitment; two years is longer than any order book this plans against.
_FORWARD_HINTS = ("request", "promise", "delivery", "eta", "planned", "due", "expected")
FORWARD_DATE_LIMIT_DAYS = 730

_QTY_HINTS = ("qty", "quantity")
_MONEY_HINTS = ("amount", "value", "cost", "price")
_DATE_HINTS = ("date",)

# Money columns whose name contains a quantity hint, or vice versa, would otherwise be
# classified twice. Unit cost is a price, not a quantity, whatever it is called.
_MONEY_FIRST = ("unit_cost", "std_cost", "avg_cost", "unit_price")


def _columns_like(df: pd.DataFrame, hints, exclude=()) -> List[str]:
    out = []
    for col in df.columns:
        name = str(col).lower()
        if col in exclude:
            continue
        if any(h in name for h in hints):
            out.append(col)
    return out


@dataclass
class ColumnReading:
    column: str
    kind: str                       # qty | money | date
    non_null: int = 0
    total: float = np.nan
    median: float = np.nan
    largest: float = np.nan
    integer_share: float = np.nan
    earliest: Any = None
    latest: Any = None
    flags: List[str] = dc_field(default_factory=list)


@dataclass
class DocumentSummary:
    doc_type: str
    rows: int
    skus: int
    readings: List[ColumnReading] = dc_field(default_factory=list)
    implied_unit_value: Optional[float] = None
    flags: List[str] = dc_field(default_factory=list)

    @property
    def suspect(self) -> bool:
        return bool(self.flags) or any(r.flags for r in self.readings)


@dataclass
class IntakeSummary:
    documents: List[DocumentSummary] = dc_field(default_factory=list)
    overlap: Dict[str, Any] = dc_field(default_factory=dict)
    anchor: Any = None

    @property
    def anchor_flag(self) -> Optional[str]:
        """
        The run's "today" cannot be in the future.

        `latest_observed_date` clamps staleness at zero, so an anchor a year ahead
        reports as perfectly fresh and no staleness warning fires. Everything measured
        against it — on-time delivery, past-due, backlog realization, stockout risk —
        is then measured from a date that has not happened.
        """
        if self.anchor is None:
            return None
        ahead = (pd.Timestamp(self.anchor) - pd.Timestamp.today().normalize()).days
        if ahead <= 0:
            return None
        return (f"the run is anchored to {str(self.anchor)[:10]}, {ahead:,} days in the "
                f"future. A date column was mapped to a promise rather than an event; "
                f"every date-based figure below is measured from a day that has not "
                f"happened.")

    @property
    def suspect(self) -> List[DocumentSummary]:
        return [d for d in self.documents if d.suspect]

    def summary(self) -> str:
        lines = ["  Intake summary — check these before trusting the run",
                 "  " + "-" * 66]
        if self.anchor_flag:
            lines.append(f"    ⚠ {self.anchor_flag}")
            lines.append("")
        for doc in self.documents:
            lines.append(f"    {doc.doc_type}: {doc.rows:,} rows, {doc.skus:,} SKUs")
            for r in doc.readings:
                if r.kind == "date":
                    lines.append(f"        {r.column:<26} {str(r.earliest)[:10]} → "
                                 f"{str(r.latest)[:10]}")
                elif r.kind == "qty":
                    lines.append(f"        {r.column:<26} Σ {r.total:>18,.0f}  "
                                 f"med {r.median:>12,.1f}  whole {r.integer_share:>5.0%}")
                else:
                    lines.append(f"        {r.column:<26} Σ {r.total:>18,.2f}  "
                                 f"med {r.median:>12,.2f}")
                for flag in r.flags:
                    lines.append(f"          ⚠ {flag}")
            if doc.implied_unit_value is not None and np.isfinite(doc.implied_unit_value):
                lines.append(f"        {'implied unit value':<26} "
                             f"{doc.implied_unit_value:>18,.2f} (median)")
            for flag in doc.flags:
                lines.append(f"          ⚠ {flag}")

        if self.overlap:
            lines.append("")
            lines.append("    SKU overlap with the demand history:")
            for doc_type, share in sorted(self.overlap.items()):
                mark = " ⚠ joins will mostly miss" if share < 0.10 else ""
                lines.append(f"        {doc_type:<26} {share:>5.0%}{mark}")

        if not self.suspect and not self.anchor_flag:
            lines.append("")
            lines.append("    Nothing implausible in the totals. This is not a check "
                         "that the mapping is right,")
            lines.append("    only that no column is obviously the wrong kind of thing.")
        return "\n".join(lines)


def summarise_intake(documents: Dict[str, pd.DataFrame], anchor=None) -> IntakeSummary:
    """
    `documents` maps doc_type to the loaded canonical frame. `anchor` is the run's
    "today" — a date beyond it is reported, because a backward-looking column cannot
    contain one and a forward-looking one is not evidence of when the extract was taken.
    """
    out = IntakeSummary(anchor=anchor)
    reference: Optional[pd.Series] = None

    for doc_type, df in documents.items():
        if df is None or not len(df):
            continue
        skus = df["sku"].nunique() if "sku" in df.columns else 0
        doc = DocumentSummary(doc_type=doc_type, rows=len(df), skus=skus)

        money_cols = _columns_like(df, _MONEY_HINTS)
        qty_cols = [c for c in _columns_like(df, _QTY_HINTS)
                    if c not in money_cols and c not in _MONEY_FIRST]
        date_cols = _columns_like(df, _DATE_HINTS)

        for col in qty_cols:
            doc.readings.append(_numeric_reading(df[col], col, "qty"))
        for col in money_cols:
            doc.readings.append(_numeric_reading(df[col], col, "money"))
        for col in date_cols:
            doc.readings.append(_date_reading(df[col], col, anchor))

        doc.implied_unit_value = _implied_unit_value(df, qty_cols, money_cols, doc)
        out.documents.append(doc)

        if doc_type == "sales_history" and "sku" in df.columns:
            reference = df["sku"].dropna().astype(str).unique()

    if reference is not None and len(reference):
        ref = set(reference)
        for doc_type, df in documents.items():
            if df is None or not len(df) or "sku" not in df.columns or doc_type == "sales_history":
                continue
            own = set(df["sku"].dropna().astype(str).unique())
            if own:
                out.overlap[doc_type] = len(own & ref) / len(own)
    return out


def _numeric_reading(series: pd.Series, col: str, kind: str) -> ColumnReading:
    values = pd.to_numeric(series, errors="coerce").dropna()
    reading = ColumnReading(column=col, kind=kind, non_null=int(len(values)))
    if not len(values):
        reading.flags.append("no numeric values at all")
        return reading

    reading.total = float(values.sum())
    reading.median = float(values.median())
    reading.largest = float(values.abs().max())

    if kind == "qty":
        whole = float(np.isclose(values % 1, 0).mean())
        reading.integer_share = whole
        if whole < INTEGER_SHARE_FLOOR:
            reading.flags.append(
                f"only {whole:.0%} of these are whole numbers — a quantity column "
                f"usually is. Check it was not mapped to a money column."
            )
    return reading


def _date_reading(series: pd.Series, col: str, anchor) -> ColumnReading:
    values = pd.to_datetime(series, errors="coerce").dropna()
    reading = ColumnReading(column=col, kind="date", non_null=int(len(values)))
    if not len(values):
        reading.flags.append("no parseable dates")
        return reading

    reading.earliest, reading.latest = values.min(), values.max()
    # Judge against the anchor, or today when the anchor is itself in the future —
    # otherwise the corrupted column becomes the standard its own corruption is
    # measured against, and nothing trips.
    today = pd.Timestamp.today().normalize()
    horizon = min(pd.Timestamp(anchor), today) if anchor is not None else today
    ahead = (reading.latest - horizon).days
    forward = any(h in col.lower() for h in _FORWARD_HINTS)
    if forward:
        # A request or promise date belongs in the future — that is what it is for.
        # Flagging it teaches the reader to skip the warnings, which costs more than
        # the check is worth. Only a date beyond any commercial commitment is evidence
        # of a parsing error rather than of a customer.
        if ahead > FORWARD_DATE_LIMIT_DAYS:
            reading.flags.append(
                f"latest is {ahead:,} days ahead — too far to be a real commitment. "
                f"Check the date parsing rather than the mapping."
            )
    elif ahead > 0:
        reading.flags.append(
            f"latest is {ahead:,} days ahead of the run date — this column reads as "
            f"backward-looking, so a future date in it means it was mapped to a "
            f"promise rather than an event"
        )
    return reading


def _implied_unit_value(df, qty_cols, money_cols, doc) -> Optional[float]:
    """
    Money over quantity on the same row — a price, and prices have a plausible range.

    The check that would have caught a value column read as a quantity: divide one by
    the other and the answer comes back at or near 1.0, because the two are the same
    number.
    """
    if not qty_cols or not money_cols:
        return None
    qty = pd.to_numeric(df[qty_cols[0]], errors="coerce")
    money = pd.to_numeric(df[money_cols[0]], errors="coerce")
    usable = qty.notna() & money.notna() & (qty > 0)
    if not usable.any():
        return None

    implied = (money[usable] / qty[usable]).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(implied):
        return None
    median = float(implied.median())

    low, high = IMPLIED_UNIT_VALUE_RANGE
    if np.isclose(median, 1.0, rtol=0.02):
        doc.flags.append(
            f"{money_cols[0]} / {qty_cols[0]} = {median:.2f} — a unit value of one "
            f"means the two columns hold the same number. One of them is mapped wrong."
        )
    elif median < low or median > high:
        doc.flags.append(
            f"{money_cols[0]} / {qty_cols[0]} = {median:,.2f} per unit, which is not a "
            f"price. Check which column is the quantity."
        )
    return median
