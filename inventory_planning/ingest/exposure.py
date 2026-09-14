"""
Exposure — how much money rests on a figure nobody measured.

The run already says which of its inputs were declared by a person and which were
assumed by the pipeline. What it does not say is how much turns on each one, and
without that the list cannot be acted on: an assumption governing four rows of a
sample file and an assumption governing every line of the stock snapshot are printed
in the same typeface, at the same size, in the same order.

That is the shape of the failure this module exists for. A stock export with no
currency column is taken to be in the reporting currency; the note saying so has been
correct and visible for some time. What it never carried is the sentence that makes a
person stop reading and go check: *this is 12.4 million of stock*. Sorting by the money
behind a figure is the whole feature, because the reader's attention is the scarce
resource and it was being spent uniformly.

Two rules keep the number honest.

**The measure comes from the contract's declared `unit`, never from the column name.**
A field is money because it says `unit: currency`, a quantity because it says
`unit: base_uom`. Reading the header instead is how `Still to be delivered (value)`
became an open quantity, and this module would be the second place to make that mistake.

**One field, never a sum of fields.** `po_history` carries two `unit: currency` fields;
adding them reports twice the exposure. The most-populated one is used and its name is
printed beside the total, so the reader can see which number they are being shown
rather than trusting an aggregate they cannot decompose.

**A document with no money in its contract has no total, and is not given one.** The
first run of this module against a real shape reported an item master's currency
assumption as `Σ min_order_qty × unit_cost 150,058` — a number that computes, sorts and
means nothing: the value of one minimum order of every item. The rule that produces it
is the useful one on a purchase order whose value column the export dropped, so the
derivation is kept and gated on what the *contract* says: money can be reconstructed for
a document that is supposed to carry it, never invented for one that never did.

Which leaves a master, where the assumption is real — a standard cost read in the wrong
currency misprices every EOQ — and no total can express it. That reports the count of
priced rows and says it is a rate rather than a total. Unknown, and unsizeable, are
legitimate answers. A guessed magnitude would order the list, and ordering the list is
the one thing the list is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

UNIT_MONEY = "currency"
UNIT_UNIT_PRICE = "currency_per_base_uom"
UNIT_QTY = "base_uom"

BASIS_DECLARED = "declared"
BASIS_ASSUMED = "assumed"

# Whether the figure governs every row of the document or only the rows the export left
# blank. A default fills blanks, so where the export also carried the field the rows it
# carried cannot be told apart afterwards and the exposure is an upper bound. Where the
# field was never mapped at all, the figure governs everything and the number is exact.
GOVERNS_ALL = "all"
GOVERNS_BLANKS = "blanks"


@dataclass(frozen=True)
class Assumption:
    """One field the run could not measure, and what it was set to instead."""

    doc_type: str
    field: str
    value: Any
    basis: str = BASIS_ASSUMED
    governs: str = GOVERNS_ALL
    # What the field would have read without this — "USD (reporting currency)". The
    # point of a declaration is the difference it made, and a declaration that restates
    # the default made none.
    instead_of: str = ""
    reason: str = ""
    by: str = ""

    @property
    def exact(self) -> bool:
        return self.governs == GOVERNS_ALL


@dataclass(frozen=True)
class Exposure:
    """The magnitude behind one assumption, in the document's own units."""

    rows: int = 0
    row_share: float = 0.0
    money: Optional[float] = None
    money_field: Optional[str] = None
    money_derived: bool = False
    qty: Optional[float] = None
    qty_field: Optional[str] = None
    upper_bound: bool = False
    # Rows carrying a unit price on a document whose contract declares no extended
    # value. A per-unit rate in the wrong currency misprices everything downstream, and
    # there is no total that says so — the count is the most that can honestly be shown.
    priced_rows: Optional[int] = None
    priced_field: Optional[str] = None

    @property
    def known(self) -> bool:
        return (self.money is not None or self.qty is not None
                or self.priced_rows is not None)

    @property
    def rank(self) -> float:
        """Sort key. Money outranks quantity, and an unmeasurable exposure sorts last."""
        if self.money is not None and np.isfinite(self.money):
            return abs(float(self.money))
        return -1.0


@dataclass(frozen=True)
class Resting:
    assumption: Assumption
    exposure: Exposure


def money_field(frame: pd.DataFrame, contract) -> Optional[str]:
    """The document's principal money column: most populated among `unit: currency`."""
    return _most_populated(frame, _fields_with_unit(contract, UNIT_MONEY))


def qty_field(frame: pd.DataFrame, contract) -> Optional[str]:
    return _most_populated(frame, _fields_with_unit(contract, UNIT_QTY))


def _fields_with_unit(contract, unit: str) -> List[str]:
    fields = getattr(contract, "fields", None) or {}
    return [name for name, spec in fields.items()
            if str(getattr(spec, "unit", "") or "") == unit]


def _most_populated(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    best, best_count = None, 0
    for name in candidates:
        if name not in frame.columns:
            continue
        count = int(pd.to_numeric(frame[name], errors="coerce").notna().sum())
        if count > best_count:
            best, best_count = name, count
    return best


def _sum(frame: pd.DataFrame, column: str) -> Optional[float]:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.notna().any():
        return None
    return float(values.sum())


def _rows_governed(frame: pd.DataFrame, assumption: Assumption) -> pd.DataFrame:
    """
    The subset the assumption is responsible for.

    Where it governs everything the whole frame is its responsibility, and no row test
    is needed — which matters, because a value written into a frame is not always equal
    to the value declared once the adapter has normalised it, and a mismatch there would
    silently report zero exposure on the assumption most worth reading.
    """
    if assumption.exact or assumption.field not in frame.columns:
        return frame
    declared = assumption.value
    column = frame[assumption.field]
    try:
        mask = column.astype(str).str.strip() == str(declared).strip()
    except Exception:
        return frame
    return frame[mask]


def measure_one(frame: Optional[pd.DataFrame], contract,
                assumption: Assumption) -> Resting:
    """What rests on one assumption, given the frame it governs and that frame's contract."""
    if frame is None or not len(frame) or contract is None:
        return Resting(assumption=assumption, exposure=Exposure())

    governed = _rows_governed(frame, assumption)
    rows = len(governed)
    share = rows / len(frame) if len(frame) else 0.0
    if not rows:
        return Resting(assumption=assumption,
                       exposure=Exposure(rows=0, row_share=0.0,
                                         upper_bound=not assumption.exact))

    price_col = _most_populated(governed, _fields_with_unit(contract, UNIT_UNIT_PRICE))

    # Whether this kind of document holds money at all, asked of the contract rather
    # than of the frame. A master does not, and a total invented for one is a plausible
    # number with no referent — the failure this pipeline is built to refuse.
    if not _fields_with_unit(contract, UNIT_MONEY):
        priced = None
        if price_col:
            priced = int(pd.to_numeric(governed[price_col], errors="coerce").notna().sum())
        return Resting(
            assumption=assumption,
            exposure=Exposure(rows=rows, row_share=share,
                              priced_rows=priced, priced_field=price_col,
                              upper_bound=not assumption.exact),
        )

    money_col = money_field(frame, contract)
    money = _sum(governed, money_col) if money_col else None
    derived = False

    # The contract says this document carries an extended value and the export did not.
    # Reconstructing it from quantity × price is a real exposure — an open PO report
    # without its value column is the case — and is marked computed rather than read,
    # because it is only as good as the mapping of the two columns it multiplies.
    qty_col = qty_field(frame, contract)
    if money is None and qty_col and price_col:
        product = (pd.to_numeric(governed[qty_col], errors="coerce")
                   * pd.to_numeric(governed[price_col], errors="coerce"))
        if product.notna().any():
            money, money_col, derived = float(product.sum()), \
                f"{qty_col} × {price_col}", True

    qty = _sum(governed, qty_col) if qty_col else None

    return Resting(
        assumption=assumption,
        exposure=Exposure(
            rows=rows, row_share=share,
            money=money, money_field=money_col, money_derived=derived,
            qty=qty, qty_field=qty_col,
            upper_bound=not assumption.exact,
        ),
    )


@dataclass
class RestingLedger:
    """Everything this run is resting on, largest exposure first."""

    items: List[Resting] = dc_field(default_factory=list)
    reporting_currency: str = ""

    @property
    def ordered(self) -> List[Resting]:
        return sorted(self.items,
                      key=lambda r: (r.exposure.rank, r.exposure.rows),
                      reverse=True)

    @property
    def declared(self) -> List[Resting]:
        return [r for r in self.items if r.assumption.basis == BASIS_DECLARED]

    @property
    def assumed(self) -> List[Resting]:
        return [r for r in self.items if r.assumption.basis == BASIS_ASSUMED]

    def summary(self) -> str:
        if not self.items:
            return ""
        cur = f" {self.reporting_currency}" if self.reporting_currency else ""
        lines = [
            "  What this run is resting on — largest exposure first",
            "  " + "-" * 66,
        ]
        for item in self.ordered:
            a, e = item.assumption, item.exposure
            mark = "declared" if a.basis == BASIS_DECLARED else "assumed"
            who = f" by {a.by}" if a.by else ""
            lines.append(f"    {a.doc_type:<16} {a.field} = {a.value!r:<12} {mark}{who}")

            bound = "≤ " if e.upper_bound else ""
            magnitude = []
            if e.money is not None and np.isfinite(e.money):
                how = " (computed)" if e.money_derived else ""
                magnitude.append(f"Σ {e.money_field} {e.money:,.0f}{cur}{how}")
            if e.qty is not None and np.isfinite(e.qty) and not e.money_derived:
                magnitude.append(f"Σ {e.qty_field} {e.qty:,.0f}")
            if e.priced_rows:
                magnitude.append(f"{e.priced_rows:,} priced on {e.priced_field} — "
                                 f"a rate, not a total")
            if not magnitude:
                magnitude.append("no money or quantity field to measure it against")
            lines.append(f"        {bound}{e.rows:,} rows ({e.row_share:.0%})   "
                         + "   ".join(magnitude))

            if a.instead_of:
                lines.append(f"        without it: {a.instead_of}")
            if a.reason:
                lines.append(f"        {a.reason.strip().splitlines()[0]}")

        unmeasured = [r for r in self.items if not r.exposure.known]
        if unmeasured:
            lines.append("")
            lines.append("    An exposure shown as unmeasurable is not a small one. It is a "
                         "document with no")
            lines.append("    money or quantity field to size it against, and it has to be "
                         "judged some other way.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reporting_currency": self.reporting_currency,
            "resting_on": [
                {
                    "doc_type": r.assumption.doc_type,
                    "field": r.assumption.field,
                    "value": r.assumption.value,
                    "basis": r.assumption.basis,
                    "governs": r.assumption.governs,
                    "instead_of": r.assumption.instead_of,
                    "reason": r.assumption.reason,
                    "by": r.assumption.by,
                    "rows": r.exposure.rows,
                    "row_share": round(r.exposure.row_share, 4),
                    "money": r.exposure.money,
                    "money_field": r.exposure.money_field,
                    "money_derived": r.exposure.money_derived,
                    "qty": r.exposure.qty,
                    "qty_field": r.exposure.qty_field,
                    "priced_rows": r.exposure.priced_rows,
                    "priced_field": r.exposure.priced_field,
                    "upper_bound": r.exposure.upper_bound,
                }
                for r in self.ordered
            ],
        }


def assemble(declared: Sequence[Assumption],
             assumed_doc_types: Sequence[str],
             reporting_currency: str = "") -> List[Assumption]:
    """
    Everything this run did not measure: what a person declared, plus what defaulted.

    The two belong in one list because they are the same statement made by different
    parties — *this money is in CNY* is the same claim whether a planner asserted it or
    the pipeline inferred it from silence, and only the second one is unattributed. A
    currency declaration is also annotated with what the field would have read without
    it, since a declaration restating the default changed nothing and should say so
    rather than take credit for the number it did not move.
    """
    fallback = (f"{reporting_currency} (reporting currency, assumed)"
                if reporting_currency else "the reporting currency, assumed")
    out: List[Assumption] = []
    for a in declared:
        if a.field == "currency" and not a.instead_of:
            a = Assumption(
                doc_type=a.doc_type, field=a.field, value=a.value, basis=a.basis,
                governs=a.governs, instead_of=fallback, reason=a.reason, by=a.by,
            )
        out.append(a)

    spoken_for = {(a.doc_type, a.field) for a in out}
    for doc_type in assumed_doc_types:
        # A document carrying a currency declaration is not also assuming one. The
        # declaration is why it stopped assuming, and printing both would show the same
        # exposure twice under two contradictory labels.
        if (doc_type, "currency") in spoken_for:
            continue
        out.append(Assumption(
            doc_type=doc_type, field="currency", value=reporting_currency,
            basis=BASIS_ASSUMED, governs=GOVERNS_ALL,
            reason="no currency column in the export — taken to be in the reporting "
                   "currency already",
        ))
    return out


def measure(assumptions: Sequence[Assumption],
            frames: Dict[str, pd.DataFrame],
            contracts: Dict[str, Any],
            reporting_currency: str = "") -> RestingLedger:
    """
    Size every assumption against the frame it governs.

    `frames` and `contracts` are both keyed by doc_type. A missing contract yields an
    exposure of unknown rather than a guess: the units are what make the number mean
    anything, and a document whose contract is not to hand has none.
    """
    items = [measure_one(frames.get(a.doc_type), contracts.get(a.doc_type), a)
             for a in assumptions]
    return RestingLedger(items=items, reporting_currency=reporting_currency)
