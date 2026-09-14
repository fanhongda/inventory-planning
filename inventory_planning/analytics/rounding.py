"""
Whole units — because a forecast of 33.4 of a countable thing is not a forecast.

A quantity the business counts cannot be fractional, and carrying the decimal through
the plan produces a safety stock of 12.4 and a reorder point of 45.8 that nobody can act
on. Rounding them is not cosmetic: it is the difference between a figure a buyer can put
on an order and one they have to interpret first.

## Rounded unless the unit says otherwise, not the other way round

The natural reading of "round it when the unit of measure is a piece" is to test the UoM
and round when it says EA. Measured against the real extracts, that rule fires on
nothing: of the four exports this plans from, the 83-column master carries no unit of
measure, the stock export carries none, and the sales history carries none. A rule
conditioned on a field nobody supplies is an annotation that changes nothing.

So the test is inverted. A quantity is rounded unless its unit is *known* to be a
continuous measure — a weight, a volume, a length. A SKU with no unit at all is taken to
be countable, which is what a distribution centre planning parts almost always holds.

That inversion is an assumption and it is treated as one: the counts come back on
`Rounding.report`, separating the rows that were rounded on a unit that said so from the
rows that were rounded because nothing said anything. The second number is the size of
the assumption, and it is the number to look at on the day a unit of measure finally
arrives in an extract.

## Why `round` and not `ceil`

`lot_sizing.round_to_lot` deliberately rounds *up*, because there the asymmetry is real:
EOQ is flat near its optimum and ordering slightly short risks a second order and a
second order cost. Here there is no such asymmetry — a demand forecast is an estimate of
a quantity, not a decision about one, and rounding every SKU's forecast up would put a
systematic bias into every figure derived from it. Nearest, and the error cancels.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

INTEGER = "integer"
NONE = "none"

# Units that measure rather than count. Deliberately a small, explicit list: everything
# absent from it is treated as countable, so an unfamiliar code errs towards a whole
# number rather than towards a decimal nobody can order.
DEFAULT_CONTINUOUS_UOM = (
    "KG", "G", "MG", "T", "TO", "TON", "LB", "OZ",
    "L", "ML", "CL", "GAL", "M3", "CBM",
    "M", "CM", "MM", "KM", "FT", "IN", "M2", "SQM",
)


@dataclass
class Rounding:
    """The convention in force, and what applying it did."""

    mode: str = INTEGER
    continuous: frozenset = dc_field(default_factory=lambda: frozenset(DEFAULT_CONTINUOUS_UOM))
    report: Dict[str, int] = dc_field(default_factory=lambda: {
        "rounded_on_a_stated_unit": 0,
        "rounded_with_no_unit": 0,
        "left_fractional": 0,
    })

    @property
    def active(self) -> bool:
        return self.mode == INTEGER

    @classmethod
    def from_conventions(cls, conventions: Dict[str, Any] = None) -> "Rounding":
        """
        Read `quantity_rounding` and `continuous_uom` from the parameter file's
        conventions block — the one place this project keeps decisions that change every
        figure, rather than in code where they cannot be reviewed.
        """
        conventions = conventions or {}
        mode = str(conventions.get("quantity_rounding", INTEGER) or INTEGER).strip().lower()
        if mode not in (INTEGER, NONE):
            raise ValueError(
                f"quantity_rounding is {mode!r}; it must be {INTEGER!r} or {NONE!r}.")
        declared = conventions.get("continuous_uom")
        units = (frozenset(str(u).strip().upper() for u in declared if str(u).strip())
                 if declared else frozenset(DEFAULT_CONTINUOUS_UOM))
        return cls(mode=mode, continuous=units)

    # ── Applying ─────────────────────────────────────────────────────────────

    def is_continuous(self, uom: pd.Series) -> pd.Series:
        """Which rows carry a unit this project measures rather than counts."""
        text = uom.astype("object").where(uom.notna(), "").astype(str).str.strip().str.upper()
        return text.isin(self.continuous)

    def apply(self, qty: pd.Series, uom: pd.Series = None,
              decimals: int = 1) -> pd.Series:
        """
        Round `qty` to whole units where the unit does not forbid it.

        `decimals` is what a fractional row keeps — the precision this pipeline used
        before the convention existed, so a continuous-measure SKU is unchanged by it.
        """
        values = pd.to_numeric(qty, errors="coerce")
        if not self.active:
            return values.round(decimals)

        if uom is None:
            units = pd.Series(pd.NA, index=values.index, dtype="object")
        elif isinstance(uom, pd.Series):
            units = uom.reindex(values.index)
        else:
            units = pd.Series(uom, index=values.index, dtype="object")

        continuous = self.is_continuous(units)
        stated = units.notna() & (units.astype(str).str.strip() != "")

        measurable = values.notna()
        self.report["left_fractional"] += int((continuous & measurable).sum())
        self.report["rounded_on_a_stated_unit"] += int(
            (~continuous & stated & measurable).sum())
        self.report["rounded_with_no_unit"] += int(
            (~continuous & ~stated & measurable).sum())

        return pd.Series(np.where(continuous, values.round(decimals), values.round(0)),
                         index=values.index)

    # ── Saying what it did ───────────────────────────────────────────────────

    def summary(self) -> str:
        if not self.active:
            return "  Quantities keep their decimals (quantity_rounding: none)."
        stated = self.report["rounded_on_a_stated_unit"]
        blind = self.report["rounded_with_no_unit"]
        left = self.report["left_fractional"]
        if not (stated or blind or left):
            return ""
        lines = [f"  Quantities rounded to whole units: {stated + blind:,} figure(s)"]
        if blind:
            lines.append(
                f"    · {blind:,} of them carried no unit of measure and were taken to "
                f"be countable. That is the assumption this convention rests on, and "
                f"the number to look at if a unit ever arrives in an extract.")
        if left:
            lines.append(f"    · {left:,} left fractional — their unit measures rather "
                         f"than counts.")
        return "\n".join(lines)


def uom_by_sku(*frames: Optional[pd.DataFrame]) -> pd.Series:
    """
    One unit of measure per SKU, from whichever document supplies it first.

    Ordered by authority rather than convenience: a master's unit is the item's, a
    stock row's is whatever that warehouse happened to record. Frames are tried in the
    order given and the first non-empty answer per SKU wins.
    """
    out: Dict[str, str] = {}
    for frame in frames:
        if frame is None or not len(frame) or "sku" not in frame.columns:
            continue
        column = next((c for c in ("uom", "uom_raw", "base_uom") if c in frame.columns),
                      None)
        if column is None:
            continue
        pairs = frame[["sku", column]].dropna()
        for sku, value in zip(pairs["sku"].astype(str), pairs[column].astype(str)):
            text = value.strip()
            if text and sku not in out:
                out[sku] = text
    return pd.Series(out, dtype="object", name="uom")
