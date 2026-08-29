"""
Normalising the values a dimension is grouped by.

A dimension column arrives from an ERP as free text that several people have typed
over several years, and the same business unit appears in it more than once. In one
real master `NBU` holds both `Water` and `Water Segment`, and both `Mechanical` and
`Mechanical Segment`; in the sales history beside it `Billto Country` holds `India`
and `INDIA`. Nothing is missing and nothing is malformed — the file looks clean.

What it does to a report is not cosmetic. Every rollup splits: one business unit
becomes two rows, each with part of the revenue, and the larger one looks like a
smaller business than it is. Worse for the forecast — a series segmented on the raw
value is split into two shorter, sparser series, each modelled on half its history,
and the two forecasts do not add back to what one would have been.

So values are canonicalised before anything groups by them, and every merge is
reported rather than done quietly. The merge changes what the numbers mean, and the
person reading them is the one who can say whether `Water` and `Water Segment` really
are one thing. They almost always are. It is still their call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import pandas as pd

# Words an ERP appends to a dimension value without changing which thing it names.
# `Water` and `Water Segment` are one business unit; `Valves` and `Valves Group` are
# one product line. Stripped only from the *end*, and only when something is left:
# `Segment` on its own is a value, not a suffix.
_NOISE_SUFFIXES = ("segment", "group", "division", "bu", "business unit", "category",
                   "line", "family")


def canonical(value: object) -> Optional[str]:
    """
    The comparison form of one dimension value.

    Case, surrounding space, repeated space and punctuation carry no meaning in these
    columns — `INDIA`, `India` and `india ` are one country by any reading. A trailing
    noise word carries none either.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    key = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    key = re.sub(r"\s+", " ", key).strip()
    for suffix in _NOISE_SUFFIXES:
        if key.endswith(" " + suffix):
            trimmed = key[: -len(suffix) - 1].strip()
            if trimmed:
                key = trimmed
                break
    return key or None


@dataclass
class Collision:
    """Several spellings that name one thing, and the one they were folded onto."""

    column: str
    canonical_key: str
    kept: str
    merged: List[str]
    counts: Dict[str, int] = dc_field(default_factory=dict)

    def describe(self) -> str:
        parts = ", ".join(f"{v!r} ({self.counts.get(v, 0):,})" for v in
                          [self.kept] + self.merged)
        return f"{self.column}: {parts} → {self.kept!r}"


@dataclass
class NormalisedDimension:
    """One dimension column after folding, with what was folded."""

    column: str
    values: pd.Series
    collisions: List[Collision] = dc_field(default_factory=list)

    @property
    def distinct_before(self) -> int:
        return int(len(self.collisions) + self.values.nunique(dropna=True)) \
            if self.collisions else int(self.values.nunique(dropna=True))


def normalise(series: pd.Series, column: str) -> NormalisedDimension:
    """
    Fold a dimension column onto one spelling per thing.

    The surviving spelling is the most frequent one, not the first seen and not the
    alphabetically smallest. Frequency is the only one of the three that reflects how
    the business actually writes it: `India` outnumbers `INDIA` seventeen to one, and
    keeping `INDIA` because it sorts first would put the odd spelling on every report.
    """
    keys = series.map(canonical)
    counts = series.dropna().astype(str).value_counts()

    winners: Dict[str, str] = {}
    collisions: List[Collision] = []
    for key, group in pd.DataFrame({"key": keys, "raw": series}).dropna(
            subset=["key"]).groupby("key")["raw"]:
        spellings = group.astype(str).value_counts()
        # Frequency first, then the shorter spelling. The tie-break matters more than
        # it looks: `Water` and `Water Segment` appearing equally often is common on a
        # master someone half-finished renaming, and value_counts leaves that order to
        # the input. Preferring the shorter one keeps the name and drops the suffix,
        # which is the direction the fold is going anyway.
        spellings = spellings.sort_values(
            key=lambda c: c, ascending=False, kind="stable")
        kept = min([str(v) for v in spellings[spellings == spellings.max()].index],
                   key=lambda v: (len(v), v))
        winners[str(key)] = kept
        if len(spellings) > 1:
            collisions.append(Collision(
                column=column,
                canonical_key=str(key),
                kept=kept,
                # Everything except the survivor, by identity rather than by position:
                # the survivor is chosen by a tie-break, not by being first.
                merged=[str(v) for v in spellings.index if str(v) != kept],
                counts={str(k): int(counts.get(k, 0)) for k in spellings.index},
            ))

    return NormalisedDimension(
        column=column,
        values=keys.map(lambda k: winners.get(k) if k is not None else None),
        collisions=sorted(collisions, key=lambda c: c.kept),
    )


def normalise_frame(df: pd.DataFrame, columns) -> tuple:
    """Apply `normalise` to each named column present, returning the frame and merges."""
    if df is None or not len(df):
        return df, []
    out = df.copy()
    all_collisions: List[Collision] = []
    for column in columns:
        if column not in out.columns:
            continue
        result = normalise(out[column], column)
        out[column] = result.values
        all_collisions.extend(result.collisions)
    return out, all_collisions
