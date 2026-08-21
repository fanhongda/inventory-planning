"""
Deterministic data profiler.

Produces a *data portrait* — column names, inferred types, cardinality, null rates,
parseability, candidate keys, and table shape — with no LLM involvement. Two reasons
that separation matters:

1.  Compliance. The portrait is metadata by construction: numeric columns are reduced
    to ranges, string columns to character patterns (``AAA-9999``). Nothing that
    leaves this module contains a raw business value unless the caller explicitly
    opts in via ``include_samples``. Per the architecture doc this has to be true from
    day one or the tool cannot be used on company data at all.

2.  Reproducibility. The same file always yields the same portrait, so a fingerprint
    computed from it can route a file to a frozen adapter deterministically.

The shape detector is the part that earns its keep day to day: it recognises a
wide period-column layout — the format a planner produces when they have already
bucketed demand into a SKU time series — and routes it to the demand_timeseries
contract without the caller having to know that a different loader exists.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Header normalisation ─────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[#./()\\\[\]{}:;,'\"*&%$@!?<>|+=~`^]")
_SPACE_RE = re.compile(r"[\s_\-]+")


def normalize_header(s: Any) -> str:
    """Lowercase, strip punctuation, collapse separators. Shared by profiler and contract."""
    text = str(s).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


# ── Period-column recognition ────────────────────────────────────────────────

_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Ordered most- to least-specific; first match wins.
_PERIOD_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(\d{4})[\-/\s]?(\d{1,2})$"), "ym"),                       # 2026-01, 2026/1, 202601
    (re.compile(r"^(\d{1,2})[\-/\s](\d{4})$"), "my"),                        # 01-2026
    (re.compile(r"^([a-z]{3,9})[\-/\s]?(\d{2,4})$"), "mony"),                # Jan-26, Jan 2026, jan2026
    (re.compile(r"^(\d{4})[\-/\s]?([a-z]{3,9})$"), "ymon"),                  # 2026-Jan
    (re.compile(r"^(\d{4})[\-/\s]?q([1-4])$"), "yq"),                        # 2026-Q1
    (re.compile(r"^q([1-4])[\-/\s]?(\d{2,4})$"), "qy"),                      # Q1-2026
    (re.compile(r"^(\d{4})[\-/\s](\d{1,2})[\-/\s](\d{1,2})$"), "ymd"),       # 2026-01-31
]


def _as_excel_serial(header: Any) -> Optional[int]:
    """The header as an Excel date serial, or None if it is not plausibly one."""
    if isinstance(header, bool):
        return None
    if isinstance(header, (int, float, np.integer, np.floating)):
        value = float(header)
    else:
        text = str(header).strip()
        if not re.fullmatch(r"\d{5}", text):
            return None
        value = float(text)
    if value != int(value):
        return None
    serial = int(value)
    return serial if _EXCEL_SERIAL_MIN <= serial <= _EXCEL_SERIAL_MAX else None


def parse_period_header(header: Any) -> Optional[pd.Period]:
    """
    Return the monthly Period a column header denotes, or None if it is not a period.

    Deliberately conservative: a bare ``2026`` or ``M1`` is rejected, because integer
    codes and year totals appear as ordinary columns often enough that accepting them
    would misclassify normal tables as time series.
    """
    # Excel writes a date-typed header cell as a serial number, so a demand matrix
    # whose last columns were entered as real dates arrives as `45992`, `46023`. Those
    # columns are then not recognised as periods and are dropped from the reshape —
    # silently losing the most recent months, which are the ones the forecast leans on
    # hardest. The serial range cannot collide with a year header.
    serial = _as_excel_serial(header)
    if serial is not None:
        return (_EXCEL_EPOCH + pd.Timedelta(days=serial)).to_period("M")

    spaced = normalize_header(header)
    if not spaced:
        return None
    # Try the separator-preserving form first, then the collapsed one. `01-2026`
    # normalises to `01 2026`, which only the separated pattern can disambiguate,
    # while `202601` only matches once the separator is gone.
    for pattern, kind in _PERIOD_PATTERNS:
        m = pattern.match(spaced) or pattern.match(spaced.replace(" ", ""))
        if not m:
            continue
        try:
            if kind == "ym":
                year, month = int(m.group(1)), int(m.group(2))
            elif kind == "my":
                month, year = int(m.group(1)), int(m.group(2))
            elif kind == "mony":
                month = _MONTH_NAMES.get(m.group(1))
                year = _two_digit_year(m.group(2))
                if month is None:
                    continue
            elif kind == "ymon":
                year = int(m.group(1))
                month = _MONTH_NAMES.get(m.group(2))
                if month is None:
                    continue
            elif kind == "yq":
                year, month = int(m.group(1)), (int(m.group(2)) - 1) * 3 + 1
            elif kind == "qy":
                month = (int(m.group(1)) - 1) * 3 + 1
                year = _two_digit_year(m.group(2))
            elif kind == "ymd":
                year, month = int(m.group(1)), int(m.group(2))
            else:
                continue

            if not (1 <= month <= 12) or not (1990 <= year <= 2100):
                continue
            return pd.Period(year=year, month=month, freq="M")
        except (ValueError, TypeError):
            continue
    return None


def _two_digit_year(text: str) -> int:
    value = int(text)
    if value < 100:
        # 70-99 -> 1970s-90s, 00-69 -> 2000s-60s. Standard windowing.
        return 1900 + value if value >= 70 else 2000 + value
    return value


# ── Value masking (compliance) ───────────────────────────────────────────────

def mask_value(value: Any) -> str:
    """
    Reduce a value to its shape: letters -> A, digits -> 9, everything else kept.
    ``ACME-10023`` becomes ``AAAA-99999``. Enough to recognise an ID format, not
    enough to identify a customer.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "<null>"
    out = []
    for ch in str(value)[:40]:
        if ch.isalpha():
            out.append("A")
        elif ch.isdigit():
            out.append("9")
        else:
            out.append(ch)
    return "".join(out)


# ── Locale detection ─────────────────────────────────────────────────────────
#
# Both of these corrupt data *silently*, which is what makes them worth detecting
# rather than documenting. `05.01.2026` read as MDY is off by four months and still
# parses cleanly; `1.200,50` read with US conventions becomes NaN and disappears into
# a null-rate statistic nobody reads.

_DMY_RE = re.compile(r"^\s*(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\s*$")

# Both patterns require the *comma* to appear, because a comma is the only
# unambiguous evidence. Dots alone are not: `516.850.900` is a European 516 million,
# a US part number, or a hierarchical code, and real ERP item masters are full of the
# latter. Treating dot-grouped text as European numbers made an entire sales extract
# claim comma decimals on the strength of its part numbers alone — which would then
# read a genuine US `1.234` as 1.234 instead of 1234.
_EU_NUM_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*,\d+$|^-?\d+,\d{1,2}$")
_US_NUM_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$|^-?\d+\.\d{1,2}$")

# Excel serial-date range, 2000-01-01 → 2050-01-01. Bare years (2018, 2026) fall far
# below it, so a year header can never be mistaken for a serial.
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_EXCEL_SERIAL_MIN = 36526
_EXCEL_SERIAL_MAX = 54789


def detect_date_order(values: pd.Series) -> Optional[bool]:
    """
    True when the column is day-first, False when month-first, None when ambiguous.

    Decided by counting values whose first or second component exceeds 12 — those are
    the only ones that carry evidence. A column of purely ambiguous dates (all
    components <= 12) correctly returns None rather than guessing.
    """
    day_first = month_first = 0
    for raw in values.dropna().astype(str).head(400):
        m = _DMY_RE.match(raw)
        if not m:
            continue
        first, second = int(m.group(1)), int(m.group(2))
        if first > 12 and second <= 12:
            day_first += 1
        elif second > 12 and first <= 12:
            month_first += 1
    if day_first == month_first == 0:
        return None
    return day_first > month_first


# ── Internal consistency ─────────────────────────────────────────────────────
#
# One column, two ways of writing the same thing. The assumption that a column speaks
# one convention is usually safe and, when it is wrong, wrong in a way that produces no
# error — one representation parses and the other quietly becomes null.
#
# Excel is the usual cause. Open a US-format export under a day-first locale and it
# converts the cells it *can* read as day-first — the ones where both components are
# 12 or less — into real dates, silently swapping month and day as it goes, and leaves
# the rest as text because `10/14/2024` is not a valid day-first date. The column is
# then half native dates and half text, and the two halves disagree about what the
# numbers mean. The real sales extract had 25,391 text values against 8,737 converted
# ones in `Invdate Date`, and the same split again in `Orddate Date`.
#
# This does not attempt to repair anything. It reports that a column contains more than
# one kind of value, and leaves the judgement to a person, because the correct response
# differs: a date mix is repairable, a decimal-style mix usually means two systems were
# pasted together, and a numeric-and-text mix is often a footnote row.

_ISO_DATE_RE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}([ T].*)?$")
_DELIM_DATE_RE = re.compile(r"^\s*\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\s*$")
_PLAIN_NUM_RE = re.compile(r"^-?[\d,.\s]+$")

# A representation has to hold this share of the column before the mix is worth
# raising. Below it the odd value is a footnote or a stray "N/A", not a second
# convention, and flagging it trains the reader to skip the warning.
MIX_MIN_SHARE = 0.02
MIX_MIN_COUNT = 10


def classify_representation(raw: Any) -> Optional[str]:
    """Which way of writing a value this is, or None when it carries no signal."""
    text = str(raw).strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return None
    if _ISO_DATE_RE.match(text):
        return "iso_date"
    if _DELIM_DATE_RE.match(text):
        return "delimited_date"
    if _EU_NUM_RE.match(text):
        return "eu_number"
    if _US_NUM_RE.match(text):
        return "us_number"
    if _PLAIN_NUM_RE.match(text):
        return "plain_number"
    return "text"


def detect_representation_mix(values: pd.Series) -> Optional[Dict[str, int]]:
    """
    Counts per representation when a column holds more than one, else None.

    Reads the whole column rather than a sample. A partial Excel conversion follows
    whichever values were ambiguous, not row order, so the first 500 rows of a split
    column are frequently all one representation — sampling would report exactly the
    files that look tidiest at the top and miss the split underneath.

    Only *incompatible* pairings are reported. A column of dates alongside a handful of
    free-text remarks is untidy; a column of dates written two ways is a correctness
    problem, because whichever parser is chosen silently discards the other half.
    """
    seen = values.dropna()
    if seen.empty:
        return None

    text = seen.astype("string").str.strip()
    text = text[text.notna() & (text != "") & ~text.str.lower().isin(["nan", "nat", "none"])]
    if text.empty:
        return None

    # Matched in the same priority order as `classify_representation`, each mask
    # excluding everything already claimed, so the counts partition the column.
    counts: Dict[str, int] = {}
    remaining = pd.Series(True, index=text.index)
    for kind, pattern in (("iso_date", _ISO_DATE_RE),
                          ("delimited_date", _DELIM_DATE_RE),
                          ("eu_number", _EU_NUM_RE),
                          ("us_number", _US_NUM_RE),
                          ("plain_number", _PLAIN_NUM_RE)):
        hit = remaining & text.str.match(pattern).fillna(False)
        n = int(hit.sum())
        if n:
            counts[kind] = n
        remaining &= ~hit
    leftover = int(remaining.sum())
    if leftover:
        counts["text"] = leftover

    if len(counts) < 2:
        return None

    total = sum(counts.values())
    material = {
        k: v for k, v in counts.items()
        if v >= MIX_MIN_COUNT and v / total >= MIX_MIN_SHARE
    }
    if len(material) < 2:
        return None

    # Which combinations actually corrupt. Two ways of writing a date, or two decimal
    # conventions, mean half the column parses wrong. Numbers next to text usually
    # means a total row, which `_count_total_rows` already reports.
    date_forms = {"iso_date", "delimited_date"} & set(material)
    number_forms = {"eu_number", "us_number"} & set(material)
    if len(date_forms) < 2 and len(number_forms) < 2:
        return None
    return dict(sorted(material.items(), key=lambda kv: -kv[1]))


def detect_decimal_style(values: pd.Series) -> Optional[str]:
    """
    Distinguish European (1.200,50) from US (1,200.50) numeric formatting.

    Only values that unambiguously match one convention are counted, so a column of
    plain integers yields None and the caller keeps its default.
    """
    eu = us = 0
    for raw in values.dropna().astype(str).head(400):
        text = raw.strip()
        if _EU_NUM_RE.match(text):
            eu += 1
        elif _US_NUM_RE.match(text):
            us += 1
    if eu == us == 0:
        return None
    return "eu" if eu > us else "us"


def _collapse_mask(mask: str) -> str:
    """Run-length collapse: AAAA-99999 -> A{4}-9{5}, so patterns group cleanly."""
    if not mask:
        return mask
    parts, prev, count = [], mask[0], 1
    for ch in mask[1:]:
        if ch == prev:
            count += 1
        else:
            parts.append(f"{prev}{{{count}}}" if count > 1 else prev)
            prev, count = ch, 1
    parts.append(f"{prev}{{{count}}}" if count > 1 else prev)
    return "".join(parts)


# ── Source-system detection ──────────────────────────────────────────────────
#
# Which ERP wrote a file is worth knowing on its own, because some header words mean
# different things in different systems and no amount of data shape resolves the
# disagreement. `Item` is the case that forced this: in SAP every transactional table
# is keyed on `<document> + <item>`, and that item number renders as `Item` in ALV
# output — VBAP-POSNR, EKPO-EBELP, MSEG-ZEILE, QMFE-FENUM all of them. In SAP `Item`
# is never the material. In Oracle, NetSuite and most WMS exports it usually is.
#
# So the ambiguity is not in the data, it is in the dialect, and it is resolvable
# once — here — rather than guessed at per column.

SYSTEM_SAP = "sap"
SYSTEM_UNKNOWN = "unknown"

# One of these settles it. Either an SAP data-dictionary field name — which only
# reaches a spreadsheet via SE16, a query or an ABAP report — or a display phrase
# that no other ERP produces.
_SAP_STRONG = frozenset({
    # Data dictionary names, by module
    "matnr", "maktx", "mtart", "matkl", "meins", "bismt", "charg",           # MARA/MAKT
    "werks", "lgort", "dispo", "dismm", "beskz", "plifz", "webaz",           # MARC
    "eisbe", "minbe", "disls", "bstmi", "bstfe", "bstrf", "mabst", "lgrad",
    "labst", "insme", "speme",                                               # MARD
    "stprs", "verpr", "vprsv", "peinh", "salk3", "lbkum", "bwkey",           # MBEW
    "ebeln", "ebelp", "etenr", "eindt", "bedat", "netpr", "elikz", "pstyp",  # EKKO/EKPO/EKET
    "lifnr", "ekorg", "ekgrp", "bstae", "wemng",
    "vbeln", "posnr", "auart", "kunnr", "vdatu", "kwmeng", "vrkme", "netwr", # VBAK/VBAP
    "waerk", "pstyv", "abgru", "uepos", "kdmat", "vkorg", "vtweg", "spart",
    "wmeng", "bmeng", "edatu", "mbdat", "lddat", "wadat", "wadat_ist",       # VBEP/LIKP
    "lfimg", "vgbel", "vgpos", "fkimg", "fkdat", "fkart", "aubel", "aupos",  # LIPS/VBRP
    "omeng", "vbtyp", "gbsta", "lfsta",
    "mblnr", "mjahr", "zeile", "bwart", "shkzg", "budat", "sobkz", "dmbtr",  # MKPF/MSEG
    "qmnum", "qmart", "fenum", "fegrp", "fecod", "otgrp", "oteil",           # QMEL/QMFE
    "aufnr", "aufpl", "aplzl", "vornr", "arbid", "gltrp", "gstrp",           # AUFK/AFKO/AFVC
    "rsnum", "rspos", "bdmng", "enmng", "bdter",                             # RESB
    "equnr", "sernr", "tplnr", "obknr",                                      # EQUI/IFLOT/OBJK
    "bukrs", "waers", "menge", "zterm", "inco1", "inco2",
    # Display phrases produced by SAP and effectively nothing else
    "sold to party", "ship to party", "bill to party", "payer",
    "purchasing document", "purchasing doc", "purch doc", "purchasing group",
    "purchasing organization", "purchasing organisation",
    "material document", "movement type", "storage location", "special stock",
    "mrp controller", "mrp type", "lot size", "valuation area", "valuation class",
    "moving average price", "planned delivery time", "gr processing time",
    "base unit of measure", "wbs element", "functional location",
    "unrestricted", "unrestricted use", "unrestricted stock",
    "higher level item", "schedule line", "reason for rejection",
    "goods recipient", "notification number", "reservation number",
})

# Two of these together settle it. Individually they are ordinary supply-chain words
# that a non-SAP export can carry — which is exactly why one is not enough.
_SAP_MEDIUM = frozenset({
    "plant", "batch", "company code", "profit center", "profit centre",
    "material description", "material group", "material type",
    "material number", "material classification",
    "sales document", "billing document", "sales organization",
    "sales organisation", "distribution channel", "division",
    "item category", "document date", "posting date", "requirement date",
    "storage bin", "condition type", "cost center", "cost centre",
    "created by", "created on",
})

# The other decisive signal, and the one that survives translation: SAP stores keys in
# fixed-width character fields and pads them with zeros, so a material number leaves
# the system as `000000000006013908`. No other convention produces that.
_SAP_PADDED_KEY = re.compile(r"^0{2,}\d+$")
_SAP_PADDED_MIN_LEN = 8
_SAP_PADDED_MIN_SHARE = 0.8


# Bounds on what can pass for a line number. The widest of SAP's own item fields is
# six digits (VBAP-POSNR); EKPO-EBELP is five and MSEG-ZEILE four.
_ORDINAL_MAX_VALUE = 999_999
# Above this many distinct values the column is a catalogue, not a position within a
# document — no order has a thousand lines.
_ORDINAL_MAX_DISTINCT = 200
# A line number is drawn from a small vocabulary reused by every document, so the rows
# must outnumber the distinct values several times over. An item number does not.
_ORDINAL_MIN_REPEAT = 3.0


def _looks_like_a_line_number(values: pd.Series, distinct: int, rows: int) -> bool:
    """
    Whether an integer column has the shape of a document line number.

    Three things have to hold together, and the third is what carries the weight.
    Small values and heavy repetition are necessary but common — a plant code or a
    status flag has both. What distinguishes a position number is that its values
    form a *series*: either SAP's default increment of ten (10, 20, 30 …) or a dense
    count from one. A numeric part number satisfies neither.

    Deliberately conservative, because the verdict does bite: a column judged a series
    is refused as an item key even where that leaves a required field unmapped. A miss
    costs nothing — the other two layers still apply — while a false positive would
    refuse a real key and stop a run that should have gone through.
    """
    if distinct < 2 or distinct > _ORDINAL_MAX_DISTINCT:
        return False
    if rows < distinct * _ORDINAL_MIN_REPEAT:
        return False

    unique = np.sort(values.unique())
    if unique[0] < 0 or unique[-1] > _ORDINAL_MAX_VALUE:
        return False

    by_tens = bool(unique[-1] >= 10 and (unique % 10 == 0).all())
    # A dense count from 0 or 1 with room for only a few gaps.
    from_one = bool(unique[0] <= 1 and unique[-1] <= distinct * 2)
    return by_tens or from_one


def detect_source_system(
    header_tokens: List[str], columns: List[ColumnProfile]
) -> Tuple[str, List[str]]:
    """
    Which ERP wrote this file, and the evidence for saying so.

    Deliberately biased toward `unknown`. A false positive is expensive — it turns on
    dialect rules that forbid mappings, so calling a non-SAP file SAP can leave a
    genuine `Item`-keyed source with no item key at all. A false negative only means
    the generic rules apply, which is where every file started.
    """
    tokens = set(header_tokens)
    evidence: List[str] = []

    strong = sorted(tokens & _SAP_STRONG)
    medium = sorted(tokens & _SAP_MEDIUM)
    padded = [c.name for c in columns if c.zero_padded_code]

    if strong:
        evidence.append(f"SAP field names in headers: {', '.join(strong[:4])}")
    if padded:
        evidence.append(f"zero-padded SAP keys in {', '.join(padded[:3])}")
    if len(medium) >= 2:
        evidence.append(f"SAP vocabulary: {', '.join(medium[:4])}")

    if strong or padded or len(medium) >= 2:
        return SYSTEM_SAP, evidence
    return SYSTEM_UNKNOWN, []


# ── Column profile ───────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    # The DataFrame's actual column label, which is not always a string — Excel hands
    # back integers for date-typed headers. `name` stays a string for display and
    # matching; anything that indexes the frame must use this instead, or the lookup
    # misses exactly the columns that were entered as real dates.
    label: Any
    position: int
    normalized: str
    non_null: int
    null_rate: float
    distinct: int
    distinct_rate: float
    inferred_type: str                       # string | integer | decimal | date | boolean | empty
    numeric_parse_rate: float = 0.0
    date_parse_rate: float = 0.0
    value_range: Optional[Tuple[float, float]] = None
    negative_rate: float = 0.0
    zero_rate: float = 0.0
    top_patterns: List[Tuple[str, int]] = dc_field(default_factory=list)
    is_period_column: bool = False
    period: Optional[str] = None
    distinct_values: Optional[List[str]] = None   # only for low-cardinality categoricals
    samples: Optional[List[str]] = None           # only when include_samples=True
    dayfirst_evidence: Optional[bool] = None      # True=DMY, False=MDY, None=ambiguous
    decimal_style: Optional[str] = None           # "eu" (1.200,50) | "us" (1,200.50)
    # Two or more incompatible ways of writing the same value in one column. Whichever
    # parser wins, the other representation becomes null without raising.
    representation_mix: Optional[Dict[str, int]] = None
    # An integer column holding a short, repeating run of small numbers — the shape of
    # an ERP line/position number (10, 20, 30 … or 1, 2, 3 …). Recorded because that
    # shape is the difference between a line number and an item number, and the header
    # alone cannot tell them apart: SAP spells both of them `Item`.
    is_small_ordinal: bool = False
    # Values are numbers written with leading zeros to a fixed width — SAP's own
    # storage convention for keys, e.g. `000000000006013908`. Feeds source-system
    # detection, and is itself a warning: joined against an unpadded export of the
    # same key, it matches nothing.
    zero_padded_code: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "position": self.position,
            "inferred_type": self.inferred_type,
            "null_rate": round(self.null_rate, 4),
            "distinct": self.distinct,
            "distinct_rate": round(self.distinct_rate, 4),
        }
        if self.value_range is not None:
            lo, hi = self.value_range
            out["value_range"] = [_round_sig(lo), _round_sig(hi)]
        if self.negative_rate:
            out["negative_rate"] = round(self.negative_rate, 4)
        if self.zero_rate:
            out["zero_rate"] = round(self.zero_rate, 4)
        if self.date_parse_rate:
            out["date_parse_rate"] = round(self.date_parse_rate, 4)
        if self.numeric_parse_rate:
            out["numeric_parse_rate"] = round(self.numeric_parse_rate, 4)
        if self.top_patterns:
            out["top_patterns"] = [{"pattern": p, "count": c} for p, c in self.top_patterns]
        # Distinct values stay on the in-process object for the local drafter, but are
        # withheld from the exportable portrait unless samples were explicitly
        # requested. A low-cardinality column is usually a status vocabulary — but it
        # can equally be a short vendor list, and the two are indistinguishable from
        # cardinality alone. Sending only what is provably safe is the wrong trade
        # here; withholding by default and opting in is the right one.
        if self.distinct_values is not None and self.samples is not None:
            out["distinct_values"] = self.distinct_values
        if self.is_period_column:
            out["is_period_column"] = True
            out["period"] = self.period
        if self.dayfirst_evidence is not None:
            out["date_order"] = "DMY" if self.dayfirst_evidence else "MDY"
        if self.decimal_style:
            out["decimal_style"] = self.decimal_style
        if self.representation_mix:
            out["representation_mix"] = self.representation_mix
        if self.is_small_ordinal:
            out["is_small_ordinal"] = True
        if self.zero_padded_code:
            out["zero_padded_code"] = True
        if self.samples is not None:
            out["samples"] = self.samples
        return out


def _round_sig(x: float, sig: int = 3) -> float:
    """Round to significant figures — a range of [0, 12483] becomes [0, 12500]."""
    if x is None or not np.isfinite(x) or x == 0:
        return 0.0
    from math import floor, log10

    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


# ── Table profile ────────────────────────────────────────────────────────────

@dataclass
class TableProfile:
    """Everything deterministically knowable about a file, without reading its meaning."""

    source_name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    header_hash: str
    header_tokens: List[str]
    shape: str                                # "long" | "wide_periods"
    period_columns: List[str] = dc_field(default_factory=list)
    period_range: Optional[Tuple[str, str]] = None
    id_columns: List[str] = dc_field(default_factory=list)
    candidate_keys: List[List[str]] = dc_field(default_factory=list)
    suspected_total_rows: int = 0
    notes: List[str] = dc_field(default_factory=list)
    # Which ERP wrote the file, and why we think so. Drives the dialect rules in the
    # contracts — the ones that say a header word means something different here.
    system: str = SYSTEM_UNKNOWN
    system_evidence: List[str] = dc_field(default_factory=list)

    def column(self, name: str) -> Optional[ColumnProfile]:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def dayfirst(self) -> bool:
        """
        Table-level verdict on date order, by majority of columns that carry evidence.
        Applied file-wide because one export is written by one system with one locale.
        """
        votes = [c.dayfirst_evidence for c in self.columns if c.dayfirst_evidence is not None]
        return bool(votes) and sum(votes) > len(votes) / 2

    @property
    def decimal_style(self) -> Optional[str]:
        """Table-level numeric convention, by majority of columns that carry evidence."""
        votes = [c.decimal_style for c in self.columns if c.decimal_style]
        if not votes:
            return None
        return "eu" if votes.count("eu") > votes.count("us") else "us"

    def to_dict(self) -> Dict[str, Any]:
        """Metadata-only portrait — this is the payload safe to show a model."""
        out = {
            "source_name": self.source_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "header_hash": self.header_hash,
            "shape": self.shape,
            "columns": [c.to_dict() for c in self.columns if not c.is_period_column],
            "candidate_keys": self.candidate_keys,
        }
        if self.system != SYSTEM_UNKNOWN:
            out["system"] = self.system
            out["system_evidence"] = self.system_evidence
        if self.shape == "wide_periods":
            out["period_columns_count"] = len(self.period_columns)
            out["period_range"] = list(self.period_range) if self.period_range else None
            out["id_columns"] = self.id_columns
        if self.suspected_total_rows:
            out["suspected_total_rows"] = self.suspected_total_rows
        if self.notes:
            out["notes"] = self.notes
        return out

    def summary(self) -> str:
        lines = [
            f"  Profile — {self.source_name}",
            f"    shape        : {self.shape}",
            f"    rows x cols  : {self.row_count:,} x {self.column_count}",
        ]
        if self.shape == "wide_periods":
            rng = f"{self.period_range[0]} -> {self.period_range[1]}" if self.period_range else "?"
            lines.append(f"    periods      : {len(self.period_columns)} ({rng})")
            lines.append(f"    id columns   : {', '.join(self.id_columns) or '(none)'}")
        if self.candidate_keys:
            keys = " | ".join("+".join(k) for k in self.candidate_keys[:3])
            lines.append(f"    candidate key: {keys}")
        if self.suspected_total_rows:
            lines.append(f"    ⚠ suspected total/subtotal rows: {self.suspected_total_rows}")
        for note in self.notes:
            lines.append(f"    note         : {note}")
        return "\n".join(lines)


# ── Profiler ─────────────────────────────────────────────────────────────────

class Profiler:
    """Builds a TableProfile from a raw DataFrame."""

    # A table is treated as wide when this share of its columns parse as periods.
    WIDE_PERIOD_MIN_COUNT = 3
    WIDE_PERIOD_MIN_SHARE = 0.30
    # Below this distinct-value count a string column is reported as a controlled
    # vocabulary — that is what makes status/incoterm value domains discoverable.
    CATEGORICAL_MAX_DISTINCT = 25
    SAMPLE_ROWS = 500

    def profile(
        self,
        df: pd.DataFrame,
        source_name: str = "<dataframe>",
        include_samples: bool = False,
    ) -> TableProfile:
        cols: List[ColumnProfile] = []
        total = max(len(df), 1)

        for pos, name in enumerate(df.columns):
            cols.append(self._profile_column(df[name], name, pos, total, include_samples))

        period_cols = [c for c in cols if c.is_period_column]
        shape, period_range = self._detect_shape(cols, period_cols)

        id_columns: List[str] = []
        if shape == "wide_periods":
            id_columns = [c.label for c in cols if not c.is_period_column]

        notes: List[str] = []
        suspected_totals = self._count_total_rows(df, cols, notes)
        candidate_keys = self._candidate_keys(df, cols, shape)

        header_tokens = [normalize_header(c) for c in df.columns]
        header_hash = self.header_hash(df.columns)
        system, system_evidence = detect_source_system(header_tokens, cols)

        if shape == "wide_periods":
            notes.append(
                f"Wide period layout detected ({len(period_cols)} period columns) — "
                f"routed as pre-aggregated demand, not a transactional extract"
            )

        if system != SYSTEM_UNKNOWN:
            notes.append(f"Source system: {system.upper()} ({'; '.join(system_evidence)})")

        dmy_votes = [c.name for c in cols if c.dayfirst_evidence is True]
        eu_votes = [c.name for c in cols if c.decimal_style == "eu"]
        if dmy_votes:
            notes.append(f"Day-first dates detected (evidence in {dmy_votes[:3]})")
        if eu_votes:
            notes.append(f"European decimal format detected (evidence in {eu_votes[:3]})")

        # Raised loudly and per column, because this is the class of defect that
        # produces a clean-looking run over half the data.
        for col in cols:
            if not col.representation_mix:
                continue
            detail = ", ".join(f"{k.replace('_', ' ')} x{v:,}"
                               for k, v in col.representation_mix.items())
            notes.append(
                f"⚠ MIXED FORMATS in {col.name!r}: {detail}. One column, two conventions "
                f"— whichever one is parsed, the other becomes null without raising. "
                f"Check this column by hand before trusting anything derived from it."
            )

        return TableProfile(
            source_name=source_name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=cols,
            header_hash=header_hash,
            header_tokens=header_tokens,
            shape=shape,
            period_columns=[c.label for c in period_cols],
            period_range=period_range,
            id_columns=id_columns,
            candidate_keys=candidate_keys,
            suspected_total_rows=suspected_totals,
            notes=notes,
            system=system,
            system_evidence=system_evidence,
        )

    # ── Fingerprinting ───────────────────────────────────────────────────────

    @staticmethod
    def header_hash(columns) -> str:
        """
        Stable hash over the normalised, *sorted* header set.

        Sorted because ERP exports reorder columns between runs far more often than
        they rename them; hashing positionally would invalidate a working adapter for
        a cosmetic change.
        """
        tokens = sorted({normalize_header(c) for c in columns if normalize_header(c)})
        digest = hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()
        return f"sha256:{digest[:16]}"

    # ── Column-level ─────────────────────────────────────────────────────────

    def _profile_column(
        self, series: pd.Series, name: Any, pos: int, total: int, include_samples: bool
    ) -> ColumnProfile:
        non_null_series = series.dropna()
        # Treat whitespace-only cells as null; ERP exports are full of them
        if non_null_series.dtype == object:
            non_null_series = non_null_series[non_null_series.astype(str).str.strip() != ""]

        non_null = len(non_null_series)
        distinct = int(non_null_series.nunique()) if non_null else 0
        period = parse_period_header(name)

        prof = ColumnProfile(
            name=str(name),
            label=name,
            position=pos,
            normalized=normalize_header(name),
            non_null=non_null,
            null_rate=1.0 - (non_null / total),
            distinct=distinct,
            distinct_rate=(distinct / non_null) if non_null else 0.0,
            inferred_type="empty",
            is_period_column=period is not None,
            period=str(period) if period is not None else None,
        )

        if non_null == 0:
            return prof

        sample = non_null_series.head(self.SAMPLE_ROWS)

        # Judged on the raw text, before any numeric inference. `000000000006013908`
        # parses cleanly as an integer, so by the time a type has been inferred the
        # padding — the whole signal — is gone.
        padded = sample.astype(str).str.strip()
        prof.zero_padded_code = bool(
            (padded.str.len() >= _SAP_PADDED_MIN_LEN).mean() >= _SAP_PADDED_MIN_SHARE
            and padded.str.match(_SAP_PADDED_KEY).mean() >= _SAP_PADDED_MIN_SHARE
        )

        prof.decimal_style = detect_decimal_style(sample)
        # Judged over more rows than the other detectors use. A partial Excel
        # conversion is not evenly distributed — it follows whichever values happened
        # to be ambiguous — so a 500-row sample off the top of the file can be entirely
        # one representation while the column as a whole is split.
        prof.representation_mix = detect_representation_mix(non_null_series)
        # Strip whichever thousands separator this column actually uses before
        # judging numeric-ness, or every European-formatted amount reads as text.
        stripped = sample.astype(str).str.strip()
        if prof.decimal_style == "eu":
            stripped = stripped.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            stripped = stripped.str.replace(",", "", regex=False)
        numeric = pd.to_numeric(stripped, errors="coerce")
        prof.numeric_parse_rate = float(numeric.notna().mean())

        if prof.numeric_parse_rate < 0.9:
            # Order must be decided from the raw text, not from whether pandas can
            # parse it. A day-first column like `13.09.2026` fails month-first parsing
            # outright, so testing parseability first would classify the most
            # unambiguously DMY columns as plain strings and never reach this check.
            prof.dayfirst_evidence = detect_date_order(sample)
            with warnings.catch_warnings():
                # Mixed formats are the norm in ERP exports; here we only want the
                # parse *rate*, so per-element fallback is acceptable.
                warnings.simplefilter("ignore", UserWarning)
                dates = pd.to_datetime(
                    sample, errors="coerce", dayfirst=bool(prof.dayfirst_evidence)
                )
                rate = float(dates.notna().mean())
                if rate < 0.8 and prof.dayfirst_evidence is None:
                    # No textual evidence either way — try the other order before
                    # concluding this is not a date column at all.
                    alt = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                    if float(alt.notna().mean()) > rate:
                        rate = float(alt.notna().mean())
                        prof.dayfirst_evidence = True
            prof.date_parse_rate = rate
        else:
            prof.date_parse_rate = 0.0

        if prof.numeric_parse_rate >= 0.9:
            full_text = non_null_series.astype(str).str.strip()
            if prof.decimal_style == "eu":
                full_text = (full_text.str.replace(".", "", regex=False)
                                      .str.replace(",", ".", regex=False))
            else:
                full_text = full_text.str.replace(",", "", regex=False)
            full_numeric = pd.to_numeric(full_text, errors="coerce").dropna()
            if len(full_numeric):
                prof.value_range = (float(full_numeric.min()), float(full_numeric.max()))
                prof.negative_rate = float((full_numeric < 0).mean())
                prof.zero_rate = float((full_numeric == 0).mean())
                is_int = bool((full_numeric % 1 == 0).all())
                prof.inferred_type = "integer" if is_int else "decimal"
                if is_int:
                    prof.is_small_ordinal = _looks_like_a_line_number(
                        full_numeric, distinct, total
                    )
        elif prof.date_parse_rate >= 0.8:
            prof.inferred_type = "date"
        else:
            prof.inferred_type = "string"
            lowered = set(sample.astype(str).str.strip().str.lower().unique())
            if lowered <= {"y", "n", "yes", "no", "true", "false", "0", "1", "x", ""}:
                prof.inferred_type = "boolean"

        if prof.inferred_type in ("string", "boolean"):
            masks = sample.astype(str).map(lambda v: _collapse_mask(mask_value(v)))
            prof.top_patterns = [
                (pattern, int(count)) for pattern, count in masks.value_counts().head(3).items()
            ]
            if distinct <= self.CATEGORICAL_MAX_DISTINCT:
                # Status codes and incoterms are business vocabulary, not personal data.
                # Surfacing them is what lets a value domain be written for this source.
                prof.distinct_values = sorted(
                    non_null_series.astype(str).str.strip().unique().tolist()
                )[: self.CATEGORICAL_MAX_DISTINCT]

        if include_samples:
            prof.samples = sample.head(5).astype(str).tolist()

        return prof

    # ── Shape ────────────────────────────────────────────────────────────────

    def _detect_shape(
        self, cols: List[ColumnProfile], period_cols: List[ColumnProfile]
    ) -> Tuple[str, Optional[Tuple[str, str]]]:
        if len(period_cols) < self.WIDE_PERIOD_MIN_COUNT:
            return "long", None
        if len(period_cols) / max(len(cols), 1) < self.WIDE_PERIOD_MIN_SHARE:
            return "long", None
        # A period column that holds text is a date column with an odd name, not a
        # measure — the wide layout requires the period columns to carry quantities.
        numeric_periods = [c for c in period_cols if c.inferred_type in ("integer", "decimal", "empty")]
        if len(numeric_periods) < len(period_cols) * 0.7:
            return "long", None

        periods = sorted(pd.Period(c.period, freq="M") for c in period_cols if c.period)
        rng = (str(periods[0]), str(periods[-1])) if periods else None
        return "wide_periods", rng

    # ── Keys and totals ──────────────────────────────────────────────────────

    def _candidate_keys(
        self, df: pd.DataFrame, cols: List[ColumnProfile], shape: str
    ) -> List[List[str]]:
        """
        Single columns that are unique, then pairs of identifier-ish columns that are.
        Feeds the structural test: a declared grain must match an actual unique key,
        which is how a PO-header extract masquerading as PO lines gets caught.
        """
        if len(df) == 0:
            return []
        candidates: List[List[str]] = []
        id_like = [
            c for c in cols
            if not c.is_period_column
            and c.non_null > 0
            and c.inferred_type in ("string", "integer")
            and c.null_rate < 0.05
        ]

        for c in id_like:
            if c.distinct == len(df):
                candidates.append([c.name])
        if candidates:
            return candidates[:3]

        # Pairs — capped because this is O(n^2) over columns
        ranked = sorted(id_like, key=lambda c: -c.distinct_rate)[:8]
        for i, a in enumerate(ranked):
            for b in ranked[i + 1:]:
                try:
                    if len(df.drop_duplicates(subset=[a.name, b.name])) == len(df):
                        candidates.append([a.name, b.name])
                except (TypeError, KeyError):
                    continue
                if len(candidates) >= 3:
                    return candidates
        return candidates[:3]

    def _count_total_rows(
        self, df: pd.DataFrame, cols: List[ColumnProfile], notes: List[str]
    ) -> int:
        """
        Spot subtotal / grand-total rows: an identifier cell containing 'total' or
        being blank while the measures are populated. These inflate every aggregate
        and are invisible once the file is summed.
        """
        if len(df) == 0:
            return 0
        id_cols = [
            c.name for c in cols
            if c.inferred_type == "string" and not c.is_period_column and c.distinct_rate > 0.1
        ][:2]
        if not id_cols:
            return 0

        mask = pd.Series(False, index=df.index)
        for name in id_cols:
            text = df[name].astype(str).str.strip().str.lower()
            mask |= text.str.contains(r"\b(?:total|subtotal|grand total|sum|合计|总计)\b",
                                      regex=True, na=False)
        count = int(mask.sum())
        if count:
            notes.append(
                f"{count} row(s) look like totals/subtotals — exclude before aggregating"
            )
        return count


def profile_frame(
    df: pd.DataFrame, source_name: str = "<dataframe>", include_samples: bool = False
) -> TableProfile:
    """Convenience wrapper for the common single-call case."""
    return Profiler().profile(df, source_name=source_name, include_samples=include_samples)
