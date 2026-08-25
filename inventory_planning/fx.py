"""
Currency normalisation — every money column in one currency before anything sums it.

A multi-country purchase organisation raises POs in the supplier's currency. SAP
exports that faithfully: one `Currency` column, one `PO Total Value` per line, and no
conversion. Summing that column adds ₹7,004,449,000 to $240,323,000 and prints the
result with a dollar sign. Every number built on it — ABC classification, EOQ holding
cost, excess value, value at risk, the frontier — inherits the error, and it is
invisible because the total still looks like money.

The conversion is deliberately conservative in three ways:

  A rate the source booked wins.   When the export carries its own exchange rate that
                                   is the rate the transaction actually settled at.
                                   The rate table is the fallback, not the authority.

  Rates are effective-dated.       Purchase history spans years. One rate applied to
                                   eight years of INR turns a currency move into a
                                   procurement trend. Rates carry a start date and the
                                   row is converted at the rate in force when it was
                                   raised.

  An unknown currency is not 1.0.  A code with no rate blanks the money columns and is
                                   reported. Defaulting it to the reporting currency
                                   is how ₹126,550 becomes $126,550, and nothing
                                   downstream can tell that it happened.

The one assumption made silently: a document with no currency column at all is taken
to be single-currency and already in the reporting currency. That is the ordinary case
for a single-entity export, and the alternative — refusing to run — would break every
extract that has never needed the field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

DEFAULT_REPORTING_CURRENCY = "USD"

# Money columns per document type, and the date that decides which rate applies.
# `unit_cost` and `unit_price` are per-unit money, so they convert on the same rate as
# the line value they were derived from.
MONEY_COLUMNS: Dict[str, Sequence[str]] = {
    "open_po": ("open_amount", "unit_cost"),
    "po_history": ("po_amount", "unit_cost", "freight_cost"),
    "open_so": ("open_amount", "unit_price"),
    "sales_history": ("amount", "unit_price"),
    "inventory": ("unit_cost", "inventory_value"),
    "item_master": ("unit_cost",),
    "planning_master": ("unit_cost",),
}

# Which timestamp dates the rate. Master data has none — it is a current-state
# document, so it converts at the latest rate.
RATE_DATE_COLUMNS: Dict[str, Sequence[str]] = {
    "open_po": ("order_date", "committed_delivery"),
    "po_history": ("po_date", "receive_date"),
    "open_so": ("order_date", "customer_request_date"),
    "sales_history": ("demand_date", "ship_date", "order_date"),
    "inventory": ("snapshot_date",),
    "item_master": (),
    "planning_master": (),
}

CURRENCY_COLUMN = "currency"
ROW_RATE_COLUMN = "fx_rate"


# ── Rate table ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FxQuote:
    """One rate, valid from `effective_from` until the next quote for that currency."""

    effective_from: pd.Timestamp
    rate: float
    source: str = ""
    # A rate nobody measured. It converts, so the run completes and the figures are the
    # right order of magnitude — but it is reported on every run that uses it, because
    # a placeholder that goes unnoticed is the same failure as defaulting to 1.0, only
    # harder to spot: the numbers look reasonable.
    placeholder: bool = False


class FxTable:
    """
    Rates into one reporting currency, optionally effective-dated.

    The config accepts either shorthand or the dated form, because most tenants have a
    single planning rate and should not have to write a list to say so:

        "GBP": 1.3229
        "INR": [{"effective_from": "2024-04-01", "rate": 0.01199},
                {"effective_from": "2025-04-01", "rate": 0.01156}]
    """

    def __init__(
        self,
        reporting_currency: str = DEFAULT_REPORTING_CURRENCY,
        quotes: Dict[str, List[FxQuote]] = None,
        source_path: Optional[Path] = None,
    ):
        self.reporting_currency = (reporting_currency or DEFAULT_REPORTING_CURRENCY).upper()
        self.quotes: Dict[str, List[FxQuote]] = quotes or {}
        self.source_path = source_path
        # The reporting currency converts to itself. Stating it explicitly means a
        # config that forgets to list it does not blank its own domestic lines.
        self.quotes.setdefault(
            self.reporting_currency,
            [FxQuote(pd.Timestamp.min, 1.0, "identity")],
        )

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        config_dir: Union[str, Path] = None,
        reporting_currency: str = None,
    ) -> "FxTable":
        """
        Read `fx_rates.json` from the config directory.

        A missing file is not an error — it produces an empty table, which converts
        nothing and reports every non-reporting currency as unrated. That is the
        correct behaviour: the planner finds out that rates are needed by reading a
        report that says so, not by getting a plausible wrong total.
        """
        config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "config"
        path = config_dir / "fx_rates.json"
        if not path.exists():
            return cls(reporting_currency or DEFAULT_REPORTING_CURRENCY, {}, None)

        raw = json.loads(path.read_text(encoding="utf-8"))
        currency = reporting_currency or raw.get("reporting_currency") or DEFAULT_REPORTING_CURRENCY
        return cls(currency, cls._parse_rates(raw.get("rates", {}) or {}), path)

    @staticmethod
    def _parse_rates(raw: Dict[str, Any]) -> Dict[str, List[FxQuote]]:
        quotes: Dict[str, List[FxQuote]] = {}
        for code, spec in raw.items():
            # Every config file in this repo documents itself with `_`-prefixed keys.
            # Reading one as a currency turns a comment into a crash three stages later.
            if str(code).startswith("_"):
                continue
            code = str(code).strip().upper()
            if isinstance(spec, (int, float)):
                entries = [FxQuote(pd.Timestamp.min, float(spec), "flat")]
            else:
                entries = []
                for item in spec or []:
                    rate = item.get("rate")
                    if rate is None:
                        continue
                    start = pd.to_datetime(item.get("effective_from"), errors="coerce")
                    entries.append(FxQuote(
                        pd.Timestamp.min if pd.isna(start) else start,
                        float(rate),
                        str(item.get("source", "")),
                        bool(item.get("placeholder", False)),
                    ))
                entries.sort(key=lambda q: q.effective_from)
            if entries:
                quotes[code] = entries
        return quotes

    # ── Lookup ───────────────────────────────────────────────────────────────

    @property
    def currencies(self) -> List[str]:
        return sorted(self.quotes)

    def rate_for(self, currency: str, when: Any = None) -> Optional[float]:
        """The rate in force for `currency` at `when`, or None if the code is unrated."""
        entries = self.quotes.get(str(currency).strip().upper())
        if not entries:
            return None
        stamp = pd.to_datetime(when, errors="coerce") if when is not None else pd.NaT
        if pd.isna(stamp):
            return entries[-1].rate          # no date to place it — use the newest
        applicable = [q for q in entries if q.effective_from <= stamp]
        # A row older than the earliest quote takes that quote rather than nothing:
        # an approximate rate on a 2018 line beats blanking eight years of history.
        return (applicable[-1] if applicable else entries[0]).rate

    def is_placeholder(self, currency: str) -> bool:
        """True when every quote for this code is a stand-in rather than a measurement."""
        entries = self.quotes.get(str(currency).strip().upper())
        return bool(entries) and all(q.placeholder for q in entries)

    def rate_series(self, currency: pd.Series, when: pd.Series = None) -> pd.Series:
        """Vectorised `rate_for` — NaN wherever the code has no quote."""
        codes = currency.astype("string").str.strip().str.upper()
        out = pd.Series(np.nan, index=currency.index, dtype="float64")
        stamps = (
            pd.to_datetime(when, errors="coerce")
            if when is not None
            else pd.Series(pd.NaT, index=currency.index)
        )

        for code, entries in self.quotes.items():
            mask = codes == code
            if not mask.any():
                continue
            if len(entries) == 1:
                out.loc[mask] = entries[0].rate
                continue
            # Compared as datetime64 rather than as integers. pandas 3 stores
            # microseconds where pandas 2 stored nanoseconds, so an int64 comparison
            # is out by a factor of 1000 on one of them — and it fails silently, by
            # putting every row on the earliest quote.
            starts = np.array([q.effective_from.to_datetime64() for q in entries],
                              dtype="datetime64[ns]")
            rates = np.array([q.rate for q in entries])
            rows = stamps.loc[mask]
            # searchsorted picks the last quote starting at or before the row date;
            # clip(0) puts a pre-history row on the earliest quote instead of index -1.
            idx = np.searchsorted(
                starts, rows.to_numpy(dtype="datetime64[ns]"), side="right"
            ) - 1
            idx = np.clip(idx, 0, len(entries) - 1)
            picked = rates[idx]
            # A row with no usable date takes the newest quote, matching `rate_for`.
            picked = np.where(rows.isna().to_numpy(), rates[-1], picked)
            out.loc[mask] = picked

        return out


# ── Conversion ───────────────────────────────────────────────────────────────


@dataclass
class ConversionReport:
    """What the conversion did, and what it could not do."""

    doc_type: str
    reporting_currency: str
    columns: List[str] = dc_field(default_factory=list)
    rows_total: int = 0
    rows_converted: int = 0          # rate != 1, money actually restated
    rows_domestic: int = 0           # already in the reporting currency
    rows_from_row_rate: int = 0      # used the rate the export itself carried
    rows_rate_rejected: int = 0      # export carried a rate that was not one
    rows_unrated: int = 0            # currency present, no rate — money blanked
    unrated: Dict[str, int] = dc_field(default_factory=dict)
    # code -> lines converted at a rate nobody measured
    placeholder_rates: Dict[str, int] = dc_field(default_factory=dict)
    value_before: Dict[str, float] = dc_field(default_factory=dict)
    value_after: Dict[str, float] = dc_field(default_factory=dict)
    assumed_reporting_currency: bool = False
    skipped: bool = False            # nothing to convert in this document

    @property
    def has_gap(self) -> bool:
        return self.rows_unrated > 0

    @property
    def is_multi_currency(self) -> bool:
        return self.rows_converted > 0 or self.rows_unrated > 0

    def summary(self) -> List[str]:
        if self.skipped:
            return []
        if self.assumed_reporting_currency:
            # The one assumption this module makes silently, and the one that burned a
            # real run: a CNY standard-cost column with no currency field beside it was
            # taken for USD, and every figure built on unit cost — annual value, ABC
            # class, EOQ, the whole report — inherited the error with nothing to show
            # for it. Say the fix, not just the assumption.
            return [f"  ⚠ {self.doc_type}: no currency column — assumed already in "
                    f"{self.reporting_currency}. If this document is not in "
                    f"{self.reporting_currency}, declare it in the adapter "
                    f"(`defaults: {{currency: XXX}}`); nothing downstream can tell."]

        lines_out = self._converted_lines()
        if self.placeholder_rates:
            detail = ", ".join(f"{code} ({n:,} lines)"
                               for code, n in sorted(self.placeholder_rates.items()))
            lines_out.append(
                f"      ⚠ {detail} converted at a placeholder rate — a stand-in, not a "
                f"measured or published rate. The magnitude is right and the figure is "
                f"not. Replace it in config/fx_rates.json."
            )
        return lines_out

    def _converted_lines(self) -> List[str]:

        lines = [
            f"  {self.doc_type}: {self.rows_converted:,} of {self.rows_total:,} lines "
            f"restated to {self.reporting_currency} "
            f"({self.rows_domestic:,} already domestic"
            + (f", {self.rows_from_row_rate:,} at the rate in the export" if self.rows_from_row_rate else "")
            + ")"
        ]
        if self.rows_rate_rejected:
            lines.append(
                f"      {self.rows_rate_rejected:,} lines carry an exchange rate that is "
                f"not one (1.0 on a foreign line, or an inverted quote) — the configured "
                f"rate was used instead."
            )
        for col in self.columns:
            before, after = self.value_before.get(col), self.value_after.get(col)
            if before is None or after is None or not before:
                continue
            lines.append(f"      {col}: {before:,.0f} raw → {after:,.0f} {self.reporting_currency}")
        if self.unrated:
            detail = ", ".join(f"{code} ({n:,} lines)" for code, n in sorted(self.unrated.items()))
            lines.append(
                f"      ⚠ no rate for {detail} — money blanked on those lines rather "
                f"than counted as {self.reporting_currency}. Add them to config/fx_rates.json."
            )
        return lines


def convert_money(
    df: pd.DataFrame,
    doc_type: str,
    table: FxTable,
    money_columns: Sequence[str] = None,
    date_columns: Sequence[str] = None,
) -> tuple:
    """
    Restate a document's money columns into the reporting currency.

    Returns `(frame, report)`. The frame gains `currency_original` (the code the line
    was raised in) and `fx_rate_applied`, so any converted figure can be traced back to
    the rate that produced it and un-converted if a rate is later corrected.
    """
    money_columns = [
        c for c in (money_columns if money_columns is not None
                    else MONEY_COLUMNS.get(doc_type, ()))
        if c in df.columns
    ]
    report = ConversionReport(doc_type=doc_type, reporting_currency=table.reporting_currency,
                              columns=list(money_columns), rows_total=len(df))
    if not money_columns or df.empty:
        report.skipped = True
        return df, report

    # No currency column: a single-entity export in its own currency. Recorded as an
    # assumption rather than a fact, because that is exactly what it is.
    if CURRENCY_COLUMN not in df.columns or df[CURRENCY_COLUMN].isna().all():
        report.assumed_reporting_currency = True
        report.rows_domestic = len(df)
        df = df.copy()
        df["currency_original"] = table.reporting_currency
        df["fx_rate_applied"] = 1.0
        return df, report

    df = df.copy()
    codes = df[CURRENCY_COLUMN].astype("string").str.strip().str.upper()
    # A blank code on a line whose neighbours all agree is the export omitting a
    # repeated value, not a different currency.
    codes = codes.fillna(table.reporting_currency)

    rate = table.rate_series(codes, _rate_date(df, doc_type, date_columns))

    # The export's own rate is what the transaction settled at, so it outranks the
    # table — but only once it has been shown to be a rate at all. Two ways it is not:
    #
    #   exactly 1.0 on a foreign line   The column is a placeholder the export fills in
    #                                   rather than a conversion. The real backlog
    #                                   extract carries `Exchange Rate` = 1 on all
    #                                   3,169 INR lines, and taking it at face value
    #                                   left every one of them in rupees while the run
    #                                   log reported them as restated.
    #
    #   an order of magnitude off       ERP rate columns are frequently quoted the
    #                                   other way up — 94.7 rupees per dollar where the
    #                                   conversion needs 0.01056. Inverted, it inflates
    #                                   the line by ~9,000x instead of shrinking it.
    #
    # Both are rejected in favour of the table, and counted so the rejection is visible.
    if ROW_RATE_COLUMN in df.columns:
        booked = pd.to_numeric(df[ROW_RATE_COLUMN], errors="coerce")
        foreign = codes != table.reporting_currency
        plausible = booked.notna() & (booked > 0) & ~np.isclose(booked.fillna(0.0), 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            versus_table = booked / rate
        # Only judged where the table has an opinion to judge against.
        implausible = rate.notna() & ((versus_table < 0.1) | (versus_table > 10))
        plausible &= ~implausible

        usable = foreign & plausible
        report.rows_from_row_rate = int(usable.sum())
        report.rows_rate_rejected = int((foreign & booked.notna() & ~plausible).sum())
        rate = rate.mask(usable, booked)

    domestic = codes == table.reporting_currency
    rate = rate.mask(domestic, 1.0)

    unrated = rate.isna()
    report.rows_unrated = int(unrated.sum())
    report.rows_domestic = int(domestic.sum())
    report.rows_converted = int((~unrated & ~domestic).sum())
    if report.rows_unrated:
        report.unrated = codes[unrated].value_counts().to_dict()

    converted_codes = codes[~unrated & ~domestic]
    if len(converted_codes):
        report.placeholder_rates = {
            code: int(n) for code, n in converted_codes.value_counts().items()
            if table.is_placeholder(code)
        }

    for col in money_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        report.value_before[col] = float(values.sum(skipna=True))
        # Unrated lines become NaN, not zero. Zero is a claim that the line was free;
        # NaN is the truth, that its value is not known in the reporting currency.
        df[col] = values * rate
        report.value_after[col] = float(pd.to_numeric(df[col], errors="coerce").sum(skipna=True))

    df["currency_original"] = codes
    df["fx_rate_applied"] = rate
    return df, report


def _rate_date(df: pd.DataFrame, doc_type: str, date_columns: Sequence[str] = None) -> Optional[pd.Series]:
    """First available dating column for the document, or None to use the latest rate."""
    candidates = date_columns if date_columns is not None else RATE_DATE_COLUMNS.get(doc_type, ())
    for col in candidates:
        if col in df.columns:
            stamps = pd.to_datetime(df[col], errors="coerce")
            if stamps.notna().any():
                return stamps
    return None


@dataclass
class FxSummary:
    """Every document's conversion, for the run log and the report."""

    reporting_currency: str
    reports: Dict[str, ConversionReport] = dc_field(default_factory=dict)
    rates_configured: bool = True

    @property
    def multi_currency(self) -> bool:
        return any(r.is_multi_currency for r in self.reports.values())

    @property
    def gaps(self) -> Dict[str, int]:
        """Currency codes with no rate, and how many lines they left unvalued."""
        out: Dict[str, int] = {}
        for report in self.reports.values():
            for code, n in report.unrated.items():
                out[code] = out.get(code, 0) + n
        return out

    def summary(self) -> str:
        lines = [f"  Currency — reporting in {self.reporting_currency}"]
        if not self.rates_configured:
            lines.append("      No config/fx_rates.json found. Any non-"
                         f"{self.reporting_currency} line will be reported unvalued.")
        for report in self.reports.values():
            lines.extend(report.summary())
        return "\n".join(lines)
