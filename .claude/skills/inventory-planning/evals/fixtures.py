"""
Build eval fixtures from the synthetic sample data, with defects injected on purpose.

Every eval used to point at `/Users/fanhongda/Downloads/*.xlsx` — absolute paths to
un-anonymised customer data. That made the suite unrunnable anywhere but one laptop and
unshareable anywhere at all. Everything here is generated from `sample_data/`, which is
synthetic (`SKU-002`, `CUST-018`), so the evals travel with the repo.

The defect variants matter more than the clean one. A pipeline is easy to test on data
that is already correct; what has to be verified is that it *notices* when the data is
not, because every real corruption found so far produced a complete run and a confident
wrong number rather than an error.

    python -m skill.evals.fixtures <target_dir>

Each variant writes a full set of five documents, so an eval can be handed a directory
and told to plan against it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "sample_data"

VARIANTS = ("clean", "multi_currency", "unrated_currency", "mangled_dates")


def _load() -> dict:
    return {p.stem: pd.read_csv(p) for p in sorted(SAMPLE.glob("*.csv"))}


def _write(frames: dict, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(target / f"{name}.csv", index=False)
    return target


def _money_columns(df: pd.DataFrame) -> list:
    """
    Money columns, found by name rather than listed.

    Listing them meant the fixture silently did nothing when the sample data used
    `Net Value` and the list said `Net Amount` — the currency codes were written, the
    amounts were not restated, and the eval passed while exercising nothing.
    """
    words = ("value", "amount", "price", "cost", "revenue")
    return [
        c for c in df.columns
        if any(w in str(c).lower() for w in words)
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]


def multi_currency(frames: dict) -> dict:
    """
    Purchase history in three currencies, with rates configured for all of them.

    Summed raw this is meaningless; converted it is correct. Both totals are valid
    numbers, which is the whole difficulty — nothing about the wrong one looks wrong.
    """
    frames = {k: v.copy() for k, v in frames.items()}
    po = frames["po_history"]
    codes = ["USD", "GBP", "INR"]
    po["Currency"] = [codes[i % 3] for i in range(len(po))]
    # Restate the line values *into* each currency, so the underlying economics are
    # unchanged and only the units differ. A fixture where INR lines are also cheap
    # would not exercise the failure.
    rate = po["Currency"].map({"USD": 1.0, "GBP": 1 / 1.25, "INR": 1 / 0.012})
    money = _money_columns(po)
    if not money:
        raise SystemExit("po_history has no money column to restate — fixture would be inert")
    for col in money:
        po[col] = (pd.to_numeric(po[col], errors="coerce") * rate).round(2)
    return frames


def unrated_currency(frames: dict) -> dict:
    """One currency with no entry in fx_rates.json — its money must be excluded and named."""
    frames = multi_currency(frames)
    po = frames["po_history"]
    po.loc[po.index[::7], "Currency"] = "JPY"
    return frames


def mangled_dates(frames: dict) -> dict:
    """
    A date column half-converted by a spreadsheet under a mismatched locale.

    Values whose day is 12 or less are ambiguous, so the spreadsheet reads them as
    day-first, swaps month and day, and stores a real date. The rest it cannot parse
    and leaves as text. Whichever parser the pipeline picks, the other half becomes
    null — silently, and only for the dates that happened to be ambiguous.
    """
    frames = {k: v.copy() for k, v in frames.items()}
    sales = frames["sales_history"]
    # Ship date by preference: `demand_date` derives from it, so that is where a
    # half-null date column actually costs demand. Mangling `Order Date` instead
    # produces a fixture that looks broken and changes nothing downstream.
    dates = [c for c in sales.columns if "date" in str(c).lower()]
    col = next((c for c in dates if "ship" in str(c).lower()), dates[0] if dates else None)
    if col is None:
        return frames
    stamps = pd.to_datetime(sales[col], errors="coerce")
    out = []
    for stamp in stamps:
        if pd.isna(stamp):
            out.append("")
        elif stamp.day <= 12:
            swapped = pd.Timestamp(year=stamp.year, month=stamp.day, day=stamp.month)
            out.append(swapped.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            out.append(f"{stamp.month}/{stamp.day}/{stamp.year}")
    sales[col] = out
    return frames


BUILDERS = {
    "clean": lambda f: f,
    "multi_currency": multi_currency,
    "unrated_currency": unrated_currency,
    "mangled_dates": mangled_dates,
}


def build(target_root: Path, variants=VARIANTS) -> dict:
    """Write each variant into `target_root/<variant>/` and return the paths."""
    base = _load()
    if not base:
        raise SystemExit(f"no sample data found in {SAMPLE}")
    out = {}
    for name in variants:
        frames = BUILDERS[name]({k: v.copy() for k, v in base.items()})
        out[name] = _write(frames, target_root / name)
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "output" / "eval-fixtures"
    if root.exists():
        shutil.rmtree(root)
    for name, path in build(root).items():
        print(f"{name:<18} {path}")
