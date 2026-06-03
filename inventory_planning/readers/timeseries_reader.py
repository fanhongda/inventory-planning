"""
Pre-compiled time series reader.
Handles wide-format demand pivot tables exported from S&OP/BI tools:
  - Rows = SKUs
  - Columns = monthly periods (with inconsistent naming conventions)
  - Values = demand qty per period

Supports messy period headers like:
  "2019 Jan", "2020 June", "Dec 2022", "Jan 2023",
  "Nov-2023", "March-2024", "April-2025", and Excel serial date integers.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..readers.base_reader import load_file

# Metadata columns that are NOT period data
METADATA_COLS = {
    "item", "sku", "description", "desc", "std cst", "std cost", "standard cost",
    "cost", "sig", "sbu", "s&op classification", "s&op class", "classification",
    "group", "item group", "uom", "unit", "category", "brand",
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    # Full names
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Excel epoch: Jan 1, 1900 (with Lotus 1-2-3 leap year bug)
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _parse_period_header(col: str) -> Optional[pd.Period]:
    """
    Try to parse a column header as a monthly period.
    Returns pd.Period('YYYY-MM', 'M') or None if not a period column.
    """
    s = str(col).strip()

    # Excel serial date (integer-like)
    try:
        serial = int(float(s))
        if 30000 < serial < 60000:  # plausible Excel date range ~1982–2064
            dt = EXCEL_EPOCH + pd.Timedelta(days=serial)
            return pd.Period(dt, freq="M")
    except (ValueError, TypeError):
        pass

    # Normalize: remove dots, extra spaces, convert to lowercase
    s_norm = re.sub(r"[\.\s]+", " ", s.lower()).strip()

    # Patterns to try (year + month or month + year)
    patterns = [
        # "2023 Jan", "2020 June", "2021 July"
        r"^(\d{4})\s+([a-z]+)$",
        # "Jan 2023", "Dec 2022", "March 2024", "April 2025"
        r"^([a-z]+)\s+(\d{4})$",
        # "Jan-2023", "Nov-2023", "March-2024"
        r"^([a-z]+)-(\d{4})$",
        # "2023-01", "2023-1"
        r"^(\d{4})-(\d{1,2})$",
    ]

    for pat in patterns:
        m = re.match(pat, s_norm)
        if m:
            a, b = m.group(1), m.group(2)
            # Determine which is year and which is month
            if a.isdigit() and len(a) == 4:
                year, month_str = int(a), b
            elif b.isdigit() and len(b) == 4:
                year, month_str = int(b), a
            elif a.isdigit() and b.isdigit():
                # "2023-01" pattern
                year, month_num = int(a), int(b)
                if 1 <= month_num <= 12:
                    return pd.Period(f"{year}-{month_num:02d}", freq="M")
                continue
            else:
                continue

            if month_str.isdigit():
                month_num = int(month_str)
            else:
                month_num = MONTH_MAP.get(month_str[:3], None)
                if month_num is None:
                    continue

            if 2000 <= year <= 2040 and 1 <= month_num <= 12:
                return pd.Period(f"{year}-{month_num:02d}", freq="M")

    return None


def _is_metadata_col(col: str) -> bool:
    return str(col).strip().lower() in METADATA_COLS


class TimeSeriesReader:
    """
    Reads a wide-format pre-compiled time series file.
    Detects metadata vs period columns automatically.
    Outputs the same pivot format as SalesHistoryReader.to_time_series().
    """

    def __init__(self, config_dir: Union[str, Path] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parents[2] / "config"

    def read(self, path: Union[str, Path],
             rolling_months: int = 36,
             interactive: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and parse a wide-format time series file.

        Args:
            path: path to xlsx/csv file
            rolling_months: how many recent months to keep (default 36)
            interactive: show mapping preview and prompt user

        Returns:
            (pivot_df, metadata_df, quality_report)
            pivot_df: period index (pd.Period) × SKU columns, values = qty
            metadata_df: SKU → Description, std_cost, classification, etc.
        """
        path = Path(path)
        raw = load_file(path)

        # --- Detect SKU column ---
        sku_col = self._find_sku_col(raw)
        if sku_col is None:
            raise ValueError("Cannot find SKU/Item column in time series file")

        # --- Classify all columns ---
        period_cols: Dict[str, pd.Period] = {}
        meta_cols: List[str] = []

        for col in raw.columns:
            if col == sku_col:
                continue
            p = _parse_period_header(col)
            if p is not None:
                period_cols[col] = p
            else:
                meta_cols.append(col)

        # Sort period columns chronologically
        sorted_period_cols = sorted(period_cols.keys(), key=lambda c: period_cols[c])

        if interactive:
            self._print_preview(sku_col, sorted_period_cols, period_cols, meta_cols, rolling_months)
            ans = input("Accept? [Y/n]: ").strip().lower()
            if ans in ("n", "no"):
                raise ValueError("User rejected time series mapping")

        # --- Build pivot ---
        df = raw.copy()
        df[sku_col] = df[sku_col].astype(str).str.strip().str.upper()
        df = df[df[sku_col].notna() & (df[sku_col] != "") & (df[sku_col] != "NAN")]

        pivot_data = {}
        parse_errors = 0
        for col in sorted_period_cols:
            period = period_cols[col]
            vals = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
            vals = vals.clip(lower=0)  # no negative demand
            pivot_data[period] = vals.values

        pivot = pd.DataFrame(pivot_data, index=df[sku_col].values).T
        pivot.index = pd.PeriodIndex(pivot.index, freq="M")
        pivot.index.name = "period"
        pivot.columns.name = "sku"

        # Apply rolling window — keep most recent N months
        if rolling_months and len(pivot) > rolling_months:
            pivot = pivot.tail(rolling_months)

        # Remove SKUs with zero demand across all kept periods
        active_skus = pivot.columns[pivot.sum(axis=0) > 0]
        n_inactive = len(pivot.columns) - len(active_skus)
        pivot = pivot[active_skus]

        # --- Metadata ---
        meta_df = df[[sku_col] + [c for c in meta_cols if c in df.columns]].copy()
        meta_df = meta_df.rename(columns={sku_col: "sku"})
        meta_df = meta_df.drop_duplicates("sku").set_index("sku")

        # Standardize known metadata column names
        col_remap = {}
        for c in meta_df.columns:
            cn = c.strip().lower()
            if cn in ("description", "desc"):
                col_remap[c] = "description"
            elif cn in ("std cst", "std cost", "standard cost", "cost"):
                col_remap[c] = "std_cost"
            elif cn == "sig":
                col_remap[c] = "sig"
            elif "classification" in cn or "s&op" in cn:
                col_remap[c] = "sopc_classification"
        meta_df = meta_df.rename(columns=col_remap)

        quality = {
            "file": path.name,
            "total_skus_in_file": len(df),
            "active_skus_with_demand": len(active_skus),
            "inactive_skus_removed": n_inactive,
            "period_columns_found": len(period_cols),
            "period_range": f"{pivot.index[0]} → {pivot.index[-1]}",
            "periods_kept": len(pivot),
            "rolling_months_applied": rolling_months,
            "metadata_columns": list(meta_df.columns),
        }
        self._print_quality(quality)
        return pivot, meta_df, quality

    def _find_sku_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            cn = str(col).strip().lower()
            if cn in ("item", "sku", "item code", "item_code", "material", "part_no"):
                return col
        return None

    def _print_preview(self, sku_col, sorted_period_cols, period_cols, meta_cols, rolling_months):
        first = period_cols[sorted_period_cols[0]] if sorted_period_cols else "?"
        last  = period_cols[sorted_period_cols[-1]] if sorted_period_cols else "?"
        print(f"\n{'='*60}")
        print(f"  Time Series File Mapping Preview")
        print(f"{'='*60}")
        print(f"  SKU column     : {sku_col}")
        print(f"  Period columns : {len(sorted_period_cols)} ({first} → {last})")
        print(f"  Periods kept   : last {rolling_months} months")
        print(f"  Metadata cols  : {meta_cols}")
        print(f"  Sample periods : {[str(period_cols[c]) for c in sorted_period_cols[:3]]} ... "
              f"{[str(period_cols[c]) for c in sorted_period_cols[-3:]]}")
        print(f"{'='*60}")

    def _print_quality(self, q: Dict):
        print(f"\n  Time Series Quality Report")
        print(f"  File            : {q['file']}")
        print(f"  SKUs in file    : {q['total_skus_in_file']}")
        print(f"  Active SKUs     : {q['active_skus_with_demand']} (demand > 0 in window)")
        print(f"  Inactive removed: {q['inactive_skus_removed']}")
        print(f"  Period range    : {q['period_range']} ({q['periods_kept']} months kept)")
