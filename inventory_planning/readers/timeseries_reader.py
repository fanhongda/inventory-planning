"""
Pre-compiled time series reader.
Handles wide-format demand pivot tables exported from S&OP/BI tools:
  - Rows = SKUs
  - Columns = monthly periods (with inconsistent naming conventions)
  - Values = demand qty per period

Period headers are read by `ingest.profiler.parse_period_header`, which covers the
messy spellings a worksheet carries — "2019 Jan", "Dec 2022", "Nov-2023", "2026年1月",
Excel serial date integers — in the one place every entry point reads them.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from ..ingest.profiler import parse_period_header as _parse_period_header
from ..readers.base_reader import load_file

# Metadata columns that are NOT period data
METADATA_COLS = {
    "item", "sku", "description", "desc", "std cst", "std cost", "standard cost",
    "cost", "sig", "sbu", "s&op classification", "s&op class", "classification",
    "group", "item group", "uom", "unit", "category", "brand",
}

# Period headers are parsed by `ingest.profiler.parse_period_header`, not here.
#
# There used to be a second implementation in this module with its own month table and
# its own Excel-serial window. It was narrower in exactly the way that matters: a
# worksheet headed `2026年1月` parsed in neither, and a header the profiler recognises
# but this did not meant the flag path and the contract path disagreed about how many
# months a file carried. One parser, so a header added to it is added once.


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
