"""
PO history reader.
Calculates actual replenishment lead times per SKU × supplier.
Supports both computed LT (receive_date - po_date) and pre-computed LT column.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from .base_reader import BaseReader

# Used when `stocking_policy.json` is absent or does not state a window. Six months is
# what this function defaulted to for as long as it existed, so nothing moves for a
# site that never set it.
DEFAULT_LT_ROLLING_MONTHS = 6


class POHistoryReader(BaseReader):
    doc_type = "po_history"

    def _load_supplier_incoterms(self) -> dict:
        cfg = self.config_dir / "supplier_incoterm.json"
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            return data.get("suppliers", {}), data.get("default", "FOB")
        return {}, "FOB"

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["sku", "supplier", "po_date"])

        # Use pre-computed LT column if available, else compute from dates
        if "lt_days_precalc" in df.columns:
            df["lt_days_precalc"] = pd.to_numeric(df["lt_days_precalc"], errors="coerce")
            valid_precalc = df["lt_days_precalc"].notna() & (df["lt_days_precalc"] > 0)
            if valid_precalc.sum() > len(df) * 0.5:
                df["lead_time_days"] = df["lt_days_precalc"]
                print(f"  Using pre-computed LT column ({valid_precalc.sum()} valid rows)")
            else:
                df = self._compute_lt_from_dates(df)
        elif "receive_date" in df.columns:
            df = self._compute_lt_from_dates(df)
        else:
            print("  Warning: no receive_date or LT column found — cannot compute lead time")
            df["lead_time_days"] = np.nan
            return df

        # Remove nonsensical LTs
        bad = (df["lead_time_days"] <= 0) | (df["lead_time_days"] > 730)
        if bad.sum():
            print(f"  Note: {bad.sum()} PO rows excluded (lead_time ≤ 0 or > 730 days)")
        df = df[~bad].copy()
        return df

    def _compute_lt_from_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        if "receive_date" not in df.columns:
            df["lead_time_days"] = np.nan
            return df
        df["lead_time_days"] = (df["receive_date"] - df["po_date"]).dt.days
        print(f"  Computed LT from receive_date - po_date")
        return df

    def _lt_rolling_months(self) -> int:
        """
        The measurement window, in months, from `stocking_policy.json`.

        Read rather than hardcoded because the file has carried `lt_rolling_months`
        since before this function did, and nothing read it: a planner who set it to 24
        got 6, and no line of output said the setting was inert.
        """
        cfg = self.config_dir / "stocking_policy.json"
        if not cfg.exists():
            return DEFAULT_LT_ROLLING_MONTHS
        try:
            raw = json.loads(cfg.read_text(encoding="utf-8")).get("lt_rolling_months")
        except (ValueError, OSError):
            return DEFAULT_LT_ROLLING_MONTHS
        months = pd.to_numeric(raw, errors="coerce")
        if pd.isna(months) or months <= 0:
            return DEFAULT_LT_ROLLING_MONTHS
        return int(months)

    def compute_supplier_lt(self, df: pd.DataFrame, rolling_months: int = None) -> pd.DataFrame:
        """
        Weighted moving average lead time per SKU × supplier.
        Weight = 1/rank (most recent = highest weight).

        The window is measured back from the newest PO date in the extract rather than
        from today, so an extract pulled last month is not read as having no recent
        history. It is a *global* date, though, and that is the part that needed fixing:
        it used to be applied as a filter before grouping, so an item bought quarterly
        against a catalogue bought monthly had every one of its receipts cut away by
        other items' order dates and disappeared from this table entirely — silently,
        and precisely for the long-lead-time items where a measured lead time is worth
        the most.

        So the window selects, and the pair's own history is the fallback. A SKU ×
        supplier with no receipt inside it keeps the lead time its own receipts measure,
        labelled `measured_outside_window`, instead of dropping out and taking a stated
        value from master data that carries no sigma at all.
        """
        if rolling_months is None:
            rolling_months = self._lt_rolling_months()
        newest = df["po_date"].max()
        cutoff = newest - pd.DateOffset(months=rolling_months)
        # Carried as a column rather than as a mask indexed against `df`, which is not
        # guaranteed to have a unique index by the time it reaches here.
        work = df.copy()
        work["_in_window"] = work["po_date"] >= cutoff

        supplier_incoterms, default_inco = self._load_supplier_incoterms()

        results, fell_back = [], []
        for (sku, supplier), pair in work.groupby(["sku", "supplier"]):
            grp = pair[pair["_in_window"]]
            in_window = grp["lead_time_days"].notna().any()
            if not in_window:
                # Nothing received inside the window. Measure what this pair does have
                # rather than losing the item — and label it, because a lead time from
                # three-year-old receipts is evidence of a different grade.
                grp = pair
            grp = grp.sort_values("po_date")
            lts = grp["lead_time_days"].dropna().values
            if len(lts) == 0:
                continue
            if not in_window:
                fell_back.append(grp["po_date"].max())
            n = len(lts)
            weights = np.arange(1, n + 1, dtype=float)
            weights /= weights.sum()
            wma_lt = np.dot(weights, lts)
            lt_std = lts.std() if n > 1 else 0.0
            # Incoterm: from data first, then config lookup, then default
            if "incoterm" in grp.columns and grp["incoterm"].notna().any():
                incoterm = grp["incoterm"].mode()[0]
            else:
                # Match supplier ID prefix (e.g. "SIM001 TYCO..." matches "SIM001")
                sup_str = str(supplier).strip()
                incoterm = next(
                    (v for k, v in supplier_incoterms.items() if sup_str.upper().startswith(k.upper())),
                    default_inco
                )
            results.append({
                "sku": sku,
                "supplier": supplier,
                "location_id": (grp["location_id"].iloc[0] if "location_id" in grp.columns
                                else self.location_id),
                "wma_lead_time_days": round(wma_lt, 1),
                "lt_std_days": round(lt_std, 1),
                "sample_count": n,
                "min_lt_days": int(lts.min()),
                "max_lt_days": int(lts.max()),
                "incoterm": incoterm,
                "last_po_date": grp["po_date"].max(),
                # Which grade of measurement this is. `_fill_lead_time_from_masters`
                # adds `item_master` / `planning_master` for the items with no receipt
                # history at all, so the column reads as one scale end to end.
                "lt_source": "measured" if in_window else "measured_outside_window",
                "lt_window_months": int(rolling_months),
            })

        out = pd.DataFrame(results)
        if fell_back:
            print(f"  Lead-time window: {rolling_months} months back from "
                  f"{newest:%Y-%m-%d} (lt_rolling_months)")
            print(f"  {len(fell_back)} SKU×supplier pairs had no receipt inside it — "
                  f"measured from their own older history instead (oldest last receipt "
                  f"{min(fell_back):%Y-%m-%d}), labelled measured_outside_window.")
        return out
