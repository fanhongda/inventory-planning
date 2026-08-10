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

    def compute_supplier_lt(self, df: pd.DataFrame, rolling_months: int = 6) -> pd.DataFrame:
        """
        Weighted moving average lead time per SKU × supplier.
        Weight = 1/rank (most recent = highest weight).
        """
        cutoff = df["po_date"].max() - pd.DateOffset(months=rolling_months)
        recent = df[df["po_date"] >= cutoff].copy()
        if len(recent) == 0:
            print(f"  Warning: no PO history in last {rolling_months} months; using all history")
            recent = df.copy()

        supplier_incoterms, default_inco = self._load_supplier_incoterms()

        results = []
        for (sku, supplier), grp in recent.groupby(["sku", "supplier"]):
            grp = grp.sort_values("po_date")
            lts = grp["lead_time_days"].dropna().values
            if len(lts) == 0:
                continue
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
                "location_id": grp["location_id"].iloc[0] if "location_id" in grp.columns else "DC-01",
                "wma_lead_time_days": round(wma_lt, 1),
                "lt_std_days": round(lt_std, 1),
                "sample_count": n,
                "min_lt_days": int(lts.min()),
                "max_lt_days": int(lts.max()),
                "incoterm": incoterm,
                "last_po_date": grp["po_date"].max(),
            })
        return pd.DataFrame(results)
