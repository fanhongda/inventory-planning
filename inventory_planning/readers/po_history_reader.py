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

# The vendor-confirmation category, where the export carries one. Used only to name the
# codes in the finding — the consolidation keys on the goods receipt, not on this.
CONFIRMATION_CATEGORY = "confirmation_category"


def consolidate_to_receipt_grain(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drop the rows of a PO line that carry no goods receipt, where the line has one.

    The contract's grain is `po_receipt_line`, and some ERP exports are finer than
    that in a way nothing downstream can see. SAP is the common case: with
    confirmation control active, EKES holds one row per vendor confirmation, so one
    PO line is exported twice — an `AB` row (Auftragsbestätigung, the order
    acknowledgement, no receipt against it) and an `LA` row (Lieferavis, the shipping
    notification, which is what the goods receipt is posted to). Other systems repeat
    a line per schedule line or per status event with the same effect.

    The lead time survives this untouched: the acknowledgement row has no receipt
    date, so `lead_time_days` is null on it and the weighted average already drops it.
    The quantities do not. `po_qty` and `delivered_qty` are line-level values repeated
    on every row of the line, so summing them counts each order twice — and every
    figure built on the ordering *pattern* moves: how much was bought against demand,
    the over-ordered finding, lot-size conformance, the replenishment cadence. None of
    it errors, the numbers stay internally consistent, and the grain test does not
    catch it because `receive_date` is part of the natural key and the two rows differ
    on exactly that column.

    What is dropped is the row, not the order. The retained row carries the same
    line-level `po_qty`, so nothing about what was ordered is lost. The acknowledgement
    *date* is lost, and that is deliberate: nothing downstream reads it today, and a
    column kept for a consumer that does not exist is how a grain gets confused again.

    Deliberately conservative. Three cases are left alone:

      no receipt column, or no receipt anywhere   An extract with no goods-receipt date
          is still a valid record of what was ordered, how often and how much — the
          contract says so, and `capability_requires` already withholds the lead-time
          signal from it. Dropping its rows would empty the file.

      every row has a receipt   Nothing is mixed. Several receipts against one line are
          *partial deliveries*, legitimately separate rows at this grain, and are kept.

      no line identifier   Without `po_number` there is no line to reason about, and a
          guess here would delete real receipts.
    """
    if df is None or len(df) == 0 or "receive_date" not in df.columns:
        return df
    key = [c for c in ("po_number", "po_line_number", "sku") if c in df.columns]
    if "po_number" not in key:
        return df

    has_receipt = df["receive_date"].notna()
    if not has_receipt.any() or has_receipt.all():
        return df

    # `dropna=False` so a line whose number or position is blank is still grouped
    # rather than silently dissolving into one row per null.
    line_has_receipt = has_receipt.groupby(
        [df[c] for c in key], dropna=False
    ).transform("any")
    drop = line_has_receipt & ~has_receipt
    if not drop.any():
        return df

    if verbose:
        lines = len(df.loc[drop, key].drop_duplicates())
        # Name the codes where the export carried them, rather than asserting AB/LA at
        # a planner whose system may spell it differently or not at all.
        named = ""
        if CONFIRMATION_CATEGORY in df.columns:
            codes = sorted({str(c).strip() for c in df.loc[drop, CONFIRMATION_CATEGORY]
                            if str(c).strip() and str(c).strip().lower() != "nan"})
            if codes:
                named = f", coded {', '.join(codes)}"
        repeated = ""
        if "po_qty" in df.columns:
            qty = pd.to_numeric(df.loc[drop, "po_qty"], errors="coerce").sum()
            repeated = (f" They repeat the line's quantity — {qty:,.0f} units that "
                        f"would have been counted twice in the ordering-pattern "
                        f"figures.")
        print(f"  Dropped {int(drop.sum()):,} PO row(s) across {lines:,} line(s) that "
              f"appear both with and without a goods receipt{named} — a "
              f"confirmation-level export, where the receipt is posted against the "
              f"shipping notification and not the acknowledgement.{repeated}")
    return df[~drop].copy()


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
        # Before the lead-time arithmetic, so `lt_days_precalc`'s "is more than half of
        # this column usable?" test is asked of receipt rows rather than of a frame
        # padded with acknowledgement rows that carry no lead time by construction.
        df = consolidate_to_receipt_grain(df)

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
