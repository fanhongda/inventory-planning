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

# The supplier's own date, where the extract has no goods receipt. Never a receipt —
# `receive_date` refuses the bare "delivery date" spellings on purpose.
CONFIRMED_DELIVERY_DATE = "confirmed_delivery_date"

# The planner's table: transport mode and transit days per supplier. A CSV because it
# is filled in for every supplier and a spreadsheet is the quickest way to do that.
TRANSIT_FILE = "supplier_transit.csv"
# The row that answers for every supplier not named. Same convention as the "*" in
# `location_status.json`, and reported the same way — as a blanket rather than an
# inventory of suppliers.
TRANSIT_WILDCARD = "*"
# Written beside the run's other outputs when suppliers are missing from the table, so
# the planner fills in a list rather than compiling one.
TRANSIT_TEMPLATE = "supplier_transit_template.csv"

# Which clock produced `lead_time_days`. Orthogonal to `lt_source`, which says whether
# the number was measured at all and from which window.
BASIS_RECEIPT = "receipt"
# Named "promised" because that is what the date is. `confirmed_delivery_date` is the
# date the supplier *committed* to, and a supplier who is chronically late looks
# punctual in it — so a lead time built on it is a commitment, never a measurement, and
# `lt_source` puts it in the `supplier_confirmed` authority tier rather than `measured`.
BASIS_PROMISED_PLUS_TRANSIT = "promised_dispatch_plus_transit"
BASIS_PRECALC = "precalc_column"

# Lead-time sources, by what stands behind the number. The window suffix says the
# receipts or confirmations were older than the measurement window, not that the grade
# of evidence is different.
SOURCE_MEASURED = "measured"
SOURCE_MEASURED_OLD = "measured_outside_window"
SOURCE_CONFIRMED = "supplier_confirmed"
SOURCE_CONFIRMED_OLD = "supplier_confirmed_outside_window"

# What `lt_std_days` actually measures, where that is not the obvious thing.
SIGMA_PROMISE_FLOOR = "promise_spread_floor"


def _lt_source(grp: pd.DataFrame, in_window: bool) -> str:
    """
    Which grade of evidence stands behind this pair's lead time.

    A commitment and a receipt are not the same claim, and `crosscheck` ranks them
    apart: `measured` outranks everything, `supplier_confirmed` outranks the masters
    but never a receipt. Labelling a commitment `measured` would let an optimistic
    promise beat the planned delivery time a person maintains — which is the one place
    the promise is most likely to be wrong and the master most likely to be right.
    """
    promised = ("lt_basis" in grp.columns
                and (grp["lt_basis"] == BASIS_PROMISED_PLUS_TRANSIT).any())
    if promised:
        return SOURCE_CONFIRMED if in_window else SOURCE_CONFIRMED_OLD
    return SOURCE_MEASURED if in_window else SOURCE_MEASURED_OLD


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

    def _load_transit_times(self) -> tuple:
        """
        Transport mode and transit days per supplier, from `config/supplier_transit.csv`.

        A CSV rather than JSON because this is a table a planner fills in for every
        supplier they buy from, and the quickest way to fill a table is a spreadsheet.
        The same reason `sample_data/templates/README.md` gives for the substitution
        list: a person who will not open a text editor can still supply this.

        Returns `(table, default)` where `table` maps an upper-cased supplier prefix to
        `(mode, days)` and `default` is the `*` row, or `(None, None)` where there is
        none. Prefix matching with longest-wins, the rule `supplier_incoterm.json`
        already uses, because export columns routinely carry the code and the name in
        one field.
        """
        cfg = self.config_dir / TRANSIT_FILE
        if not cfg.exists():
            return {}, (None, None)
        try:
            raw = pd.read_csv(cfg, comment="#", dtype=str)
        except (ValueError, OSError, pd.errors.ParserError):
            return {}, (None, None)
        if "supplier" not in raw.columns or "transit_days" not in raw.columns:
            return {}, (None, None)

        table, default = {}, (None, None)
        for row in raw.itertuples(index=False):
            key = str(getattr(row, "supplier", "") or "").strip().upper()
            days = pd.to_numeric(getattr(row, "transit_days", None), errors="coerce")
            if not key or pd.isna(days) or days < 0:
                continue
            mode = str(getattr(row, "transport_mode", "") or "").strip().upper() or None
            if key == TRANSIT_WILDCARD:
                default = (mode, float(days))
            else:
                table[key] = (mode, float(days))
        return table, default

    @staticmethod
    def _transit_for(supplier, table: dict, default):
        """
        The `(mode, days)` for one supplier: longest matching prefix, else the default.

        Longest wins so a specific vendor code can override a group — `SIM001` beats
        `SIM`, which is how one plant of a supplier gets its own transit time without
        restating every other.
        """
        text = str(supplier or "").strip().upper()
        if not text:
            return default
        matches = [(len(k), v) for k, v in table.items() if text.startswith(k)]
        return max(matches, key=lambda m: m[0])[1] if matches else default

    def _lead_time_from_dispatch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measure the clock as far as the supplier's dispatch, and state the rest.

        For an extract that carries a confirmed-delivery or dispatch date and no goods
        receipt. `receive_date` deliberately refuses the bare "delivery date" spellings,
        so without this such a file yields no measured lead time at all and every item
        falls back to the planned delivery time in the item master.

        What this produces is a *commitment*, not a measurement, and every label says
        so: `lt_basis` is `promised_dispatch_plus_transit` and `lt_source` puts the
        pair in the `supplier_confirmed` authority tier, below a receipt and above the
        masters. `transit_days` carries the stated leg, which
        `policy.should_be._transit_days` then uses in place of its
        `transit_share_of_lt` convention, so the pipeline-stock figure stops guessing.

        **The variability is the part to be careful about.** The spread of
        `confirmed_delivery_date - po_date` across orders is how much the supplier's
        *promise* varies, not how much their performance does: a supplier who always
        promises 45 days and is always ten days late shows a spread of zero. So the
        number is kept and reported as a floor — the true lead-time sigma is at least
        this and probably more — rather than either zeroed, which would shrink safety
        stock on the strength of a promise, or passed off as measured variability.
        `lt_sigma_basis` names it.

        A supplier with no transit entry gets no lead time rather than a short one.
        Treating a missing transit time as zero understates the reorder point, and that
        is the direction that stocks out.
        """
        table, default = self._load_transit_times()
        dispatch = (df[CONFIRMED_DELIVERY_DATE] - df["po_date"]).dt.days
        suppliers = (df["supplier"] if "supplier" in df.columns
                     else pd.Series(None, index=df.index))
        found = suppliers.map(lambda v: self._transit_for(v, table, default))
        transit = found.map(lambda t: t[1] if t else np.nan).astype("float64")
        mode = found.map(lambda t: t[0] if t else None)

        df["transit_days"] = transit
        # The purchase export has no mode column, so this is the only place it comes
        # from. Carried because it is what a reader needs to make sense of a 35-day
        # transit leg, and because `diagnostics._chronic_air` already looks for it.
        # Nothing is inferred from it: it does not change an arithmetic result.
        if "transport_mode" not in df.columns:
            df["transport_mode"] = mode
        else:
            df["transport_mode"] = df["transport_mode"].where(
                df["transport_mode"].notna(), mode)
        df["lead_time_days"] = dispatch + transit
        df["lt_basis"] = np.where(df["lead_time_days"].notna(),
                                  BASIS_PROMISED_PLUS_TRANSIT, "")
        df["lt_sigma_basis"] = np.where(df["lead_time_days"].notna(),
                                        SIGMA_PROMISE_FLOOR, "")

        known = int(transit.notna().sum())
        print(f"  No goods-receipt date — lead time taken from the supplier's confirmed "
              f"dispatch ({CONFIRMED_DELIVERY_DATE}) plus a stated transit leg. This is "
              f"what the supplier committed to, not what they delivered.")
        if known:
            by_mode = (pd.DataFrame({"mode": mode, "days": transit})
                       .dropna(subset=["days"]).fillna({"mode": "unstated"})
                       .groupby("mode")["days"].agg(["size", "median"]))
            spread = ", ".join(f"{m} {r['median']:.0f}d x{int(r['size']):,}"
                               for m, r in by_mode.iterrows())
            print(f"      {known:,} of {len(df):,} rows: transit from {TRANSIT_FILE} "
                  f"({spread})"
                  + (f", default {default[1]:g}d for suppliers not named"
                     if default and default[1] is not None else ""))
            print(f"      Median total {df['lead_time_days'].median():.0f}d "
                  f"({dispatch.median():.0f}d to dispatch + "
                  f"{transit.median():.0f}d transit)")
            print(f"      lt_std_days here is a FLOOR, not a measurement: it is the "
                  f"spread of the supplier's promise, and a supplier who is reliably "
                  f"late shows no spread at all. The transit leg is a constant and adds "
                  f"none either. Safety stock covers less lead-time risk than the number "
                  f"suggests — post goods receipts and the measurement becomes real.")
        missing = (sorted(df.loc[transit.isna(), "supplier"].dropna().astype(str).unique())
                   if known < len(df) and "supplier" in df.columns else [])
        if missing:
            shown = ", ".join(missing[:4])
            print(f"      {len(df) - known:,} rows across {len(missing):,} supplier(s) "
                  f"have no transit time and so no lead time from this path — they fall "
                  f"back to the item master's planned delivery time. Add them to "
                  f"{TRANSIT_FILE}: {shown}"
                  + (" ..." if len(missing) > 4 else ""))
        self._write_transit_template(df, table, default)
        return df

    def _write_transit_template(self, df: pd.DataFrame, table: dict, default) -> None:
        """
        Write a starter copy of the transit table, listing every supplier in the extract.

        Compiling the supplier list is the slow half of filling this file in, and it is
        the half the run can do. Written whenever a supplier is unlisted — including
        when a `*` row is covering for them, because a blanket is a reason to finish the
        table rather than a reason to stop.

        Never written into `config/`: that would either overwrite a table someone is
        maintaining or look like it had been accepted. It goes beside the run's other
        outputs, and the planner moves it.
        """
        if "supplier" not in df.columns or self.output_dir is None:
            return
        named = {str(s).strip().upper(): str(s).strip()
                 for s in df["supplier"].dropna().unique() if str(s).strip()}
        unlisted = {raw for key, raw in named.items()
                    if not any(key.startswith(k) for k in table)}
        if not unlisted:
            return
        rows = [{"supplier": raw,
                 "transport_mode": (default[0] or "") if default else "",
                 "transit_days": (default[1] if default and default[1] is not None
                                  else "")}
                for raw in sorted(unlisted)]
        path = Path(self.output_dir) / TRANSIT_TEMPLATE
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write(f"# {len(rows)} supplier(s) with no row in "
                             f"{TRANSIT_FILE}. Fill transport_mode and transit_days,\n")
                handle.write(f"# then move this file to config/{TRANSIT_FILE}.\n")
                handle.write("# Any value already shown came from the \"*\" row and is "
                             "a blanket, not an answer for this supplier.\n")
                pd.DataFrame(rows).to_csv(handle, index=False)
        except OSError:
            return
        print(f"      Starter table for the {len(rows):,} unlisted supplier(s): {path}")

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

        return self.ensure_lead_time(df)

    def ensure_lead_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Put `lead_time_days` and `lt_basis` on the frame, by the best basis it supports.

        Public and idempotent because the two entry paths meet nowhere earlier. The
        reader calls it from `_post_process`; the intake path never runs the reader —
        the contract derives `lead_time_days` itself from `days_between(receive_date,
        po_date)`, which is exactly the derivation that cannot evaluate on an extract
        with no goods receipt — so the orchestrator calls it too. Without that second
        call the dispatch basis below would be unreachable on the path the pipeline
        actually uses.
        """
        if df is None or len(df) == 0 or "po_date" not in df.columns:
            return df
        # Already measured, by this or by the contract's own derivation. Re-running
        # would overwrite a receipt-based lead time with a dispatch-based one on a
        # frame that has both.
        if ("lead_time_days" in df.columns
                and pd.to_numeric(df["lead_time_days"], errors="coerce").notna().any()):
            if "lt_basis" not in df.columns:
                df = df.copy()
                df["lt_basis"] = BASIS_RECEIPT
            return df

        # Which clock to run, best basis first. Presence of a column is not enough:
        # a `receive_date` that is present and entirely empty used to take this branch
        # and produce a column of nulls, which reads downstream as "no lead time
        # measurable" when a dispatch date sitting in the same file could have measured
        # most of one.
        usable = lambda col: col in df.columns and df[col].notna().any()
        if "lt_days_precalc" in df.columns:
            df["lt_days_precalc"] = pd.to_numeric(df["lt_days_precalc"], errors="coerce")
            valid_precalc = df["lt_days_precalc"].notna() & (df["lt_days_precalc"] > 0)
            if valid_precalc.sum() > len(df) * 0.5:
                df["lead_time_days"] = df["lt_days_precalc"]
                df["lt_basis"] = BASIS_PRECALC
                print(f"  Using pre-computed LT column ({valid_precalc.sum()} valid rows)")
            elif usable(CONFIRMED_DELIVERY_DATE) and not usable("receive_date"):
                df = self._lead_time_from_dispatch(df)
            else:
                df = self._compute_lt_from_dates(df)
        elif usable("receive_date"):
            df = self._compute_lt_from_dates(df)
        elif usable(CONFIRMED_DELIVERY_DATE):
            # Measured as far as the dispatch, stated the rest of the way. Better than
            # no lead time at all, which is what this file produced before — and worse
            # than a receipt date, which is why it is third.
            df = self._lead_time_from_dispatch(df)
        else:
            print("  Warning: no receive_date, dispatch date or LT column found — "
                  "cannot compute lead time")
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
        df["lt_basis"] = BASIS_RECEIPT
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
                "lt_source": _lt_source(grp, in_window),
                "lt_window_months": int(rolling_months),
                # Which clock produced it. `lt_source` says whether it was measured and
                # from which window; this says how far the measurement actually reached.
                "lt_basis": (grp["lt_basis"].dropna().iloc[0]
                             if "lt_basis" in grp.columns and grp["lt_basis"].notna().any()
                             else BASIS_RECEIPT),
                "transit_days": (pd.to_numeric(grp["transit_days"], errors="coerce").median()
                                 if "transit_days" in grp.columns else np.nan),
                "lt_sigma_basis": (grp["lt_sigma_basis"].dropna().iloc[0]
                                   if "lt_sigma_basis" in grp.columns
                                   and grp["lt_sigma_basis"].notna().any() else ""),
            })

        out = pd.DataFrame(results)
        if fell_back:
            print(f"  Lead-time window: {rolling_months} months back from "
                  f"{newest:%Y-%m-%d} (lt_rolling_months)")
            print(f"  {len(fell_back)} SKU×supplier pairs had no receipt inside it — "
                  f"measured from their own older history instead (oldest last receipt "
                  f"{min(fell_back):%Y-%m-%d}), labelled measured_outside_window.")
        return out
