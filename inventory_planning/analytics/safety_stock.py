"""
Safety stock calculator.

Formula (MIT CTL §7 — combined demand × lead time variability):
  SS = Z * sqrt(exposure * σ_fc² + demand_avg² * σ_LT²)

Where:
  σ_fc = forecast RMSE (preferred) or demand std dev (fallback if RMSE unavailable)
  demand_avg = unconditional mean demand per month (including zero periods)

## The exposure period is R + LT, not LT

Under periodic review the position is only seen once every R days, so a shortfall that
appears the day after a review cannot be acted on until the next one. The window the
safety stock has to survive is therefore the review period *plus* the lead time. Using
LT alone understates it by √((R+LT)/LT) — 41% for a monthly review against a 30-day lead
time, and the error is largest on exactly the short-lead-time items where the buyer
assumes they are safe.

`review_period_days` comes from the resolved planning parameters, per SKU, so an A item
on a 7-day review and a C item on a 90-day review get different exposures from the same
demand. When no review period is supplied the exposure falls back to LT alone — that is
the continuous-review reading, and it is what a caller who knows their policy watches
the position continuously should get. It is stated in `ss_exposure_basis` on every row
rather than left to be inferred.

This is the same convention `policy/should_be.py` applies, read from the same
`safety_stock_exposure` setting. Two definitions of one exposure in one pipeline is how
the recommended order and the target stock come to disagree by construction.

Service level note (see config/stocking_policy.json → service_level_metric):
  CSL: k = NORMSINV(CSL) — z_score from config, probability of no stockout per cycle
  IFR: requires solving G(k) = Q*(1-IFR)/σDL — not yet implemented, defaults to CSL

Supplier LT selection:
  Uses the ordering supplier (from config/supplier_incoterm.json) when available.
  Falls back to shortest WMA LT if no preferred supplier defined.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


class SafetyStockCalculator:

    def __init__(self, config_dir: Path = None):
        self._preferred_supplier: dict = {}
        if config_dir is not None:
            si_path = Path(config_dir) / "supplier_incoterm.json"
            if si_path.exists():
                data = json.loads(si_path.read_text(encoding="utf-8"))
                # supplier_incoterm.json maps supplier_id → incoterm; no preferred supplier list yet.
                # When a sku-level preferred supplier config is added, load it here.

    @staticmethod
    def _review_period(df: pd.DataFrame, review_period_days) -> pd.Series:
        """
        Per-SKU review period, from a frame, a Series, a number, or nothing.

        Returning NaN where it is unknown is deliberate: a missing review period must
        leave the exposure at the lead time and say so, rather than borrow a default
        that would quietly inflate every safety stock in the run.
        """
        if review_period_days is None:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        if isinstance(review_period_days, pd.DataFrame):
            if "sku" not in review_period_days.columns or \
                    "review_period_days" not in review_period_days.columns:
                raise ValueError(
                    "review_period_days frame needs 'sku' and 'review_period_days' columns"
                )
            lookup = (review_period_days.drop_duplicates("sku")
                      .set_index("sku")["review_period_days"])
            return pd.to_numeric(df["sku"].map(lookup), errors="coerce")
        if isinstance(review_period_days, pd.Series):
            return pd.to_numeric(df["sku"].map(review_period_days), errors="coerce")
        return pd.Series(float(review_period_days), index=df.index, dtype="float64")

    def calculate(self, classified_demand: pd.DataFrame, supplier_lt: pd.DataFrame,
                  forecast_summary: pd.DataFrame = None,
                  review_period_days=None,
                  exposure: str = "review_plus_lt") -> pd.DataFrame:
        """
        classified_demand: output of DemandClassifier.classify()
        supplier_lt:       output of POHistoryReader.compute_supplier_lt()
        forecast_summary:  output of Forecaster.summary() — provides forecast_rmse per SKU.
                           When provided, forecast_rmse is used as σ_fc.
                           When absent, demand_std_rolling is used (overestimates SS).
        review_period_days: per-SKU review period. A DataFrame with `sku` and
                           `review_period_days`, a Series indexed by SKU, or one number
                           for every SKU. Omitted means the caller has not told us how
                           often the position is seen, and the exposure stays at LT.
        exposure:          `review_plus_lt` (periodic review, the default) or `lt_only`
                           (continuous review — the position is watched, so the gap
                           between reviews is not part of the risk).

        Returns per-SKU: safety_stock, reorder_point, ss_coverage_days, sigma_source,
        ss_exposure_days, ss_exposure_basis
        """
        # ── Supplier LT selection ─────────────────────────────────────────────
        # Preferred: use the supplier with the most PO history (by order count) as proxy
        # for the "ordering supplier". Shortest LT can be an unreliable outlier supplier.
        # Fallback: shortest WMA LT (original behaviour).
        if "order_count" in supplier_lt.columns:
            ordering_lt = (
                supplier_lt.sort_values("order_count", ascending=False)
                .groupby("sku")
                .first()
                .reset_index()[["sku", "wma_lead_time_days", "lt_std_days"]]
            )
        else:
            ordering_lt = (
                supplier_lt.sort_values("wma_lead_time_days")
                .groupby("sku")
                .first()
                .reset_index()[["sku", "wma_lead_time_days", "lt_std_days"]]
            )

        df = classified_demand.merge(ordering_lt, on="sku", how="left")

        no_lt = df["wma_lead_time_days"].isna().sum()
        if no_lt:
            print(f"  Warning: {no_lt} SKUs have no supplier LT data — safety stock set to 0, review manually")
        df["wma_lead_time_days"] = df["wma_lead_time_days"].fillna(0)
        df["lt_std_days"] = df["lt_std_days"].fillna(0)

        # ── Merge forecast RMSE if available ─────────────────────────────────
        if forecast_summary is not None and "forecast_rmse" in forecast_summary.columns:
            rmse_map = forecast_summary.set_index("sku")["forecast_rmse"]
            df["forecast_rmse"] = df["sku"].map(rmse_map).fillna(np.nan)
            df["sigma_source"] = df["forecast_rmse"].apply(
                lambda v: "forecast_rmse" if pd.notna(v) and v > 0 else "demand_std"
            )
        else:
            df["forecast_rmse"] = np.nan
            df["sigma_source"] = "demand_std"

        # ── Exposure period ───────────────────────────────────────────────────
        df["review_period_days"] = self._review_period(df, review_period_days)
        use_review = (exposure == "review_plus_lt") & df["review_period_days"].notna()
        df["ss_exposure_days"] = np.where(
            use_review,
            df["wma_lead_time_days"] + df["review_period_days"].fillna(0.0),
            df["wma_lead_time_days"],
        )
        df["ss_exposure_basis"] = np.where(use_review, "review_plus_lt", "lt_only")

        # ── Safety stock calculation ──────────────────────────────────────────
        df["lt_months"] = df["wma_lead_time_days"] / 30.0
        df["lt_std_months"] = df["lt_std_days"] / 30.0
        df["exposure_months"] = df["ss_exposure_days"] / 30.0

        def _ss(row):
            if row["stocking_class"] == "non-stocking" or pd.isna(row["z_score"]) or row["z_score"] is None:
                return 0.0
            z = float(row["z_score"])
            exposure_m = float(row["exposure_months"])
            lt_std = float(row["lt_std_months"])
            # demand_mean_rolling is the unconditional monthly mean (including zero months)
            d_mean = float(row["demand_mean_rolling"])
            # Use forecast RMSE when available; fall back to demand std dev
            if pd.notna(row["forecast_rmse"]) and row["forecast_rmse"] > 0:
                sigma_fc = float(row["forecast_rmse"])
            else:
                sigma_fc = float(row["demand_std_rolling"])
            # Combined variability formula (MIT CTL §7). The demand term runs over the
            # whole exposure window; the lead-time term does not, because σ_LT is a
            # property of the supplier and does not grow with how often we look.
            variance = exposure_m * (sigma_fc ** 2) + (d_mean ** 2) * (lt_std ** 2)
            return round(z * np.sqrt(max(variance, 0)), 1)

        df["safety_stock"] = df.apply(_ss, axis=1)

        # ROP uses unconditional mean (demand_mean_rolling already set to unconditional
        # in classifier). `demand_during_lt` is kept because pipeline stock is genuinely
        # a lead-time quantity; the reorder point is not, and is built on the exposure
        # window so that it and the safety stock beside it answer the same question.
        df["demand_during_lt"] = (df["demand_mean_rolling"] * df["lt_months"]).round(1)
        df["demand_during_exposure"] = (
            df["demand_mean_rolling"] * df["exposure_months"]
        ).round(1)
        df["reorder_point"] = (df["demand_during_exposure"] + df["safety_stock"]).round(1)

        daily_demand = df["demand_mean_rolling"] / 30.0
        df["ss_coverage_days"] = (df["safety_stock"] / daily_demand.replace(0, np.nan)).round(0)

        return df
