"""
Assemble the per-SKU attribute frame the policy layer reasons over.

Everything downstream — rule scopes, should-be, levers, target actions — reads one
table with one row per SKU. Building it in a single place means a rule author and the
should-be engine are guaranteed to be talking about the same `lead_time_days`, and it
gives one obvious answer to "where does this column come from".

Column naming is deliberately plain (`lead_time_days`, not `wma_lead_time_days`),
because these names are the vocabulary a planner writes rule scopes in and they end up
in `planning_parameters.md` documentation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .parameters import PlanningParameters


def build_sku_attributes(
    classified_demand: pd.DataFrame,
    supplier_lt: pd.DataFrame = None,
    inventory: pd.DataFrame = None,
    forecast_summary: pd.DataFrame = None,
    timeseries_meta: pd.DataFrame = None,
    params: PlanningParameters = None,
) -> pd.DataFrame:
    """
    One row per SKU, carrying demand statistics, supply parameters, cost and
    segmentation. Missing sources degrade the frame rather than break it — a rule
    whose scope needs an unavailable column simply matches nothing, and says so.
    """
    df = classified_demand.copy()

    # The classifier emits both `demand_std` (whole-history) and `demand_std_rolling`
    # (the rolling window actually used for planning). Renaming without dropping the
    # former leaves two columns of the same name, and every later `df[col]` silently
    # becomes a DataFrame.
    rename = {
        "demand_mean_rolling": "demand_mean",
        "demand_std_rolling": "demand_std",
    }
    rename = {k: v for k, v in rename.items() if k in df.columns}
    df = df.drop(columns=[v for v in rename.values() if v in df.columns]).rename(columns=rename)

    # ── Supply parameters ────────────────────────────────────────────────────
    if supplier_lt is not None and len(supplier_lt):
        # One supplier per SKU: the one actually used most, not the fastest. Picking
        # the shortest lead time flatters the plan with a supplier who may handle a
        # fraction of the volume.
        sort_col = "order_count" if "order_count" in supplier_lt.columns else "sample_count"
        chosen = (
            supplier_lt.sort_values(sort_col, ascending=False)
            if sort_col in supplier_lt.columns
            else supplier_lt
        ).groupby("sku", as_index=False).first()

        keep = {"sku": "sku", "wma_lead_time_days": "lead_time_days",
                "lt_std_days": "lt_sigma_days", "supplier": "supplier",
                "incoterm": "incoterm"}
        available = {k: v for k, v in keep.items() if k in chosen.columns}
        df = df.merge(chosen[list(available)].rename(columns=available), on="sku", how="left")

    for col, default in (("lead_time_days", np.nan), ("lt_sigma_days", 0.0)):
        if col not in df.columns:
            df[col] = default

    # ── Cost and current position ────────────────────────────────────────────
    if inventory is not None and len(inventory):
        cost_col = next((c for c in ("unit_cost", "std_cost", "avg_cost")
                         if c in inventory.columns), None)
        agg: Dict[str, Any] = {}
        if cost_col:
            agg[cost_col] = "first"     # cost is per SKU, not additive across bins
        for c in ("qty_on_hand", "qty_in_transit"):
            if c in inventory.columns:
                agg[c] = "sum"
        if agg:
            inv = inventory.groupby("sku", as_index=False).agg(agg)
            if cost_col and cost_col != "unit_cost":
                inv = inv.rename(columns={cost_col: "unit_cost"})
            df = df.merge(inv, on="sku", how="left")

    if "unit_cost" not in df.columns:
        df["unit_cost"] = np.nan

    # ── Forecast error as the demand sigma ───────────────────────────────────
    # Forecast RMSE is the right sigma for safety stock: it measures the error the
    # stock actually has to absorb. Demand std over-states it by counting variation
    # the forecast successfully predicted.
    df["demand_sigma"] = np.nan
    if forecast_summary is not None and "forecast_rmse" in forecast_summary.columns:
        rmse = forecast_summary.set_index("sku")["forecast_rmse"]
        df["demand_sigma"] = df["sku"].map(rmse)
    df["sigma_source"] = np.where(df["demand_sigma"].notna(), "forecast_rmse", "demand_std")
    if "demand_std" in df.columns:
        df["demand_sigma"] = df["demand_sigma"].fillna(df["demand_std"])
    df["demand_sigma"] = df["demand_sigma"].fillna(0.0)

    # ── Descriptive attributes for rule scopes ───────────────────────────────
    if timeseries_meta is not None and len(timeseries_meta):
        meta = timeseries_meta.reset_index() if timeseries_meta.index.name == "sku" else timeseries_meta
        meta_cols = [c for c in ("sku", "description", "sopc_classification", "product_family")
                     if c in meta.columns]
        if "sku" in meta_cols and len(meta_cols) > 1:
            df = df.merge(meta[meta_cols].drop_duplicates("sku"), on="sku", how="left")

    if "product_family" not in df.columns:
        df["product_family"] = _infer_family(df)

    # ── Segmentation ─────────────────────────────────────────────────────────
    df["annual_value"] = (
        pd.to_numeric(df["demand_mean"], errors="coerce").fillna(0.0) * 12
        * pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0)
    )
    params = params or PlanningParameters()
    df["abc_class"] = params.assign_abc(df)
    if "demand_cv" in df.columns:
        df["volatility_class"] = params.assign_volatility(
            pd.to_numeric(df["demand_cv"], errors="coerce").fillna(0.0)
        )

    return df


def _infer_family(df: pd.DataFrame) -> pd.Series:
    """
    Derive a product family from the SKU prefix when no explicit column exists.

    Most ERP part numbers are prefixed by family, so this makes family-scoped rules
    usable without extra master data. It is a guess, and a rule that depends on it is
    only as good as the numbering scheme — which is why the inferred value is exposed
    as a column the planner can inspect rather than used silently.
    """
    if "description" in df.columns and df["description"].notna().any():
        first_word = df["description"].astype(str).str.strip().str.split().str[0]
        if first_word.nunique() > 1:
            return first_word.str.lower()

    prefix = df["sku"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False)
    return prefix.str.lower().fillna("unknown")
