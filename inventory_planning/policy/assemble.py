"""
Assemble the per-SKU attribute frame the policy layer reasons over.

Everything downstream — rule scopes, should-be, levers, target actions — reads one
table with one row per SKU. Building it in a single place means a rule author and the
should-be engine are guaranteed to be talking about the same `lead_time_days`, and it
gives one obvious answer to "where does this column come from".

Column naming is deliberately plain (`lead_time_days`, not `wma_lead_time_days`),
because these names are the vocabulary a planner writes rule scopes in and they end up
in `planning_parameters.md` documentation.

Where an item master or a planner worksheet is also supplied, the same attribute
arrives from several sources at once. Choosing between them is not done inline here —
it goes through `policy.crosscheck`, so the precedence rule is stated once and every
disagreement is recorded rather than resolved silently.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .crosscheck import (
    CONFIG,
    ITEM_MASTER,
    MEASURED,
    PLANNING_MASTER,
    CrossCheckResult,
    SourceResolver,
)
from .parameters import PlanningParameters

# Planner-set parameters carried through untouched. They are the benchmark the
# suggestion engine compares against — never an input to a calculation, because a
# pipeline that consumes the planner's safety stock will always agree with it.
PLANNER_BENCHMARK_COLUMNS = (
    "planner_safety_stock", "planner_reorder_point", "planner_min_qty",
    "planner_max_qty", "planner_review_period_days", "planner_service_level",
    "planner_stocking_class", "planner_notes",
    # The planner's own usage figures. Not a demand source — the history is — but the
    # gap between the two is the most diagnostic number in the whole cross-check: when
    # they disagree at scale, the two are counting different events and every parameter
    # derived from either inherits the difference.
    "planner_monthly_demand", "planner_annual_demand",
)


def build_sku_attributes(
    classified_demand: pd.DataFrame,
    supplier_lt: pd.DataFrame = None,
    inventory: pd.DataFrame = None,
    forecast_summary: pd.DataFrame = None,
    timeseries_meta: pd.DataFrame = None,
    params: PlanningParameters = None,
    item_master: pd.DataFrame = None,
    planning_master: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, CrossCheckResult]:
    """
    One row per SKU, carrying demand statistics, supply parameters, cost and
    segmentation. Missing sources degrade the frame rather than break it — a rule
    whose scope needs an unavailable column simply matches nothing, and says so.

    `item_master` and `planning_master` are optional. Where they overlap with what the
    transactions measured, `policy.crosscheck` decides which value to use by authority
    and records every material disagreement; where the transactions are silent, they
    fill the gap. The planner's own parameters are additionally kept on the frame under
    `planner_*` names so the suggestion engine can compare against them — they are a
    benchmark, never an input.

    Returns `(attributes, crosscheck)`.
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

        # `sample_count` travels with the lead time because a sigma computed from two
        # receipts is not a distribution, and every parameter derived from it inherits
        # that weakness. Carrying the count is what lets a suggestion say so.
        keep = {"sku": "sku", "wma_lead_time_days": "lead_time_days",
                "lt_std_days": "lt_sigma_days", "supplier": "supplier",
                "incoterm": "incoterm", "sample_count": "lt_samples"}
        available = {k: v for k, v in keep.items() if k in chosen.columns}
        df = df.merge(chosen[list(available)].rename(columns=available), on="sku", how="left")

    for col, default in (("lead_time_days", np.nan), ("lt_sigma_days", 0.0),
                         ("lt_samples", 0)):
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

    # ── Master data: fill the gaps, cross-check the overlaps ─────────────────
    df, crosscheck = _merge_masters(df, item_master, planning_master)

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

    if "product_family" not in df.columns or df["product_family"].isna().all():
        df["product_family"] = _infer_family(df)
    else:
        # Master data wins where it speaks; the prefix guess fills the rest.
        df["product_family"] = df["product_family"].fillna(_infer_family(df))

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

    _crosscheck_demand(df, crosscheck)
    return df, crosscheck


# ── Master data ──────────────────────────────────────────────────────────────


def _merge_masters(
    df: pd.DataFrame,
    item_master: pd.DataFrame = None,
    planning_master: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, CrossCheckResult]:
    """
    Bring the two optional master documents onto the attribute frame.

    Numeric parameters that exist in more than one place go through the resolver, so
    the precedence rule is applied once and the disagreements are collected. Purely
    descriptive columns are filled in where the frame is silent — there is nothing to
    cross-check about an item description.
    """
    resolver = SourceResolver()

    item = _one_row_per_sku(item_master)
    plan = _one_row_per_sku(planning_master)
    if item is None and plan is None:
        return df, resolver.result

    df = df.copy()
    item_cols = _lookup(df, item)
    plan_cols = _lookup(df, plan)

    # Numeric parameters, resolved by authority.
    resolutions = {
        "lead_time_days": (
            df.get("lead_time_days"),
            item_cols.get("lead_time_days"),
            plan_cols.get("planner_lead_time_days"),
        ),
        "unit_cost": (
            df.get("unit_cost"),
            item_cols.get("unit_cost"),
            plan_cols.get("unit_cost"),
        ),
        "min_order_qty": (
            df.get("min_order_qty"),
            item_cols.get("min_order_qty"),
            plan_cols.get("planner_min_order_qty"),
        ),
        "order_multiple": (
            df.get("order_multiple"),
            item_cols.get("order_multiple"),
            plan_cols.get("planner_order_multiple"),
        ),
    }
    for attribute, (measured, from_item, from_plan) in resolutions.items():
        if from_item is None and from_plan is None:
            continue
        resolved = resolver.resolve(
            attribute,
            skus=df["sku"],
            candidates={MEASURED: measured, ITEM_MASTER: from_item, PLANNING_MASTER: from_plan},
            # Only lead time has a sample count behind it; the rest are stated values
            # wherever they come from.
            sample_counts=df.get("lt_samples") if attribute == "lead_time_days" else None,
        )
        df[attribute] = resolved["value"]
        df[f"{attribute}_source"] = resolved["source"].fillna(CONFIG)

    # Descriptive attributes: fill, do not arbitrate.
    for column, sources in (
        ("description", ("description", "description")),
        ("supplier", ("supplier", "supplier")),
        ("incoterm", ("incoterm", None)),
        ("product_family", ("product_family", None)),
        ("item_status", ("item_status", None)),
        ("planner_code", ("planner_code", None)),
    ):
        item_key, plan_key = sources
        candidate = item_cols.get(item_key)
        if plan_key and plan_cols.get(plan_key) is not None:
            candidate = candidate if candidate is not None else plan_cols[plan_key]
        if candidate is None:
            continue
        df[column] = df[column].fillna(candidate) if column in df.columns else candidate

    # The planner's own decisions, carried through untouched for comparison.
    for column in PLANNER_BENCHMARK_COLUMNS:
        if plan_cols.get(column) is not None:
            df[column] = plan_cols[column]

    if "planner_service_level" in df.columns:
        # Both 0.95 and 95 appear in real worksheets, and reading 95 as a probability
        # would put z off the end of the table.
        sl = pd.to_numeric(df["planner_service_level"], errors="coerce")
        df["planner_service_level"] = sl.where(sl <= 1.0, sl / 100.0)

    coverage = []
    if item is not None:
        coverage.append(f"item master covers {df['sku'].isin(item['sku']).sum():,} of "
                        f"{len(df):,} SKUs")
    if plan is not None:
        coverage.append(f"planner worksheet covers {df['sku'].isin(plan['sku']).sum():,} of "
                        f"{len(df):,} SKUs")
    for line in coverage:
        resolver.note(line)

    return df, resolver.result


def _crosscheck_demand(df: pd.DataFrame, crosscheck: CrossCheckResult) -> None:
    """
    Compare the planner's stated usage against the demand the history actually shows.

    This one is reported but never used to override anything. A large, one-directional
    gap almost always means the two are counting different events — orders raised vs
    goods issued, or a different date basis — and that difference propagates into every
    parameter derived from either. Naming it is more useful than reconciling it
    automatically, which would only hide which definition won.
    """
    if "demand_mean" not in df.columns:
        return

    stated = None
    if "planner_monthly_demand" in df.columns:
        stated = pd.to_numeric(df["planner_monthly_demand"], errors="coerce")
    elif "planner_annual_demand" in df.columns:
        stated = pd.to_numeric(df["planner_annual_demand"], errors="coerce") / 12.0
    if stated is None or stated.notna().sum() == 0:
        return

    resolver = SourceResolver()
    resolver.result = crosscheck
    resolver.resolve(
        "monthly_demand",
        skus=df["sku"],
        candidates={
            MEASURED: pd.to_numeric(df["demand_mean"], errors="coerce"),
            PLANNING_MASTER: stated,
        },
    )


def _one_row_per_sku(master: pd.DataFrame = None) -> Optional[pd.DataFrame]:
    """A master keyed on SKU. Duplicates would multiply the attribute frame on merge."""
    if master is None or len(master) == 0 or "sku" not in master.columns:
        return None
    if master["sku"].duplicated().any():
        master = master.drop_duplicates("sku", keep="first")
    return master


def _lookup(df: pd.DataFrame, master: pd.DataFrame = None) -> Dict[str, pd.Series]:
    """
    Every master column aligned to the attribute frame's SKUs and index.

    Aligning by map rather than merging keeps the frame's shape fixed — a merge against
    a master with an unexpected duplicate silently adds rows, and every quantity joined
    on `sku` afterwards is then counted twice.
    """
    if master is None:
        return {}
    indexed = master.set_index("sku")
    return {
        column: df["sku"].map(indexed[column])
        for column in indexed.columns
        if column != "sku"
    }


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
