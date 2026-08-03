"""
Bridge between the canonical ingest layer and the existing analytics modules.

This module exists to keep a seam clean. `inventory_planning/ingest/` imports nothing
from the rest of this package — it only knows about contracts, adapters and pandas —
so it can be lifted out into a shared `sc-canonical` package (architecture doc, P0)
without a rewrite when the reconciliation skill needs it too. Everything that knows
about *this* pipeline's column expectations lives here instead.

The analytics modules were written against the old readers' output, so the mapping
below is deliberately explicit rather than clever: each rename records a decision
about which canonical field feeds which downstream calculation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .ingest.intake import Intake, IntakeResult


# Canonical field -> the name the existing analytics expect.
_DOWNSTREAM_RENAMES: Dict[str, Dict[str, str]] = {
    "inventory": {
        # The inventory export's own open-PO column, used only when no standalone
        # open PO document was supplied.
        "qty_on_order": "open_po_qty_inv",
    },
    "open_so": {},
    "open_po": {},
    "po_history": {},
    "sales_history": {},
}


class IngestBridge:
    """Turns an IntakeResult into the frames `InventoryPlanner.run_planning` expects."""

    def __init__(self, config_dir: Union[str, Path] = None, verbose: bool = True):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "config"
        self.verbose = verbose
        self.location_id = self._load_location_id()

    def _load_location_id(self) -> str:
        cfg = self.config_dir / "node_config.json"
        if cfg.exists():
            return json.loads(cfg.read_text()).get("location_id", "DC-01")
        return "DC-01"

    # ── Public API ───────────────────────────────────────────────────────────

    def adapt(self, result: IntakeResult) -> Dict[str, Any]:
        """
        Produce the keyword arguments for `run_planning`, plus the intake metadata the
        report needs in order to explain which numbers are estimates.
        """
        out: Dict[str, Any] = {
            "sales_df": None,
            "po_history_df": None,
            "open_so_df": None,
            "open_po_df": None,
            "inventory_df": None,
            "timeseries_pivot": None,
            "timeseries_meta": None,
        }

        for doc_type, doc in result.documents.items():
            frame = self._prepare(doc.frame.copy(), doc_type)
            if doc_type == "inventory":
                out["inventory_df"] = frame
            elif doc_type == "open_po":
                out["open_po_df"] = self._prepare_open_po(frame)
            elif doc_type == "open_so":
                out["open_so_df"] = frame
            elif doc_type == "po_history":
                out["po_history_df"] = self._prepare_po_history(frame)
            elif doc_type == "sales_history":
                out["sales_df"] = frame
            elif doc_type == "demand_timeseries":
                pivot, meta = self._prepare_timeseries(frame)
                out["timeseries_pivot"] = pivot
                out["timeseries_meta"] = meta

        out["_intake"] = result
        out["_intake_plan"] = result.plan
        return out

    def load(
        self,
        paths: List[Union[str, Path]],
        hints: Dict[str, str] = None,
        tenant: str = "default",
        baseline_path: Union[str, Path] = None,
    ) -> Dict[str, Any]:
        """Read files and adapt in one call — the normal entry point."""
        intake = Intake(tenant=tenant, baseline_path=baseline_path, verbose=self.verbose)
        return self.adapt(intake.load_files(paths, hints=hints))

    # ── Per-document preparation ─────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame, doc_type: str) -> pd.DataFrame:
        df = df.rename(columns=_DOWNSTREAM_RENAMES.get(doc_type, {}))
        # Every downstream output carries location_id for multi-echelon readiness;
        # only the inventory contract sources it from the data, so the rest inherit
        # the configured node.
        if "location_id" not in df.columns:
            df["location_id"] = self.location_id
        else:
            df["location_id"] = df["location_id"].fillna(self.location_id)
        return df

    def _prepare_po_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lead time is already derived by the contract (from dates or a pre-computed
        column). What remains is the outlier trim the analytics assume has happened.
        """
        if "lead_time_days" not in df.columns:
            df["lead_time_days"] = np.nan
            return df

        df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce")
        bad = (df["lead_time_days"] <= 0) | (df["lead_time_days"] > 730)
        if bad.sum() and self.verbose:
            print(f"  Excluded {int(bad.sum())} PO rows with implausible lead time (<=0 or >730d)")
        df = df[~bad].copy()

        if "po_qty" in df.columns:
            df["po_qty"] = pd.to_numeric(df["po_qty"], errors="coerce")

        # Drop only on the identifying columns this source actually has. Naming a
        # column that did not map raises rather than filtering, turning a reported
        # contract-test failure into an unhandled crash further downstream.
        required = [c for c in ("sku", "supplier", "po_date") if c in df.columns]
        return df.dropna(subset=required) if required else df

    def _prepare_open_po(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag which arrival dates are real and which will have to be estimated, so the
        report can distinguish a supplier commitment from the pipeline's own guess.
        """
        if "committed_delivery" not in df.columns:
            df["committed_delivery"] = pd.NaT
        df["delivery_estimated"] = df["committed_delivery"].isna()

        n_estimated = int(df["delivery_estimated"].sum())
        if n_estimated and self.verbose:
            print(f"  {n_estimated} open PO lines have no committed date — ETA will be estimated")
        return df

    def _prepare_timeseries(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Long sku/period/qty back into the SKU x period matrix the forecaster consumes,
        plus a per-SKU metadata frame carrying whatever descriptive columns the
        planner's file happened to include.
        """
        pivot = frame.pivot_table(
            index="period", columns="sku", values="qty", aggfunc="sum", fill_value=0.0
        )
        pivot.index = pd.PeriodIndex(pivot.index, freq="M")
        pivot = pivot.sort_index()

        meta_cols = [c for c in ("description", "sopc_classification") if c in frame.columns]
        if meta_cols:
            meta = (
                frame.drop_duplicates("sku")
                .set_index("sku")[meta_cols]
                .rename(columns={"sopc_classification": "sopc_classification"})
            )
        else:
            meta = pd.DataFrame(index=pd.Index(pivot.columns, name="sku"))

        if self.verbose:
            print(f"  Demand series: {len(pivot.columns)} SKUs x {len(pivot)} periods "
                  f"({pivot.index[0]} -> {pivot.index[-1]})")
        return pivot, meta
