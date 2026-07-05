"""
FeedbackCollector: records actual outcomes against a prior snapshot.

Usage — called the month after a planning run, once actual data is available:

    from inventory_planning.feedback.collector import FeedbackCollector

    collector = FeedbackCollector("output/history/2026-06/snapshot_20260603_2335.json")
    collector.record_actuals(
        actual_sales_df,     # same format as sales_history — actual shipments this month
        actual_inventory_df, # end-of-month inventory snapshot
    )
    # Actuals are written back into the snapshot JSON.

Actuals recorded per SKU:
  actual_demand      — units actually consumed / shipped in the planning month
  actual_eom_inv     — end-of-month on-hand inventory
  actual_receipt_qty — purchase orders received during the month
"""

import json
from pathlib import Path
import pandas as pd


class FeedbackCollector:

    def __init__(self, snapshot_path: str | Path):
        self.path = Path(snapshot_path)
        with open(self.path) as f:
            self.snapshot = json.load(f)

    def record_actuals(
        self,
        actual_sales_df: pd.DataFrame,
        actual_inventory_df: pd.DataFrame,
        actual_receipts_df: pd.DataFrame = None,
    ) -> None:
        """
        actual_sales_df:
          Must have columns: sku, qty (units shipped/sold in the planning month)

        actual_inventory_df:
          Must have columns: sku, qty_on_hand (end-of-month physical count)

        actual_receipts_df (optional):
          Must have columns: sku, qty (PO receipts during the month)
        """
        sales_map = self._agg(actual_sales_df, "qty", "actual_demand")
        inv_map = self._agg(actual_inventory_df, "qty_on_hand", "actual_eom_inv")
        receipt_map = self._agg(actual_receipts_df, "qty", "actual_receipt_qty") if actual_receipts_df is not None else {}

        actuals = {}
        for sku in self.snapshot["skus"]:
            actuals[sku] = {
                "actual_demand": sales_map.get(sku),
                "actual_eom_inv": inv_map.get(sku),
                "actual_receipt_qty": receipt_map.get(sku),
            }

        self.snapshot["actuals"] = actuals
        self._write()
        print(f"  Actuals recorded for {len(actuals)} SKUs → {self.path}")

    def _agg(self, df: pd.DataFrame, value_col: str, label: str) -> dict:
        if df is None or value_col not in df.columns:
            return {}
        return df.groupby("sku")[value_col].sum().to_dict()

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.snapshot, indent=2, ensure_ascii=False))
