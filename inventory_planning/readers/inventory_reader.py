"""
Inventory snapshot reader.
Handles incoterm-aware GIT counting: EXW/FOB/CIF → GIT is buyer's; DDP/DAP → don't count.
"""

import json
from pathlib import Path
import pandas as pd
from .base_reader import BaseReader


class InventoryReader(BaseReader):
    doc_type = "inventory"

    def __init__(self, config_dir=None):
        super().__init__(config_dir)
        cfg = self.config_dir / "incoterm_rules.json"
        self.incoterm_rules = json.loads(cfg.read_text())["rules"]
        self.default_incoterm = json.loads(cfg.read_text()).get("default_incoterm", "FOB")

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Blank on-hand = zero stock (not truly missing — ERP omits 0s)
        if "qty_on_hand" in df.columns:
            df["qty_on_hand"] = pd.to_numeric(df["qty_on_hand"], errors="coerce").fillna(0)
            n_zero_filled = (df["qty_on_hand"] == 0).sum()
            if n_zero_filled:
                print(f"  Note: {n_zero_filled} SKUs with blank/zero on-hand treated as 0")

        # Aggregate duplicate SKUs (same SKU across multiple bins/locations sums up)
        agg_cols = {"qty_on_hand": "sum"}
        if "qty_in_transit" in df.columns:
            agg_cols["qty_in_transit"] = "sum"
        if "open_po_qty_inv" in df.columns:
            agg_cols["open_po_qty_inv"] = "sum"
        grp_cols = ["sku"]
        if "location_id" in df.columns:
            grp_cols.append("location_id")
        df = df.groupby(grp_cols, as_index=False).agg(agg_cols)
        return df

    def effective_inventory(self, inv_df: pd.DataFrame, open_po_df: pd.DataFrame,
                            supplier_params: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute effective inventory position per SKU considering incoterm rules.

        Logic:
        - Always include qty_on_hand
        - qty_in_transit: include only if incoterm → include_git_in_inventory = True
        - open_po_qty: always include (represents committed future receipts not yet in transit
          or not yet counting as buyer inventory)
        - If both GIT and open PO exist for buyer-owned incoterms, they are additive
          (GIT = shipped but not received; open PO = ordered but not shipped)

        Returns per-SKU: qty_on_hand, qty_in_transit (adjusted), total_open_po_qty,
                         effective_position
        """
        df = inv_df.copy()

        # Determine incoterm per SKU from supplier_params if available
        if supplier_params is not None and "incoterm" in supplier_params.columns:
            sku_incoterm = (
                supplier_params.dropna(subset=["incoterm"])
                .groupby("sku")["incoterm"]
                .agg(lambda x: x.mode()[0])
            )
            df = df.merge(sku_incoterm.rename("incoterm"), on="sku", how="left")
        elif "incoterm" not in df.columns:
            df["incoterm"] = None

        # Fill missing incoterm with default and warn
        missing_inco = df["incoterm"].isna().sum()
        if missing_inco:
            print(f"  Note: {missing_inco} SKUs have unknown incoterm — defaulting to '{self.default_incoterm}'")
            df["incoterm"] = df["incoterm"].fillna(self.default_incoterm)

        # Apply GIT rule
        def git_rule(row):
            rule = self.incoterm_rules.get(str(row["incoterm"]).upper(), self.incoterm_rules.get(self.default_incoterm))
            git = row.get("qty_in_transit", 0) or 0
            return git if rule["include_git_in_inventory"] else 0

        if "qty_in_transit" not in df.columns:
            df["qty_in_transit"] = 0
        df["qty_in_transit_adj"] = df.apply(git_rule, axis=1)

        # Merge open PO (prefer standalone open PO file; fall back to embedded column)
        if open_po_df is not None and "total_open_po_qty" in open_po_df.columns:
            df = df.merge(open_po_df[["sku", "total_open_po_qty"]], on="sku", how="left")
            df["total_open_po_qty"] = df["total_open_po_qty"].fillna(0)
        elif "open_po_qty_inv" in df.columns:
            # Inventory report embeds open PO qty directly (e.g. OpenPOQuantity column)
            df["total_open_po_qty"] = pd.to_numeric(df["open_po_qty_inv"], errors="coerce").fillna(0)
            print("  Using embedded OpenPOQuantity from inventory report (no standalone open PO file)")
        else:
            df["total_open_po_qty"] = 0

        df["effective_position"] = df["qty_on_hand"] + df["qty_in_transit_adj"] + df["total_open_po_qty"]

        cols = ["sku", "location_id", "qty_on_hand", "qty_in_transit",
                "qty_in_transit_adj", "incoterm", "total_open_po_qty", "effective_position"]
        return df[[c for c in cols if c in df.columns]]
