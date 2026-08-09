"""
Inventory snapshot reader.
Handles incoterm-aware GIT counting: EXW/FOB/CIF → GIT is buyer's; DDP/DAP → don't count.

Grain is the other thing this module owns. A warehouse export is one row per
SKU × storage location — normal stock in 01, quarantine in 02 — and that is the
honest grain to read at. But everything downstream of here plans a single node and
joins on `sku` alone, so a SKU present in two locations fans out into two planning
rows, each of which then matches the *same* open PO and the *same* backlog. The
result is one SKU appearing twice with contradictory advice: pull in on one row,
push out on the other. `consolidate_to_planning_grain` collapses the snapshot to one
row per SKU before it reaches the analytics, and the projector refuses to run on a
frame that has not been through it.
"""

import json
from pathlib import Path
import pandas as pd
from .base_reader import BaseReader

# Quantities that are additive across storage locations. Anything not listed here is
# a dimension and takes the first non-null value — notably `unit_cost`, which is a
# master-data attribute of the SKU rather than of the bin it happens to sit in.
_ADDITIVE_MEASURES = (
    "qty_on_hand", "qty_in_transit", "qty_allocated", "qty_available",
    "qty_on_order", "open_po_qty_inv", "inventory_value",
)


def consolidate_to_planning_grain(
    inv_df: pd.DataFrame,
    planning_location: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collapse a SKU × storage-location snapshot to one row per SKU.

    Quantities are summed across locations; every other column takes its first
    non-null value. The location codes that were merged are kept in
    `stock_locations` so the detail is visible rather than discarded — and so the
    later multi-node work has something to build on.

    Note what this does *not* do: it makes no distinction between sellable stock and
    quality-quarantine stock. Everything in the snapshot counts toward the position.
    Where an inspection or blocked location holds material amounts, the position is
    overstated by exactly that much. Separating them needs a supply-chain topology
    the pipeline does not have yet (see TODO.md).
    """
    if inv_df is None or len(inv_df) == 0 or "sku" not in inv_df.columns:
        return inv_df

    df = inv_df.copy()
    has_location = "location_id" in df.columns

    # Source codes per row. On a frame that has already been through here, location_id
    # is the planning node and `stock_locations` is the only remaining record of where
    # the stock actually was, so it wins where it says anything — consolidating twice
    # must not quietly rewrite the provenance to "DC-01". Where it is blank, the row
    # has not been consolidated before and location_id is the real code.
    fallback = (df["location_id"].fillna("").astype(str) if has_location
                else pd.Series("", index=df.index))
    if "stock_locations" in df.columns:
        prior = df["stock_locations"].fillna("").astype(str)
        codes = prior.where(prior.str.strip() != "", fallback)
    else:
        codes = fallback
    df = df.drop(columns=["stock_locations"], errors="ignore")

    def _union(values) -> str:
        seen = {c.strip() for v in values for c in str(v).split(",") if c.strip()}
        return ",".join(sorted(seen))

    if not df["sku"].duplicated().any():
        # Already one row per SKU. Still stamp the provenance column so downstream
        # consumers can rely on it existing.
        df["stock_locations"] = codes.values
        return df

    locations = (
        df.assign(_codes=codes.values)
        .groupby("sku")["_codes"]
        .agg(_union)
        .rename("stock_locations")
    )

    agg = {}
    for col in df.columns:
        if col == "sku":
            continue
        agg[col] = "sum" if col in _ADDITIVE_MEASURES else "first"

    before = len(df)
    out = df.groupby("sku", as_index=False, dropna=False).agg(agg)
    out = out.merge(locations, on="sku", how="left")
    if has_location and planning_location:
        out["location_id"] = planning_location

    if verbose:
        multi = int((out["stock_locations"].str.count(",") > 0).sum())
        print(f"  Consolidated inventory to one row per SKU: {before:,} -> {len(out):,} rows"
              + (f" ({multi} SKUs held in more than one location)" if multi else ""))
        if multi:
            print("     Quarantine / inspection locations are summed into the available "
                  "position — see TODO.md, supply-chain topology.")
    return out


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

        # Aggregate duplicate SKUs across bins/locations. Grouping by sku+location_id
        # here used to leave one row per warehouse, and `BaseReader.read` then stamped
        # the configured node over `location_id` — producing rows with an identical key
        # that every downstream join duplicated.
        return consolidate_to_planning_grain(df, planning_location=self.location_id)

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

        The snapshot is consolidated to one row per SKU first. Without it a SKU held in
        two locations gets the full open PO quantity merged onto each of its rows and
        the position is counted twice.
        """
        df = consolidate_to_planning_grain(
            inv_df, planning_location=self.location_id, verbose=False
        ).copy()

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

        cols = ["sku", "location_id", "stock_locations", "qty_on_hand", "qty_in_transit",
                "qty_in_transit_adj", "incoterm", "total_open_po_qty", "effective_position"]
        return df[[c for c in cols if c in df.columns]]
