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
from ..node import (
    SELLABLE, STOCK_STATUSES, StockStatusMap, UndeclaredStockLocation,
    load_stock_status,
)

# Quantities that are additive across storage locations. Anything not listed here is
# a dimension and takes the first non-null value — notably `unit_cost`, which is a
# master-data attribute of the SKU rather than of the bin it happens to sit in.
_ADDITIVE_MEASURES = (
    "qty_on_hand", "qty_in_transit", "qty_allocated", "qty_available",
    "qty_on_order", "open_po_qty_inv", "inventory_value",
)

# Of those, the ones describing *what is in the bin* rather than what is moving toward
# the node. Only these are partitioned by stock status, and the distinction is
# load-bearing.
#
# `qty_in_transit`, `qty_on_order` and `open_po_qty_inv` are inbound flows. Goods in
# transit are not in a storage location at all, and an inbound quantity reported on a
# row coded `02` has not been quarantined — it has not arrived. Withholding those from
# the position would understate inbound and over-order, which is the mirror of the
# defect this is here to fix.
#
# `qty_allocated` and `qty_available` never reach `effective_position` either way, but
# they are views of the same on-hand quantity. They are partitioned with it rather than
# left whole beside a reduced on-hand, contradicting it on the face of the row.
_BIN_CONTENTS = ("qty_on_hand", "qty_allocated", "qty_available", "inventory_value")

# What a consolidated frame carries about stock that did not net. Always present once a
# split has run, because `qty_not_sellable == 0` says something a missing column does
# not: the codes were declared, and nothing was withheld.
NOT_SELLABLE_QTY = "qty_not_sellable"
NOT_SELLABLE_VALUE = "value_not_sellable"
EXCLUDED_LOCATIONS = "stock_locations_excluded"
_SPLIT_MARKERS = (NOT_SELLABLE_QTY, NOT_SELLABLE_VALUE, EXCLUDED_LOCATIONS)


def status_column(status: str) -> str:
    """The reported column that one non-sellable status lands in."""
    return f"qty_{status}"


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """One measure as numbers, zero where the column is absent or unparseable."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _refuse_undeclared(df: pd.DataFrame, codes: pd.Series, undeclared,
                       status_map: StockStatusMap) -> None:
    """
    Raise, naming the codes and the stock standing behind each one.

    The message is the whole value of the refusal. A planner who knows their warehouse
    answers this in under a minute — but only if they are told which codes, how much is
    sitting in them, and what the words mean; a refusal that merely says a declaration
    is missing gets the shortest declaration that makes it stop.
    """
    on_hand = _numeric(df, "qty_on_hand")
    detail = []
    for code in undeclared:
        rows = codes.str.strip().str.upper() == code
        n = int(rows.sum())
        detail.append(f"      {code} — {on_hand[rows].sum():,.0f} on hand "
                      f"across {n:,} row{'' if n == 1 else 's'}")
    known = (", ".join(f"{c} = {s}" for c, s in sorted(status_map.statuses.items()))
             or "none")
    vocabulary = "\n".join(f"      {name:<20} {why}"
                           for name, why in STOCK_STATUSES.items())
    # Worded off the codes actually in front of the planner. A generic example invites
    # a generic answer, and the first guess — that the lowest code is the sellable one
    # — is the one worth putting in their hands to confirm or correct.
    pairs = [f'"{undeclared[0]}": "sellable"']
    if len(undeclared) > 1:
        pairs.append(f'"{undeclared[1]}": "quarantine"')
    example = "{" + ", ".join(pairs) + "}"
    raise UndeclaredStockLocation(
        f"The stock snapshot holds {codes.str.strip().ne('').sum():,} located rows "
        f"under {len(undeclared)} location code(s) whose meaning has not been "
        f"declared:\n" + "\n".join(detail) + "\n\n"
        f"    Already declared: {known}\n"
        f"    Declared with: config/location_status.json, under \"statuses\" — "
        f"e.g. {example}. \"*\" declares every code at once.\n\n"
        f"    The statuses, and what each one does:\n{vocabulary}\n\n"
        f"    Refused rather than summed. Only `sellable` nets into the planning "
        f"position, and stock of unknown status added to it overstates the position — "
        f"so the run would order less than it should, and nothing on the report would "
        f"say anything had been assumed."
    )


def _partition_by_status(df: pd.DataFrame, codes: pd.Series,
                         stock_status: StockStatusMap, verbose: bool):
    """
    Move non-sellable bin contents out of the netting measures and into reported ones.

    Done per row, before any aggregation, so the existing sum does the arithmetic and
    there is only one place where quantities are added up.
    """
    status_map = stock_status or StockStatusMap(statuses={}, source="not supplied")
    upper = codes.str.strip().str.upper()
    distinct = sorted({c for c in upper if c})

    undeclared = status_map.undeclared(distinct)
    if undeclared and len(distinct) > 1:
        _refuse_undeclared(df, codes, undeclared, status_map)

    # One code, or none: nothing to separate. The snapshot is the position by
    # construction, and a single-node planner is not made to describe a distinction
    # they do not have.
    statuses = upper.map(lambda c: status_map.status_of(c) or SELLABLE)
    sellable = statuses == SELLABLE

    df[NOT_SELLABLE_QTY] = 0.0
    df[NOT_SELLABLE_VALUE] = 0.0
    df[EXCLUDED_LOCATIONS] = ""
    if sellable.all():
        return df, []

    on_hand, value = _numeric(df, "qty_on_hand"), _numeric(df, "inventory_value")
    held = ~sellable
    df.loc[held, NOT_SELLABLE_QTY] = on_hand[held]
    df.loc[held, NOT_SELLABLE_VALUE] = value[held]
    df.loc[held, EXCLUDED_LOCATIONS] = upper[held]

    notes = []
    for status in sorted(set(statuses[held])):
        rows = held & (statuses == status)
        column = status_column(status)
        df[column] = 0.0
        df.loc[rows, column] = on_hand[rows]
        notes.append(f"     {on_hand[rows].sum():,.0f} units in "
                     f"{', '.join(sorted(set(upper[rows])))} are {status} — reported "
                     f"as {column}, not counted in the position")

    # Zero the netting measures on the withheld rows. The quantity is not lost: it is
    # in the reported columns above, and the codes are in `stock_locations_excluded`.
    for measure in _BIN_CONTENTS:
        if measure in df.columns:
            df.loc[held, measure] = 0.0

    if verbose and status_map.wildcard:
        notes.append(f"     Location statuses answered by the \"*\" wildcard "
                     f"({status_map.wildcard}), not code by code")
    return df, notes


def consolidate_to_planning_grain(
    inv_df: pd.DataFrame,
    planning_location: str = None,
    verbose: bool = True,
    stock_status: StockStatusMap = None,
) -> pd.DataFrame:
    """
    Collapse a SKU × storage-location snapshot to one row per SKU, counting only the
    stock that can actually be sold.

    Quantities are summed across locations; every other column takes its first
    non-null value. The location codes that were merged are kept in `stock_locations`
    so the detail is visible rather than discarded — and so the later multi-node work
    has something to build on.

    `stock_status` declares what each location code means — `config/location_status.json`,
    read by `node.load_stock_status`. Only `sellable` stock nets into the position;
    quarantine, blocked, consignment and transfer staging are summed into reported
    columns (`qty_quarantine` and so on, plus a `qty_not_sellable` total), so the stock
    stays visible without being counted on.

    **A snapshot carrying more than one location code with any of them undeclared is
    refused.** Not defaulted. Summing them is wrong in one direction and it is the
    expensive one: quarantine counted as available overstates the position, so the run
    orders less than it should and no figure on the report shows that anything was
    assumed. A single-code snapshot needs nothing declared — there is nothing to
    separate — so a single-node planner is never made to fill in a file describing a
    distinction they do not have.

    Idempotent. On a frame that has already been through here the partition is not run
    again: the reported columns are carried through the aggregation, and
    `stock_locations` is read as provenance rather than as a fresh claim about where
    the stock is.
    """
    if inv_df is None or len(inv_df) == 0 or "sku" not in inv_df.columns:
        return inv_df

    df = inv_df.copy()
    has_location = "location_id" in df.columns
    already_split = any(c in df.columns for c in _SPLIT_MARKERS)

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

    notes = []
    if not already_split:
        df, notes = _partition_by_status(df, codes, stock_status, verbose)

    def _union(values) -> str:
        seen = {c.strip() for v in values for c in str(v).split(",") if c.strip()}
        return ",".join(sorted(seen))

    # The reported columns are additive like any other quantity, and they exist only
    # after a split — so the set is assembled per call rather than fixed at import.
    additive = set(_ADDITIVE_MEASURES) | {NOT_SELLABLE_QTY, NOT_SELLABLE_VALUE}
    additive |= {status_column(s) for s in STOCK_STATUSES if s != SELLABLE}

    if not df["sku"].duplicated().any():
        # Already one row per SKU. Still stamp the provenance column so downstream
        # consumers can rely on it existing.
        df["stock_locations"] = codes.values
        if verbose:
            for note in notes:
                print(note)
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
        if col == EXCLUDED_LOCATIONS:
            agg[col] = _union
        else:
            agg[col] = "sum" if col in additive else "first"

    before = len(df)
    out = df.groupby("sku", as_index=False, dropna=False).agg(agg)
    out = out.merge(locations, on="sku", how="left")
    if has_location and planning_location:
        out["location_id"] = planning_location

    if verbose:
        multi = int((out["stock_locations"].str.count(",") > 0).sum())
        print(f"  Consolidated inventory to one row per SKU: {before:,} -> {len(out):,} rows"
              + (f" ({multi} SKUs held in more than one location)" if multi else ""))
        for note in notes:
            print(note)
    return out
    return out


class InventoryReader(BaseReader):
    doc_type = "inventory"

    def __init__(self, config_dir=None):
        super().__init__(config_dir)
        cfg = self.config_dir / "incoterm_rules.json"
        self.incoterm_rules = json.loads(cfg.read_text(encoding="utf-8"))["rules"]
        self.default_incoterm = json.loads(cfg.read_text(encoding="utf-8")).get("default_incoterm", "FOB")
        # What each storage-location code means. Loaded here rather than at each call
        # site so one reader cannot disagree with another about whether `02` is
        # sellable — the kind of split that produces two positions for one warehouse.
        self.stock_status = load_stock_status(self.config_dir)

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
        return consolidate_to_planning_grain(
            df, planning_location=self.location_id, stock_status=self.stock_status
        )

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
            inv_df, planning_location=self.location_id, verbose=False,
            stock_status=self.stock_status,
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
