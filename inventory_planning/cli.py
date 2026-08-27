"""
CLI entry point: inventory-plan
Usage:
  inventory-plan --sales <path> --po-history <path> --open-so <path>
                 --open-po <path> --inventory <path>
                 [--timeseries <path>] [--item-master <path>]
                 [--planning-master <path>] [--output <dir>] [--parameters <path>]
                 [--no-interactive]
"""

import argparse
import sys
from pathlib import Path

from .orchestrator import InventoryPlanner


def main():
    parser = argparse.ArgumentParser(
        prog="inventory-plan",
        description="DC Inventory Planning — demand classification, safety stock, forecast, purchase recommendations",
    )
    parser.add_argument("--sales",        required=True,  help="Sales history file (CSV/xlsx)")
    parser.add_argument("--po-history",   required=True,  help="PO history file (CSV/xlsx)")
    parser.add_argument("--open-so",      required=True,  help="Open sales orders file (CSV/xlsx)")
    parser.add_argument("--open-po",      required=True,  help="Open purchase orders file (CSV/xlsx)")
    parser.add_argument("--inventory",    required=True,  help="Inventory snapshot file (CSV/xlsx)")
    parser.add_argument("--timeseries",   default=None,   help="Pre-compiled time series file (wide format, optional)")
    parser.add_argument("--item-master",  default=None,   help="ERP item master (optional): supplier, lead time, MOQ, cost")
    parser.add_argument("--planning-master", default=None, help="Planner worksheet (optional): safety stock, min/max, LT — compared, not consumed")
    parser.add_argument("--ts-months",    type=int, default=36, help="Rolling months for time series (default 36)")
    parser.add_argument("--output",       default="output", help="Output directory (default: ./output)")
    parser.add_argument("--config",       default=None,   help="Config directory (default: ./config)")
    parser.add_argument("--parameters",   default=None,
                        help="Rule set to plan under (default: <config>/planning_parameters.md). "
                             "A different one is a scenario: it changes the review period and "
                             "service level the recommendations are built on, and the run records "
                             "it so the two can be compared afterwards.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip column-mapping confirmation prompts")

    args = parser.parse_args()

    planner = InventoryPlanner(
        config_dir=args.config,
        output_dir=args.output,
        interactive=not args.no_interactive,
        parameters_file=args.parameters,
    )

    print("Loading input files...")
    sales_df,   _ = planner.load_sales_history(args.sales)
    po_hist_df, _ = planner.load_po_history(args.po_history)
    open_so_df, _ = planner.load_open_so(args.open_so)
    open_po_df, _ = planner.load_open_po(args.open_po)
    inv_df,     _ = planner.load_inventory(args.inventory)

    ts_pivot = ts_meta = None
    if args.timeseries:
        ts_pivot, ts_meta, _ = planner.load_timeseries(args.timeseries, rolling_months=args.ts_months)

    # The two masters go through the contract-driven intake rather than a legacy
    # reader: there is no fixed schema to write one against, and the contract already
    # knows every alias an ERP export or a planner's spreadsheet is likely to use.
    item_master_df = planning_master_df = None
    if args.item_master or args.planning_master:
        from .ingest_bridge import IngestBridge
        paths = [p for p in (args.item_master, args.planning_master) if p]
        loaded = IngestBridge(config_dir=args.config, verbose=True).load(paths)
        item_master_df = loaded.get("item_master_df")
        planning_master_df = loaded.get("planning_master_df")

    results = planner.run_planning(
        sales_df, po_hist_df, open_so_df, open_po_df, inv_df,
        timeseries_pivot=ts_pivot,
        timeseries_meta=ts_meta,
        item_master_df=item_master_df,
        planning_master_df=planning_master_df,
    )
    planner.run_policy_analysis(
        results, inventory_df=inv_df, open_po_df=open_po_df,
        item_master_df=item_master_df, planning_master_df=planning_master_df,
    )


if __name__ == "__main__":
    main()
