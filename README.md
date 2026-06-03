# DC Inventory Planning

Single-stage distribution-centre inventory planning pipeline — demand classification, safety stock, ARIMA forecasting, and purchase recommendations. Multi-echelon ready (all outputs carry `location_id`).

## What it does

| Step | Output |
|------|--------|
| Reads 5 input files (sales, PO history, open SO, open PO, inventory) | Cleaned canonical DataFrames |
| Auto-detects column names across ERP formats | Column mapping preview + confirmation |
| Classifies SKUs: stocking-high / stocking-med / non-stocking | Based on demand frequency over 36-month window |
| Calculates safety stock | Combined demand × lead-time variability formula |
| Projects inventory position | On-hand + GIT (incoterm-aware) + open PO vs should-be |
| 6-month demand forecast | ARIMA → ETS → SMA fallback per SKU |
| Purchase recommendations | PURCHASE-REQUEST / ORDER-FOR-BACKLOG / PUSH-OUT-OPEN-PO / HOLD |
| HTML dashboard | 9 embedded charts + sortable data tables, self-contained |

## Quick start

```bash
pip install -e ".[dev]"

# Run with your files
inventory-plan \
  --sales       sales_history.xlsx \
  --po-history  po_history.xlsx \
  --open-so     open_so.xlsx \
  --open-po     open_po.xlsx \
  --inventory   inventory.xlsx \
  --timeseries  timeseries_3yr.xlsx   # optional pre-compiled wide-format TS
```

Or use the Python API:

```python
from inventory_planning import InventoryPlanner

planner = InventoryPlanner(output_dir="output/", interactive=True)

sales_df,   _ = planner.load_sales_history("sales_history.xlsx")
po_hist_df, _ = planner.load_po_history("po_history.xlsx")
open_so_df, _ = planner.load_open_so("open_so.xlsx")
open_po_df, _ = planner.load_open_po("open_po.xlsx")
inv_df,     _ = planner.load_inventory("inventory.xlsx")

# Optional: pre-compiled 3-year time series (wide format — SKU rows × month columns)
ts_pivot, ts_meta, _ = planner.load_timeseries("timeseries.xlsx", rolling_months=36)

results = planner.run_planning(
    sales_df, po_hist_df, open_so_df, open_po_df, inv_df,
    timeseries_pivot=ts_pivot,
    timeseries_meta=ts_meta,
)
```

## Input file formats

All files: CSV or Excel (.xlsx / .xls / .xlsm). Column names are auto-detected — the reader shows a mapping preview and asks for confirmation before processing.

| File | Required columns |
|------|-----------------|
| Sales history | SKU/Item, Order Date or Ship Date, Qty |
| PO history | SKU, Supplier, PO Date, Receive Date (or pre-computed LT column) |
| Open SO | SKU, Open Qty |
| Open PO | SKU, Order Qty, Delivered Qty, Closed Status |
| Inventory | SKU, On-Hand Qty |
| Time series *(optional)* | Item/SKU column + monthly period columns (wide format) |

## Configuration

All thresholds live in `config/` — no code changes needed:

| File | Controls |
|------|----------|
| `stocking_policy.json` | Frequency thresholds (9/12, 6/12), service levels (95%, 90%), forecast horizon |
| `incoterm_rules.json` | EXW/FOB/DDP GIT counting rules |
| `supplier_incoterm.json` | Supplier ID → incoterm (when PO files lack incoterm column) |
| `node_config.json` | Location ID, currency — one entry per DC |

## Output files

```
output/<timestamp>/
├── inventory_report_<ts>.html           ← self-contained HTML dashboard (open this)
├── purchase_recommendations_<ts>.csv
├── inventory_projection_<ts>.csv
├── forecast_detail_<ts>.csv
├── sku_planning_params.csv              ← persisted: stocking class, SS, ROP per SKU
└── supplier_params.csv                  ← persisted: WMA lead time per SKU × supplier
```

## Project structure

```
inventory_planning/
├── orchestrator.py          Main pipeline coordinator
├── schema.py                Canonical column schemas + aliases
├── cli.py                   CLI entry point
├── readers/                 5 document readers + wide-format time series reader
├── analytics/               Demand classifier, safety stock, projector, forecaster, recommender
└── reporting/               Matplotlib charts + Jinja2 HTML report generator
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Multi-echelon expansion

All outputs carry `location_id`. To expand to multi-echelon:
1. Update `config/node_config.json` with node topology
2. Run one `InventoryPlanner` per DC
3. A future `network_optimizer` module will aggregate across nodes

## Requirements

- Python ≥ 3.9
- pandas, numpy, scipy, statsmodels, openpyxl, matplotlib, seaborn, jinja2
