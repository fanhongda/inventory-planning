# Inventory Planning — Domain Skill

> Part of a personal supply chain agent infrastructure designed to demonstrate how frontier AI can operate meaningfully in real-world logistics contexts.

---

## Design Philosophy

### The Bigger Picture: Supply Chain Agent Infrastructure

This skill sits within a broader research direction: building AI agents that can reason about supply chains not as optimisation puzzles, but as **socio-technical systems** — where demand is noisy, organisations respond with lag, supplier capacity has hard ceilings, and the gap between what a planner's KPI measures and what the supply chain actually needs is often the root cause of failure.

The long-term architecture separates three layers:

```
Infrastructure Layer    Claude Code + skills framework
                        Readers, orchestrator, eval harness

Domain Skill Layer      ← this repo
                        Inventory planning with MIT SCM principles
                        + deviation attribution feedback loop

World Model Layer       (in progress)
                        Causal loop models of bullwhip dynamics,
                        organisational behaviour, KPI misalignment
```

This repo is the **domain skill layer**: grounded in classical supply chain theory, designed to be used by a human expert, and built to accumulate learning over time without blindly trusting its own outputs.

---

### Why Deviation Attribution Matters

Most inventory planning tools treat the gap between recommendation and actuals as a signal to adjust the model. That assumption is almost always wrong.

In real supply chains, a deviation between what the system recommends and what actually happens can have at least three distinct causes:

| Cause | Example | Correct response |
|---|---|---|
| **Operator non-execution** | Buyer ignores buy signal due to budget freeze or warehouse space | Maintain recommendation. Flag to ops. Do not adjust model. |
| **Cumulative supply gap** | Demand exceeds supply for 3 consecutive months — supplier has no flex capacity | Escalate to supply chain review. Not a model error. |
| **Model bias** | Forecast systematically over-estimates for erratic SKUs | Investigate model. Consider parameter adjustment. |

The failure mode this guards against is **spurious learning**: a system that observes an operator consistently buying less than recommended, then gradually lowers its recommendations to match operator behaviour — effectively learning to sanction underperformance.

This is a specific instance of the broader alignment problem: when the observable signal (what was purchased) diverges from the true target (what should be purchased), a naïve learning loop reinforces the wrong behaviour.

Real supply chain crises tend to share two structural features: (1) **persistent cumulative gaps** — supply falling behind demand for multiple consecutive months without correction, and (2) **supplier capacity exhaustion** — when a buyer represents a large share of a supplier's output, there is no buffer when demand accelerates. Both conditions, especially when combined with an external shock (semiconductor shortage, port disruption), create the conditions for bullwhip-driven collapse.

**The feedback module in this repo tracks cumulative supply and demand gaps across months, distinguishes operator deviation from model error, and only suggests parameter changes when the deviation cannot be explained by execution failure or supply constraint.**

---

## What This Skill Does

Single-stage DC inventory planning pipeline, grounded in MIT CTL MicroMasters supply chain principles (`principles/sc-principles.md`).

| Step | Method | Output |
|---|---|---|
| Demand characterisation | Frequency + CV-based classification | stocking-high / stocking-med / non-stocking |
| Demand pattern routing | CV thresholds | smooth / intermittent / erratic / lumpy |
| Forecasting | ETS (Holt-Winters) or Croston's method per pattern | 6-month forward forecast + t+1 point forecast |
| Safety stock | Combined demand × lead-time variability (MIT CTL §7) | SS = k · √(LT·σ_fc² + μ² · σ_LT²) |
| Inventory projection | Days-of-supply based excess detection | EXCESS / SHORTAGE-RISK / OK |
| Purchase recommendation | Net requirement using t+1 point forecast | PURCHASE-REQUEST / PUSH-OUT / HOLD |
| Feedback loop | Cumulative gap attribution across months | OPERATOR_DEVIATION / SUPPLY_GAP / MODEL_BIAS |
| HTML dashboard | Self-contained report with 9 charts | inventory_report_<ts>.html |

### Key design decisions vs standard approaches

**Croston's method for intermittent demand.** ETS and ARIMA applied to sparse demand (e.g., 6 out of 12 active months) produce systematically unstable forecasts. Per MIT CTL, Croston's method separates demand magnitude from inter-arrival frequency, producing stable estimates for slow-moving and erratic SKUs.

**t+1 point forecast for purchase quantity.** Using the 6-month average forecast as the purchase signal introduces systematic bias in trending demand. This skill uses the next-period point forecast (`forecast_next_period`) to drive net requirement, with the 6-month summary retained for planning horizon visibility.

**Forecast RMSE as σDL for safety stock.** The standard approximation uses historical demand standard deviation as σDL. When a good forecast exists, the relevant uncertainty is the *residual* after forecasting, not raw demand variability. This skill uses in-sample forecast RMSE when available, falling back to demand std dev.

**Days-of-supply excess threshold.** EXCESS inventory is flagged when DOS > configurable threshold (default 90 days), not by an arbitrary percentage over ROP. This is a policy parameter, not a model assumption.

---

## Reference Files

```
principles/sc-principles.md    MIT CTL MicroMasters key concepts + formulas
                                Theoretical anchor for all analytics decisions

policy/policy.md               Company-specific hard constraints
                                MOQ, approved suppliers, KPI guardrails
                                (blank — populate for your supply chain)

AUDIT.md                       Principles-vs-code audit log
                                Records every logic deviation found and fixed
```

---

## Quick Start

```bash
pip install -e ".[dev]"

inventory-plan \
  --sales       sales_history.xlsx \
  --po-history  po_history.xlsx \
  --open-so     open_so.xlsx \
  --open-po     open_po.xlsx \
  --inventory   inventory.xlsx \
  --timeseries  timeseries_3yr.xlsx   # optional pre-compiled wide-format TS
```

Python API:

```python
from inventory_planning import InventoryPlanner

planner = InventoryPlanner(output_dir="output/", interactive=True)

sales_df,   _ = planner.load_sales_history("sales_history.xlsx")
po_hist_df, _ = planner.load_po_history("po_history.xlsx")
open_so_df, _ = planner.load_open_so("open_so.xlsx")
open_po_df, _ = planner.load_open_po("open_po.xlsx")
inv_df,     _ = planner.load_inventory("inventory.xlsx")

results = planner.run_planning(
    sales_df, po_hist_df, open_so_df, open_po_df, inv_df
)
# Snapshot saved automatically to output/history/YYYY-MM/
```

---

## Feedback Loop Usage

After each run, a snapshot is saved to `output/history/YYYY-MM/`. The following month, once actuals are available:

```python
from inventory_planning.feedback.collector import FeedbackCollector
from inventory_planning.feedback.loss import LossCalculator

# Record actuals against last month's snapshot
collector = FeedbackCollector("output/history/2026-06/snapshot_20260603_2335.json")
collector.record_actuals(actual_sales_df, actual_inventory_df)

# Compute loss + deviation attribution
calc = LossCalculator("output/history/2026-06/snapshot_20260603_2335.json")
result = calc.compute()
```

Output:
```
============================================================
  FEEDBACK REPORT — 2026-06
============================================================

  Forecast:  MD=+12.3  MAD=28.1  MAPE=18.4%  (over-forecast)
  Inventory: avg DOS realized=67d  excess rate=12%

  Cumulative gap analysis (3 months, threshold=3):
  ⚠  OPERATOR DEVIATION:   4 SKUs — model recommendation maintained
  🔴 SUPPLY RISK:          2 SKUs — escalate to supply chain review

  Suggestions:
  ► [OPERATOR_DEVIATION] SKU=XYZ-001
    Maintain system recommendation unchanged.
    Review with operations: storage constraints, budget freeze, or manual override?
    → No model parameter change.

  ► [CUMULATIVE_SUPPLY_GAP] SKU=ABC-042
    Escalate to supply chain review. Check supplier capacity and lead time.
    → No model parameter change.
============================================================
```

---

## Input Formats

All files accept CSV or Excel (.xlsx / .xls / .xlsm). Column names are auto-detected across ERP formats.

| File | Required columns |
|---|---|
| Sales history | SKU/Item, Order Date or Ship Date, Qty |
| PO history | SKU, Supplier, PO Date, Receive Date (or pre-computed LT column) |
| Open SO | SKU, Open Qty |
| Open PO | SKU, Order Qty, Delivered Qty, Closed Status |
| Inventory | SKU, On-Hand Qty |
| Time series *(optional)* | Item/SKU column + monthly period columns (wide format) |

## Configuration

| File | Controls |
|---|---|
| `config/stocking_policy.json` | Stocking tiers, service levels, CV thresholds, DOS excess threshold |
| `config/incoterm_rules.json` | EXW/FOB/DDP goods-in-transit counting rules |
| `config/supplier_incoterm.json` | Supplier ID → incoterm mapping |
| `config/node_config.json` | Location ID, currency — one entry per DC |
| `policy/policy.md` | Company-level hard constraints (human-readable, Claude-interpreted) |

## Output

```
output/<timestamp>/
├── inventory_report_<ts>.html           ← open this
├── purchase_recommendations_<ts>.csv
├── inventory_projection_<ts>.csv
├── forecast_detail_<ts>.csv
├── sku_planning_params.csv
├── supplier_params.csv
└── history/YYYY-MM/snapshot_<ts>.json  ← feedback loop input
```

## Project Structure

```
inventory_planning/
├── orchestrator.py
├── analytics/
│   ├── demand_classifier.py   CV + frequency → demand pattern
│   ├── forecaster.py          ETS / Croston / ARIMA / SMA
│   ├── safety_stock.py        MIT CTL combined variability formula
│   ├── inventory_projector.py DOS-based excess detection
│   └── purchase_recommender.py  t+1 net requirement
├── feedback/
│   ├── snapshot.py            auto-saves planning state after each run
│   ├── collector.py           records actuals against prior snapshot
│   └── loss.py                cumulative gap analysis + deviation attribution
├── readers/                   5 document readers + time series reader
└── reporting/                 charts + HTML report generator
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Requirements

- Python ≥ 3.9
- pandas, numpy, scipy, statsmodels, openpyxl, matplotlib, seaborn, jinja2
