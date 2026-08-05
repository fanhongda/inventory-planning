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

### Data intake as program synthesis, not schema matching

Getting a second ERP's data in took longer than analysing it. Splitting that pain
apart gave three different problems that had been sharing one code path:

| Failure | Example | Does an alias list solve it? |
|---|---|---|
| **Lexical** — same meaning, new name | `Bal. Qty` / `Outstanding Quantity` | Yes, but by unbounded enumeration |
| **Derivational** — the field isn't in the source | Open PO has no open qty; needs `order_qty − delivered_qty` | **No.** This is program synthesis |
| **Grain / value domain** — the row means something else | PO header vs line vs schedule line; `CLSD` / `C` / `99` / `Closed` | **No.** Needs declared grain and a value map |

The old `schema.py` alias dict handled only the first. The other two required editing
a reader for every new ERP.

So the knowledge moved out of code and into two data artefacts. A **contract** states
what a field means — its type, its grain, the expressions it can be *reconstructed*
from, and the invariants it must satisfy. An **adapter** states how one specific export
satisfies that contract. Adapters are generated once, verified by assertions, reviewed,
then frozen into git; from then on every run is deterministic with no model in the loop.

The verification half is what makes the generation half safe. Three layers of test run
on every load:

- **structural** — required fields present; declared grain matches an actual unique key
- **semantic** — `open_qty >= 0`, `open_qty <= order_qty`, lead time within a plausible band
- **reconciliation** — control totals, and distribution drift against the previous load

The grain test earns its place repeatedly. A PO *schedule line* export carries two or
three rows per PO line; summed without a rollup, every quantity inflates, the position
looks healthy, and the pipeline quietly stops recommending purchases — with no error
anywhere, because the numbers stay internally consistent.

### Should-be inventory, and why the lever ranking is computed

The planner's core operation is not "forecast then order" — it is "what *should* stock
be, and where is the gap". So should-be is the central object, decomposed by lever:

```
should-be = cycle(R, lot size) + safety(z, σ_D, σ_LT, R+LT) + pipeline(buyer-owned transit)
```

Two decisions in that formula change the answer more than the arithmetic does.

**Ownership boundary.** Goods the supplier has not shipped are on nobody's finished-goods
books; goods in transit are the buyer's only when the incoterm passes title at origin.
Counting a full `D·LT` of pipeline overstates the balance sheet, usually by a lot, since
production dominates lead time. Getting this right reverses the headline conclusion:

| Lever | Pipeline counted in full | Ownership boundary applied |
|---|---|---|
| Review 30d → 7d | −12% | **−38%** |
| σ_LT 15d → 5d | — | −12% |
| Mean LT 75d → 45d | **−29%** | −5% |

Mean lead time stops being the big lever, because it now acts only through the
√(R+LT) term. **Lead-time variability becomes several times more valuable than
lead-time speed** — which makes the supplier conversation about reliability, not
about going faster.

**Net, not gross.** A lever is ranked on annual net benefit, not on the cash it frees.
The stock reduction is one-off; the ordering cost it incurs repeats every year. On real
data "review weekly" routinely frees five figures of stock and destroys six figures of
value — a result the gross ranking gets exactly backwards.

EOQ lives here as a **constraint, not an objective**. Planners order to a review
cadence, not to EOQ; EOQ's job is to say when that cadence has passed the economic
point. So it appears in the output only when violated, and it is evaluated against the
cadence a lever *proposes*, not the one it starts from.

### On-time delivery, split four ways

OTD is measured, not modelled: a line is on time when it shipped on or before the
customer's request date. What earns it a module is that the failures are not one thing.

| State | Owner |
|---|---|
| On time | — |
| Shipped late | supply |
| Past due, no stock | supply — a genuine miss |
| **Past due, stock available** | **not supply** — the goods were there, uncollected |

That last row is the one that makes the metric credible. It is not a planning failure,
but counting it as one both overstates the supply miss and hides that the stock is
committed, aged and immobile. Two owners, two fixes; a combined number serves neither.
The report prints the fair reading beside the naive one and names the difference.

Request-date quality is measured rather than caveated. A request date equal to the
order date is "as soon as possible", not a requirement — OTD against it is unwinnable
by construction. Those lines, plus ones already past due when raised, are counted and
reported, and a clean-lines-only OTD is shown when it differs. If the two readings
diverge, the metric is measuring order-entry habits and the fix is upstream of planning.

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

Hand it the files in any order without saying what they are:

```python
planner = InventoryPlanner(output_dir="output/")
inputs  = planner.load_all(glob("exports/*"))
results = planner.run_planning(**inputs)
```

Each file is profiled, routed to a contract, transformed by an adapter, and verified
by contract tests before it reaches the analytics. CSV and Excel are both accepted.

### Requirements are capabilities, not filenames

| Capability | Required | Satisfied by |
|---|---|---|
| `demand_signal` | yes | pre-compiled demand time series **or** sales history |
| `position_signal` | yes | inventory snapshot |
| `lead_time_signal` | no | PO history |
| `inbound_signal` | no | open POs |
| `commitment_signal` | no | open sales orders |
| `cost_signal` | no | inventory **or** PO history |
| `order_pattern_signal` | no | PO history **or** open POs |
| `service_signal` | no | open sales orders |

A planner-supplied SKU × period matrix satisfies `demand_signal` by itself; the wide
layout is recognised from the header, so no separate loader has to be called. Missing
optional inputs do not fail the run — they produce a specific statement of what the
analysis can no longer tell you:

```
○ lead_time_signal     fallback: config default lead time, zero lead-time variance

What this run cannot tell you:
  • Safety stock loses its lead-time variability term and will be understated
  • Open PO arrival dates cannot be estimated when the export omits them
```

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
├── ingest_bridge.py           canonical frames → analytics column expectations
├── policy/                    ← should-be, levers, targets, decision loop
│   ├── parameters.py          planning_parameters.md parser + scoped rule engine
│   ├── assemble.py            one per-SKU attribute frame the whole layer reads
│   ├── should_be.py           cycle + safety + buyer-owned pipeline, incoterm-aware
│   ├── levers.py              per-SKU lever ranking on net annual benefit; EOQ guard
│   ├── target.py              hard-target action frontier with burn-down limits
│   ├── service.py             OTD measured, split four ways; request-date quality
│   ├── diagnostics.py         over-ordering, chronic air, stockout risk, slow burn
│   └── decisions.py           accept/reject log → constraint candidates
├── ingest/                    ← contract-driven intake (no imports from this package)
│   ├── contracts/*.yaml       canonical fields: type, grain, derivable_from, assertions
│   ├── adapters/*/*.yaml      how one ERP export satisfies a contract (frozen, versioned)
│   ├── expressions.py         AST-whitelisted evaluator for derivations + assertions
│   ├── profiler.py            deterministic portrait; wide/long + locale detection
│   ├── adapter.py             map → parse → value-map → derive → filter → roll up
│   ├── registry.py            fingerprint routing; drafts an adapter when none matches
│   ├── contract_tests.py      structural / semantic / reconciliation assertions
│   ├── capabilities.py        capability resolution + degradation reporting
│   └── intake.py              entry point: files in, canonical frames out
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
├── readers/                   legacy per-document readers (superseded by ingest/)
│
│   (outside the package, at repo root:)
│   skill/SKILL.md             Claude Code workflow — how to run it and what to relay
│   .github/copilot-instructions.md   VS Code + GitHub Copilot — invariants and layout
└── reporting/
    ├── kpi_report.py          two-chapter review: what happened / what is coming
    ├── charts.py              matplotlib charts for the planning report
    └── html_report.py         planning report generator
```

`ingest/` imports nothing from the rest of the package, so it can be lifted into a
shared `sc-canonical` distribution when the reconciliation skill needs the same
machinery. `ingest_bridge.py` is the only place that knows this pipeline's column names.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Requirements

- Python ≥ 3.9
- pandas, numpy, scipy, statsmodels, openpyxl, matplotlib, seaborn, jinja2
