# Inventory Planning — Domain Skill

> Part of a personal supply chain agent infrastructure designed to demonstrate how frontier AI can operate meaningfully in real-world logistics contexts.

A single-stage DC inventory planning pipeline grounded in MIT CTL MicroMasters supply
chain principles. It reads whatever exports a planner actually has, works out what
stock *should* be, ranks the actions that would close the gap, and says plainly which
of its numbers are measured and which are assumptions.

---

## Design Philosophy

### The bigger picture: supply chain agent infrastructure

This skill sits within a broader research direction: building AI agents that can reason
about supply chains not as optimisation puzzles, but as **socio-technical systems** —
where demand is noisy, organisations respond with lag, supplier capacity has hard
ceilings, and the gap between what a planner's KPI measures and what the supply chain
actually needs is often the root cause of failure.

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

This repo is the **domain skill layer**: grounded in classical supply chain theory,
designed to be used by a human expert, and built to accumulate learning over time
without blindly trusting its own outputs.

### The recurring principle: never agree with what you were shown

Most of the design decisions below are instances of one idea. A planning system earns
its place by disagreeing usefully with the organisation around it — and there are
several ways to lose that ability by accident:

- learning from an operator's non-execution until you recommend what they already do
- consuming a hand-set safety stock as an input, and then confirming it
- adding the forecast to the backlog, so the order book validates itself
- taking a master lead time as fact when the receipts say otherwise

Each is a place where an observable signal diverges from the true target, and where a
naïve pipeline reinforces the wrong behaviour. The modules that look like extra
complexity — deviation attribution, source authority ranking, forecast consumption,
backlog realization — exist to keep those four doors shut.

### Why deviation attribution matters

Most inventory planning tools treat the gap between recommendation and actuals as a
signal to adjust the model. That assumption is almost always wrong.

In real supply chains, a deviation between what the system recommends and what actually
happens can have at least three distinct causes:

| Cause | Example | Correct response |
|---|---|---|
| **Operator non-execution** | Buyer ignores buy signal due to budget freeze or warehouse space | Maintain recommendation. Flag to ops. Do not adjust model. |
| **Cumulative supply gap** | Demand exceeds supply for 3 consecutive months — supplier has no flex capacity | Escalate to supply chain review. Not a model error. |
| **Model bias** | Forecast systematically over-estimates for erratic SKUs | Investigate model. Consider parameter adjustment. |

The failure mode this guards against is **spurious learning**: a system that observes an
operator consistently buying less than recommended, then gradually lowers its
recommendations to match — effectively learning to sanction underperformance.

Real supply chain crises tend to share two structural features: **persistent cumulative
gaps** (supply falling behind demand for months without correction) and **supplier
capacity exhaustion** (when a buyer represents a large share of a supplier's output,
there is no buffer when demand accelerates). Both, especially combined with an external
shock, create the conditions for bullwhip-driven collapse.

The feedback module tracks cumulative supply and demand gaps across months,
distinguishes operator deviation from model error, and only suggests parameter changes
when the deviation cannot be explained by execution failure or supply constraint.

---

## What it does

| Step | Method | Output |
|---|---|---|
| Demand characterisation | Frequency + CV classification | stocking-high / stocking-med / non-stocking |
| Demand pattern routing | CV thresholds | smooth / intermittent / erratic / lumpy |
| Forecasting | ETS, ARIMA or Croston per pattern | 6-month forecast + t+1 point forecast |
| Safety stock | Combined demand × lead-time variability (MIT CTL §7) | SS = k · √((R+LT)·σ_fc² + μ²·σ_LT²) |
| Inventory projection | Days-of-supply excess detection | EXCESS / SHORTAGE-RISK / OK |
| Backlog realization | Share of the order book that actually ships, measured from uncollected past-due lines | per-SKU realization rate |
| Purchase recommendation | Forecast consumption: max(t+1 forecast, realizable backlog due in horizon) + SS | PURCHASE-REQUEST / ORDER-FOR-BACKLOG / PUSH-OUT / HOLD |
| Should-be inventory | cycle + safety + buyer-owned pipeline, incoterm-aware | gap vs actual, per lever |
| Lever ranking | Net annual benefit, not cash freed | ordered action list with what each is worth |
| Parameter suggestion | What the data supports vs what is in force | per-SKU CSV + paste-able rule blocks |
| Source cross-check | Measured vs ERP master vs planner worksheet | every material disagreement, with both values |
| Service attribution | OTD measured and split four ways | who owns each failure |
| Feedback loop | Cumulative gap attribution across months | OPERATOR_DEVIATION / SUPPLY_GAP / MODEL_BIAS |

---

## Decisions that change the answer

### Data intake as program synthesis, not schema matching

Getting a second ERP's data in took longer than analysing it. Splitting that pain apart
gave three different problems that had been sharing one code path:

| Failure | Example | Does an alias list solve it? |
|---|---|---|
| **Lexical** — same meaning, new name | `Bal. Qty` / `Outstanding Quantity` | Yes, but by unbounded enumeration |
| **Derivational** — the field isn't in the source | Open PO has no open qty; needs `order_qty − delivered_qty` | **No.** This is program synthesis |
| **Grain / value domain** — the row means something else | PO header vs line vs schedule line; `CLSD` / `C` / `99` / `Closed` | **No.** Needs declared grain and a value map |

So the knowledge moved out of code and into two data artefacts. A **contract** states
what a field means — its type, its grain, the expressions it can be *reconstructed*
from, and the invariants it must satisfy. An **adapter** states how one specific export
satisfies that contract. Adapters are generated once, verified by assertions, reviewed,
then frozen into git; from then on every run is deterministic with no model in the loop.

The verification half is what makes the generation half safe. Three layers run on every
load: **structural** (required fields present, declared grain matches an actual unique
key), **semantic** (`open_qty >= 0`, `open_qty <= order_qty`, lead time in a plausible
band), and **reconciliation** (control totals, distribution drift against the previous
load).

### Grain: the failure that produces contradictory advice

A warehouse export is one row per SKU × storage location — sellable stock in `01`,
quality quarantine in `02`. That is the honest grain to read at, and the contract
declares it. But planning here is single-node, and every join downstream is on `sku`
alone. Left unhandled, one SKU fans out into two planning rows that each match the
*same* open PO and the *same* backlog:

```
sku    on_hand  open_po  eff_pos  status          action
ABC-1     20      100      120    SHORTAGE-RISK   PURCHASE-REQUEST  (110)
ABC-1    500      100      600    EXCESS          PUSH-OUT-OPEN-PO  (100)
```

Pull in and push out, same item, same run — and the 100-unit PO counted twice. The
snapshot is now consolidated to one row per SKU before it reaches the analytics, the
merged location codes are kept in `stock_locations`, and the projector **raises** rather
than fanning out if it is ever handed a duplicated key. A silent version of this bug is
worse than a crash, because the numbers stay internally consistent.

Quarantine stock is currently summed into the available position, which overstates it.
Separating sellable from blocked needs a supply-chain topology the pipeline does not
have yet — see [TODO.md](TODO.md).

### Stated parameters are ranked, not merged

Planners keep an ERP **item master** (supplier, planned lead time, MOQ, standard cost)
and their own **planning worksheet** (the safety stock, min/max and lead time actually
in use). Both are optional inputs. Both will disagree with the transaction history and
with each other. Rather than picking one and moving on, sources are ranked by what kind
of claim they make:

```
measured         derived from this run's transactions   — what happened
item_master      a standing parameter in the ERP        — an intention
planning_master  a value a person maintains by hand     — a decision
config           a pipeline-wide default                — a guess
```

Higher authority wins, with one exception: a lead time measured from fewer than three
receipts is a coincidence, not a distribution, so it yields to a stated value. This
matters most for a SKU never bought through the export — it used to get lead time 0 and
therefore safety stock 0, the pipeline confidently recommending nothing for a
three-month-lead-time item.

Every material disagreement is recorded with both values, because a master lead time
that no longer resembles what suppliers deliver is a finding rather than a nuisance:
every MRP run in the ERP is planning on it.

```
SKU-003  lead_time_days  measured 46.1  vs  ERP item master 120.0  (+160%)
SKU-010  monthly_demand  measured 307.1 vs  planner worksheet 40.0  (−87%)
⚠ monthly_demand: 50% of comparisons disagree. That is not per-item drift — the two
  sources are measuring different things, and the definition is worth settling.
```

**The planner's own parameters are never consumed as inputs.** Safety stock is fully
determined by demand variability, lead time and a service level, so a hand-set value is
a position to be measured, not an input. A pipeline that consumed it would agree with
whatever it was shown.

### Forecast consumption, not forecast plus backlog

The obvious net requirement is `forecast + safety stock + backlog`. It over-orders twice
over.

The forecast is fitted on *shipment* history, and those shipments came from orders that
were open backlog before they shipped — so the two terms count the same demand twice,
and the error grows with how much backlog is carried, which is exactly when the buy
matters most. Second, it assumes the whole book converts. Where customers do not collect
on schedule, the order book is permanently larger than what will move, so the inflation
is a standing bias, not noise that averages out.

The requirement is therefore the larger of the two estimates, not their sum, with the
backlog side scoped to the lines due inside the horizon and discounted by a **measured**
realization rate:

```
gross = max(forecast_t+1, backlog_due × realization_rate) + safety_stock
```

The realization rate is not assumed. An open line past its request date **with the stock
on the shelf** is a line the customer has demonstrably not pulled; its share of the book
is the discount. Per-SKU rates are shrunk toward the portfolio rate by line count, so one
line cannot carry a confident 0%, and floored so that a collection problem cannot zero
out purchasing entirely. `demand_basis` in `config/stocking_policy.json` selects the
treatment; the legacy additive behaviour is still reachable.

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

Mean lead time stops being the big lever, because it now acts only through the √(R+LT)
term. **Lead-time variability becomes several times more valuable than lead-time
speed** — which makes the supplier conversation about reliability, not about going
faster.

**Net, not gross.** A lever is ranked on annual net benefit, not on the cash it frees.
The stock reduction is one-off; the ordering cost it incurs repeats every year. On real
data "review weekly" routinely frees five figures of stock and destroys six figures of
value — a result the gross ranking gets exactly backwards.

EOQ lives here as a **constraint, not an objective**. Planners order to a review cadence,
not to EOQ; EOQ's job is to say when that cadence has passed the economic point. So it
appears in the output only when violated, and it is evaluated against the cadence a lever
*proposes*, not the one it starts from.

### On-time delivery, split four ways

OTD is measured, not modelled: a line is on time when it shipped on or before the
customer's request date. What earns it a module is that the failures are not one thing.

| State | Owner |
|---|---|
| On time | — |
| Shipped late | supply |
| Past due, no stock | supply — a genuine miss |
| **Past due, stock available** | **not supply** — the goods were there, uncollected |

That last row is what makes the metric credible. It is not a planning failure, but
counting it as one both overstates the supply miss and hides that the stock is committed,
aged and immobile. Two owners, two fixes; a combined number serves neither. The report
prints the fair reading beside the naive one and names the difference.

Request-date quality is measured rather than caveated. A request date equal to the order
date is "as soon as possible", not a requirement — OTD against it is unwinnable by
construction. Those lines, plus ones already past due when raised, are counted and
reported, and a clean-lines-only OTD is shown when it differs. If the two readings
diverge, the metric is measuring order-entry habits and the fix is upstream of planning.

### Parameters are suggested, never applied

`config/planning_parameters.md` holds the parameters a planner has *decided*. The
suggestion engine produces the ones the data *supports*, so the two can be compared —
measured lead time and its sample count, the safety stock that lead time and forecast
error justify, the review period EOQ implies, the replenishment method the demand
pattern calls for.

The rules file it emits is not prose about rules; it is rules. The suggestion engine
*is* a rule set, declared in the same scope language `planning_parameters.md` uses and
validated by constructing the same objects its parser produces, so a generated block
pastes in and means what it meant here. Tests re-parse the output and assert it selects
the same SKUs with the same values.

Where a planner worksheet is supplied, each suggestion is expressed as a change from what
is in force, with the capital attached:

```
Against the safety stock the planner set by hand:
  planner holds more than justified     4 SKUs
  planner holds less than justified     5 SKUs

  SKU-009   set   700   justified     0   $105,000 tied up
  SKU-001   set 1,500   justified   680   $ 98,376 tied up
```

Nothing is ever written back to the parameter file. A parameter that changed because a
script decided it should is a parameter nobody can defend in a review.

### Smaller decisions

**Croston's method for intermittent demand.** ETS and ARIMA on sparse demand (6 active
months out of 12) produce systematically unstable forecasts. Croston separates demand
magnitude from inter-arrival frequency, giving stable estimates for slow movers.

**t+1 point forecast for purchase quantity.** The 6-month average underestimates in
uptrending demand. Net requirement uses `forecast_next_period`; the 6-month summary is
retained for horizon visibility only.

**Forecast RMSE as σ_DL for safety stock.** When a good forecast exists, the uncertainty
stock must absorb is the *residual*, not raw demand variability. Demand std dev is the
fallback, and the source is reported per SKU.

**Exposure period is R + LT, not LT.** Under periodic review a shortfall discovered just
after a review cannot be acted on until the next one. Using LT alone understates safety
stock by √((R+LT)/LT) — 41% for a monthly review with a 30-day lead time.

**Days-of-supply excess threshold.** EXCESS is flagged when DOS exceeds a configurable
threshold (default 90 days), not by an arbitrary percentage over ROP. A policy parameter,
not a model assumption.

**Anchored to the data, not the clock.** "Now" is the newest real date in the extract. An
export taken months ago scored against today marks every open line past due and drives
OTD to zero — a confident, entirely artificial answer.

---

## Quick Start

```bash
pip install -e ".[dev]"
```

Hand it the files in any order without saying what they are:

```python
from glob import glob
from inventory_planning import InventoryPlanner

planner = InventoryPlanner(output_dir="output/")
inputs  = planner.load_all(glob("exports/*"))      # profiled, routed, verified
results = planner.run_planning(**inputs)

policy  = planner.run_policy_analysis(
    results,
    inventory_df=inputs["inventory_df"],
    open_po_df=inputs["open_po_df"],
    item_master_df=inputs["item_master_df"],
    planning_master_df=inputs["planning_master_df"],
)
review  = planner.run_kpi_review(policy, sales_df=inputs["sales_df"], ...)
```

Or name the files explicitly:

```bash
inventory-plan \
  --sales            sales_history.xlsx \
  --po-history       po_history.xlsx \
  --open-so          open_so.xlsx \
  --open-po          open_po.xlsx \
  --inventory        inventory.xlsx \
  --timeseries       timeseries_3yr.xlsx    # optional pre-compiled wide-format TS
  --item-master      item_master.xlsx       # optional ERP master
  --planning-master  planner_sheet.xlsx     # optional planner worksheet
```

CSV and Excel are both accepted. Each file is profiled, routed to a contract,
transformed by an adapter and verified by contract tests before it reaches the analytics.

---

## Inputs

### Requirements are capabilities, not filenames

| Capability | Required | Satisfied by |
|---|---|---|
| `demand_signal` | yes | pre-compiled demand time series **or** sales history |
| `position_signal` | yes | inventory snapshot |
| `lead_time_signal` | no | PO history **or** item master **or** planner worksheet |
| `inbound_signal` | no | open POs |
| `commitment_signal` | no | open sales orders |
| `cost_signal` | no | inventory **or** PO history **or** item master |
| `order_pattern_signal` | no | PO history **or** open POs |
| `service_signal` | no | open sales orders |
| `item_dimension` | no | item master **or** planner worksheet |
| `planner_baseline` | no | planner worksheet |

A planner-supplied SKU × period matrix satisfies `demand_signal` by itself; the wide
layout is recognised from the header, so no separate loader has to be called.

Missing optional inputs do not fail the run — they produce a specific statement of what
the analysis can no longer tell you:

```
○ commitment_signal   fallback: backlog treated as zero

What this run cannot tell you:
  • Net requirement excludes backlog and will be understated
  • Past-due backlog cannot be measured, so on-time delivery is modelled rather than observed
```

---

## Output

```
output/
├── kpi_review_<ts>.html                 ← open this
├── parameter_suggestions_<ts>.csv       suggested parameters vs those in force, per SKU
├── suggested_rules_<ts>.md              the same, as paste-able planning_parameters.md rules
├── source_crosscheck_<ts>.csv           where two sources disagree, and by how much
├── purchase_recommendations_<ts>.csv
├── inventory_projection_<ts>.csv
├── backlog_realization_<ts>.csv         per-SKU realization rate and the evidence
├── forecast_detail_<ts>.csv
├── sku_planning_params.csv              persisted: stocking class, SS, ROP per SKU
├── supplier_params.csv                  persisted: WMA lead time per SKU × supplier
└── history/YYYY-MM/snapshot_<ts>.json   feedback loop input
```

---

## Configuration

| File | Controls |
|---|---|
| `config/planning_parameters.md` | Segmentation boundaries, conventions, scoped override rules — **the file a planner edits** |
| `config/stocking_policy.json` | Stocking tiers, service levels, CV thresholds, DOS excess threshold, `demand_basis`, backlog realization |
| `config/incoterm_rules.json` | EXW/FOB/DDP goods-in-transit counting rules |
| `config/supplier_incoterm.json` | Supplier ID → incoterm mapping |
| `config/node_config.json` | Location ID, currency — one entry per DC |
| `policy/policy.md` | Company-level hard constraints (human-readable, Claude-interpreted) |

`planning_parameters.md` is markdown with fenced YAML, not a config format, because the
knowledge of *which SKU gets which policy* is business judgment that changes far more
often than the arithmetic does. Every rule must carry a rationale — three months on, the
reason is the only thing that lets anyone judge whether the rule still applies. Every run
prints which rules hit which SKUs, and flags where two rules fight over one parameter.

---

## Feedback loop

After each run a snapshot is saved to `output/history/YYYY-MM/`. The following month,
once actuals are available:

```python
from inventory_planning.feedback.collector import FeedbackCollector
from inventory_planning.feedback.loss import LossCalculator

collector = FeedbackCollector("output/history/2026-06/snapshot_20260603_2335.json")
collector.record_actuals(actual_sales_df, actual_inventory_df)

result = LossCalculator("output/history/2026-06/snapshot_20260603_2335.json").compute()
```

```
  Forecast:  MD=+12.3  MAD=28.1  MAPE=18.4%  (over-forecast)
  Inventory: avg DOS realized=67d  excess rate=12%

  Cumulative gap analysis (3 months, threshold=3):
  ⚠  OPERATOR DEVIATION:   4 SKUs — model recommendation maintained
  🔴 SUPPLY RISK:          2 SKUs — escalate to supply chain review

  ► [OPERATOR_DEVIATION] SKU=XYZ-001
    Maintain system recommendation unchanged.
    Review with operations: storage constraints, budget freeze, or manual override?
    → No model parameter change.
```

---

## Project structure

```
inventory_planning/
├── orchestrator.py            run_planning → run_policy_analysis → run_kpi_review
├── ingest_bridge.py           canonical frames → analytics column expectations
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
├── policy/                    ← should-be, levers, targets, suggestions
│   ├── parameters.py          planning_parameters.md parser + scoped rule engine
│   ├── assemble.py            one per-SKU attribute frame the whole layer reads
│   ├── crosscheck.py          source authority ranking + disagreement reporting
│   ├── should_be.py           cycle + safety + buyer-owned pipeline, incoterm-aware
│   ├── levers.py              per-SKU lever ranking on net annual benefit; EOQ guard
│   ├── target.py              hard-target action frontier with burn-down limits
│   ├── suggestions.py         suggested parameters + paste-able rule blocks
│   ├── service.py             OTD measured, split four ways; request-date quality
│   ├── diagnostics.py         over-ordering, chronic air, stockout risk, slow burn
│   └── decisions.py           accept/reject log → constraint candidates
├── analytics/
│   ├── demand_classifier.py   CV + frequency → demand pattern
│   ├── forecaster.py          ETS / Croston / ARIMA / SMA
│   ├── safety_stock.py        MIT CTL combined variability formula
│   ├── inventory_projector.py DOS-based excess detection
│   ├── backlog_realization.py what share of the open order book actually ships
│   └── purchase_recommender.py  forecast consumption → net requirement
├── feedback/
│   ├── snapshot.py            auto-saves planning state after each run
│   ├── collector.py           records actuals against prior snapshot
│   └── loss.py                cumulative gap analysis + deviation attribution
├── readers/                   legacy per-document readers (superseded by ingest/)
└── reporting/
    └── kpi_report.py          two-chapter review: what happened / what is coming

  (at repo root:)
  skill/SKILL.md                     Claude Code workflow — how to run it, what to relay
  .github/copilot-instructions.md    VS Code + Copilot — invariants and layout
  principles/sc-principles.md        MIT CTL key concepts + formulas
  policy/policy.md                   company-specific hard constraints (blank — populate)
  AUDIT.md                           principles-vs-code audit log
  TODO.md                            known gaps, next structural work
```

`ingest/` imports nothing from the rest of the package, so it can be lifted into a shared
`sc-canonical` distribution when the reconciliation skill needs the same machinery.
`ingest_bridge.py` is the only place that knows this pipeline's column names.

---

## Known gaps

Tracked in [TODO.md](TODO.md). The structural one: **supply chain topology**. The
pipeline plans a single node and sums every storage location of a SKU into one position,
so quarantine and blocked stock read as available and the position is overstated by
exactly that much. Fixing it properly needs ERP node identifiers, stock status per
location, and upstream/downstream relationships between nodes — not a location whitelist.

---

## Tests

```bash
python -m pytest tests/ -q
```

236 tests, no network, no fixtures beyond `sample_data/`.

## Requirements

- Python ≥ 3.9
- pandas, numpy, scipy, statsmodels, openpyxl, pyyaml
