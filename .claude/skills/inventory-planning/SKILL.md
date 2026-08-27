---
name: inventory-planning
description: >
  Full distribution-centre inventory planning pipeline. Use this skill whenever
  the user mentions inventory planning, stock replenishment, safety stock,
  purchase recommendations, demand forecasting, or supply chain planning —
  even if they just say "run inventory planning", "plan my stock", "what should
  I buy", "what POs should I push out", or "check my inventory position".
  Also trigger when the user uploads or references any of: sales history,
  PO history, open sales orders, open purchase orders, or an inventory report,
  and wants analysis done on them. Do NOT wait for the user to name all five
  files — start the skill as soon as the intent is clear and ask for missing
  files during the confirmation step.
---

# DC Inventory Planning Skill

You are running a single-stage distribution-centre inventory planning pipeline.
The package is the `inventory-planning` repository this skill ships inside, so **every
path below is relative to the repository root — run from there.** All logic is in
`inventory_planning/`; do not rewrite it, orchestrate it.

> **Canonical location:** `.claude/skills/inventory-planning/SKILL.md`, inside the repo,
> so this document versions together with the code it describes. That path is not an
> arbitrary one: `.claude/skills/` is a default discovery directory for both Claude Code
> and VS Code's Agent Skills, so cloning the repo installs the skill and there is no
> per-machine step. The directory name has to stay equal to the `name` in the
> frontmatter — a mismatch makes the skill fail to load, silently.
>
> The same repo carries `.github/copilot-instructions.md` for VS Code + GitHub Copilot,
> which cannot read this file. The two are different documents for different consumers —
> this one is the workflow, that one is the invariants — but they describe one system.
> **Change a public entry point or an invariant and both need updating.**

## What this skill produces

1. **Stocking classification** — high-service / med-service / non-stocking per SKU,
   plus a suggested **MTS/MTO** for every SKU with a demand series, with the evidence
   (`suggested_stocking_policy`, `policy_basis`). It is compared against the ERP's
   policy and never overwrites it — relay the disagreements, both directions: held
   as MTO but demand recurs, and held as MTS but demand is sporadic.
2. **Safety stock** — combined demand × lead-time variability formula
3. **Inventory projection** — should-be vs current position (on-hand + GIT + open PO)
4. **6-month forecast** per SKU, plus `forecast_<run_id>.csv` — one row per SKU with its
   history and the six months that follow, side by side, and the model that produced
   them. For make-to-stock items the model is chosen by backtest; make-to-order items
   use Croston. `vs_naive` on the row says whether the model beat repeating last
   month — where it reads 1.00, it did not, and the forecast deserves no more trust
   than that.
5. **Purchase recommendations** — EXPEDITE-INBOUND / PURCHASE-REQUEST /
   ORDER-FOR-BACKLOG / PUSH-OUT-OPEN-PO / HOLD / NO-ACTION. **Relay every
   `EXPEDITE-INBOUND` first and by name.** It means the shelf runs dry before the
   next delivery lands, so the ask is a ship date confirmed with the supplier —
   daily — and a price for air freight, not a purchase order. The run prints the
   days of cover left, how long the shelf is bare, and how much is already past
   due; those three are what the buyer takes to the supplier.
6. **Backlog realization rate** — measured share of the open order book that actually ships
   and, where an item master or planner worksheet is supplied, a source cross-check and a
   per-SKU comparison against the parameters the planner set by hand
7. **Parameter suggestions** — what the data says the policy parameters should be, as a
   per-SKU CSV and as rule blocks that paste straight into `planning_parameters.md`
8. **KPI review** — two-chapter HTML: what happened and who caused it, what is coming
9. **CSV outputs** — supplier_params, sku_planning_params, projection, forecast, recommendations

---

## Inputs — think in capabilities, not file slots

The pipeline does not require a fixed list of five files. It requires **information**,
and each requirement can be satisfied by more than one document. Ingest identifies
every file automatically — do not ask the user to say which file is which.

| Capability | Required | Satisfied by | Without it |
|---|---|---|---|
| `demand_signal` | **yes** | pre-compiled demand time series **or** sales history | cannot run |
| `position_signal` | **yes** | inventory snapshot | cannot run |
| `lead_time_signal` | no | PO history **or** item master **or** planner worksheet | safety stock loses its lead-time variability term |
| `inbound_signal` | no | open POs | position excludes goods in transit; no push-out advice |
| `commitment_signal` | no | open sales orders | net requirement runs on the forecast alone; the backlog realization rate cannot be measured |
| `cost_signal` | no | inventory **or** PO history **or** item master | DIOH cannot be expressed in currency |
| `order_pattern_signal` | no | PO history **or** open POs | EOQ conformance not assessed |
| `service_signal` | no | open sales orders | on-time delivery modelled, not measured |
| `item_dimension` | no | item master **or** planner worksheet | suggested quantities not rounded to a real MOQ; obsolete items still replenished |
| `planner_baseline` | no | planner worksheet | cannot say whether the safety stock in use is above or below what the data justifies |
| `substitution_signal` | no | a substitution list — **only** this document merges anything | a part that changed number is planned as two items |

**A pre-compiled SKU time series fully satisfies `demand_signal` on its own.** When a
planner has already bucketed demand themselves, sales history is not needed — and the
wide period layout is detected from the header, so nothing special has to be called.

### The two master documents, and why they are treated differently

Planners routinely keep an **ERP item master** (supplier, planned lead time, MOQ,
order multiple, standard cost, lifecycle status) and their **own planning worksheet**
(the safety stock, min/max, review period and lead time actually in use, usually
alongside a few columns of usage history). Both are optional; both change what the run
can say. Hand them over with everything else — ingest identifies them.

They are not interchangeable, and the pipeline ranks them:

```
measured         derived from this run's transactions   — what happened
item_master      a standing parameter in the ERP        — an intention
planning_master  a value a person maintains by hand     — a decision
config           a pipeline-wide default                — a guess
```

Higher authority wins, with one exception: a lead time "measured" from fewer than
three receipts is not a distribution, so it yields to a stated value.

What this buys, in the order a planner cares about:

1. **Gaps get filled.** A SKU that has never been bought through this export gets a
   real lead time instead of a zero — and therefore a real safety stock. Those rows
   still carry no lead-time *variability*, so their safety stock is understated; the
   run says so and gives the count.
2. **Sources get cross-checked.** Where two sources disagree by more than 25%, both
   values are reported with the gap. A master lead time that no longer resembles what
   suppliers deliver is a finding in its own right — every MRP run in the ERP is
   planning on it. Written to `source_crosscheck_<run_id>.csv`.
3. **The planner's parameters get compared.** The safety stock, review period and
   service level in the worksheet are measured against what the data justifies, per
   SKU, with the capital or exposure attached. This is the main reason to load it.

**The planner's numbers are never consumed as inputs.** Safety stock is fully
determined by demand variability, lead time and a service level, so a hand-set value
is a position to be compared, never an input — a pipeline that consumed it would agree
with whatever it was shown. Say this if the user asks why their safety stock did not
"take".

### The third document: which item numbers are the same material

A part renumbered from 7100014 to 7100015 arrives as two SKUs, and every figure that
follows is wrong with no error anywhere — the old number holds stock and open POs
against a history that stops, so it reads as dead stock with a year of cover; the new
number has three months of history, is classified non-stocking, forecast flat, and
given almost no safety stock. **Ask for this list whenever the planner mentions a part
change, a phase-in, or an item that "used to be" something else.**

A file with two item columns is enough — old and new, one row per pair. Optional
columns, all of which change the answer:

| Column | What it does |
|---|---|
| relation | `supersede` (renumbering — merge everything) or `phase` (both numbers trade at once — annotate, never merge). Absent means `supersede`. |
| ratio | Quantity of the new number equal to one of the old. **Ask for it explicitly whenever the pack size or UoM changed** — the default of 1 is silently wrong by an order of magnitude if it did. |
| effective_date | When the switch happened. Not needed to merge; it is what lets the run test the claim. |
| rationale | Why these two are the same material. Required by convention, for the same reason every rule in `planning_parameters.md` carries one. |

**Check the ERP before asking anyone to type it.** SAP already models this on the
material's plant view — follow-up material `MARC-NFMAT`, effective-out date
`MARC-AUSDT`, discontinuation indicator `MARC-KZAUS`. An item master carrying those
columns saves the retyping, but it does **not** merge anything on its own: the run
reads it and prints the pairs as a substitution list to check and hand back. Relay that
list and ask whether it is right — that is the confirmation, and it is deliberate. The
master was handed over for lead times and MOQs, that column is maintained years before
anyone plans on it, and merging item numbers restructures every document in the run.

What the run does with it, and what it will not do:

- Every occurrence of the old number in **every** document is rewritten before anything
  is computed, and rows that become duplicates are recombined. Quantities add; a master
  row does not — two 90-day lead times are not a 180-day lead time, so the successor's
  master row wins and the predecessor's only fills its gaps.
- **Pairs are never inferred.** Do not offer to detect them from the data. Pairing a
  SKU going to zero with one ramping up would be right often and wrong silently, and a
  wrong pair adds two unrelated materials' stock and history together and produces a
  complete, self-consistent, wrong report.
- **Only a substitution list merges anything.** Not an item master, not a transfer
  document, not a thin-margin route. If a run merged nothing and the planner expected
  it to, the answer is always in the `Material supersessions` block — read it rather
  than guessing. Never work around the gate by hand-editing frames after `load_all`.
- A **split** (one number, two successors) or a **loop** is dropped and reported, never
  resolved by choosing. The rest of the map still applies.
- Declarations are **challenged, not trusted**: an old number still transacting after
  its own effective date is reported. Relay that one — it usually means the pair is
  really a phase pair and a live material has just been folded into another.
- `output/<ts>/supersessions_<run_id>.csv` records what each old number contributed to each
  document. **Quote it when reporting on a merged SKU**: the stock is still on the shelf
  under the old label and the open POs are still raised against it, so a buyer told to
  order 400 needs to know how much of the cover is sitting under the number they will
  not find in the report.

`phase` pairs are read, counted and otherwise inert today — coexisting materials are
not merged, and the phase-in/phase-out handling is not built yet. Say so rather than
implying the annotation changed a recommendation.

Accepts CSV or Excel (.xlsx / .xls / .xlsm).

**If the intake summary shows a file on the wrong doc_type**, or prints `⚠ Routed on a
thin margin`, run `python -m inventory_planning.explain <file-or-folder>` and relay the
per-contract breakdown. A misroute passes every contract test — the arithmetic is
correct on the wrong premise — so `OK` in the summary is not evidence the file is what
the router thinks. Check the routing line before anything else; a PO history filed as
an open PO silently removes the measured lead time.

---

## Step-by-step execution

### Step 1 — Collect whatever the user has

Ask for the files they have; do not enumerate five specific documents. If files are
already attached, use them. Only the two required capabilities need chasing — and only
name the *information*, not a filename:

> "I have demand and stock. I don't have purchase history — without it I'll use an
> assumed lead time and safety stock will be understated. Do you have a PO extract?"

### Step 2 — Run intake and report what it found

```python
import sys
sys.path.insert(0, '.')          # repository root; run from there
from inventory_planning.orchestrator import InventoryPlanner

planner = InventoryPlanner(
    output_dir='output/<YYYYMMDD_HHMM>',
    interactive=False,
)
inputs = planner.load_all([<every file path the user gave, in any order>])
```

`load_all` profiles each file, routes it to a contract, applies an adapter, and runs
contract tests. It prints an intake summary. **Read it and relay three things:**

1. **What each file was identified as**, and at what confidence. If a file was routed
   to something surprising, say so before continuing.
2. **Contract test failures.** A `FAILED` status means the data contradicts the
   contract — a negative outstanding quantity, a grain finer than declared, a lead
   time beyond 400 days. Show the failure and ask before proceeding.
3. **Degradations** — the "What this run cannot tell you" block. These are not
   warnings to bury; they are the honest limits of the analysis and belong in the
   final summary.
4. **Key warnings** — "keys on something the other documents do not use", or "keyed on
   only N distinct item number(s)". Every join in the pipeline is on `sku`, so a key
   that is wrong produces a complete report full of zeroes and no error anywhere.
   **Stop and fix the mapping before running.** The warning names the column that
   would have joined instead; Step 2b is how to apply it.
5. **The `Material supersessions` block**, when a substitution list was supplied. Say
   which numbers were merged into which before quoting any figure about them — the
   report will name only the survivor, and the planner is looking for the other one.
   Relay any challenge or dropped pair under it verbatim; both mean the map, not the
   data, needs a decision.

Do **not** walk the user through column-by-column mapping confirmation. The mapping is
inferred, tested and logged. Surface it only when a test fails, confidence is low, or
one of the key warnings above appears.

On SAP exports a `required_field:sku` failure usually means the guard did its job
rather than that anything is missing: `Item` is SAP's line number in every module, it
is refused as the item key, and the material is in a column the aliases did not
recognise. Look at the headers, find the material column, and map it in an adapter.

### Step 2b — When routing or mapping is wrong

Inspect the provenance, which records every transform:

```python
print(planner._intake.get('open_po').explain())
```

To correct a mapping, edit the drafted adapter and freeze it so the fix is permanent
for that source:

```python
from inventory_planning.ingest.registry import AdapterRegistry
reg = AdapterRegistry()
adapter = planner._intake.get('open_po').adapter
adapter.column_map['sku'] = 'Material'      # the correct source column
adapter.status = 'verified'
reg.save(adapter)     # -> ingest/adapters/<tenant>__<system>/open_po.v1.yaml
```

Once saved, that source is recognised by fingerprint on every future run and the
mapping never has to be confirmed again. **This is the mechanism that makes the second
import of an ERP fast — always offer to freeze the adapter after a successful run.**

### Step 3 — Incoterm confirmation

Before running, ask:
> "Do any of your suppliers use EXW or ExWorks terms? If yes, which supplier IDs?"

Update `config/supplier_incoterm.json` with the answer (already pre-populated for SIM001 and SRM005 as EXW).
If the user doesn't know, default to FOB and note this in the report.

### Step 4 — Run the pipeline

`load_all` already returned exactly the arguments `run_planning` needs:

```python
results = planner.run_planning(**inputs)
```

<details>
<summary>Legacy path — one loader per named file</summary>

Still supported for a single known file, but it requires the caller to know which
document is which and which loader a pre-aggregated series needs. Prefer `load_all`.

```python
sales_df,   _ = planner.load_sales_history('<path>')
po_hist_df, _ = planner.load_po_history('<path>')
open_so_df, _ = planner.load_open_so('<path>')
open_po_df, _ = planner.load_open_po('<path>')
inv_df,     _ = planner.load_inventory('<path>')
ts_pivot, ts_meta, _ = planner.load_timeseries('<path>', rolling_months=36)

results = planner.run_planning(
    sales_df, po_hist_df, open_so_df, open_po_df, inv_df,
    timeseries_pivot=ts_pivot, timeseries_meta=ts_meta,
)
```
</details>

### Step 4b — Policy analysis (should-be, levers, targets)

This is what makes the output an analysis rather than a report. Run it whenever the
user asks any of: *how much stock should we have? / where's the excess? / how do I hit
an inventory target? / should we review more often? / air or sea?*

```python
policy = planner.run_policy_analysis(
    results,
    inventory_df=inputs["inventory_df"],
    open_po_df=inputs["open_po_df"],
    target_value=5_000_000,          # optional — only for a hard target
    deadline=date(2026, 12, 31),     # optional
)
```

Produces three things, in order:

**`should_be`** — what stock ought to be, split into `cycle` / `safety` /
`pipeline (buyer-owned)`. The split is the point: each part answers to a different
lever. Compare to actual; the gap is the analysis.

Pipeline counts **only buyer-owned goods in transit**, decided per SKU by incoterm.
Goods the supplier has not shipped are on nobody's books. Actual is measured on the
same boundary — measuring the two sides differently manufactures a gap that is not
there.

**`levers`** — what each action is worth, ranked on **net annual benefit**, not on
stock freed. The stock reduction is one-off; the ordering cost repeats every year.
Expect the ranking to contradict intuition, and relay it:

- Reviewing weekly frees cash but often costs far more in ordering than it frees.
- Cutting lead-time **variability** beats cutting lead-time **mean**, usually several
  times over — because once pipeline is on the ownership boundary, mean LT acts only
  through the √(R+LT) term. **The supplier ask is reliability, not speed.**
- The best lever differs per SKU. Report `best_lever_per_sku()`, not just the total.

Levers whose cost cannot be priced from the data (freight premium, service loss) are
marked `unpriced` rather than assumed free. Say so — do not present them as wins.

**`frontier`** (when `target_value` is given) — the ordered set of moves, free and
reversible first, service-affecting last. Read out four things:

1. The **sufficient set** — how far down the list they need to go, and the worst
   service risk it incurs
2. **Burn-down limits** — excess only becomes cash as fast as demand consumes it. A
   theoretical $666k of excess may only yield $353k by December.
3. **Too slow** — a supplier reliability programme is the right move on the wrong
   horizon. Start it, but it does not count toward this target.
4. **Exclusions** — what was deliberately not recommended, and why. The user needs
   this to answer "did you consider…".

If the target is unreachable, say so plainly with the shortfall. Do not pad the list
with service cuts to make it close on paper.

### Step 4bb — KPI attribution and forward risk (the report)

```python
kpi = planner.run_kpi_review(
    policy,
    sales_df=inputs["sales_df"], open_so_df=inputs["open_so_df"],
    open_po_df=inputs["open_po_df"], inventory_df=inputs["inventory_df"],
    po_history_df=inputs["po_history_df"],
)
```

Writes a two-chapter HTML report and returns the underlying objects.

**Chapter 1 — what happened.** OTD is *measured*: a line is on time when it shipped
on or before the customer's request date. Failures split four ways, and the split is
the point:

| State | Whose problem |
|---|---|
| On time | — |
| Shipped late | supply |
| Past due, **no stock** | supply — a genuine miss |
| Past due, **stock available** | **not supply.** The goods were there; the customer has not collected |

**Always relay both readings.** The report prints the fair OTD and what a naive count
would have shown. That difference is stock sitting in the warehouse being blamed on
planning — it belongs in the inventory-efficiency conversation, with a different owner.

Request-date quality is measured, not caveated: lines with request date = order date
("as soon as possible"), lines already past due when raised, and lines with no date at
all are counted and reported. If under 70% are clean, say plainly that the metric is
indicative and the fix is order-entry discipline, upstream of planning.

Also in chapter 1: which SKUs drive the excess capital, over-ordering (bought more days
of supply than any policy justifies), chronic air freight (a planning gap paid for at
the freight desk), and erratic lot sizing.

**Replenishment cadence** sits next to those and answers what lot-size CV cannot: did
the buying keep pace with the selling, and how many orders were spent doing it. The
monthly `po_qty − sales_qty` balance is accumulated across the window; a curve that
returns to ≈ 0 means the two run rates match, which is a stronger statement than any
single month's accuracy. Four readings come out of its shape — **controlled**,
**chasing** (persistently behind), **losing_control** (persistent positive bias, so
nothing is closing the loop), **oscillating** (over-correcting, bullwhip at SKU scale) —
plus **unearned_frequency**.

Relay that last one carefully, because it is the one a planner can act on this week:
a monthly review entitles them to twelve orders a year, and ordering above the cadence
is worth paying for on a critical part and on nothing else. Items ordered far above
their cadence *and* not A-class are listed with the annual ordering cost it wastes, and
the fix is a longer review period rather than a better forecast.

Read the notes under the section before quoting its direction. If a share of demand or
purchase rows carry no usable date, the balance is biased by that much and the section
says so — the imbalance is in the ingest, not the buying, and the date mapping has to
be fixed first.

**Chapter 2 — what is coming.** Stockout risk ranked by *demand value at risk* — a
stockout costs sales, not inventory — with whether an inbound PO actually covers it or
lands too late. Then slow burn: stock with over a year of cover, flagging any that
**still has inbound POs**, which are the first push-out or cancel candidates.

Then the work list, which is what the planner actually acts on: **materials to act on**
(the recommendations that are not HOLD, ranked by the money the action moves) and
**open POs to change** (push out or cancel where the stock it lands in already has a
year of cover; pull in where the SKU runs out before it arrives). Both are read off the
same forward projection as the sections above, so the three cannot disagree.

Last, **parameter suggestions** — what the data supports next to what is in force, with
the capital the safety-stock change would move. Relay these as proposals, never as
findings: the pipeline cannot see why a parameter was set by hand, and a review period
stretched to match a supplier's shipping window looks identical to a careless one.

If OTD cannot be measured (no sales history carrying `ship_date` and
`customer_request_date`), the report says so rather than printing a blank. Do not
present a missing metric as a passing one.

### Step 4c — Record the decision (closes the loop)

After the user reacts, record it — **especially rejections**:

```python
from inventory_planning.policy.decisions import DecisionLog, decisions_from_frontier
log = DecisionLog(planner.output_dir.parent / "decisions.jsonl")
log.record_many(decisions_from_frontier(run_id, policy["frontier"], {
    "push_out_po": {"decision": "rejected",
                    "reason_code": "supplier_wont_reschedule",
                    "note": "V100 契约锁定排产"},
    "stop_buying": "accepted",
}))
```

A rejection carries more information than an acceptance: it names a constraint that
exists in no ERP extract. After a few runs, `log.summary(attributes)` surfaces
repeated patterns as **constraint candidates** with a paste-ready rule stub for
`planning_parameters.md`. Offer to add the strongest.

Never assume silence is agreement — unrecorded actions stay unrecorded.

### Step 5 — Present results

After the pipeline completes, tell the user:

1. **Open the KPI review**: `open <output_dir>/kpi_review_<timestamp>.html`
2. **Summary headline**: show the PURCHASE RECOMMENDATIONS SUMMARY block
3. **Top 5 actions** — most urgent purchase requests and biggest push-out candidates
4. **Data quality warnings** — flag anything the user should investigate (e.g. missing incoterms, short history, SKUs with no LT data)

Use this summary template:
```
✅ Planning complete — <N> SKUs analysed

📊 Report: <path_to_html>

📐 Parameter suggestions: <path_to_suggested_rules_md>
   (nothing is applied — these are proposals for the planner to review)

🛒 Action summary:
  • Purchase requests  : <N> SKUs  (<total_qty> units)
  • Order for backlog  : <N> SKUs
  • Push-out open POs  : <N> SKUs  (<total_qty> units to defer)
  • On target / hold   : <N> SKUs

⚠️  Notes to review:
  • <any quality warnings>
```

---

## Configuration

All tuning lives in `config/` — no code changes needed:

| File | What it controls |
|------|-----------------|
| `stocking_policy.json` | Frequency thresholds (9/12, 6/12), service levels (95%, 90%), forecast horizon |
| `incoterm_rules.json` | EXW/FOB/DDP GIT counting rules per incoterm |
| `supplier_incoterm.json` | Supplier ID → incoterm mapping (when PO file lacks incoterm column) |
| `node_config.json` | Location ID, reporting currency — set `location_id` for each DC |
| `fx_rates.json` | Rates into the reporting currency, effective-dated. A code absent here is reported unvalued, never counted at face value — check the report's currency callout before quoting any total |

**Planning policy lives in `config/planning_parameters.md`** — segmentation boundaries,
calculation conventions, and scoped overrides, in markdown a planner edits directly:

```markdown
### R-002 · Actuator 类周度 review 且提高服务水平
scope: product_family == "actuator"
set:  {review_period_days: 7, service_level: 0.98}
rationale: 涉及数据中心业务,客户停机成本远高于我方持有成本
```

Every rule **must** carry a rationale — it is what lets anyone judge later whether the
rule still applies, and what lets the system ask if a rule's scope should extend to a
newly-appeared segment. Rule hits are audited every run: which rule changed which
parameter on which SKUs, and where a later rule overrode an earlier one.

Two conventions in that file change every number downstream, so state which is in force
when presenting results:

| Convention | Options | Effect |
|---|---|---|
| `cycle_stock_basis` | `peak` (D·R) / `average` (D·R/2) | Factor of two on cycle stock. `peak` matches a target-ceiling discipline; `average` matches a DIOH snapshot. |
| `safety_stock_exposure` | `review_plus_lt` / `lt_only` | Using LT alone understates safety stock by √((R+LT)/LT) — 41% at monthly review with a 30-day LT. |

Data-shape knowledge lives in `inventory_planning/ingest/` and is also data, not code:

| Location | What it holds |
|---|---|
| `ingest/contracts/*.yaml` | Canonical fields per document: type, grain, `derivable_from`, `never`, assertions, value domains. Change these to change what the pipeline *means* by a field. `substitution.yaml` is the one that declares identity rather than content — which item numbers are the same material. |
| `ingest/adapters/<tenant>__<system>/*.yaml` | How one specific export satisfies a contract. Add a file here to support a new ERP — no code change. |

**Adding a new ERP is a YAML file, not a code change.** Run once, review the drafted
adapter, correct it, freeze it.

---

## Common issues & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| All SKUs = non-stocking | < 6 months of history | Request 24-month sales extract or use pre-compiled TS |
| Forecast model = SMA (flat) | < 8 periods | Same — more history needed |
| 0 GIT counted | Incoterm defaults to DDP/DAP | Confirm incoterm with user and update `supplier_incoterm.json` |
| SKU missing from projection | In time series but not in inventory file | SKU has no current stock — effective position = 0 (correct) |
| No committed delivery in open PO | ERP doesn't export ETA | Estimated from order_date + WMA LT; flagged as `delivery_estimated` |

### Intake-specific failures

| Test failure | What it means | Fix |
|---|---|---|
| `grain:<x>` — rows share a natural key | Source is finer than the contract (PO schedule lines, WMS bins). Left alone, every quantity is inflated and nothing downstream notices. | Intake proposes `rollup_to` automatically — verify it in the adapter notes before freezing |
| `required_field:<x>` absent | No column matched and no derivation applied | Add a `column_map` entry, or a `derivations` expression, to the adapter |
| `required_field:sku` absent on an SAP export | Usually the guard working, not a gap. `Item` is SAP's line number in every module — VBAP-POSNR, EKPO-EBELP, MSEG-ZEILE, QMFE-FENUM — so it is refused as the item key, and the real material column is spelled something the aliases do not list | Read the headers, find the material column, map `sku` to it in the adapter. Do **not** map `sku` to `Item` to make the error go away — that is the failure the refusal exists to prevent |
| `⚠ keyed on only N distinct item number(s)` | The item key holds a series (10, 20, 30 …), not identifiers — a line number reached `sku` under a header that looked right, or a frozen adapter points at the wrong column | Stop. The note names the columns that would have joined. Fix the adapter before running: every join is on `sku`, so the report would otherwise be complete, zero-filled and silent |
| `open_qty >= 0` violated | `order_qty` mapped to the wrong column, or over-receipts in the source | Check the mapping first; genuine over-receipt should be clamped in the adapter |
| `committed_delivery >= order_date` violated | Date order misparsed (DMY read as MDY) | Set `parsing.dayfirst` in the adapter; intake usually detects this from the data |
| `⚠ MIXED FORMATS` note | One source column holds two conventions — typically a spreadsheet opened under a locale that disagreed with the file, which converted the values it could read (swapping month and day) and left the rest as text. Whichever parser wins, the other half nulls out silently. | **Relay this to the user every time.** Where evidence was conclusive the parser restored the swapped values — check the transform log for `swapped` — but the source file is still wrong and the next export will be too. The fix is upstream: export dates in ISO, or as a real date column rather than text |
| `⚠ <file> is not UTF-8; it was read as <codec>` | The file is in a legacy codepage and the codec was inferred. Chinese, Japanese and accented names depend on that inference being right | Check a few supplier names in the output. If they look wrong, re-export as UTF-8 — the guess cannot be made reliable, only visible |
| `⚠ location_id is 'DC-01' … placeholder` | Neither the export nor `config/node_config.json` names a warehouse, so every row carries the shipped placeholder | Set `location_id` in `config/node_config.json`, or map the source's plant column. Do not report the location as if it were data |
| `distribution_drift:<col>` | A column's *meaning* changed while its name did not — a UoM, currency or filter change at the source | Do not just re-run. Confirm with whoever produces the extract |
| Unmapped status codes warning | This source uses a status vocabulary the value domain doesn't cover | Add them to `value_maps` in the adapter — an unmapped closed code counts closed POs as live supply |

### Data never leaves the machine

The profiler sends only metadata — column names, inferred types, ranges, null rates,
and character patterns (`AAA-9999`). Raw values are withheld unless `include_samples=True`
is passed explicitly. Keep it that way when working with company data.

---

## Multi-echelon note

All outputs carry `location_id`. When expanding to multi-echelon:
1. Update `node_config.json` with child/parent node relationships
2. Run one `InventoryPlanner` per DC node
3. A future `network_optimizer` skill will aggregate across nodes

---

## Output file reference

```
output/<timestamp>/
├── kpi_review_<run_id>.html             ← self-contained review (open this)
├── parameter_suggestions_<run_id>.csv   ← suggested parameters vs those in force, per SKU
├── suggested_rules_<run_id>.md          ← the same as paste-able planning_parameters.md rules
├── source_crosscheck_<run_id>.csv       ← where two sources disagree, and by how much
├── supersessions_<run_id>.csv           ← old number -> new, and what each contributed to
                                       each document. Written only where a renumbering
                                       was declared and matched something.
├── purchase_recommendations_<run_id>.csv
├── inventory_projection_<run_id>.csv
├── backlog_realization_<run_id>.csv     ← per-SKU realization rate and the evidence
├── forecast_detail_<run_id>.csv
├── sku_planning_params.csv          stocking class, SS, ROP per SKU
├── supplier_params.csv              WMA LT per SKU×supplier
└── history/YYYY-MM/snapshot_<run_id>.json
                                     ← the durable record: plan + the lead time it
                                       planned on. Read back by the feedback loop and
                                       by feedback.drift for lead-time movement.
```
