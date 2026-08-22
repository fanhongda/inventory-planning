# Inventory planning — repo instructions

Distribution-centre inventory planning: ingest messy ERP exports, forecast demand, size
safety stock, and answer what stock *should* be versus what it is.

## Running it

```python
import sys; sys.path.insert(0, '.')
from inventory_planning.orchestrator import InventoryPlanner

planner = InventoryPlanner(output_dir='output/run1', interactive=False)

inputs  = planner.load_all(sorted(Path('/path/to/exports').iterdir()))  # any order, undeclared
results = planner.run_planning(**inputs)
policy  = planner.run_policy_analysis(results,
                                      inventory_df=inputs['inventory_df'],
                                      open_po_df=inputs['open_po_df'],
                                      target_value=5_000_000)           # optional
kpi     = planner.run_kpi_review(policy, sales_df=inputs['sales_df'],
                                 open_so_df=inputs['open_so_df'],
                                 open_po_df=inputs['open_po_df'],
                                 inventory_df=inputs['inventory_df'],
                                 po_history_df=inputs['po_history_df'])
```

**There are two read paths and they must agree.** `ingest/` (contracts + adapters, via
`load_all`) is the one to change; `readers/` + `schema.py` is the legacy per-document
path, still reachable through `load_sales_history()` and friends. A fix applied to one
and not the other is how `Item` went on being read as the SKU for months after
`ingest/` had learned better — so `readers/` now imports the contracts' rules rather
than keeping its own copy. Do not restate a rule in `schema.py` that a contract states.

`load_all` identifies each file itself — do not ask which file is which, and do not
reintroduce per-document loader calls. The legacy `load_sales_history()` style still
works but requires the caller to know the answer already.

Outputs land in `output_dir`: `kpi_review_*.html` is the review, plus CSVs
(`purchase_recommendations_*`, `parameter_suggestions_*`, `inventory_projection_*`,
`forecast_detail_*`, `supplier_params`, `sku_planning_params`, `supersessions_*` where
material numbers were merged) and `suggested_rules_*.md`. There is no longer an
`inventory_report_*.html` — the older forecast-oriented report was removed, and three
skill evals went on asserting its existence for months afterwards.

The review covers, in order: service (OTD split four ways, plus OTD by month), inventory
against policy, ordering behaviour, replenishment cadence, forward risk, then the work
list — materials to act on, open POs to change, and parameter suggestions.

```bash
python3 -m pytest tests/ -q        # 668 tests, all should pass
```

Production runs **pandas 2.x**; the dev venv here is **pandas 3.x** and `pyproject.toml`
allows both. They differ in ways that fail silently — datetime resolution (`us` vs `ns`,
so `.astype("int64")` on a date column is out by 1000×), `NaT` stringification, and
`np.where` on typed columns. Test a fix under both before claiming it works.

## Where knowledge lives — data, not code

Supporting a new ERP or changing a planning rule should not require a code change.

| Location | Holds |
|---|---|
| `inventory_planning/ingest/contracts/*.yaml` | what each canonical field *means*: type, grain, `derivable_from`, `never` (headers a field must not take under a given ERP), assertions, value domains, `discriminator` |
| `inventory_planning/ingest/adapters/**/*.yaml` | how one specific export satisfies a contract; drafted on first run, reviewed, then frozen |
| `config/planning_parameters.md` | segmentation boundaries, calculation conventions, scoped policy overrides — every rule carries a rationale |
| `config/*.json` | service levels, incoterm rules, node config |
| `config/fx_rates.json` | exchange rates into the reporting currency, effective-dated. A currency absent here is reported unvalued, never counted at face value |

If you find yourself adding an `if` for a particular customer's column name or business
rule, it belongs in one of those files instead.

## Invariants — do not break these when editing

These are not style preferences. Each one was a bug that produced confident wrong
numbers, and each is covered by tests.

**Pipeline stock counts only buyer-owned goods in transit**, decided per SKU by
incoterm (`should_be.py`). Goods the supplier has not shipped are on nobody's books.
Actual stock is measured on the *same* boundary — measuring the two sides differently
manufactures a gap that is not there.

**Excess is gross, never netted against shortfall** (`ShouldBeResult.excess_value`).
Overstock on one SKU cannot offset a shortage on another; the units are not
interchangeable and neither is the cash. Netting once reported "$0 excess" for a
warehouse holding six figures of it.

**Levers rank on net annual benefit, not on stock freed** (`levers.py`). The balance
sheet reduction is one-off; the ordering cost it incurs repeats yearly. Ranking on the
gross figure systematically over-recommends frequent ordering.

**EOQ is a constraint, not an objective.** Planners order to a review cadence; EOQ's job
is to say when that cadence has passed the economic point. It appears only when
violated, and is evaluated against the cadence a lever *proposes*, not the current one.

**Safety stock exposure is R + LT**, not LT (`should_be.py::_safety`). Using LT alone
understates it by √((R+LT)/LT) — 41% at monthly review with a 30-day lead time.

**OTD is measured over completed deliveries only** (`service.py`) — the *headline*
rate. An open line past its request date could never have been on time, so including
open lines drives the rate toward zero by construction. Past-due backlog is reported as
its own metric.

**The monthly OTD series deliberately does not follow that rule** (`monthly_otd`). It
buckets on the **request** date and its denominator is every line due that month,
open ones included: of what the customer asked for in July, what share went out on or
before the date. Ship-date bucketing answers a throughput question instead — a line
requested in March and shipped in July lands on July, and one requested in July that
never shipped lands nowhere. The series reads lower than the headline, on purpose.
**Nothing past `as_of` is drawn**: under request-date bucketing the order book supplies
months that have not happened yet.

**A line past due with stock on the shelf is not a supply failure.** It is an
inventory-efficiency problem with a different owner. Never fold it into the OTD miss
rate — that blames planning for a collection problem and hides idle stock at once.

**"Now" is the newest date in the data, not `date.today()`.** A seven-month-old extract
scored against the clock marks every open line past due.

**Absence of a column is not evidence for a document type** (`registry.py`). A missing
receipt-date column made `is_null(receive_date)` vacuously true, so a sales export
scored 100% as an open PO.

**`Item` is a line number in SAP and a material almost everywhere else** (`profiler.py
::detect_source_system`, `contracts/*.yaml` `never:`). VBAP-POSNR, EKPO-EBELP,
MSEG-ZEILE and QMFE-FENUM all render as `Item`, and no data shape resolves the clash —
so the source system is decided once, from headers and SAP's zero-padded keys, and the
contract declares what the dialect forbids. Every transactional contract also has a
line-number field so `Item` has somewhere to go, and a column holding 10, 20, 30 is
refused as an item key even in the rescue pass for a required field. **Never "fix" a
`required_field:sku` failure by mapping `sku` to `Item`.** Leaving it unmapped stops the
run; mapping it produces a complete report of zeroes and no error at all.

**A renumbered material becomes one SKU at intake, in every document at once**
(`ingest/supersede.py`, called from `intake.py::_apply_supersessions`). A part that was
7100014 and is now 7100015 is one planning problem arriving as two: the old number
holds stock against a history that stopped, the new one has too little history to
forecast or to stock, and both readings are confident and wrong. The rewrite therefore
happens before any analysis and before every cross-document check — and a *partial*
rewrite is worse than none, because every join here is on `sku`, so a SKU can end up
with a forecast in one document and its position in another. After the rewrite, rows
that now share the contract's natural key are recombined, or the grain the pipeline
just broke double-counts everything downstream.

**Two rows for one material combine by what the key says they are.** A natural key of
`[sku]` alone means one row per material of its *standing attributes*, so the successor
row wins and the predecessor only fills its gaps — adding them makes a 180-day lead time
out of two 90-day ones. Every other key carries an event or a period, so quantities add;
per-unit money is quantity-weighted, never summed and never flat-averaged.

**Substitution pairs are declared, never inferred.** A rule pairing a SKU going to zero
with one ramping up would be right often and wrong silently, and the two errors do not
cost the same: a missed pair leaves the status quo, an invented one adds two unrelated
materials together and produces a complete report with no error in it anywhere. What
the pipeline *may* do is test a declaration somebody already made — an old number still
transacting after its own effective date is reported, not acted on. A split (one number,
two successors) or a loop is dropped and reported; it is never resolved by choosing.

**Exactly one document causes a merge, and it takes an act rather than an attribute.**
A substitution list, whose whole purpose is to declare identity. Everywhere else a
stated value is ranked below measurement and cross-checked; identity cannot be ranked
against anything, because merging leaves no trace in any figure downstream and nothing
to disagree with afterwards. Three consequences, each with a test:
- An **item master's `successor_sku`** (SAP `MARC-NFMAT`) is read and *proposed*, never
  applied — the master was handed over for lead times, that column is maintained years
  before anyone plans on it, and it is routinely stale. The run emits the pairs as a
  substitution list to hand back. `item_master.yaml` therefore does **not** declare
  `substitution_signal`; a ✓ there would say the renumberings had been handled.
- A **thin-margin route merges nothing** (`intake.py::_apply_supersessions`, gated on
  `LoadedDocument.route_uncertain`). Elsewhere a misroute is a wrong premise the numbers
  can still be checked against.
- **`from material` / `to material` are not aliases** on `substitution.yaml` and must
  not be added back. A material-to-material stock transfer (SAP movement 309) is headed
  exactly that way and routed to substitution at 83% until they were removed — one
  posting would have permanently merged two live materials.

**Every CSV is written UTF-8 *with a BOM*, and read by sniffing, not by a ladder**
(`ingest/encoding.py`). Without the BOM Excel decodes using the system codepage, so a
supplier written correctly as 深圳市华强电子 opens as 娣卞湷甯傚崕寮虹數瀛 on a Chinese
install. On the read side, `latin-1` maps all 256 byte values and therefore never
raises — a try-until-one-works ladder with latin-1 in it can never reach a CJK codec,
so a GBK export became mojibake with no error. Use `write_csv`, never bare `to_csv`.

**A forecasting model is chosen by competition, for make-to-stock items**
(`forecaster.py::_select_by_backtest`). Candidates are refitted on a rolling origin and
scored on periods they have not seen; lowest error wins, and a **naive forecast is
always an entrant** so "best of five" can be recognised as worse than repeating last
month (`vs_naive`; 35 of 178 on the PL30 extract are exactly that). Make-to-order goes
straight to Croston and is not backtested — nothing is stocked against a forecast there.
The policy is the ERP's `stocking_policy`, not the inferred `stocking_class`; where a
SKU has none, the old pattern routing still applies. **The winner is per SKU** —
recorded in `model_used` on every row — and the run summary's model counts are
counts of SKUs, not one choice for the run.

**A stocking policy is suggested from demand and never applied**
(`demand_classifier.py::_suggest_policy`). Any SKU with a series gets an MTS/MTO
suggestion with the evidence attached, on the same tier boundary the rest of the
pipeline uses. It sits beside the ERP's policy rather than overwriting it: that is
a decision in force, this is what twelve months did, and the disagreement is the
finding. 84% agree on the PL30 extract. Backtest three steps ahead, not
one: a one-step score cannot see a model walking away from the data.

**Covering the period in front of you outranks rebalancing the position**
(`inventory_projector.py::supply_gap`). The position is a total and says nothing about
*when* supply lands, so an order committed for four months' time counts in it exactly
as much as stock on the rack. A shelf that empties before the next delivery — or that
is empty while supply sits past due — is `EXPEDITE-INBOUND`, ranked ahead of push-out
and never deferred. Open PO quantity is phased into past-due / due-in-horizon / beyond
(`open_po_reader.inbound_schedule`); a past-due order is not near-term supply, because
it no longer has a date on it.

**Push-out defers the surplus, never the whole order.** `min(surplus_deficit,
total_open_po_qty)` lands the position on the reorder point. Deferring everything
inbound out of an empty shelf is the opposite advice, not a milder version of it.

**`location_id` is read from the source wherever the export names a plant** (contracts
declare it on every transactional document, not just inventory) **and the configured
node is stamped only where none does.** `DC-01` is the placeholder `node_config.json`
ships with; when it reaches a row it is reported as a placeholder rather than passed
off as a warehouse. Never write the literal in a reader — read `self.location_id`.

**A key too coarse to be an item number is reported, and is not evidence against
anything else** (`intake.py::_check_key_shape`). The agreement check skips documents
with fewer than ten distinct keys as too small a sample — which is precisely the shape
of a line-number key — and then counted that key as evidence, so the only warning
printed named the file that was mapped correctly.

**Money is converted before anything sums it, and an unknown rate is never 1.0**
(`fx.py`). Source lines arrive in the supplier's currency; adding them raw put ₹7.0bn
next to $240m and printed the total with a dollar sign. A currency with no configured
rate blanks its money columns and is reported — defaulting it turns ₹126,550 into
$126,550 and nothing downstream can tell. A rate the export itself carries wins only
after passing a plausibility gate: a placeholder `1.0` on a foreign line and an inverted
quote are both rejected in favour of the table.

**An implausible lead time discards the lead time, not the order** (`ingest_bridge.py`).
The receipt date is what is unusable; the order was still raised, for that quantity, on
that date. Dropping whole rows threw away 63% of the ordered quantity and made every
ordering diagnostic read the remainder as the whole.

**A column can hold two conventions at once, and the parser must not pick one**
(`adapter.py::_parse_mixed_dates`). A spreadsheet opened under a mismatched locale
converts the values it can read — swapping month and day — and leaves the rest as text.
Whichever parser wins, the other half becomes NaT silently. Repairing the swap is gated
on evidence (unambiguous text order, no day above 12 in the converted half, and the
repair must not push values outside the column's own span); without it the rows are
still recovered, only the month/day correction is withheld.

**Cadence is keyed on the order date, not the receipt** (`cadence.py`). The cumulative
`po_qty − sales_qty` balance measures the planner's decisions against the demand they
were meant to cover. Folding in supplier delivery would let a reliable planner buying
from a late supplier read as a control failure — that question belongs to `service.py`
and `diagnostics.forward`.

**Ordering above the review cadence has to be earned.** It is worth paying for on a
critical item and nothing else, so it is priced at the order cost and reported only
where the item is not A-class.

**`stocking_policy` (the ERP's MTO/MTS) is never merged with `stocking_class`** (the
inferred tier). One is the policy in force, the other the behaviour observed; where they
disagree that *is* the finding, and merging them erases it.

**Charts are inline SVG against the CSS tokens, never a library** (`kpi_report.py`). The
report is one self-contained file that opens from a share with no assets and no script.
Both themes are authored, not auto-flipped. Run the palette through the dataviz
validator rather than eyeballing contrast.

## Style

- Python 3.9 compatible, pandas-centric, no new runtime dependencies without a reason.
- Comments explain *why*, especially where a line encodes a domain judgement. Do not
  add comments that restate the code.
- Empty results keep their full column schema so callers can index them unconditionally;
  a bare `pd.DataFrame()` turns "nothing to report" into a `KeyError`.
- Report what a run cannot answer rather than emitting a blank that reads as fine.
- `.claude/skills/inventory-planning/SKILL.md` is the Claude Code workflow document,
  and the same file VS Code loads as an Agent Skill. If you change a public entry
  point or an invariant above, update it too — it and this file are the two consumers
  of the same knowledge and drift between them is the failure mode to avoid.
