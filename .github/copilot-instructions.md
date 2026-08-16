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

`load_all` identifies each file itself — do not ask which file is which, and do not
reintroduce per-document loader calls. The legacy `load_sales_history()` style still
works but requires the caller to know the answer already.

Outputs land in `output_dir`: `kpi_review_*.html` is the two-chapter review,
`inventory_report_*.html` is the older forecast-oriented report, plus CSVs.

The review covers, in order: service (OTD split four ways, plus OTD by month), inventory
against policy, ordering behaviour, replenishment cadence, forward risk, then the work
list — materials to act on, open POs to change, and parameter suggestions.

```bash
python3 -m pytest tests/ -q        # 459 tests, all should pass
```

Production runs **pandas 2.x**; the dev venv here is **pandas 3.x** and `pyproject.toml`
allows both. They differ in ways that fail silently — datetime resolution (`us` vs `ns`,
so `.astype("int64")` on a date column is out by 1000×), `NaT` stringification, and
`np.where` on typed columns. Test a fix under both before claiming it works.

## Where knowledge lives — data, not code

Supporting a new ERP or changing a planning rule should not require a code change.

| Location | Holds |
|---|---|
| `inventory_planning/ingest/contracts/*.yaml` | what each canonical field *means*: type, grain, `derivable_from`, assertions, value domains, `discriminator` |
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

**OTD is measured over completed deliveries only** (`service.py`). An open line past its
request date could never have been on time, so including open lines drives the rate
toward zero by construction. Past-due backlog is reported as its own metric.

**A line past due with stock on the shelf is not a supply failure.** It is an
inventory-efficiency problem with a different owner. Never fold it into the OTD miss
rate — that blames planning for a collection problem and hides idle stock at once.

**"Now" is the newest date in the data, not `date.today()`.** A seven-month-old extract
scored against the clock marks every open line past due.

**Absence of a column is not evidence for a document type** (`registry.py`). A missing
receipt-date column made `is_null(receive_date)` vacuously true, so a sales export
scored 100% as an open PO.

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
- `skill/SKILL.md` is the Claude Code workflow document. If you change a public entry
  point or an invariant above, update it too — it and this file are the two consumers
  of the same knowledge and drift between them is the failure mode to avoid.
