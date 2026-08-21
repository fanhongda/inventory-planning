# Inventory Planning — Domain Skill

> Part of a personal supply chain agent infrastructure designed to demonstrate how frontier AI can operate meaningfully in real-world logistics contexts.

A single-stage DC inventory planning pipeline grounded in MIT CTL MicroMasters supply
chain principles. It reads whatever exports a planner actually has, works out what
stock *should* be, ranks the actions that would close the gap, and says plainly which
of its numbers are measured and which are assumptions.

Roughly half of it is not planning arithmetic. Real ERP extracts are wrong in ways
nothing raises: a material number padded in one report and not the next, five currencies
summed as though they were one, a spreadsheet that silently swapped month and day on the
40% of dates it could reinterpret. None of these produce an error — they produce a clean
run, a confident total, and a number that is off by a factor nobody can see. So the
intake layer is treated as a first-class problem rather than glue, and every figure the
report prints carries whether it was measured, derived, or assumed.

The other half is the refusal to agree. A planning system earns its place by disagreeing
usefully with the organisation around it, and most of the modules that look like extra
complexity exist to keep that ability: measured values outrank stated ones and every
disagreement is reported, the planner's own safety stock is compared but never consumed,
and a deviation between recommendation and reality is attributed rather than learned
from.

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
| **Intake** — routing | Contract scoring per document, adapters frozen per export | which file is which, decided from content not filename |
| **Intake** — mixed-format detection | Whole-column representation census at profile time | any column holding two conventions is named for manual check |
| **Intake** — date repair | Evidence-gated month/day restore on spreadsheet-mangled columns | both halves of a split date column survive, or the repair declines and says so |
| **Intake** — currency normalisation | Per-line transaction currency → one reporting currency, effective-dated | every money column comparable; unrated codes named, never defaulted |
| **Intake** — capability report | What each run can and cannot answer, given what arrived | the questions this extract cannot support, stated up front |
| Demand characterisation | Frequency + CV classification | stocking-high / stocking-med / non-stocking |
| Demand pattern routing | CV thresholds | smooth / intermittent / erratic / lumpy |
| Forecasting | ETS, ARIMA or Croston per pattern | 6-month forecast + t+1 point forecast |
| Safety stock | Combined demand × lead-time variability (MIT CTL §7) | SS = k · √((R+LT)·σ_fc² + μ²·σ_LT²) |
| Inventory projection | Days-of-supply excess detection | EXCESS / SHORTAGE-RISK / OK |
| Backlog realization | Share of the order book that actually ships, measured from uncollected past-due lines | per-SKU realization rate |
| Purchase recommendation | Forecast consumption: max(t+1 forecast, realizable backlog due in horizon) + SS | PURCHASE-REQUEST / ORDER-FOR-BACKLOG / PUSH-OUT / HOLD |
| Should-be inventory | cycle + safety + buyer-owned pipeline, incoterm-aware | gap vs actual, per lever |
| Lever ranking | Net annual benefit, not cash freed | ordered action list with what each is worth |
| Service attribution | OTD measured and split four ways, and tracked by month | who owns each failure, and when it moved |
| Ordering behaviour | Over-ordering, chronic air freight, erratic lot sizing | what the buying pattern cost |
| Replenishment cadence | Cumulative monthly PO qty − sales qty, against the order count the review period allows | controlled / chasing / positive bias / over-correcting, priced |
| Stocking policy vs behaviour | ERP make-to-stock / make-to-order flag, carried alongside the inferred stocking class | where policy and observed demand disagree |
| Parameter suggestion | What the data supports vs what is in force | per-SKU CSV + paste-able rule blocks |
| Source cross-check | Measured vs ERP master vs planner worksheet | every material disagreement, with both values |
| Feedback loop | Cumulative gap attribution across months | OPERATOR_DEVIATION / SUPPLY_GAP / MODEL_BIAS |
| Lead-time drift | Successive snapshots compared; drift separated from source change | which suppliers moved, and by how much |

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

### A column can hold two conventions at once, and the parser must not pick one

The intake assumed a column speaks one convention. Reasonable, and wrong in a way that
produces no error: whichever parser is chosen, the other representation becomes null.

Excel is the usual cause. Opened under a locale that disagrees with the file, it
converts the cells it *can* read as local order — the ones where both components are 12
or less — into real dates, **swapping month and day** as it does so, and leaves the rest
as text because `10/14/2024` is not a valid day-first date. The column comes back half
`2024-04-10 00:00:00` and half `10/14/2024`.

On the real sales extract this removed 8,737 of 34,128 rows from `Invdate Date` and
13,163 of 32,800 from `Orddate Date` — 9.7% of all shipped quantity across 759 of 1,256
SKUs. Nothing looked wrong downstream, because the demand that survived was real; there
was simply less of it. The survivors were exactly the values whose day exceeded 12.

Two separate responses, because they answer different needs:

- **The profiler flags any column carrying two incompatible representations**, reading
  the whole column rather than a sample — a partial conversion follows whichever values
  were ambiguous, not row order, so a split column is frequently uniform for its first
  several hundred rows. The flag survives the repair: rescuing this run's numbers does
  not fix the file, and the next export will be broken the same way.
- **The parser repairs the swap, but only on evidence.** Three things must hold: the
  text half states an unambiguous order, the converted half contains no day above 12 at
  all — the fingerprint of having been filtered to the ambiguous values — and restoring
  the values must not push them outside the span the rest of the column occupies. That
  last check is a veto rather than a requirement, so it refuses a bad repair without
  refusing every export whose halves already overlap.

The fingerprint is the proof. A mixed column can only arise when the spreadsheet's
locale disagreed with the file — had it agreed, every value would have converted and no
text half would remain to compare against. Where evidence is absent the rows are still
recovered; only the month/day correction is withheld.

### Charts are inline SVG, and a shape is not a summary statistic

The report is one self-contained file that has to open from a network share with no
assets and no script, so there is no plotting library available — and none would be
appropriate if there were. Every chart is inline SVG against the same CSS tokens as the
rest of the page, authored for both themes rather than auto-flipped.

Two of them replace numbers that were hiding something:

- **On-time delivery over time.** A single OTD figure answers "how did we do" and hides
  "when did it change", which is the question that decides whether anything needs
  fixing. 82% flat for two years and 82% because the last four months collapsed are the
  same number describing different situations. The volume strip beneath is not
  decoration — a month at 100% on four lines looks identical to one at 100% on four
  hundred, and without the counts the eye reads the thin months as the good ones.
- **Replenishment pattern.** The cadence diagnosis *is* a shape, and rendering it as
  three columns of statistics made the reader rebuild the picture in their head. Small
  multiples of the cumulative curve, diverging around zero, read by comparison: six flat
  months below the line then a vertical correction is one glance.

The axis on the OTD chart is focused on the range in play rather than anchored at zero.
That is correct for a line encoding a rate — the length from zero carries no meaning, so
there is nothing for a zero baseline to protect — and the floor is labelled and never
rises above 80%, so the zoom is stated rather than implied. The rule it would break is
the one about *bars*, where length is the encoding.

### Money is converted before it is added, and an unknown rate is not 1.0

A purchase organisation spanning several countries raises each PO in the supplier's
currency. SAP exports that faithfully — one `Currency` column, one line value, no
conversion — and nothing about the result looks wrong: every figure is a valid number
and so is their sum. On the real extract that sum was ₹7,004,449,000 added to
$240,323,000 and printed with a dollar sign. Restated, purchase history is $402,281,808.

What the mixture destroys is not the total but the *ranking*. A rupee line outranks
every dollar line on value alone, so ABC classification, excess value, value at risk and
the whole efficient frontier get sorted by which currency a supplier happens to bill in.

Three rules keep the conversion honest:

- **A rate the source booked wins — once it is shown to be a rate.** It is what the
  transaction settled at. But the backlog extract carries `Exchange Rate` = 1 on all
  3,169 of its INR lines, and ERP rate columns are frequently quoted upside down
  (94.7 rupees per dollar where the conversion needs 0.01056). A placeholder 1.0 on a
  foreign line, or a quote an order of magnitude away from the configured rate, is
  rejected in favour of the table — and the rejection is reported, not silent.
- **Rates are effective-dated.** Purchase history spans eight years. One rate across all
  of it renders a currency move as a procurement trend.
- **An unrated currency blanks its money rather than defaulting to 1.0.** Defaulting is
  how ₹126,550 becomes $126,550, and nothing downstream can tell that it happened. The
  lines are excluded from every money figure and named in the report instead.

The one assumption made silently: a document with no currency column is taken to be
single-currency. That is the ordinary single-entity export, and it is recorded as an
assumption rather than a fact.

### Order-size CV cannot see timing; the cumulative flow balance can

`diagnostics.erratic` scores lot-size consistency. A planner can pass it perfectly and
still be a month behind demand every month of the year, because consistent lots placed
at the wrong times produce exactly the stockouts erratic ones do.

`policy/cadence.py` accumulates `po_qty − sales_qty` month by month instead. The shape
of that curve separates four failures the CV cannot:

| Shape | Reading | Fix |
|---|---|---|
| Returns to ≈ 0 | Run rates match. Errors that cancel are errors the inventory never carried | none — this is the target |
| Sits negative for months | Buying behind selling and staying behind | replenishment timing |
| Climbs and stays up | Persistent positive bias — nothing closes the loop | the review itself, not the forecast |
| Swings both ways | Over-correction; bullwhip at single-SKU scale | both, and it costs both |

Alongside it sits the question the CV never asks: **how many orders were spent getting
there**. A monthly review entitles the planner to twelve orders a year. Twelve that land
the cumulative on zero is control; forty that land on the same zero is the same result
at three times the ordering cost, and usually means the cadence is being overridden by
expediting.

Two choices keep it from crying wolf. The tolerance band is at least **one typical lot**,
because a periodic review swings by a full lot even when working perfectly — a part
bought quarterly sits a quarter's demand down for most of every quarter, and that is the
policy operating, not failing. And **ordering above the cadence has to be earned**: it is
worth paying for on a critical part and on nothing else, so it is priced at the order
cost and reported only where the item is not A-class. On the real extract that is 40 SKUs
and $81,900 a year of ordering cost, all of it on C-class items.

What it deliberately does not model is the lag between raising a PO and receiving it.
The balance is keyed on the **order** date, so it measures the planner's decisions
against the demand they were meant to cover. Supplier delivery against those decisions
is a different question that `service` and `diagnostics.forward` already answer, and
mixing the two would let a reliable planner buying from a late supplier read as a
control failure.

### Material identity is declared, and the merge happens before anything is computed

A part renumbered from 7100014 to 7100015 is one planning problem that arrives as two,
and nothing in the transactions says so. Left alone it produces two confident answers,
both wrong, and no error anywhere:

```
7100014   150 on hand, 500 on order, demand history stops in March
          → a year of cover, no forward demand: dead stock, push out the PO
7100015   three months of history, no receipts
          → non-stocking, forecast flat, safety stock ≈ 0
```

So the rewrite happens at intake, before any analysis, and — this is the part that
matters — **in every document at once**. Every join here is on `sku`, so rewriting the
demand history and not the stock report is worse than leaving both alone: it gives one
SKU a forecast with no position and another a position with no forecast.

Rows that become duplicates of each other afterwards are recombined, or the grain the
pipeline itself just broke double-counts everything downstream. **How they combine
follows from the key, not from the document's name.** A natural key of `[sku]` alone
means one row per material of its standing attributes, so two such rows are two
descriptions of one thing: the successor's row wins and the predecessor's only fills
its gaps. Adding them would make a 180-day lead time out of two 90-day ones. Every
other key in the contract set carries an event or a period alongside the SKU, so
quantities add — and a per-unit price is quantity-weighted, never summed and never
flat-averaged.

Where the renumbering also changed the pack size, a conversion ratio scales the
quantities and divides the per-unit money, so the total value does not move. The
default of 1 is the common case and the dangerous one: a loose piece becoming a
ten-pack is wrong by an order of magnitude with nothing to catch it.

**The pairs are never inferred.** The data does contain a tempting signal — one SKU
going to zero in the same month another ramps up — and a rule built on it would be
right often. The two ways of being wrong do not cost the same. A missed pair leaves the
status quo. An invented pair adds two unrelated materials' stock and history together
and produces a complete, self-consistent, wrong report. Only a planner knows, so only a
planner declares it.

**And declaring it takes an act, not an attribute.** Exactly one document causes a
merge: a substitution list, whose entire purpose is to state identity. This is the one
place the pipeline does not treat a stated value as something to rank and cross-check,
because identity cannot be ranked against anything — merging restructures every
document in the run, no figure downstream carries a trace of it, and there is nothing
left to disagree with afterwards.

That gate has three teeth. An **item master's follow-up-material column** is read and
proposed, never applied: SAP maintains one (`MARC-NFMAT`, with `MARC-AUSDT` and
`MARC-KZAUS`), it is maintained at the moment of a phase-out decision and consulted
years later, and the master was handed over to supply lead times. The run turns whatever
it finds there into a substitution list to check and hand back — the confirmation, and
the saving of the retyping, at once. A **thin-margin route** merges nothing and says so;
for every other document a misroute is a wrong premise the numbers can still be checked
against, but here it is a rewrite of identity that nothing downstream can see. And the
**aliases refuse ambiguity**: `From Material` / `To Material` are not aliases here,
because a material-to-material stock transfer is headed exactly that way and routed here
at 83% until they were removed — one movement posting would have merged two live
materials.

What the pipeline *does* do is test the declaration, the same way a stated lead time is
tested: an old number still transacting after its own effective date is reported, because
that is a claim the transactions do not support and the merge has just folded a live
material into another one. A split — one number with two successors — and a loop are
dropped and reported rather than resolved by choosing; the rest of the map still applies,
since one mistyped row should not cost the other forty merges.

Coexisting pairs are a different problem with different arithmetic and are not merged.
They are read and counted; the phase-in / phase-out handling is not built yet.

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
| `substitution_signal` | no | a substitution list — only this document causes a merge |

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

## When a file routes to the wrong contract

Routing is automatic, and the cost of that is a failure mode with no obvious handle: a
file lands on the wrong contract, every number downstream is computed correctly from
the wrong premise, and the run reports `OK`. The intake summary now flags a thin margin,
but when it happens:

```bash
python -m inventory_planning.explain "path/to/folder"
```

It prints, per contract, the score it got, which required fields could not be mapped
and to what the mapped ones went, which identifying fields are absent, whether the
content discriminator could even be evaluated, and which source columns matched
nothing. The last two are usually the answer — a document loses either because a column
it needs is named something the aliases do not know, or because the column exists but
holds something unexpected.

The fix is a data change: add the alias to the contract, or freeze an adapter for that
export. Not a code change.

**A contract must state what makes it itself.** `identifying_any` lists field groups of
which at least one must be present. It exists because a contract whose only required
field is `sku` matches every file with an item column, scores full marks on required
coverage, and outranks the document that genuinely fits — which is how a sales-history
export gets filed as an item master, taking the demand signal with it. An item master
earns the name by carrying a planning parameter; a file with none is a list of item
numbers, whatever else is on it.

**A contract can also state what a field must never be.** Aliases answer "what else
might this column be called?", which is the wrong question when one header word means
two different things in two ERPs. `Item` is the case: in SAP it is the position within
a document — VBAP-POSNR, EKPO-EBELP, MSEG-ZEILE, QMFE-FENUM all render as `Item` — and
never the material, while most other systems use it for the material. No amount of
data shape settles that, so the profiler decides *which ERP wrote the file* once, from
header vocabulary and SAP's zero-padded keys, and the contract declares the mapping the
dialect forbids:

```yaml
  sku:
    aliases: [sku, material, material number, ..., matnr, matl, item]
    never:
      sap: [item, itm, item no, position, pos, line item, 项目]
```

Three layers back it up, because each one alone has a hole:

| layer | what it catches | where it is blind |
|---|---|---|
| a line-number field on every transactional contract | `Item` alongside `Material` | an export with no material column — `Item` is then the only candidate for a required field |
| the `never` rule | any spelling, whatever the values look like | a file that never says which ERP wrote it |
| the shape verdict — a column holding 10, 20, 30 is a series, not identifiers | an unidentified source | a line number that is not written as a series |

When they all miss, intake says so rather than proceeding: a document keyed on fewer
than ten distinct item numbers is reported by name, with the values, the reason, and
the columns that would have joined instead. Refusing to map is deliberate — the
required-field test then fails and stops the run, which is a question a human answers
in a minute, where a backlog keyed on 10, 20, 30 produces a full report of confident
zeroes and no error at all.

---

## Output

```
output/
├── kpi_review_<ts>.html                 ← open this
├── parameter_suggestions_<ts>.csv       suggested parameters vs those in force, per SKU
├── suggested_rules_<ts>.md              the same, as paste-able planning_parameters.md rules
├── source_crosscheck_<ts>.csv           where two sources disagree, and by how much
├── supersessions_<ts>.csv               old item number -> new, and what each old
│                                        number contributed to each document
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
| `config/node_config.json` | Location ID, reporting currency — one entry per DC |
| `config/fx_rates.json` | Exchange rates into the reporting currency, effective-dated. A currency absent here is reported unvalued, never counted at face value |
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

  Lead-time drift (3 snapshots): 1 SKUs moved materially, 1 lengthening
     SKU-A                45d ->    66d  +21d / +47%  [ACME]
     A supplier that moved is a supplier conversation — the plan absorbed it silently.

  ► [OPERATOR_DEVIATION] SKU=XYZ-001
    Maintain system recommendation unchanged.
    Review with operations: storage constraints, budget freeze, or manual override?
    → No model parameter change.
```

### Lead-time drift

Each snapshot also records the lead time the run planned on, with its sigma, sample
count and source. This is the one input most likely to move without anyone noticing:
every run measures it afresh from recent receipts and builds an internally consistent
plan around the new number, so a supplier sliding from 45 days to 62 over four months
never triggers anything — the reorder point, safety stock and exposure period all move
with it.

`feedback.drift` compares successive snapshots and reports the movement. It needs no
actuals, so it is readable the moment a second month exists:

```python
from inventory_planning.feedback.loss import LossCalculator

print(LossCalculator(snapshot_path).lead_time_drift().summary())
```

Two things it deliberately keeps apart. **Drift** is the measured lead time itself
moving — a supplier conversation. A **source change** is last month's figure coming
from an item master and this month's being measured, or the reverse: nothing happened
at the supplier, only what the pipeline knew about it. Reporting the second as drift
would manufacture a supplier problem out of a data improvement.

A move must clear both a relative and an absolute floor (20% and 5 days by default) to
be reported. A relative threshold alone flags 2 days becoming 4; an absolute one alone
misses 90 days becoming 110, which is the more expensive event.

---

## Project structure

```
inventory_planning/
├── orchestrator.py            run_planning → run_policy_analysis → run_kpi_review
├── ingest_bridge.py           canonical frames → analytics column expectations
├── fx.py                      transaction currency → reporting currency, effective-dated
├── explain.py                 why a file routed where it did — per-contract scoring
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
│   ├── cadence.py             cumulative PO − sales balance vs the cadence's order count
│   └── decisions.py           accept/reject log → constraint candidates
├── analytics/
│   ├── demand_classifier.py   CV + frequency → demand pattern
│   ├── forecaster.py          ETS / Croston / ARIMA / SMA
│   ├── safety_stock.py        MIT CTL combined variability formula
│   ├── inventory_projector.py DOS-based excess detection
│   ├── backlog_realization.py what share of the open order book actually ships
│   └── purchase_recommender.py  forecast consumption → net requirement
├── feedback/
│   ├── snapshot.py            auto-saves planning state + the lead time planned on
│   ├── collector.py           records actuals against prior snapshot
│   ├── drift.py               lead-time movement across months; drift vs source change
│   └── loss.py                cumulative gap analysis + deviation attribution
├── readers/                   legacy per-document readers (superseded by ingest/)
└── reporting/
    └── kpi_report.py          two-chapter review: what happened / what is coming

  (at repo root:)
  .claude/skills/inventory-planning/ the skill itself — SKILL.md is the workflow, and
                                     .claude/skills is where Claude Code and VS Code
                                     both look, so cloning the repo installs it
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

285 tests, no network, no fixtures beyond `sample_data/`.

## Requirements

- Python ≥ 3.9 — Windows, macOS and Linux. All text I/O states `encoding="utf-8"`
  explicitly rather than inheriting the platform default, and a test enforces it:
  `config/planning_parameters.md` is written in Chinese, so a locale-dependent read
  fails outright on a Western Windows install (cp1252).
- pandas, numpy, scipy, statsmodels, openpyxl, pyyaml
