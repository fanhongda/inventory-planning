# TODO

## Supply chain topology (P0)

The pipeline currently plans one node and knows nothing about where stock physically
sits. `readers.inventory_reader.consolidate_to_planning_grain` sums every storage
location of a SKU into a single position, which is what makes the numbers add up —
but it is the wrong answer for two cases the data already contains:

- **Quality quarantine.** Location `02` holds material under inspection or blocked.
  It is summed into the available position today, so the position is overstated by
  exactly that quantity and the pipeline under-orders. There is no way to tell a
  quarantine location from a sellable one without knowing what the location codes
  mean.
- **Multiple real nodes.** Two DCs in one export are two planning problems, not one.
  Summing them hides a shortage at one behind a surplus at the other.

What is needed is a topology model, not a location whitelist:

- ERP identifiers per node — company code, plant, storage location — and how they map
  onto planning nodes
- Node type and stock status per location: sellable, quarantine, blocked, consignment,
  in-transit staging. Only sellable nets into the position; the rest are reported.
- Upstream / downstream relationships between nodes, so a transfer is modelled as
  supply at the receiving node and demand at the sending one
- Which node owns replenishment for which SKU

`config/node_config.json` already carries `parent_node` / `child_nodes` placeholders
and every output carries `location_id`, so the shape is anticipated. The blockers are
the master data and the multi-node arithmetic, not the plumbing.

Until then: `consolidate_to_planning_grain` prints how many SKUs were merged from more
than one location, and `stock_locations` on the inventory outputs lists the codes that
went in, so the exposure is at least visible.

## Phase-in / phase-out (P1)

The `substitution` contract carries two relations and only one is acted on.
`supersede` — a renumbering — merges everything. `phase` — two numbers trading side by
side while one ramps up and the other winds down — is read, counted, and otherwise
inert. That is deliberate: the two have different arithmetic and merging a phase pair
folds a live material into another one. But an annotation that changes nothing is just
a label, and there are four specific things it should change:

- **A phase-in item must not be classified non-stocking.** Short rising history is what
  the classifier reads as a small item, so the new product is systematically
  under-stocked at exactly the moment it is ramping. The annotation says the short
  history is not evidence of a small item; the forecast should be marked unreliable
  rather than returned flat.
- **A phase-out item's excess is not an ordering failure.** It lands in over-ordering
  and slow burn today, which puts a product decision on the buyer's KPI. It belongs in
  its own section — planned obsolescence — with the run-out or write-off as the action,
  not a push-out.
- **Cap the buy on a phase-out item** at what is needed before the phase-out date. This
  is the one item on the list that saves money this week, and it is why a phase pair
  needs an end date rather than just a start.
- **Report the pair adjacent**, with combined cover as a *reported* figure only.

Explicitly not in scope: pooling. Phase-in/phase-out is not interchangeability. If the
old customers only buy the old number, the two stocks cannot cover for each other and
combining their σ is wrong. True interchangeability is a third relation, with the
risk-pooling arithmetic that goes with it, and it needs its own declaration.

Two smaller pieces left over from the supersede work:

- **Action lines should name the number the buyer will find in the ERP.** Planning runs
  on the survivor, correctly, but an open PO to push out was raised against the old
  number and that is what the PO says. `supersessions_<ts>.csv` answers the question;
  the recommendation row should carry it directly.
- `Adapter._apply_rollup` sums every numeric measure, including `unit_cost`. Wrong for
  a per-unit price and for a lead time — `supersede.py::_recombine` works out the right
  aggregation from the field's declared unit, and the rollup path should use the same
  rule.

## Replenishment quantity — what is left (P2)

The order quantity now covers the lead time. `s = mu(R+LT) + SS(R+LT)`, the lot is EOQ
raised to the MOQ and the order multiple, `S = s + lot`, and which arithmetic applies is
decided per SKU by `replenishment_method` from the rule engine. `policy/profile.py`
records the eight axes that decision rests on, with the evidence and the provenance
behind each reading, and reports where the evidence disagrees with the rule rather than
overwriting it. `analytics/safety_stock.py` sizes safety stock on the same R + LT
exposure `should_be.py` always did, so the target stock and the order aiming at it no
longer answer different questions.

One correction to what this section used to predict: the gate was expected to produce
*fewer* recommendation lines. It produces more, and larger ones. `IP <= s` was already
implied by `net_requirement > 0`, so making it explicit changed nothing; what changed is
that `s` now spans R + LT instead of thirty days, so more positions fall below it. That
is the defect being fixed, not a side effect.

What remains:

- **The measured ordering cadence is still not offered as evidence for R.** R comes from
  `review_period_days` in `config/planning_parameters.md` — a stated value, correctly,
  because it is a decision rather than an observation. But `policy/cadence.py` measures
  what the buyer actually does, and where the two disagree by a wide margin that is a
  finding: either the rule is aspirational or the cadence is being overridden by
  expediting. Report it beside the rule, on the same measured / stated / default ranking
  the rest of the pipeline uses. Do not consume it.
- **Slow movers are still sized on the Normal distribution.** MIT CTL §9 puts the
  boundary at `mu(DL+R) < 10` units, below which Poisson is the right distribution and
  the Normal understates the safety stock badly. The profile already records
  `demand_continuity` per SKU, so the SKUs are identifiable; the arithmetic is not there.
- **Only three of the five methods exist.** Periodic review, (s, Q) and order-on-demand
  are implemented. Base stock — one-for-one replenishment — is a real policy for a very
  high value, very short lead time item and is currently forced onto periodic review.
  Single period (newsvendor) has no place in a rolling DC plan and is deliberately
  absent; if a genuinely one-shot buy ever needs planning it needs the critical ratio,
  not this pipeline.

Explicitly decided and **not** to be built: inferring air versus sea freight to set the
review period. There is no transport-mode field in any extract, and incoterm is not a
proxy for one — EXW and FCA are mode-agnostic. Lengthening or shortening a review period
for named materials is a rule the planner writes, which is what the rule engine is for.

## Data layer (P1 — step 1 done)

Where the inputs live, and where each run's decisions go. The full reasoning is in
[DATA_LAYER.md](DATA_LAYER.md); this is the work that follows from it, in order.

The shape it settled on is **an append-only bitemporal fact store plus an editable
overrides layer**, not an editable SQL database of ERP reports. Taking "create, update,
delete on a fact" case by case, none of them needs a row updated in place: a bad import
voids a batch, an amended document is a new batch, a delivery date confirmed by phone is
*new evidence* belonging in overrides rather than a correction to a record that was
never wrong, and a genuine erasure rewrites a batch under approval. Three kinds of table
follow, with three key disciplines — facts keyed on `natural_key` + `valid_time` +
`batch_id`, parameters on entity + `effective_from`, decisions on `run_id` + entity.

**Step 1 — the natural key. Done.** As-of semantics rest entirely on being able to tell
that two rows are the same row, and the key could not. Every part but `sku` was optional,
every call site reduced the key to whatever had mapped, and an unverifiable grain was
reported as a pass. Fixed: the document line is part of the key for the four line-grain
contracts; `po_schedule_line` has a field of its own, so the key column stops moving
between a sample and its parent export; and `KeyStatus` separates *plannable* — a
degraded key still forecasts, and must keep doing so — from *storable*, which needs a
complete one.

**Step 2 — `run_id` and a run registry. Done.** `provenance.py` records what a run
read, what it resolved and what code ran, and `RunRegistry` keeps a manifest per run
plus an append-only index under `<output>/runs/`. Two fingerprints carry the weight:
`input_fingerprint` over the source files, `config_fingerprint` over the config files
and the rule ids in force. `compare()` uses them to name *why* two runs differ —
`scenario` when only the parameters moved, `new_data` when only the facts did,
`identical`, and `mixed` when more than one axis moved. That last verdict is the point
of the exercise: a policy UI must refuse to attribute a difference when the facts moved
underneath it too. The manifest also carries each input's key verdict from step 1, so
the store already has its gate recorded before it exists. Nothing here can fail a run —
a missing git binary, an unreadable config, an input with no file behind it are each
recorded as unknown.

**Scenarios work, and the rule set is now a run input.** `parameters_file` used to
reach only `run_policy_analysis`, so an alternate rule set changed the policy report and
left the purchase recommendations exactly as they were — the one thing a planner would
act on. It is a constructor argument now: `InventoryPlanner(parameters_file=...)`, or
`--parameters` on the CLI. One planner, one rule set, one run identity, fixed before the
first batch is written. Two runs over one dataset under two rule sets compare as
`scenario` and their safety stock genuinely differs (9 of 10 SKUs on the sample data).

Output files are stamped with the `run_id` rather than the minute. Four independent
`datetime.now()` calls could disagree within one run, and two runs seconds apart — a
planner trying a rule change, which is the whole point — overwrote each other's CSVs.

A UI now exists over this — see [INTERFACE.md](INTERFACE.md) — and the run diff it
serves answers only *whether* a difference is attributable, not what moved.

**Per-rule hits are retained. Done 2026-09-15.** The manifest carries, per rule, what it
matched, what was still standing once the later rules had run, the columns a skipped rule
wanted, and sample SKUs. `GET /policy` finds the newest run under the same rule *bytes* —
not the same `policy_fingerprint`, which folds in the path, so a scenario copy of the
rules still matches — and serves the counts keyed by rule id; edit the file and they
disappear rather than going stale. The standing count is the part worth keeping: `R-001`
on the sample data matches 4 A-class SKUs and keeps 2, and on the test frame keeps none,
so `matched` alone reports a dead rule as the busiest on the page.

**Macro scalars are editable. Done 2026-09-15.** `policy/macro.py` proposes a change as
a one-line diff and applies it once approved: the value is replaced where it sits, so the
comments explaining a setting and the prose keys `fx_rates.json` carries survive
untouched. The edit is validated by loading the file with the pipeline's own loader
before anything is written; the reason and the owner go to `config/macro_changes.jsonl`.
The two halves are one endpoint called twice, tied together by the digest of the file the
diff was made against — approving a change approves *those bytes*, and a file that moved
in between is refused. `PUT /policy/macro`, and an Edit on each row of the settings
table. The rules are still read-only.

**The quality gate is on the import screen. Done 2026-09-15.** `GET /gates` runs the
intake checkpoint over everything landed and renders its three severities;
`POST /gates/{check}/waivers` writes an ordinary `gate_waivers` entry, so a finding
waived by clicking is waived in a headless run and not merely hidden on the screen — a
waived finding stays on the page, downgraded, carrying who waived it and until when.
The page says the intake gate is one of four and names the other three, because they
need a time series, a forecast and a position and so cannot be answered before the run.

**Found while doing it, not fixed:** `cli.py`'s per-file path never builds an
`IntakeResult`, so the intake gate does not run there at all — only under `load_all`.
The CLI compensates for one of its checks (`product_dimension`, via a `parser.error`);
`sku_agreement`, `semantic_failure` and `dimension_spelling` simply do not fire. A
disagreeing inventory export runs to completion with exit 0 on the CLI and raises
`DataQualityError` through `load_all`.

**The rules are editable too. Done 2026-09-15.** `PUT /policy/rules` over the same
propose-then-approve contract, with edit, add and remove. Only the fields that changed
are rewritten, so a comment beside a parameter survives an edit to the parameter next to
it and a rationale nobody touched is not reflowed. The rule's own `rationale`, `owner`
and `date` move with the change — that is where the next reader looks — and the reason
for the change goes to the log, which for a removal is the only record left. Every
proposal reports the rule order the file will have afterwards, because later rules win
and where a rule sits is part of what it does. `rule_id` is not editable: the manifest
records hits against it.

**The results screen is up. Done 2026-09-15.** A fourth screen renders what a run
wrote, located from the manifest: the workbook's sheets, its text outputs verbatim, and
every file it produced with a download. It computes nothing — the numbers are cells,
shown under the workbook's own number formats with Excel's rounding, so the page and the
file cannot disagree. A test pins the writer's format constants to the reader's, because
that is the only way the two files stay in agreement without becoming one.

**The workspace seam is cut. Done 2026-09-15.** `workspace.py` resolves config, store
and output from a tenant id in one place; `--tenant` on the pipeline, the server and the
store CLI. The default tenant moves no path. It found one defect on the way: `--output`
defaulted to the string `"output"`, argparse passed it every run, so it reached the
resolver as an explicit argument and outranked the tenant — a tenant's outputs landed in
the shared directory the results screen reads while config and store were isolated. The
same literal was in the API service and its server. It also found that the API tests
were reading the repository's own `./output`, so "no runs here" was an assertion about
the developer's machine.

**The identity seam is cut. Done 2026-09-15.** `attribution.py` turns a name into an
actor in one place — six API endpoints, three CLIs and four writers now ask it instead
of reading `by` off a payload. It records *how* the name was established, not only the
name: `self_asserted` today, `verified` when a token fills it. Absent means
self-asserted, so nothing was written to disk and nothing migrated, and the 1,218
attributed records already in the store are correctly classified by the new reader. It
closed the same defect the workspace seam found, one layer down: `by` defaulted to `""`
on `ledger.restate` and `ledger.void`, so enforcement was only at the entry points.

Still a form field, and nothing behind it. The bind address remains the whole access
control — `api/__main__.py` says so where someone deciding to expose the port will read
it.

**A maintenance plan is now pinned to what it was read against. Done 2026-09-15.**
`store/migration.py`. `store restate --apply` used to select again from scratch, so it
changed whatever the predicate matched at that second rather than what was counted and
approved — and shadow write lands batches while a plan is being read. A plan is now
written to `<store>/plans/`, and `--apply --plan <file>` carries out exactly the ids in
it. The basis digests the *selection* (sorted batch ids + operation + target layer), not
the selector, which would have digested identically at both moments and checked nothing.
A batch restated by someone else in between is a refusal that names it. `--batch` alone
is exempt: ids name the batches rather than describing them.

Still open, and the rest of that seam: **a migration has no identity**. The 1,188 lines
share a reason string, so a store can only be grepped for what has run against it, not
asked. And a layout change bumps `SCHEMA_VERSION` while a restatement does not — the
first migration that happened was the kind the stamp does not cover, and its refusal
names a `migrate` command that does not exist.

**The feedback loop closes. Done 2026-09-15.** It never had: 2,430 snapshots, 63 MB,
none ever scored, and three of `feedback/`'s five modules with no caller. `record_actuals`
asked for two hand-assembled frames a month later; the store already held both.
`feedback/actuals.py` reads them at scoring time and
`python -m inventory_planning.feedback score --run <id> --sales <batch>` runs it. The
snapshot is never written to — `LossCalculator.compute` was also writing back — and the
score is its own record under `<store>/scores/`, carrying which batch and which period
produced it. The scoring period comes from `as_of`, not `planning_month`: those are
different things and using the wall clock would score a forecast against another month's
demand.

Verified on a real run rather than a fixture: a plan anchored to 2026-07-23, an August
extract landed after it, first score ever produced — MAPE 15.8%, over-forecast,
attributed to `MODEL_BIAS` with the parameter it points at.

**On real data the loop still cannot close**, and that is a data fact rather than a code
one: every extract in the store is the same July pull, so there is no later month to
score against. The refusal says so by name rather than reporting every SKU as having
sold nothing.

**The CLI routes by contract. Done 2026-09-15.** Found by running the real extract
before a production run: the documented invocation died on it with `KeyError: ['qty']`.
The sales history heads its quantity column `Shipped Quantity`; the legacy `schema.py`
the per-file readers use does not carry that alias, and the contracts do. It is not one
alias — **the contracts carry 1,216 the legacy table does not**, across every document
type. `inventory-plan <dir-or-files...>` now routes by content, which is the path that
already worked and had no command line. The flags still work and print what they cost.

Why it survived: `sample_data/sales_history.csv` heads that column `Sales Qty`, which
the legacy table knows. Every test passed while the real export failed, and there were
no CLI tests at all. There are now, and the one that matters compares the two alias
tables directly rather than checking that the CLI runs.

**Still two implementations of intake.** The right end state is the readers reading
through the contracts, so there is one alias table; that is a bigger change than this
was and it is not done. Until then the per-file path is a fallback that says so.

**Real business figures sit in a public repository.** Noticed 2026-09-16 while
checking that no real extract had reached git — none had, and `sample_data/` is
synthetic, but the *numbers* from one did. The repository is public
(`github.com/fanhongda/inventory-planning`), and several arguments quote the scale of a
real employer's data to make their point:

| Where | What |
|---|---|
| `README.md:387` | `13,163 of 32,800` rows, `9.7%` of shipped quantity, `759 of 1,256` SKUs |
| `inventory_planning/ingest/adapter.py:150` | the same figures, in the mixed-date comment |
| `ingest/adapters/default__regional_planning_master/planning_master.yaml` | `1,256 sales SKUs`, `1,028` joining — in the adapter's own description |
| `tests/test_invariants.py`, `tests/test_mixed_formats.py` | the same percentages, quoted in docstrings |

None of it is data — no material number, customer or quantity — and all of it predates
the interface work; the figures added since are about this tool's own bookkeeping
(batch counts, snapshot counts), most of which are development runs. So this is
housekeeping rather than a leak, which is why it is here and not a fix.

**The fix when it happens:** the arguments do not depend on the magnitudes. "On a real
sales extract this silently removed a tenth of shipped quantity across three-fifths of
the SKUs" makes the same point as `8,737 of 34,128`, and says nothing about whose
warehouse it was. Replace the counts with proportions or with "a real extract", and
leave the reasoning exactly as it is — the reasoning is the thing worth keeping.

Also worth deciding at the same time: whether the adapter directory should be named for
a system at all. It was `default__fsp_in30` and is now `default__regional_planning_master`,
which is already the right instinct applied once.

What this still needs: **decisions queryable** — `run_id` is in the snapshot filename and
not its body, and there is no query layer over 2,430 JSON files — and a **SKU-level diff
over two `run_id`s** — which SKUs changed class, what the safety-stock total moved by,
which recommendations flipped. Neither needs new identity work.

**Step 3 — the store. Phase one done.** `store/` holds it: `location.py` resolves the
root (`--store` / `$INVENTORY_PLANNING_STORE` / `$XDG_DATA_HOME` / `~/.local/share`),
`ledger.py` is the append-only batch record, `fact_store.py` writes parquet under
`<root>/facts/<doc_type>/<batch_id>.parquet`. Parquet rather than CSV because the store
sits exactly where the type damage happens — a CSV round trip turns
`000000000000123456` back into `123456`, which is the failure `normalize:
material_number` exists to undo.

Each batch carries both timestamps. `valid_time` comes from `latest_observed_date` —
the newest date the data itself carries, a property of the content, unlike an mtime that
a copy rewrites — and is refused rather than guessed when there is no anchor.
`transaction_time` is the load. Re-loading the same bytes is a no-op, and a void appends
a line rather than rewriting one.

The `history/` split is closed: `SnapshotSaver` takes an explicit `history_root` and the
orchestrator resolves it to the store, so the location no longer depends on the shape of
`--output`. The two snapshots that were tracked at the repository root are untracked and
`history/` is ignored; they stay on disk because `feedback.loss` reads a snapshot by
path. The 223 under `output/history/` are untouched and still readable the same way.

**The store has a read path; the pipeline is not on it.** `store/query.py` reads across
batches with both time axes and three named readings, and the interface serves them — but
planning still reads its files, which is what keeps the rollout safe. What remains:

- **Phase two** — read from the store behind a flag, and compare a run's outputs field
  by field against the file path. Two known differences to expect: the canonical frame
  applies the contract's `default_filters` and the legacy readers do not (200 po_history
  rows against 198 on the sample), and the per-flag CLI path stores `prepared` frames
  rather than canonical ones.
- **Phase three** — flip the default; keep the file path one release longer.
- **Migration** — a `_schema_version` bump needs somewhere for migrations to live. There
  is a version and a refusal, and no migration path yet; the first bump has to bring
  one.
- **Per-document `valid_time`.** One run anchor stands in for all five documents today.
  An inventory snapshot date that differs from the sales anchor is real, and belongs
  with the merge interface below rather than with a guess here.

Then the Snowflake merge interface — `plan_merge` / `apply`, never a single `sync`, with
the classification counts shown before anything is written. Those counts are the
diagnostic: expecting a backfill and seeing 12,000 corrections means the wrong extract,
and `unchanged` is a free integrity check on the key — 100% is the same file twice, 0% is
the key failing to match across sources, which between Snowflake and Excel almost always
means leading zeros.

Two items above depend on master data the coverage worksheet locates: the P0 topology
needs `MARD-INSME` / `SPEME` to tell quarantine from sellable, and the P1 phase-out cap
needs `MARC-AUSDT` for the end date.

**What a replicated source changes, and the one thing it breaks (2026-11-01).** Four
findings from scoping the move, in the order they have to be dealt with.

- **A Snowflake-sourced run would silently claim to be a repeat of the last one.**
  `input_fingerprint` means "two runs sharing this read the same bytes", and
  `InputRecord.identity` delivers that from `sha256`, failing that `batch_id`, failing
  that `<unhashed:{name}>` (`provenance.py:106-114`). A query has no file to hash, so two
  extracts a week apart would reduce to the same name and `compare()` would return
  `identical` — taking drift, scenario and feedback learning with it, since all three
  rest on that verdict. The mechanism that fixes it is already there and already
  commented for exactly this case: a landed batch carries `batch_id`. So the rule is not
  a preference. **Snowflake lands in the fact store and planning reads the store; the
  pipeline never queries Snowflake directly.** One extract per doc_type per run, which
  also makes an as-of replay possible at all.
- **The adapter's freezing mechanism is header-shaped.** `ingest/adapter.py` pins a
  mapping by header fingerprint. A SQL source has no header row; its equivalent is the
  result-set column signature plus a hash of the query text. This is a second *source
  binding* on the same contract, not a second contract. Two contracts per doc_type — an
  Excel one and a Snowflake one — would split the semantics the contracts exist to hold,
  and the divergence would show up as a reconciliation difference nobody could attribute.
- **The overrides layer stops being optional.** The coverage worksheet puts 46 of 131
  canonical fields (35%) in `planner本地` or `派生` — Snowflake will never carry them. It
  does not exist yet: `grep override` finds the ingest declarations and the parameter
  rules, not a master-data overrides table. While everything arrives in one workbook a
  planner can type a MOQ into the file; once facts come from a replicated source and
  parameters do not, there is nowhere to type it. This also forces the convergence this
  document already asks for — one SKU dimension — from advisable to required, because
  Snowflake's zero-padded MATNR and Excel's stripped one are otherwise two entities.
- **Run the probe's section 4 before section 1.** `docs/data_layer/snowflake_probe.sql`
  §4 asks `distinct_load_days`. If the replicated stock table is a current-state
  overwrite (= 1), Snowflake supplies *freshness and no history*, the store remains the
  only place history accumulates, and shadow write becomes the urgent item rather than a
  staged one — history not collected cannot be recovered later. That one query decides
  whether the weekly execution layer below is months away or a year away, so it is worth
  the first afternoon of access rather than the last.

Deliberately not here: moving the rule engine into the database. The front end this
section used to rule out was built once the reason changed from correctness to planner
adoption — [INTERFACE.md](INTERFACE.md) carries that decision and its constraint, which
is unchanged: it writes declarations and overrides, never a fact.

## What a run leaves, and what the screen lets you take back (2026-10-09)

Five findings from running the current build end to end, four of them fixed here. They
have one shape in common and it is worth naming before the list: **each is a capability
that exists in the code and is not reachable from anything anybody runs.** Not an
arithmetic error — those have tests. A method with no caller, a screen with no undo, a
parser that covers one script, a frame published at the grain it was computed at.

**The visual review had no caller. Fixed.** `reporting/kpi_report.py` is 1,978 lines
that render the only output of this pipeline that is a picture — the OTD trend with the
volume panel beneath it, should-be against actual, the cover distribution, forward risk
— and `run_kpi_review` was reachable from the README, from `SKILL.md`, and from a Python
session. No command called it. So a run left a workbook, three CSVs and a note, the
charts were dead code that passed its own tests every time, and the complaint that came
back was the accurate one: *this version threw the visualisations away*. It did, on
2026-08-09, when `ChartBuilder` and `HTMLReportGenerator` were removed — and what
replaced them was wired to nothing.

The CLI calls it now on both paths, it is recorded on the manifest, and the Results
screen renders it ahead of the workbook. A failure there cannot fail the run: it is the
last thing a run does, after everything a planner acts on is written, so it is reported
and the run still ends having produced a plan. `test_cli.py` asserts both call sites,
by counting them in the source — the defect was never subtle, and the test for it does
not need to be either.

**The import screen had no way back. Fixed.** Every upload landed and stayed. Worse than
clutter: `Service.landed_documents` takes the newest batch per document type, so a file
uploaded by mistake was the document a run would have read, and the only correction on
offer was to upload something else and hope it won. There is now a list of what has been
uploaded, which of two claims on a document type is in force, `Remove` per file and
`Start over` for everything not yet promoted. The reasoning — why a fact needs a written
reason and an upload does not — is in [INTERFACE.md](INTERFACE.md) §2 M1.

It found a quieter one on the way: `landed_at` carried **seconds**, and a batch id's
suffix is random, so two files uploaded in the same second — three dropped on the page
at once, the normal way to use it — were indistinguishable by every field that decides
which is newest. Which one a run read was whichever the filesystem happened to list
last. Milliseconds now, and the sort keyed on the id as well.

**A planner's demand matrix headed `2026年1月` was silently discarded. Fixed.** This is
the one that was costing real accuracy. `demand_signal` ranks `demand_timeseries` above
`sales_history` — a planner who bucketed the demand has already made that judgement, and
the pipeline's own bucketing should not quietly outrank it — and the precedence was
correctly implemented and unreachable, because the shape detection never fired. The
period-header parser knew seven Latin spellings and no CJK one, so the file profiled as
`long`, fell through to alias scoring, scored 39% as an inventory report, lost to the
real stock report and was dropped as a duplicate claim. The run then forecast from sales
history and said nothing about it.

Three changes: the CJK forms (`2026年1月`, `26年1月`, `2026年第2季度`); one parser instead
of two, with `readers/timeseries_reader.py` delegating to the profiler's rather than
keeping its own narrower month table; and the console now names both sources when the
matrix wins, because silence there is indistinguishable from the file having been
dropped.

And one more underneath it: **the landing layer deleted a column that was blank on every
row.** The payload omits empty cells, correctly — a wide export is mostly blanks — but
what came back was the union of the keys that happened to be filled, in order of first
appearance. On a demand matrix that is a month deleted rather than a month of zeroes,
and `demand_timeseries` says in as many words that zero is a real no-demand period and
is what drives the intermittency classification. It also scrambled the header order,
which is the one thing a reader of that layer is looking at the file to see. Both fixed
by reindexing onto the header map.

**The S&IOP sheet was published at the grain it was computed at. Fixed.** One row per
SKU per period: ten items and six months is sixty rows for ten decisions, and with a
real catalogue it is thousands of rows for hundreds. A planner holding one item and
asking when it runs short had to read six rows and compare one against another. It is
one row per item now, months across the header under the `<measure> <qty|amt> <period>`
spelling the S&OP worksheet already uses — so the two sheets a meeting opens together
sort the same way — with `first_short_period`, the floor and the buy leading the row.
The reshape is lossless and `test_workbook.py` asserts it cell by cell, because the only
honest defence of a narrower-looking sheet is that nothing left it.

**The README was describing sixteen files that stopped being written on 2026-09-04.**
Fixed here, and recorded as a class rather than a typo: the output consolidation changed
what a run leaves and nothing failed when the documentation went on describing the old
set. There is no test that the README's file list matches what a run writes, and the
manifest now records exactly that, so one is cheap. Not written.

### Left open by this pass

- **A fiscal period header is still unreadable.** `FY26 P01`, `P01 2026`, `2026-W01`.
  Deliberately not guessed: a fiscal month needs a calendar saying where the year starts,
  which is a declaration, not a regex. A **two-row header** — the year on one row, the
  month on the next — is the other common worksheet shape and is the same conversation.
- **A month column with a qualifier attached** — `Jan 2026 (EA)`, `Qty Jan 2026` — parses
  as nothing. Stripping the qualifier is tempting and wrong by default: `Amt Jan 2026`
  beside `Qty Jan 2026` would melt money into a quantity field, and the parser's whole
  stated discipline is to be conservative. It needs the block to be identified first.
- **`--timeseries` and the contract path can read the same file twice, differently.**
  Pass the flag for a file that is also in the directory and `load_all` routes it
  through the contract, then the flag path overwrites the pivot with the legacy
  reader's — which applies `rolling_months` and drops SKUs with no demand in the window,
  and the contract path does not. Which reading a run gets depends on a flag. The
  readers reading through the contracts is the end state already named under the data
  layer; this is one more reason it is the right one.
- **Nothing checks that the documented outputs are the outputs.** See above.

## Forecasting models: six measured, none adopted (2026-11-01)

Nine models from MIT CTL section 3 and the intermittent-demand literature were checked
against the pool. Six were implemented and measured. **None of them is in the pool**,
and one thing was added that cannot be chosen yet. This entry exists so that nobody
re-derives the same six from the same papers in a year and adopts them on the same
reasoning that did not survive measurement here.

### Read the noise floor first

Method: select a model on the pipeline's own rolling backtest, then score *the chosen
forecast* on three months held out of the selection entirely. 610 SKUs of the IN30
extract.

    same pool, scored at 4 backtest origins instead of 6     2.43%

That is the spread from nothing but which origins were used. Every candidate effect
measured below is smaller than it. Nothing here was decided on these numbers alone, and
nothing should be.

### At three origins, which is what ships

    neither                     318.019     —
    SBA                         317.894   -0.04%
    Cumulative                  319.615   +0.50%
    SBA + Cumulative            319.635   +0.51%

### At six origins, which does not

    none                        320.111     —
    SBA                         320.085   -0.01%
    ADIDA (quarterly buckets)   320.640   +0.17%
    Cumulative                  318.532   -0.49%
    ADIDA + Cumulative          317.606   -0.78%
    SBA + ADIDA + Cumulative    317.907   -0.69%

**`Cumulative` changes sign between the two tables** — helpful at six origins, harmful
at three. That is the thin-pool argument measured directly: a candidate pays only when
the chooser has enough evidence to tell whether it is paying, and three origins is nine
pooled points.

### Rejected, with the reason

- **`SBA` (Syntetos-Boylan).** The bias it corrects is real and the correction is a flat
  `1 - beta/2`, a 10% reduction. But **the dominant Croston over-forecast was the
  `n_hat = 1.0` initialisation, and `_croston` already fixes it** — a quarterly item
  used to come out 92% too high. Measured on the items Croston wins, its mean bias is
  already only `+1.06`; the textbook 10% on top takes it to `-3.84`. The correction is
  calibrated for an estimator this one is no longer. Tried three ways — as a rival
  entrant, as a correction inside `_croston`, and not at all — and all three scored
  within 0.05% of each other on MAE, so the bias is the only thing that distinguished
  them.
- **`Cumulative`** — MIT section 3's cumulative model. +0.50% at the shipping origin
  count. Its own theory also says it should get *worse* as history lengthens, since the
  distant past becomes less relevant, so it is unlikely to come back.
- **`Holt-fixed`** — Holt's method with the constants asserted rather than fitted
  (+2.95%). `ETS-trend` already fits the same two, and more history improves the fitted
  one, not the written-down one. Will not change.
- **Scaling the backtest origin count with history.** Built, measured, removed. The
  argument is good — nine pooled points cannot rank nine candidates — but the measured
  benefit was `320.391 -> 320.111`, **0.09%**, against **+60%** backtest cost (6.5 to
  10.5 minutes per 30,000 SKUs). Not worth it at this history length. If the pool is ever
  enlarged this is the prerequisite, and the arithmetic affords 11 origins at 22 months
  and 25 at 36.

### Worth re-testing on more data

- **Damped trend (phi)** — MIT section 3 has it, and names it as the fix for Holt's on a
  non-linear trend; the literature recommends it more strongly than anything else here.
  It measured 0/10 in the best candidate subsets and 5/5 in the worst. But trend fires on
  only 11% of this extract and the scoring window is three months, which is where
  damping has least to offer. Re-test at 36 months, scoring six steps out.
- **`ADIDA` / temporal aggregation** (Nikolopoulos 2011) and **iMAPA** (Petropoulos and
  Kourentzes 2015). The bucket width was assumed quarterly rather than chosen per item,
  and iMAPA — combining across aggregation levels — was never implemented.
- **`ETS-mul-seasonal<m>`** is **in the pool now and inert**: never chosen, because this
  extract has no cycle at any lag. MIT section 3 specifies the multiplicative form and
  the pool only fitted the additive one, so it is there for the history being
  accumulated. A candidate that cannot be chosen cannot cost anything. The thing to
  check is whether it ever wins once a real season exists.
- **Bass diffusion** for a newly launched item, where no model in the pool has history to
  work with. Needs `m`, the market potential, which no extract carries — a judgemental
  input, so it needs somewhere for a planner to state one before it needs code.

### Leads, including one withdrawn

- **The "systematic over-forecast" was ten items, and is withdrawn.** This entry used
  to say that a flat 0.9x shrinkage of the level beat every candidate tested — 315.556
  against a 320.111 baseline, the largest effect measured here — and that the pool
  therefore over-forecasts and wants diagnosing. It was checked, and it does not.

  The mean bias across 610 SKUs is `+70.72` units per SKU-month, which is what that
  reading rested on. The **median is `+0.00`**, and **48% of items are over-forecast**:
  a coin flip, which is what an unbiased forecast looks like. **Ten SKUs carry 47% of
  all absolute bias**, and they are items whose demand moved in the held-out quarter by
  amounts no model could anticipate — 57,994 to 42,982 a month, 12,294 to 3,503, 4,018
  to 22, against 40 to 3,926 the other way. The largest movers happened to fall, so the
  mean came out positive and a 10% haircut "helped". On a window where they rose it
  would hurt by as much. Shrinkage was fitting one sample, not correcting a bias.

  Three checks, in case this is proposed again: the held-out quarter is **+0.8%** on
  aggregate demand against the 19 months before it, so it is not a low window; by model,
  `Croston` is unbiased at `+1.06` while the concentration sits in `ARIMA` (+390) and
  `SES` (+349), which are chosen for the largest and most volatile items; and per-SKU
  median change over the same window is `-1.7%` against a mean of `+194%`, which is the
  same heavy tail saying the same thing.

  **The generalisable part is the mistake, not the result.** A mean over 610 items where
  ten carry half the signal does not describe the catalogue, and this was read as if it
  did — three times in one session, on three different quantities. `forecast_accuracy.py`
  already argues the same thing in its own docstring and refuses to publish a weighted
  average for it: *"a single accuracy number is a scoreboard, and a scoreboard closes a
  discussion."* It is right, and it applies to diagnostics as much as to the report.

- **What the concentration actually implies, and it is not a modelling change.** Half
  the plan's error is a handful of large items moving 26% to 99% in a quarter. No
  candidate model catches that; an item going from 4,018 a month to 22 is an event, not
  a forecasting failure. What catches it is the review loop, and what measures it is the
  per-item ranked accuracy report that already exists — `ForecastAccuracy.by_sku`,
  ranked by value deviation with direction attached, built for exactly this. The thing
  to do is get a committed plan through one S&OP cycle so that report has something to
  score, which is already the open item under the snapshot work, not to add models.

- **Measure through `_select_by_backtest`, never around it.** The first sweep
  reimplemented the chooser and reported `-4.70%` for a combination worth `-0.69%`,
  because the reimplementation ranked on MASE always where the real code ranks on MAPE
  wherever the window holds no zero actual, decided the ARIMA and Croston gates per fold
  where the real code decides them once per series, and had no "no demand in the window,
  defer to TSB" branch. A recommendation went out on that number before it was caught.

### Not diverging by accident

MIT section 3's **MSE estimate via exponential smoothing** (`MSE_t = w(x_t - F_t)^2 +
(1-w)MSE_{t-1}`) is deliberately absent: sigma-DL comes from the backtest error, which is
out-of-sample where the smoothed MSE is in-sample.


## Weekly execution layer (P1)

Decided 2026-11-01. **The planning parameters stay monthly and nothing in them is
re-derived weekly.** That is the substance of the decision and it goes first, because
the obvious reading of "plan weekly" is the wrong one. Re-estimating on weekly buckets
would move `cycles_per_year`, `demand_rolling_cycles`, `min_active_cycles`, the six
`× 12` annualisations and the Croston/TSB eligibility thresholds all at once, reclassify
a large share of the catalogue, and buy nothing: aggregation lowers CV, so a weekly fit
on a C item is worse than the monthly fit divided by 4.33. Monthly parameterisation is
the normal practice and it is kept.

What changes is that a second, thinner layer sits under it at weekly cadence, and the
two answer different questions:

| | Cadence | Question | Mechanism |
|---|---|---|---|
| **Planning** | monthly | what rate should this SKU replenish at, and how far may it drift before the rate itself is wrong | forecast, σ, safety stock, review period, lot — all unchanged |
| **Execution** | weekly | has the balance moved as that rate implies; absorb the difference or escalate it | three flow rates against a stock-snapshot difference — fits nothing, estimates nothing |

The execution layer is a difference engine, not a planning run. It reads two stock
snapshots and three transaction streams, compares the result to a rate handed down from
the month end, and decides between two actions: order at the baseline, or escalate. The
quantities it works in are rates per week; the only month-to-week conversion in the
design happens once, at the handoff below, rather than being spread through the
arithmetic.

### The handoff is the missing object

`analytics/purchase_recommender.py` publishes fifty columns and none of them is a rate.
`order_quantity`, `suggested_po_qty` and `net_requirement` are one-shot quantities;
`forecast_avg_monthly` is a demand rate, not a replenishment rate, and neither carries a
tolerance or an expiry. So the month end has nothing to hand down.

Three fields, derived from figures the recommender already computes:

```
baseline_rate   qty/week   the replenishment rate the policy implies, spread over R
band            qty/week   how far a week may deviate before it means anything
valid_until     date       when the baseline is re-set rather than deviated from
```

**The band comes down from the planning layer; the execution layer never computes its
own.** This is the load-bearing constraint of the whole arrangement. A tolerance derived
weekly from weekly data is an estimate, and an execution layer that estimates is just the
planning layer run four times a month at a grain that does not support it. Scaling the
monthly σ to a weekly band is the one legitimate conversion and it belongs here, in one
place, where it can be read.

The home for this is `feedback/snapshot.py`'s `_published_plan` — already "every period
of the horizon as it was planned" — as additional fields, not a new file. The baseline
is part of the published plan, and scoring it later is the same act as scoring the plan.

### Three flow rates, and the one thing that blocks them

The measurements the planner actually watches week to week are the balance delta and
three rates. Every field they need is already declared in the contracts; **no new
ingestion is required**:

| Weekly measure | Source |
|---|---|
| New order intake rate | `open_so.order_date` |
| Shipment rate | `sales_history.ship_date` |
| Receipt rate | `po_history` / `open_po` dates |
| Balance delta | the difference between two stock snapshots |

The first three are transactional sums over weekly buckets — robust, and nothing in them
needs a distribution. The fourth is the bottleneck, and it is a data-layer problem rather
than an analytics one:

- `store/query.py` already has both time axes and four named readings, so *"what was the
  position on the Wednesday of week 3"* is mechanically answerable today. What is missing
  is that planning is not on the store read path — data layer **phase two / three**.
- **Per-document `valid_time` is a prerequisite, not a refinement.** It is already listed
  under the data layer as open: one run anchor stands in for all five documents. For
  monthly planning that is imprecise. For a weekly difference it is fatal and silent —
  an inventory snapshot sharing the sales anchor's `valid_time` attributes stock movement
  to the wrong week, and the result does not raise; it merely looks slightly odd.

This is the precise sense in which the weekly layer depends on Snowflake, and it is
narrower than "better data". A weekly difference needs snapshots taken on a dependable
cadence with trustworthy per-document timestamps. A hand-exported snapshot will not
happen four times a month reliably, and worse, two exports taken under different filters
difference into a number that is wrong and looks fine. A replicated source supplies both
the cadence and the per-table load timestamp that fills the gap above.

### The deviation rule, and when the baseline is wrong instead

Two bands, because the layering's real question is which layer owns the problem:

- **Inside the band** — order at the baseline. The week needs no planner attention at
  all, which is the point: the default action is "same as the baseline", and attention
  goes only to exceptions.
- **Outside the band, same direction, N consecutive weeks** — do not keep deviating.
  Re-set the baseline, which returns the problem to the planning layer. Without this the
  planner chases with weekly deviations indefinitely, and that end state already has a
  name in `policy/cadence.py`: `oscillating`, the most expensive of its four verdicts.

`cadence.py` is the precedent for the arithmetic and deliberately **not** the host for
it. It computes the cumulative flow balance `cum = Σ(po_qty − sales_qty)` monthly, over
two streams, against sales. The execution layer is the same shape over three streams,
weekly, against an explicit baseline. Same curve, different parameterisation, and folding
one into the other would give the diagnostic two meanings.

### The guard

**A weekly run must not call the orchestrator's planning path.** It would re-fit the
forecast and re-derive safety stock on a series one week longer, so the monthly
parameters would drift week to week — the layering's value destroyed in the first month —
and `compare()` would report `mixed` every week, taking drift and scenario analysis down
with it.

The mechanism for pinning it already exists: the **baseline is an input**, fingerprinted
as one. `config_fingerprint` does not move, `input_fingerprint` does as new facts arrive,
and `provenance.compare()` classifies the week correctly as `new_data`. No new concept is
needed, but the baseline has to be passed in rather than recomputed for that
classification to be true.

It follows that the layer belongs in **`feedback/`, not `analytics/`**. `FeedbackCollector`
already records `actual_demand` / `actual_eom_inv` / `actual_receipt_qty` against a
published plan; the weekly layer is that same loop at weekly cadence, with three flow
rates and a baseline to compare against instead of a single monthly actual. `feedback/`
is already the layer whose premise is that it does not estimate.

### Order

1. **Per-document `valid_time`** (already open under the data layer). Nothing downstream
   is trustworthy without it.
2. **Data layer phase two / three.** The weekly layer reads two instants out of the
   store; it cannot read them from files.
3. **The baseline handoff** — `baseline_rate` / `band` / `valid_until` into
   `_published_plan`, derived in `purchase_recommender`. Independent of 1 and 2, buildable
   now, and it completes the monthly layer's output whether or not the weekly layer
   follows. Touches no existing arithmetic.
4. **The weekly `feedback` loop and the deviation rule**, scoped to A-class first —
   `R-001` already carries `review_period_days: 7`, and `cadence.py`'s
   `CRITICAL_CLASSES = ("A",)` agrees with that boundary. Weekly attention on a C-class
   consumable pays an ordering cost that dwarfs the cycle stock it saves.
5. **Snowflake** turns 1 and 2 from "can run" into "runs every week without anyone
   exporting anything".

### What is deliberately not here

- **A weekly statistical forecast.** See the opening paragraph. If a weekly demand signal
  is ever wanted, the shape is to profile the monthly forecast down to weeks — which
  preserves the model competition in `forecaster.py` — not to fit weekly from scratch.
  The noise floor measured in the section above sharpens this: 2.43% of spread comes
  from nothing but which backtest origins were used, and every candidate model
  measured there moved less than that. A weekly refit would be asking a noisier series
  for an effect that is already below the floor on the monthly one, so the result
  would be unreadable rather than merely worse.
- **A per-SKU weekly DRP grid.** Projected on-hand / planned receipts / planned orders by
  week is the natural-looking artefact and it is not what the layering asks for: it is a
  planning output at a weekly grain, which puts estimation back in the weekly layer. The
  planner's question is the deviation from a rate, and that is four numbers, not a grid.
- **Re-tuning the monthly configs.** `cycles_per_year`, `demand_rolling_cycles`,
  `min_active_cycles`, `forecast_horizon_months` and the `× 12` annualisations all stay
  as they are. The one genuine untidiness found while scoping this is that
  `DAYS_PER_MONTH` is `30.0` in `purchase_recommender.py`, `should_be.py` and
  `suggestions.py` and `30.44` in `cadence.py`, with bare `/ 30.0` inline in
  `safety_stock.py` and `inventory_projector.py`. It is a latent inconsistency worth one
  constant, and it is not a prerequisite for anything above.


## Purchase history: two cleaning layers, and what each one cannot reach (2026-11-07)

`readers.po_history_reader.consolidate_to_receipt_grain` drops the rows of a PO line
that carry no goods receipt, where that line has one. It exists because a
*confirmation-grain* export is finer than the `po_receipt_line` grain the contract
declares: SAP under confirmation control holds one EKES row per vendor confirmation,
so a line arrives twice — an `AB` acknowledgement with no receipt against it and an
`LA` shipping notification carrying the goods receipt. `po_qty` and `delivered_qty`
are line-level values repeated on both rows, so every ordering-pattern figure doubled
— bought against demand, over-ordered, lot-size conformance, replenishment cadence.
The lead time was never affected: the acknowledgement row has no receipt date, so
`lead_time_days` is null on it and the weighted average already dropped it.

Nothing caught it, and that is the part worth remembering. The grain test compares row
count against distinct natural-key count, and `receive_date` is part of that key — so
the two rows of one line differ on exactly the column that would have exposed them.

It runs in two places because the entry paths meet nowhere earlier: the reader's
`_post_process`, and `run_planning` / `run_kpi_review` in the orchestrator. The intake
path never runs the reader — the contract derives `lead_time_days` itself. The same is
true of `POHistoryReader.ensure_lead_time`, and for the same reason.

**The other layer, and why it has to be separate.** Where the extract carries *no*
goods receipt, the two confirmation rows of a line differ on nothing at all: the
natural key degrades to the line, `_apply_rollup` fires, and it **sums** — 100 + 100
into 200. That sum is not recoverable, so nothing downstream of intake can fix it.
The contract therefore declares `repeat_markers: [confirmation_category]`, and
`Adapter._collapse_repeats` drops rows identical but for a marker, between the filters
and the rollup.

Two mechanisms at two layers is not duplication; the cases are genuinely different:

| | rows differ on | rollup | fixed by |
|---|---|---|---|
| receipt in the extract | `receive_date` | does not fire — keys are distinct | `consolidate_to_receipt_grain`, downstream, keyed on the receipt |
| no receipt | nothing but the marker | fires and sums | `repeat_markers`, inside intake, before the sum |

The marker rule is deliberately an exact-duplicate test. "The measures are identical
within the group" would also describe two schedule lines that happen to be equal
halves of a line, and summing *those* is correct — only a column naming the telling
can separate the two cases. A row differing anywhere outside the marker is kept, which
is what leaves partial deliveries and schedule lines intact.

**What is still at the raw grain.** Only the receipt case, since the other is cleaned
inside intake now. Two views of such a file keep the confirmation rows:

- the fact store, which is handed `po_history_as_read` deliberately, for the same
  reason the stock snapshot is: the dropped rows are a real property of the export,
  and history not collected cannot be recovered later
- `_resting_on`, which sizes each unmeasured assumption against the intake result's
  own frames rather than the assembled ones — by the design its docstring sets out,
  because sizing an assumption against a frame whose grain has moved is how a
  plausible wrong number gets made. On a confirmation-grain export the `Σ po_qty`
  it prints behind the currency assumption is therefore twice the planning quantity.

No computed figure is affected — the sample data reshaped to confirmation grain and
the same data at receipt grain produce identical output apart from those two intake
lines — and the exposure number errs toward overstating what rests on the assumption
rather than toward false comfort. So this is a legibility gap, not a wrong plan: a
planner reconciling `Σ po_qty` against their ERP sees twice their number, with the
explanation several lines earlier instead of beside it.

**Why the receipt case is not moved into intake too.** Three options, none free:

- *Clean it in the intake as well.* The store then holds a frame that is not what was
  exported, against the principle above — unless intake hands out two frames, one
  published and one as-read, which is a change to what a document *is* at that layer
  rather than a tweak. Note the asymmetry this would remove: the repeat-marker case is
  already cleaned in intake, so the store already does not hold those rows. The
  principle is being honoured in one case and not the other because in one case the
  sum forced the issue, not because the two were weighed and decided differently.
- *Give `_resting_on` a per-document grain correction.* Narrow and cheap, but it puts
  knowledge of one document type's defect inside the exposure module, which is the
  coupling that docstring is written against.
- *Teach the grain test to see it.* Does not fix the number, but it is the cheapest
  thing that stops the next instance being silent — and there will be a next instance:
  schedule-line and status-event exports arrive in the same shape, and the related item
  under *Smaller items* (transfer and rework receipts mixed into purchase lead times)
  is the same family of defect. The signal is specific enough to test for: a natural
  key containing a timestamp that is null on some rows of an otherwise identical key.

**A routing trap next door.** A purchase export with no goods receipt needs a delivered
or open quantity column to be recognised as history: `po_history`'s discriminators are
`not_null(receive_date)` and `open_qty == 0`, and with neither evaluable such a file
scores as an open PO and is counted as inbound supply — the exact failure
`test_po_history_variants` was written for, which inflates the position and stops the
run recommending purchases. Reproduced at 87% confidence on a dispatch-date extract
stripped of its delivered-quantity column. Nothing is wrong in the code; the shape is
genuinely ambiguous. But a file one column away from being misread as its own opposite
deserves a check that says so, rather than a confidence score nobody reads.

## Lead time measured to dispatch: the variance it cannot see (2026-11-07)

`POHistoryReader.ensure_lead_time` runs the best clock the extract supports. Where
there is a confirmed-delivery or dispatch date and no goods receipt, that is
`(confirmed_delivery_date - po_date) + transit_days`, with the transit leg stated per
supplier in `config/transit_times.json`. `lt_basis` says `dispatch_plus_transit` on
every row it produced, and `transit_days` travels as far as `policy.should_be`, which
prefers it over the `transit_share_of_lt` convention it otherwise guesses with.

Two limits, both inherent rather than bugs:

- **The stated leg has no variance.** A constant adds none, so `lt_std_days` on this
  basis is the variability of the dispatch leg alone and safety stock covers less
  lead-time risk than the number suggests. Reported on every run that uses the basis.
  The fix is not a better constant — it is a second source with real arrival dates
  (an inbound delivery report, a warehouse receipt log), measured per lane.
- **A supplier absent from the table gets no lead time**, falling through to the item
  master's planned delivery time, rather than a lead time short by the whole transit
  leg. That direction is deliberate: short understates the reorder point, which is the
  one that stocks out. `default_days` exists for sites that would rather have one
  number than a gap.

Until then: the drop is reported on every run that triggers it, naming the row and line
counts, the confirmation codes actually found, and the quantity that would have been
counted twice. `confirmation_category` is read as a dimension — never filtered on, so
the rule still works for systems that have no such column — so the codes are on the
frame for anyone checking.

## Smaller items

- `IFR` (item fill rate) service metric — `config/stocking_policy.json` documents
  `service_level_metric` but only `CSL` is implemented; IFR needs `G(k) = Q(1−IFR)/σDL`
  solved numerically.
- Preferred-supplier config per SKU. `SafetyStockCalculator.__init__` reads
  `supplier_incoterm.json` and has a placeholder for it; today the supplier with the
  most PO history stands in.
- Transfer / rework receipts are not distinguished from purchase receipts in PO
  history, so lead times mix them.
