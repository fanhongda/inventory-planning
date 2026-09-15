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

What this still needs: a **SKU-level diff over two `run_id`s** — which SKUs changed class, what the safety-stock total moved by,
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

Deliberately not here: moving the rule engine into the database. The front end this
section used to rule out was built once the reason changed from correctness to planner
adoption — [INTERFACE.md](INTERFACE.md) carries that decision and its constraint, which
is unchanged: it writes declarations and overrides, never a fact.

## Smaller items

- `IFR` (item fill rate) service metric — `config/stocking_policy.json` documents
  `service_level_metric` but only `CSL` is implemented; IFR needs `G(k) = Q(1−IFR)/σDL`
  solved numerically.
- Preferred-supplier config per SKU. `SafetyStockCalculator.__init__` reads
  `supplier_incoterm.json` and has a placeholder for it; today the supplier with the
  most PO history stands in.
- Transfer / rework receipts are not distinguished from purchase receipts in PO
  history, so lead times mix them.
