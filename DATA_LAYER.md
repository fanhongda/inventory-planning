# Data layer — where the inputs should live

Most of what has gone wrong in a run has gone wrong at intake, not in the arithmetic.
This is the design decision that follows from that, taken on 2026-08-25. Nothing here
is built yet; it is written down so the next session starts from the conclusion rather
than the question.

## What a database does and does not fix

"Reading errors" is four different problems wearing one name, and only two of them are
a storage question:

| | Example | Does a typed store help? |
|---|---|---|
| **A. Wrong mapping** | which column is the material; whether `Item` is the material or the line number | **No.** That is what `ingest/contracts/` and `ingest/adapters/` already decide. Different storage, same error. |
| **B. Wrong parse** | leading zeros on MATNR, trailing minus (`100-`), comma decimals, dates that follow the user profile, display UoM ≠ base UoM | **Yes, substantially.** A typed column fails loudly where a DataFrame silently keeps a string. |
| **C. Wrong file** | last week's snapshot, one country's tab only, a paginated report truncated, two DCs summed into one | **Yes.** Snapshot metadata plus a row/qty/value reconciliation catches all of these. |
| **D. Missing master data** | MOQ, planned delivery time, whether a storage location is quarantine, phase-out date | Only an overrides layer fixes this. **This is the part that genuinely needs create/update/delete.** |

The declaration layer for A already exists — nine contracts, adapters frozen by header
fingerprint, contract tests. What is missing is persistence, not declaration.

## The decision

**Not** a general-purpose editable SQL database holding SAP reports. Instead:

- **A typed canonical store, append-only.** Facts are never `UPDATE`d. Every load is a
  snapshot with an as-of timestamp, so a past run stays reproducible — the property
  `history/snapshot_*.json` and the transform log are already reaching for.
- **An overrides table, editable.** Corrections and planner-owned parameters live here
  and are applied deterministically *after* intake, with who / when / why on every row.
  Because facts are immutable, update and delete are only ever safe in this layer.

**DuckDB, single file, to start.** It reads CSV/Excel/Parquet directly, needs no server,
speaks SQL, and — being just SQL — moves to Snowflake later without redesign. Do not put
an ORM in front of the readers; that scatters the semantics the contracts hold.

## What "create, update, delete" on a fact actually means

The instinct is a database with CRUD. Taking the cases one at a time, none of them
turns out to need a row updated in place:

| Real case | Looks like | Actually is |
|---|---|---|
| Wrong report, wrong date range, wrong adapter | delete rows | **void a batch** — the rows are never touched |
| ERP amended the document and it was re-exported | update rows | **a new batch**; an as-of read hides the old one |
| A delivery date confirmed by phone, never entered in the ERP | edit one row | **new evidence, not a correction.** The ERP record was not wrong; it said what it said. Overwriting it discards the fact that we know better, which is the whole content of the measured / stated / default ranking the pipeline already uses. Belongs in overrides. |
| A customer name that must genuinely be erased | delete a row | **rewrite a batch** — rare, needs approval, not a daily operation |

So: create is a new batch, read is an as-of query, update is either a new batch or an
override, delete is a batch marked void. What must never exist is an `UPDATE` against
a fact row.

## Why bitemporal is the floor, not a refinement

Every batch carries two timestamps, and they are routinely different:

- `valid_time` — the moment the data describes (the stock snapshot date)
- `transaction_time` — the moment it was loaded

An extract downloaded today may describe last week. With one timestamp, re-importing a
corrected version of last week's file is inexpressible: order by load time and the
correction wins (right), but then no query can reconstruct what we believed last week
(wrong), and that is exactly the question asked when reviewing a recommendation after
the fact.

Two consequences fall out of this and both are cheap only if done first:

- **Raw files must not be readable by the pipeline directly.** A missing as-of
  predicate means one PO appearing once per batch and inbound quantity multiplied by
  the number of loads. It does not raise; the result merely looks high. A `current`
  view is the only thing callers get.
- **`valid_time` must be supplied explicitly**, never inferred from file mtime.

## Where the layers actually sit

The working mental model is a five-stage pipeline: read → store → forecast →
replenish → report. The code says six, and the extra one is load-bearing:

`ingest/` → **`policy/`** → `analytics/` forecast → `analytics/` replenishment → `reporting/`

`policy/` (13 modules) decides *which arithmetic a SKU gets*: the eight evidence axes in
`profile.py`, the rule engine in `parameters.py`, the provenance cross-check in
`crosscheck.py`. Folding it into "replenishment" merges choosing the formula with
running it, and every future module — scenario, feedback learning, macro settings —
attaches to the choosing, not the running.

Two corrections to the same model:

- **The store is a substrate, not a stage.** Reporting and feedback read it directly
  without passing through forecasting. It therefore has to serve two shapes at once: a
  single-grain `current` view for the pipeline, and full history for analysis.
- **Decisions have to be stored with the same rigour as facts.** Recommendations,
  forecasts and resolved parameters are timestamped CSVs today. Feedback learning is
  the question "what did we recommend, and what then happened" — it cannot be asked of
  a CSV directory. `feedback/` already has the skeleton (`collector`, `loss`,
  `snapshot`) and no substrate under it.

### `run_id` is the pivot

A run identity binding *(fact as-of, parameter version, config hash, code sha) →
outputs* is what makes four separate wishes into one mechanism:

- **drift** — the same entity across runs
- **scenario** — the same facts across parameter versions
- **feedback learning** — a run's decisions against subsequent facts
- **a policy UI** — the impact of a rule change, which is a diff of two runs, not a
  state view. `parameters.py` already records per-rule hit counts and skip reasons; it
  prints them and throws them away.

Scenario simulation is therefore not a module to build. It is what falls out once run
identity is recorded — and no amount of UI substitutes for it if that is missing.

### Three kinds of table, three key disciplines

| | Content | Key | Time | Mutability |
|---|---|---|---|---|
| **Facts** | from the ERP / Snowflake | contract `natural_key` + `valid_time` + `batch_id` | bitemporal | append, or void a batch |
| **Parameters and masters** | overrides, topology, supplier, phase-out | entity + `effective_from` | SCD-2 intervals | editable, with who / when / why |
| **Decisions** | each run's recommendations, forecasts, resolved parameters | `run_id` + entity | immutable | new runs only |

Entity keys have to converge. Every join runs on `sku` (on `sku + location` once the
topology lands), and `normalize: material_number` is applied per file today. Persisted,
it has to be applied once into one SKU dimension that everything references — otherwise
Snowflake's zero-padded MATNR and Excel's stripped one become two entities.

## Isolation and rollout

Branching protects the code and not the data: a run started on a branch writes to the
same store. So the store path is the isolation mechanism, not a convenience —
`INVENTORY_PLANNING_STORE` pointing at a dev store is what makes branch work safe.

Four layers, only one of which is version control:

| Layer | Mechanism |
|---|---|
| Code | branch + PR |
| **Data** | separate store path — **matters more than the branch**, because the store is not regenerable |
| Runtime | shadow write, then a read flag |
| Release | `_schema_version` in the store, checked at startup, refusing rather than guessing |

**Shadow write is the rollout.** Phase 1: the pipeline still reads files; the store is
written and read by nobody. Zero risk, and history starts accumulating immediately —
history not collected cannot be recovered later. Phase 2: read from the store behind a
flag and compare field by field against the file path. Phase 3: flip the default. The
merge into `main` happens while the thing still changes nothing.

Note what git does and does not protect. Once the store is outside the repository,
`pull`, `checkout`, `merge`, `reset --hard` and even `clean -fdx` cannot reach it — but
that is not the same as an upgrade being safe. New code can change the store's schema
without git touching a byte, and the bad outcome is not a crash but a silent
misreading. Hence the schema version, from the first version rather than retrofitted.

## Scope of each layer, measured

`docs/data_layer/snowflake_coverage.csv` is generated from the live contracts by
`gen_coverage.py`. As of this writing: **131 canonical fields across 9 contracts, 21 of
them required.**

| Owner | Fields | |
|---|---|---|
| ERP fact | 61 | Snowflake may cover |
| ERP master | 24 | Snowflake may cover |
| Planner-local | 24 | **Snowflake will never have these** |
| Derived | 22 | **Should not be looked for in Snowflake** |

The 46 planner-local + derived fields (35%) are the whole scope of the overrides layer.
That number is also the answer to the front-end question below.

## 1. Snowflake

Usable, under three rules:

**Inventory coverage before connecting anything.** Check all 131 fields against what is
actually replicated. The common real case is that sales, billing and finance are
replicated well while the MRP side (planned delivery time, minimum lot size, safety
stock, schedule-line delivery dates) is not, because the warehouse was modelled for
financial reporting. If those are missing, Snowflake covers two or three of the five
required inputs and the architecture has to be mixed rather than Snowflake-first.

**Read Snowflake through an adapter too.** A Snowflake view is still someone else's
column names and it still changes upstream. The gain over an Excel export is that the
names are stable and the types are real, so the adapter can be frozen permanently and
the profiler's guessing path disappears — not that the contract can be skipped.

**Keep planner-owned data local.** Overrides, node topology, FX, supplier incoterm,
phase-out dates are planning *decisions*, not system records. In git they are
reviewable, diffable and revertible. Push them to Snowflake only when a second planner
needs to share them.

Also: always pull with an as-of timestamp into a snapshot. A live `SELECT` destroys the
ability to reproduce last week's recommendation.

### Three findings that came out of the inventory itself

1. **Two required fields are high-risk.** `open_po.open_qty` has no ready column in SAP
   — it is `EKPO-MENGE` minus goods receipts from `EKBE`, net of 102 reversals. And
   `open_so.customer_request_date` lives in `VBEP` / `VBAK-VDATU`, not `VBAP`.
   Immediately adjacent and equally fatal: `open_po.committed_delivery` is `EKET-EINDT`,
   in the schedule-line table. A replication carrying only EKKO/EKPO loses the delivery
   date column silently, and without it inbound cannot be phased at all.

2. **`inventory.snapshot_date` should be checked today, because it is lost forever.**
   MARD is a current-state table with no date. If the replication overwrites in place
   (probe query 4: `distinct_load_days = 1`), history is being discarded daily and
   cannot be reconstructed afterwards.

3. **The inventory closes two data gaps already logged in TODO.md.**
   `MARD-INSME` (quality inspection) / `SPEME` (blocked) / `UMLME` (in transit) are
   exactly what the P0 topology item needs to stop counting quarantine stock as
   sellable. `MARC-NFMAT` (successor) and `MARC-AUSDT` (effective-out date) feed the
   `substitution` contract directly and supply the end date the P1 phase-out cap
   requires.

## 2. SAP report format as input

**Fine as a transport, wrong as the canonical schema.** The customisation argument
strengthens this rather than weakening it:

- A customised t-code output is *stable for this company*, which makes it the ideal
  adapter candidate: fingerprint it once, freeze it, no guessing afterwards. Record the
  t-code and variant name in the adapter metadata — layout changes ship with a
  transport, and when a run breaks you need to know which report moved.
- SAP's native naming is actively misleading, not merely opaque. The
  `never: sap: [item, itm, position, 项目]` rule in the contracts exists because `Item`
  renders VBAP-POSNR / EKPO-EBELP / MSEG-ZEILE — a line number — while most other ERPs
  mean the material by it. Adopting SAP names as canonical would freeze that trap into
  the schema.
- Report level is lossy: pre-aggregated, carrying subtotal rows and repeated page
  headers, dates as text, quantities in display units, and company code or plant often
  hidden by the variant — which is precisely how two DCs end up summed into one.

Prefer table level (MARA/MARC, EKKO/EKPO/EKET/EKBE, MARD/MBEW, VBAK/VBAP/VBEP) wherever
it is reachable, directly or via replication. Where it is not, keep using the report and
let the adapter absorb the customisation.

Parse traps worth having in front of you: MATNR leading zeros; `NETPR` must be divided
by `PEINH` or prices are out by a factor of 10/100/1000; `MAKT` must be filtered by
language or rows multiply; goods receipts need 102 reversals netted and transfer/rework
receipts separated by document type, which is the same defect TODO.md records as lead
times being mixed.

## 3. Front end

**Not now.** The editable surface is two things and both are small: adapter review /
freeze (once per new report) and overrides (the 46 fields above, a few hundred rows).
The natural form for both is YAML/CSV in git with a validation step — it has diffs,
review and history, which no UI provides for free.

If one is built later, the minimum is Streamlit over DuckDB, roughly a day. The reason
to build it should be **planner adoption** — someone who will not edit YAML — not
correctness.

One hard constraint either way: a UI may write the overrides table only, with who /
when / why, and never `UPDATE` a fact table. An untracked edit to a fact destroys the
reproducibility the transform log and contract tests exist to provide.

## Next step

**Step 1 — make the natural key a real key. Done.** The key is the whole premise of
as-of semantics: a merge cannot classify a row as append / correct / new version unless
it can tell that two rows are the same row. It was not one. Every key part but `sku` was
optional, every call site reduced the key to whatever was present, and the grain test
reported an unverifiable grain as a pass. The line number is now part of the key for the
four line-grain documents, the schedule line has a field of its own so the key column
stops moving between a sample and its parent export, and `KeyStatus` names the
difference between *plannable* (a degraded key is fine) and *storable* (only a complete
one is).

**Step 2 — `run_id` and a run registry.** The pivot above. Verifiable in one run, so it
goes to `main` directly like step 1.

**Step 3 — the store itself**, on a branch behind a PR: move it outside the repository,
close the `history/` split, add `_schema_version`, and shadow-write. This one is not
verifiable in a single run — its value accrues over weeks — which is exactly the
distinction that decides branch versus `main`.

Then, and only then, the Snowflake merge interface: `plan_merge` / `apply`, never a
single `sync`, with the plan printed for a human before anything is written. The
classification counts are themselves the diagnostic — expecting a backfill and being
shown 12,000 corrections means the wrong extract was picked up, and **`unchanged` is a
free integrity check on the key**: 100% means the same file was loaded twice, 0% means
the key is not matching across sources, which for these two sources almost certainly
means leading zeros.

Independently of all of the above, the coverage worksheet still has to be filled in:

1. Run `docs/data_layer/snowflake_probe.sql` queries 1 and 2 to find candidate tables;
   record them in the `sf_*` columns of the CSV.
2. Run queries 3–5 per candidate table: grain, as-of support, material-number shape.
3. Run the query 6 reconciliation — same day, same plant, Snowflake versus the existing
   t-code report, comparing row count, quantity total and value total. **Do not switch a
   source before this passes.** When it does not, look first at posting date versus
   document date, 102 reversals, `LOEKZ` deletion flags, and company-code scope.

The decision rule once it is filled in: **all 21 required fields covered and inventory
carries an as-of → Snowflake-first. Any of `open_qty`, the EKET delivery date, or the
inventory snapshot missing → local canonical store first, Snowflake for history only.**

The `sap_candidate` column is a *candidate to search for*, not an assertion about this
tenant — a customised t-code and a customised replication both move things. The probe
result wins.

## Not in scope

- Editable fact tables. Corrections go through the overrides layer or they are not
  reproducible.
- Uploading the real extract anywhere to make a cloud or mobile session work. It is
  un-anonymised and deliberately outside git; `sample_data/` is synthetic and exists for
  exactly that purpose.
- An ORM over the readers.
- Moving the rule engine into the database. It already exists in `policy/parameters.py`,
  driven by `config/planning_parameters.md`, and what rules need — review, diff,
  rationale, owner — markdown in git gives for free and a table would have to rebuild.
  A scenario runs a different rule file, or a different git ref.
