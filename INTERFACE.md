# Interface layer — where a person enters the system

Three questions arrived together on 2026-09-04: should imports be constrained by a
template, what should a front end look like, and what has to be true of the backend for
a later client/server split not to hurt. They are one question wearing three hats —
**at which points does a human decision enter the pipeline, and is that decision
replayable afterwards** — and answering them separately produces three surfaces that
disagree.

Written down like [DATA_LAYER.md](DATA_LAYER.md): nothing here is built. It exists so
the next session starts from the conclusion.

---

## The governing rule

> **Every action the interface offers must emit a declaration, and nothing the interface
> knows may live only in the interface.**

A planner clicking "this column is the material" and a planner typing that into
`config/declarations.yaml` must produce byte-identical state. Not because YAML is nice,
but because three properties fall out of it and are unrecoverable if lost: a headless
run reproduces a UI run, a mapping correction stays diffable and attributable, and the
tests keep testing the thing the planner actually uses.

The corollary is the one hard constraint DATA_LAYER.md already set and which nothing
below relaxes: **the interface writes overrides, declarations and batches. It never
`UPDATE`s a fact row.**

---

## 1. Import templates

### What the surprise actually was

Uncommenting three `scope: value` currency declarations moved the run's numbers by a
large multiple, with nothing in between to say it would. The instinct — "constrain the
input with a template so this stops happening" — treats it as a mapping problem. It was
not one. Every column was mapped correctly. The export simply **did not carry the
fact**, the pipeline filled a default, and the default was wrong by 7×.

A template would not have caught it. A template with a `currency` column would have
moved the same assertion from one reviewable line in git, carrying `reason` / `by` /
`at`, to 9,000 repeated cells in a spreadsheet carrying nothing. That is worse.

So the template question and the surprise question have different answers, and both are
worth doing.

### The answer to the surprise: show the delta

Two changes, neither of them a template:

- **A declaration prints what it changed.** Applying a declaration re-resolves the
  affected documents and reports the before/after on the totals that moved:
  `inventory value 12.4m USD → 12.4m CNY (1.74m USD, ×0.14)`. A declaration that changes
  nothing is reported too — it is usually a declaration that failed to match, which
  intake already detects and should now show at the same weight.
- **Assumptions are acknowledged, not merely printed.** Intake already says which
  figures were assumed (commit `9c1fda0`). The missing half is the exposure behind each
  one — money assumed to be in the reporting currency, quantity assumed to be in base
  UoM, dates assumed to be day-first — sorted by magnitude, presented as a list a person
  clears. Where it is not cleared, the manifest records that it was not. An assumption
  worth 12m at stake and an assumption worth nothing currently read identically.

This is the same principle as measured-outranks-stated, applied to the interface: the
run should be hardest to believe exactly where it is guessing most expensively.

### The answer to templates: three tiers, and only one of them is a template

| Tier | Input | Mapping decided by | Use for |
|---|---|---|---|
| **A** | native ERP export | profiler + frozen adapter + declarations | all five required documents |
| **B** | **template workbook** | fixed by construction | planner-owned data no ERP export carries |
| **C** | `config/declarations.yaml` | a person, through M1 | facts the export omits; per-SKU parameters |

**Tier C is a storage format, not a user surface.** No target user opens the YAML. They
click "this column is the material" and the interface writes the declaration, with
`by` / `at` / `reason` filled from the session. The file keeps its properties — diffable,
attributable, replayable headless — for the same reason a ledger is a file rather than a
screen. Listing it as something a person chooses was wrong.

**Tier B is where a template belongs, and it is not a small category.** DATA_LAYER.md
measured it: of 131 canonical fields, **46 are planner-local or derived — 35% — and
Snowflake will never have them.** Node topology and stock status per storage location
(TODO P0), supplier incoterm, phase-out dates and phase pairs (TODO P1), FX, the S&OP
review worksheet. None of these has a source system. Every one of them is currently a
hand-edited JSON file or does not exist. This is the whole case for templates and it is
a good one.

### The standardisation artifact for tier A is an export spec, not a template

For a user who cannot read a mapping screen, the instinct is to constrain the file they
hand us. But a person who cannot adjudicate `sku ← Material` versus `PartNo.` equally
cannot decide which SAP column to paste into a template's `open_qty`. The template does
not remove the judgement; it relocates it — out of a place with a null rate, a
cross-document overlap, a contract test and a header fingerprint, into a spreadsheet
with none of them, performed by hand, monthly, on nine thousand rows.

What removes the judgement is making the transform happen **in the source system**: a
one-page report specification — t-code, variant name, exact field list — handed once to
whoever administers the ERP, saved as a variant. The user's monthly action becomes run
the variant, export, upload. Zero decisions, no copy-paste, and on our side the layout is
fingerprinted and frozen once, so the mapping screen never appears at all. That is
stronger standardisation than a template, and it is the right ask for this audience.

Where no such administrator is available, a template is the fallback — and then it must
be **self-checking**, because copy-paste fails in specific, detectable ways: numbers
stored as text, merged cells, a partial paste, a single column sorted independently of
its neighbours, a row count or a value total that jumps against last month's upload. The
upload page rejects these in plain language, on the spot.

**Tier A must never be re-keyed into a template.** Requiring a planner to paste an SAP
export into our layout moves the mapping error from a place with a header fingerprint,
a confidence score, a contract test and a transform log, into a place with none of
them — a copy-paste is an unrecorded transform, and it is exactly the manual step that
broke a run before (editing headers in Excel to fix a mapping, which then broke the
adapter freeze). The pipeline's premise is that it reads what the planner already has.
A mandatory template repeals it.

### How a template should be built

**Generated from `ingest/contracts/*.yaml`, never hand-written.** A hand-written
template is a fourth place the field list lives and it drifts on the first contract
change — the same defect as a frozen adapter missing a new field.

    python -m inventory_planning.ingest.templates emit --doc-type node_topology

Three sheets:

1. **data** — canonical headers, in contract order, required columns first.
2. **dictionary** — per field: required / optional, unit, whether it is part of the
   natural key, what it means, and the usual SAP source (`MARD-LGORT`, `EKET-EINDT`).
   The dictionary is the deliverable as much as the sheet is; most of these fields have
   a precise meaning that nothing currently writes down for the person filling them in.
3. **`_meta`** — `template_version`, `doc_type`, contract hash.

Intake recognises `_meta` and routes with a permanent adapter: no profiling, no
scoring, confidence 1.0. **A template is just an adapter whose headers we got to choose
in advance** — zero new machinery, it rides the frozen-adapter path.

One thing a template does *not* buy: correctness. A template file still goes through
contract tests, gates and the intake summary. Getting the mapping right is not the same
as getting the numbers right, and the totals check exists because the second one fails
on its own.

---

## 2. Front end

DATA_LAYER.md deferred this with a condition: build it for **planner adoption**, not for
correctness, since correctness is better served by YAML in git. That condition is now
met. What follows keeps the deferral's constraint while dropping its conclusion.

### M1 — Import and validate

The module with the highest return, because it is the only one that replaces something
a person currently cannot do: review a mapping before it is believed. The CLI has this
(`--no-interactive` implies an interactive path); it is a terminal prompt against a
1,000-column export.

Flow, each step a resource the API exposes:

    upload → land (lossless, store/landing.py) → profile & route
           → review resolution → declare corrections → re-resolve (no re-upload)
           → contract tests + gates → intake summary totals → promote to batch

Three things this screen must show that no current output shows together:

- **The proposed mapping beside its evidence.** `sku ← Material` with the null rate, the
  cross-document overlap coefficient, and the runner-up it beat. The scorer already
  computes all of this and prints a conclusion.
- **The raw landed row beside the canonical row.** Landing keys payload by column index
  precisely so this is recoverable. It is the fastest way a human finds a mis-map, and
  it is the answer to "让用户检查错误".
- **The intake summary totals, at the same weight as the mapping.** Row count, quantity
  total, value total, implied unit value, max date per date column. A value column
  mapped to a quantity is invisible in a mapping log and unmissable in a totals row.

#### Designed for someone who cannot adjudicate a mapping

The target user is less technical than the author of this repo and will never open an
editor. That does not change the architecture — the governing rule already routes every
click into a declaration — but it changes what the review screen is allowed to ask.

- **Ask a question they can answer.** Not "is `sku ← Material` correct?" but *"this file
  says 9,132 materials and ¥12.4m of stock — does that sound right?"* Someone who runs
  the warehouse knows the stock is twelve million and not eighty-seven. The intake
  summary becomes the primary interaction, not a section further down the page.
- **Decide by default; escalate only the close calls.** The scorer produces a confidence
  and a runner-up. Most mappings should never be shown. Show the two or three where the
  margin is thin.
- **Give the consequence, not the evidence.** Not "overlap coefficient 86.5%" but "this
  column matches your other files; the alternative matches 1% of them" — and, where it
  is computable, what being wrong would cost: *"if this is the wrong column, your open PO
  value will read about 200× high."*
- **Refuse rather than default.** Currency on a money document is a required dropdown,
  not a silent fallback to the reporting currency. "What currency is this file booked
  in" is business knowledge this user has; a 7× error is not something they can spot
  afterwards.
- **Help text is generated from the contract**, the same source as the template
  dictionary. Never a second hand-written copy.

Alongside it: a per-document *how to obtain this file* page, and a **"try this file"**
action that runs the full intake, contract tests and gates without promoting anything.
For this audience that button is the highest-leverage item on the page.

Correcting a mapping here writes a `scope: mapping` declaration keyed on **headers**,
appended to `config/declarations.yaml` with `by` / `at` / `reason` — the reason field
required, not optional. Then re-resolve from landing. No re-upload, no adapter re-freeze.

Gate findings render as their three severities (`BLOCK` / `SEVERE` / `WARN`) with the
per-check waiver available inline, `expires` mandatory as it already is in YAML.

### M2 — Macro and policy

Two kinds of setting that must not share a widget:

**Scalars** — default currency, calendar convention, `days_per_year`, FX table, node
topology. These belong in `config/*.json`, edited through a form, written back as a diff
the user approves. Git stays the store; the form is a validator with a nicer keyboard.

**Rules** — review period, service level, replenishment method by segment. These stay in
`config/planning_parameters.md`. DATA_LAYER.md's refusal to move the rule engine into a
database holds: review, diff, rationale and owner are what a rule needs, and markdown in
git gives all four for free. The UI's job here is not editing but **impact**: show the
rule table read-only, and beside each rule the SKUs it hit and the SKUs it skipped —
`policy/parameters.py` already computes both and throws them away.

And the feature that makes the page worth building at all: **a policy change is a diff
of two runs, not a state view.** Edit a parameter file → run as a scenario → diff
against the baseline `run_id`. That is the `run_id` pivot DATA_LAYER.md identified, and
without it a settings page is a form that changes a number a person cannot evaluate.

**Built, read-only.** The settings the engine actually reads with the file each came
from; the conventions, defaults and segmentation as declared; the rules with their scope,
values and rationale; the run registry; and the diff of any two runs with its `basis` —
`scenario`, `new_data`, `identical` or `mixed`. That last is the feature: a parameter
change is not a state to inspect, and whether a difference between two runs is
attributable to the change depends on whether anything else moved. Two runs of the sample
under different parameter files come back `scenario`; the same pair with a commit in
between comes back `mixed`, and refuses to attribute.

Read-only is the design rather than a stage of it, and the page says so. Editing is still
the file.

One thing the page cannot yet show, and says instead of implying: a rule's hit count.
`policy/parameters.py` computes which SKUs each rule reached and which it skipped, prints
it, and throws it away — so there is nothing to display without re-running, and an empty
count would read as "this rule matched nothing", which is a finding rather than an
absence. Retaining `ParameterSet.hits` on the run manifest is the next piece of this
screen and the one worth doing.

> **Checked before building this page, and the answer stood.** Two of the settings
> originally named — calendar versus working days, and a growth target — **do not exist
> anywhere in the engine today**
> (grepped: no `working_day` / `business_day` / `growth` in `inventory_planning/` or
> `config/`). Lead time in working days read as calendar days is a systematic ~1.4×
> error on every safety stock, so the setting is worth having; but a toggle that is
> honoured by nothing is worse than no toggle, because it is read as a guarantee.
> Implement the semantics first, expose the switch second. A growth target, separately,
> is not a macro setting — it is a demand assertion, and it belongs in the S&OP
> worksheet round-trip where sales sign for it.

### M3 — Browse the loaded data

The request is CRUD. The store is append-only. Both are satisfiable, because the four
verbs mean something here that is not an `UPDATE`:

| Verb | What it is |
|---|---|
| **Create** | a new batch — an upload, through M1 |
| **Read** | an as-of query: `doc_type` × `valid_time` × SKU × location × batch, landing row beside canonical row |
| **Update** | either a re-resolve (mapping changed, same landed bytes) or an override row (`who` / `when` / `why`) — never a fact rewritten |
| **Delete** | void a batch — appended to the ledger, original rows untouched |

Built, on the second screen. Picking a document type gives an as-of date, a named
reading, a SKU filter, the batches behind the answer with both their timestamps, and any
one batch shown before and after the adapter — the verbatim rows above, the canonical
ones below. Voiding is done from there, with the reason and the name it requires.

The one thing it promised and does not do is resolve a *single fact row* back to the
verbatim row it came from. The batch id is shared between the two layers now, but the
canonical frame carries no row number, so the pointer `(batch_id, row_no)` is followable
only at batch granularity. Adding a row number to the canonical frame crosses into the
contracts and the adapters, and belongs with P5 rather than here.

**Planning and analysis get no UI.** Excel stays the output, as asked. That is a defence
as much as a preference: the run's five-sheet workbook is what gets handed round a
meeting, and a screen that showed the same numbers would immediately become a second
place they are formatted, rounded and subtly disagreed about.

---

## 3. Backend

One new component.

    Browser (React + TS)                    ← M1/M2/M3
            │  HTTP/JSON
    FastAPI  (inventory_planning/api/)      ← the only new code
            │  in-process import
    inventory_planning/                     ← unchanged
            │
    store/  Parquet: landing · facts · identity · declarations · ledger
            │
    DuckDB  query layer — holds no state

**The API is a wrapper, never a second implementation.** Every endpoint is a thin call
into an existing function. Where an endpoint cannot be thin, the function is missing
from the package and belongs there, not in the web layer — otherwise the UI forks the
semantics the contracts hold, which is the same mistake as putting an ORM over the
readers.

**Whatever the UI can do, the CLI can do, producing identical artefacts.** The UI is a
client, not a mode. This is what stops the headless path rotting.

Surface, roughly:

    POST   /uploads                        → batch_id (landed, not promoted)
    GET    /batches?doc_type=&status=
    GET    /batches/{id}/resolution        → mapping + evidence + runner-up
    PUT    /batches/{id}/resolution        → declare; returns the re-resolve diff
    GET    /batches/{id}/summary           → intake totals + gate findings
    POST   /batches/{id}/promote           → facts
    POST   /batches/{id}/void              → ledger append
    GET    /facts/{doc_type}?as_of=&sku=&location=
    GET    /policy/macro   PUT /policy/macro          → config diff, then apply
    GET    /policy/parameters              → rules + per-rule hits and skips
    POST   /runs                           → run_id (background job)
    GET    /runs/{id}                      → status, gate verdict, workbook path
    GET    /runs/{a}/diff/{b}              → the scenario answer

Long work — a planning run, a large ingest — is a background job keyed on `run_id`,
which already exists and already means the right thing. A thread plus a job record on
disk is enough; a queue is a scaling answer to a problem this does not have.

Auth: none while it is one planner on localhost. The migration is already prepared,
because `by:` is a required field on every declaration and override — the audit trail
predates the login, which is the right order. When it becomes multi-user, put OIDC in
front and fill `by:` from the token instead of the config.

### Front-end stack

Build the API first regardless of the UI. It is the load-bearing part and it is
identical under either choice.

The argument that decides it is not React versus Streamlit. It is that **the UI must be
an HTTP client**. The tempting shortcut is Streamlit calling the package directly — a
day's work, as DATA_LAYER.md estimated — and the problem with it is that it skips the
API. Skipping the API is what makes the C/S move never happen: the UI logic ends up in
Python running in the same process as the store, and the interface's decisions never
become resources.

**M1 shipped as a no-build ES-module client served by the API itself** — three files
under `api/web/`, no toolchain, no `node_modules`. This was a change from the React +
TypeScript recommendation this section originally carried, made once the API existed and
the screen turned out to be five panels. It keeps every property the recommendation was
protecting: it talks only to HTTP, holds no state of its own, and has no second opinion
about a mapping. What it gives up is what React buys at ten times this size, and the cost
of changing course later is the contents of one directory.

Revisit that when the client grows past one screen with shared state across it — a batch
list with live status, a policy editor, and the review wizard at once is where hand-rolled
rendering stops paying.

---

## 4. What the store can actually do — today, and by design

Asked directly on 2026-09-04: can it be queried, changed, deleted. The design answer and
the code answer are different, and the gap is almost entirely on the query side.

### Query

**Designed: yes, and it is the main thing the store is for.** As-of reads over
`doc_type` × `valid_time` × entity, full history or a single-grain `current` view,
landing row beside canonical row, SQL underneath.

**Built** (`store/query.py`, DuckDB over the Parquet). Three readings, each named,
because more than one is defensible and picking silently is how a plausible wrong number
gets made:

| | |
|---|---|
| `history` | every observation, each row carrying its batch. **Never sum this** — one PO appears once per load |
| `current` | the newest observation of every key ever seen |
| `latest` | the newest batch and only it |

`current` and `latest` differ by whatever the newest export dropped, and which is right
is a property of the export — a whole-population snapshot replaces its predecessor, an
incremental window does not — which no contract declares. So `carried_forward` returns
the size of the disagreement rather than a caveat about it, and `current` refuses
outright when the natural key is incomplete, since reducing on a partial key collapses
distinct rows and looks cleaner than not reducing at all.

Both time axes are honoured: `as_of` filters `valid_time`, `known_at` filters
`transaction_time`, and the second is what reconstructs what was believed before a
correction arrived. Promotion applies `KeyStatus.storable` — the gate that was defined
for a fact store and that nothing had yet applied — so an export that cannot be keyed is
refused at write with the declaration that would fix it, rather than accepted and failing
on every read.

Two gaps remain, and both matter before a browse screen is promised:

- **`store/landing.py` is not wired into the pipeline.** The API lands every upload, but
  a CLI run does not, so "raw row beside canonical row" holds only for files that came
  in through the interface.
- **`valid_time` is one anchor for the whole run**, not per document. An inventory
  snapshot dated differently from the sales anchor is not yet expressible from the
  pipeline — the API's promote takes it per batch, which is the shape the pipeline
  should move to.

### Change

**Yes, in two forms, and neither is an `UPDATE` on a fact.** Both exist in code today:

- **Re-resolve** — the mapping was wrong, the bytes were fine. Declare a correction and
  replay. `store/declarations.py` and `config/declarations.yaml` are done (P3) and wired
  into `Intake.__init__(declarations=)` and `InventoryPlanner._run_gate`, covering
  `mapping`, `value` and `parameter` scopes.
- **Override** — the record was right and we know better. A row with who / when / why,
  applied after intake.

### Delete

**Yes, at batch granularity.** `BatchLedger.void()` appends a `VoidRecord`; `batches()`
filters voided ids out and `include_void=True` recovers them. The original rows are
never touched, so a void is itself reversible and auditable. There is no delete of a
single fact row and there will not be one — a genuine erasure rewrites a batch under
approval, which is rare and deliberate.

The short version for a user: **create is an upload, read is an as-of query, update is a
correction with your name on it, delete is withdrawing a whole import.** Four verbs, none
of them destructive, and the reason the store can be trusted to reconstruct what was
believed last week.

## 5. Query engine and scaling

**DuckDB over Parquet, as already decided — and the reason it scales is the part worth
protecting: DuckDB holds no state.**

Facts are Parquet files; batch identity is the ledger. DuckDB reads them in place. It is
a query layer, not a database, and four consequences follow:

- **Unlimited concurrent readers.** Each request opens an in-memory connection over the
  Parquet files. No DuckDB file, no write lock, no single-writer bottleneck — which is
  the constraint that would otherwise make DuckDB the wrong choice for C/S.
- **Writes stay serialized in one process.** Parquet write plus ledger append, behind one
  lock in the API server. Never two processes writing one store.
- **The storage format is engine-independent.** Moving to Trino, ClickHouse or Snowflake
  later is a connection change, because nothing is stored *inside* the engine. This is
  the property that makes "伸缩性" a configuration question rather than a migration.
- **SQL lives in one repository module**, not scattered `duckdb.connect()` calls — the
  same rule, for the same reason, as no ORM over the readers.

Partition Parquet by `doc_type` / `valid_time` so an as-of read touches one partition.

When to move off it: one planner or ten on one server → DuckDB, comfortably; this data
is small (largest real document seen is a whole-entity stock extract, and it fits in
memory). Concurrent writers from separate processes, or multi-tenant isolation, or the
store outgrowing one machine → object storage plus a distributed engine, same Parquet,
same SQL.

---

## 6. Order, and what blocks what

The dependency that decides the sequence: **M1 needs only P1–P3, which are done. M3
needs P4.**

1. **Assumption exposure + declaration impact preview.** No UI. Answers the actual
   surprise, and it is the smallest change on this page.
2. **Template generator** for the tier-B documents (node topology first — it is TODO P0
   and it has no source system at all).
3. **`inventory_planning/api/`** — the batch and resolution endpoints, over what exists.
4. **M1** on top of them. Stop here and evaluate: if a planner will not use M1, M2 and
   M3 are not worth building.
5. **P4 read path, then M3 — both done.** `store/query.py`,
   `POST /batches/{id}/promote`, `GET /facts/{doc_type}`, and the stored-facts screen.
   Not the second half of P4: the pipeline still reads its files, and swapping
   `ingest_bridge` over to the store is its own review.
6. **M2 — done, read-only.** `/policy`, `/policy/macro`, `/runs`,
   `/runs/{a}/diff/{b}`, and the policy screen. The editing half stays unbuilt on
   purpose; what is missing before it is worth revisiting is retained rule hits.

## Not in scope

- Editable fact rows. Unchanged from DATA_LAYER.md.
- Re-keying an ERP export into a template.
- A planning or analysis UI. Excel is the output.
- Settings whose semantics do not exist yet, exposed as switches.
- UI-only state of any kind — sessions, drafts, saved filters that change a result.
- A "continue anyway" button. `allow_degraded` exists for a judgement a threshold got
  wrong, and one click is not that. If it appears in the UI at all it costs a typed
  reason, an expiry, and a line on the output — the same price the YAML charges.
- Deleting a single fact row.
