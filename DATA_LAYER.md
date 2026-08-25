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

Do not write code first. Fill in the worksheet:

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
