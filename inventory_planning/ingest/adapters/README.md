# Adapter store

Frozen mappings from one specific ERP export to a canonical contract.

**This directory is the shipped half, and it is read-only.** What lives here is an
export template general enough to be reviewed in a diff and installed with the code.
Adapters drafted against *your* sources are written outside the working tree, to
`$INVENTORY_PLANNING_ADAPTERS` or `~/.local/share/inventory-planning/adapters`.
`AdapterRegistry` reads both, shipped first, and a local adapter displaces a shipped one
of the same identity and version. `save()` only ever writes to the local store.

The split exists because a frozen adapter is two things at once. It is production
configuration — `route()` tries every stored fingerprint before it classifies anything,
so a stored adapter decides how a real export is read, ahead of inference, whatever its
status. It is also a description of somebody's business: the columns their master data
is keyed on, which regions use a local code, which fields are not clean. Inside the
working tree the first makes it one `git clean -fdx` from gone, and the second makes it
one `git add -A` from published. `../location.py` has the resolution order.

An older build wrote drafts here. If `git status` shows adapter directories as
untracked, those are from that build: still read, still ahead of inference, and frozen
against whatever the contract looked like at the time. A run names them and says what
to do with them.

## Lifecycle

1. **Draft.** A file with no matching adapter is profiled and an adapter is drafted from
   its column names and data shape. It runs, and everything it guessed is recorded in
   the transform log and the adapter's `notes`.
2. **Review.** Check the drafted `column_map`, `derivations`, `rollup_to` and `parsing`
   against the contract tests. The notes call out anything unresolved.
3. **Freeze.** Set `status: verified` (or `frozen`) and save. From then on the source is
   matched by header fingerprint and the mapping is deterministic — no guessing, no
   model in the loop. `save()` returns the path it wrote, which is in the local store,
   not here.

   Status is a record of review, not a gate: `route()` prefers a `frozen` adapter only
   when two fingerprints match at the same confidence. A `draft` left in the store still
   decides how its export is read.

```python
from inventory_planning.ingest.registry import AdapterRegistry

registry = AdapterRegistry()
adapter = intake_result.get("open_po").adapter
adapter.column_map["sku"] = "Material"      # correct anything wrong
adapter.status = "verified"
registry.save(adapter)                      # -> <tenant>__<system>/open_po.v1.yaml
```

## Layout

```
inventory_planning/ingest/adapters/     # shipped, under version control, read-only
  <tenant>__<system>/
    planning_master.yaml

~/.local/share/inventory-planning/adapters/    # this installation's own, written here
  <tenant>__<system>/
    open_po.v1.yaml
    inventory.v1.yaml
```

Bump `version` rather than editing a frozen adapter in place, so a mapping change is a
reviewable diff and an old run stays reproducible.

## Why these are data, not code

Supporting a new ERP is a YAML file — here if the template is general, in the local
store if it describes one installation's export — not a change to a reader. The three failure
modes that used to need code — a renamed column, a field that must be *derived* rather
than renamed, and a grain or status vocabulary that differs — are all expressible as
`column_map`, `derivations` and `rollup_to` / `value_maps`. See `../contracts/` for what
each canonical field means.
