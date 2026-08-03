# Adapter store

Frozen mappings from one specific ERP export to a canonical contract. Empty on a fresh
checkout — adapters are drafted on the first run against a new source, reviewed, then
saved here.

## Lifecycle

1. **Draft.** A file with no matching adapter is profiled and an adapter is drafted from
   its column names and data shape. It runs, and everything it guessed is recorded in
   the transform log and the adapter's `notes`.
2. **Review.** Check the drafted `column_map`, `derivations`, `rollup_to` and `parsing`
   against the contract tests. The notes call out anything unresolved.
3. **Freeze.** Set `status: verified` (or `frozen`) and save. From then on the source is
   matched by header fingerprint and the mapping is deterministic — no guessing, no
   model in the loop.

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
adapters/
  <tenant>__<system>/
    open_po.v1.yaml
    inventory.v1.yaml
```

Bump `version` rather than editing a frozen adapter in place, so a mapping change is a
reviewable diff and an old run stays reproducible.

## Why these are data, not code

Supporting a new ERP is a YAML file here, not a change to a reader. The three failure
modes that used to need code — a renamed column, a field that must be *derived* rather
than renamed, and a grain or status vocabulary that differs — are all expressible as
`column_map`, `derivations` and `rollup_to` / `value_maps`. See `../contracts/` for what
each canonical field means.
