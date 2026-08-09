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

## Smaller items

- `IFR` (item fill rate) service metric — `config/stocking_policy.json` documents
  `service_level_metric` but only `CSL` is implemented; IFR needs `G(k) = Q(1−IFR)/σDL`
  solved numerically.
- Preferred-supplier config per SKU. `SafetyStockCalculator.__init__` reads
  `supplier_incoterm.json` and has a placeholder for it; today the supplier with the
  most PO history stands in.
- Transfer / rework receipts are not distinguished from purchase receipts in PO
  history, so lead times mix them.
