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

## Smaller items

- `IFR` (item fill rate) service metric — `config/stocking_policy.json` documents
  `service_level_metric` but only `CSL` is implemented; IFR needs `G(k) = Q(1−IFR)/σDL`
  solved numerically.
- Preferred-supplier config per SKU. `SafetyStockCalculator.__init__` reads
  `supplier_incoterm.json` and has a placeholder for it; today the supplier with the
  most PO history stands in.
- Transfer / rework receipts are not distinguished from purchase receipts in PO
  history, so lead times mix them.
