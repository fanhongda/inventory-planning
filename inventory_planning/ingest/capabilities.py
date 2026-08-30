"""
Capability-based intake.

The old pipeline demanded five named files. That framing is wrong in a specific,
costly way: it confuses *a document* with *the information the planner needs from it*.
A planner who has already bucketed demand into a SKU time series has fully satisfied
the demand requirement, yet the old intake still asked for sales history and, not
getting it, either failed or silently degraded.

Here the pipeline declares capabilities instead:

    demand_signal      what to forecast          sales_history | demand_timeseries
    lead_time_signal   how long supply takes     po_history | config default
    position_signal    what is on hand           inventory
    inbound_signal     what is already coming    open_po
    commitment_signal  what is already owed      open_so

Each capability names the documents that can supply it and what analysis degrades
when nothing does. The result is that missing inputs produce a specific, honest
statement — "no lead time source, so safety stock falls back to a 45-day assumption
and lead-time variability is excluded from the calculation" — instead of a crash or a
silently thinner number.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Set

# ── Capability definitions ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Capability:
    """One information requirement of the planning pipeline."""

    name: str
    description: str
    required: bool
    suppliers: List[str]                       # doc_types that can satisfy it
    fallback: Optional[str] = None             # what happens with none of them
    degrades: List[str] = dc_field(default_factory=list)

    def __str__(self) -> str:
        return self.name


CAPABILITIES: Dict[str, Capability] = {
    "demand_signal": Capability(
        name="demand_signal",
        description="Historical demand per SKU per period — the series that gets forecast",
        required=True,
        suppliers=["demand_timeseries", "sales_history"],
        fallback=None,   # genuinely fatal: there is nothing to plan against
        degrades=[],
    ),
    "position_signal": Capability(
        name="position_signal",
        description="Current on-hand stock per SKU",
        required=True,
        suppliers=["inventory"],
        fallback=None,
        degrades=[],
    ),
    "lead_time_signal": Capability(
        name="lead_time_signal",
        description="Realised replenishment lead time and its variability",
        required=False,
        # Ordered by authority, not convenience. A lead time computed from receipts is
        # evidence; one typed into a master is an intention. The master values are here
        # because for a SKU with no receipt history they are the only thing standing
        # between the plan and a config-wide guess.
        suppliers=["po_history", "item_master", "planning_master"],
        fallback="config default lead time, zero lead-time variance",
        degrades=[
            "Safety stock loses its lead-time variability term and will be understated",
            "Open PO arrival dates cannot be estimated when the export omits them",
            "Reorder points are based on an assumed rather than observed lead time",
        ],
    ),
    "inbound_signal": Capability(
        name="inbound_signal",
        description="Open purchase orders — supply already committed but not received",
        required=False,
        suppliers=["open_po"],
        fallback="inventory position counts on-hand only",
        degrades=[
            "Inventory position excludes goods in transit, overstating the need to buy",
            "Push-out recommendations cannot be produced",
            "Ordering-pattern diagnostics lose their most recent cycle",
        ],
    ),
    "commitment_signal": Capability(
        name="commitment_signal",
        description="Open sales orders — demand already promised to customers",
        required=False,
        suppliers=["open_so"],
        fallback="backlog treated as zero",
        degrades=[
            "Net requirement excludes backlog and will be understated",
            "Past-due backlog cannot be measured, so on-time delivery is modelled rather than observed",
            "Revenue-at-risk ranking falls back to forecast value instead of committed value",
        ],
    ),
    "cost_signal": Capability(
        name="cost_signal",
        description="Unit cost per SKU — converts quantities into money",
        required=False,
        suppliers=["inventory", "po_history", "item_master", "planning_master"],
        fallback="config default unit cost",
        degrades=[
            "Days-on-hand cannot be expressed in currency",
            "EOQ holding cost uses an assumed unit value, weakening lot-size recommendations",
        ],
    ),
    "order_pattern_signal": Capability(
        name="order_pattern_signal",
        description="Historical order sizes and frequency — the basis for EOQ conformance",
        required=False,
        suppliers=["po_history", "open_po"],
        fallback="ordering behaviour not assessed",
        degrades=[
            "Cannot compare actual order frequency against the economic order quantity",
            "Cannot detect erratic lot sizing or chronic expediting",
        ],
    ),
    "service_signal": Capability(
        name="service_signal",
        description="Evidence of delivery performance against customer request dates",
        required=False,
        suppliers=["open_so"],
        fallback="on-time delivery estimated from projected stock position only",
        degrades=[
            "On-time delivery is modelled, not measured — the KPI cannot be calibrated to actuals",
        ],
    ),
    "planner_baseline": Capability(
        name="planner_baseline",
        description=(
            "The parameters a planner has actually set — safety stock, min/max, review "
            "period — as the benchmark the computed ones are compared against"
        ),
        required=False,
        suppliers=["planning_master"],
        fallback="computed parameters are reported without a baseline to compare to",
        degrades=[
            "Cannot say whether the safety stock in use is above or below what the "
            "measured lead time and forecast error justify",
            "Cannot cross-check the planner's demand and lead-time figures against the "
            "transaction history, so a disagreement in what the two are counting stays hidden",
            "Parameter suggestions are absolute rather than expressed as a change from "
            "what is in force",
        ],
    ),
    "item_dimension": Capability(
        name="item_dimension",
        description=(
            "Master attributes per SKU — nominated supplier, MOQ, order multiple, "
            "product family, lifecycle status"
        ),
        # Required, having been optional. What it used to fall back to was not a
        # thinner number but a different one presented in the same font: MOQ and order
        # multiple from a config-wide default, so a suggested quantity was rounded to
        # a pack size the supplier does not sell; obsolete items replenished against
        # their own history because nothing said they were dead; and the product
        # family guessed from the SKU prefix. None of those announce themselves in the
        # output. A run without master data produces a complete, plausible,
        # unfalsifiable report, which is worse than no report — so it does not run.
        required=True,
        suppliers=["item_master", "planning_master"],
        fallback=None,
        degrades=[],
    ),
    "product_dimension": Capability(
        name="product_dimension",
        description=(
            "The product family each SKU belongs to — the axis sales revenue, "
            "days-on-hand and the S&OP review are grouped along"
        ),
        # Separate from item_dimension because the two fail differently and only one
        # of them is silent. A master with no MOQ column leaves `min_order_qty` empty
        # and every consumer of it can see that. A master with no family column leaves
        # nothing empty: `_infer_family` fills the gap from the SKU prefix, and a
        # revenue-by-family table built on a numbering scheme looks exactly like one
        # built on master data.
        #
        # It is also why this is a capability and not a `required: true` on the field.
        # Required fields carry 75% of the routing score, so demanding one here would
        # halve the score of a master that lacks it and route the file to some other
        # contract entirely — turning "your item master has no product family" into
        # "your item master was read as a planner worksheet".
        #
        # Required, then not. Blocking on it stopped runs a planner legitimately needed:
        # replenishment does not depend on the family at all — safety stock, reorder
        # points and purchase quantities are identical with it and without it. What the
        # family decides is how the results are *grouped*, and a plan a buyer can act on
        # should not be withheld because the rollup above it would be unreliable.
        #
        # So the run proceeds and the gate raises a SEVERE finding instead: not one line
        # among eleven, but the thing every output leads with, naming the tables that
        # are grouping on a guess.
        required=False,
        suppliers=["item_master", "planning_master"],
        fallback="the family is guessed from the SKU prefix, and every row says so "
                 "in `product_family_source`",
        degrades=[
            "Revenue and forecast value by product line group on the part-number "
            "prefix, so the lines are the numbering scheme's and not the business's",
            "Days-on-hand by product line is unusable for the same reason — the "
            "denominator is a group nobody defined",
            "The S&OP worksheet cannot be split for review by the team that owns each "
            "line, because the lines do not correspond to teams",
            "Family-scoped parameter rules in planning_parameters.md match on the "
            "guess, so a rule written for one product line silently applies to another",
        ],
    ),
    "substitution_signal": Capability(
        name="substitution_signal",
        description=(
            "Declared relationships between material numbers — which old number "
            "became which new one, and which pairs merely coexist"
        ),
        required=False,
        # Only the document whose purpose is to declare identity supplies this. An
        # item master's follow-up-material column is read and proposed, never applied,
        # so a run holding one has *not* had its renumberings handled — and a ✓ here
        # would say it had.
        suppliers=["substitution"],
        fallback="every material number is taken to be a distinct material",
        degrades=[
            "A part that changed number is planned as two items — the old number as "
            "dead stock against a history that stopped, the new one as a new item on "
            "too little history to forecast or to stock",
        ],
    ),
    "geography_dimension": Capability(
        name="geography_dimension",
        description=(
            "Where demand came from — the country or region a sale was billed to, the "
            "bottom level of the forecast and sales-review segmentation"
        ),
        # Optional, and staying optional. A single-country extract has nothing to split
        # by and is not a lesser run for it; the segmentation simply has one fewer
        # level. What is not optional is saying so, which is what the degradation does.
        required=False,
        suppliers=["sales_history"],
        fallback="the forecast and the sales review are not split by country",
        degrades=[
            "A distribution centre serving several countries forecasts them as one "
            "series, so a country growing while another shrinks reads as flat demand",
            "The sales review cannot be handed to the people who own each market — "
            "there is one sheet for everyone rather than one per country",
        ],
    ),
    "customer_dimension": Capability(
        name="customer_dimension",
        description="Customer attribution of demand",
        required=False,
        suppliers=["sales_history", "open_so"],
        fallback="analysis is SKU-level only",
        degrades=["Shortage risk cannot be ranked by customer exposure"],
    ),
    "revenue_dimension": Capability(
        name="revenue_dimension",
        description="Sales value of demand",
        required=False,
        suppliers=["sales_history", "open_so"],
        fallback="ranking falls back to quantity",
        degrades=["Shortage risk is ranked by units rather than revenue at risk"],
    ),
}


# ── Resolution ───────────────────────────────────────────────────────────────


@dataclass
class ResolvedCapability:
    """Whether one capability was satisfied, and by what."""

    capability: Capability
    satisfied: bool
    supplied_by: Optional[str] = None
    source_name: Optional[str] = None

    @property
    def name(self) -> str:
        return self.capability.name


@dataclass
class IntakePlan:
    """
    What the pipeline can and cannot do with the documents actually provided.

    This is the object the skill reports to the planner before running, and the one
    the report cites when explaining why a number is an estimate.
    """

    resolved: Dict[str, ResolvedCapability] = dc_field(default_factory=dict)
    documents: Dict[str, str] = dc_field(default_factory=dict)   # doc_type -> source name
    unrecognised: List[str] = dc_field(default_factory=list)
    # (doc_type, capability) a document declares but cannot supply from this extract.
    withheld: List[tuple] = dc_field(default_factory=list)

    # ── Queries ──────────────────────────────────────────────────────────────

    def has(self, capability: str) -> bool:
        entry = self.resolved.get(capability)
        return bool(entry and entry.satisfied)

    def source_of(self, capability: str) -> Optional[str]:
        entry = self.resolved.get(capability)
        return entry.supplied_by if entry and entry.satisfied else None

    @property
    def missing_required(self) -> List[Capability]:
        return [
            r.capability for r in self.resolved.values()
            if r.capability.required and not r.satisfied
        ]

    @property
    def missing_optional(self) -> List[Capability]:
        return [
            r.capability for r in self.resolved.values()
            if not r.capability.required and not r.satisfied
        ]

    @property
    def can_run(self) -> bool:
        return not self.missing_required

    @property
    def degradations(self) -> List[str]:
        """Every specific consequence of every missing optional capability."""
        out: List[str] = []
        for cap in self.missing_optional:
            for item in cap.degrades:
                out.append(item)
        return out

    # ── Reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["  Intake plan", "  " + "-" * 58]
        for doc_type, source in sorted(self.documents.items()):
            lines.append(f"    {doc_type:<20} <- {source}")
        if self.unrecognised:
            lines.append(f"    unrecognised        : {', '.join(self.unrecognised)}")

        lines.append("")
        for name in CAPABILITIES:
            entry = self.resolved.get(name)
            if entry is None:
                continue
            if entry.satisfied:
                lines.append(f"    ✓ {name:<20} from {entry.supplied_by}")
            else:
                mark = "✗" if entry.capability.required else "○"
                tail = "REQUIRED — cannot run" if entry.capability.required else \
                       f"fallback: {entry.capability.fallback}"
                lines.append(f"    {mark} {name:<20} {tail}")

        if self.withheld:
            lines.append("")
            lines.append("  Declared but not supplied by this extract:")
            for doc_type, cap_name in self.withheld:
                lines.append(f"    {doc_type} carries no data for {cap_name} — "
                             f"no column mapped to the field behind it, or the one "
                             f"that did arrived empty")

        if self.degradations:
            lines.append("")
            lines.append("  What this run cannot tell you:")
            for item in self.degradations:
                lines.append(f"    • {item}")

        if self.missing_required:
            lines.append("")
            for cap in self.missing_required:
                lines.append(f"  ✗ Missing {cap.name} — {cap.description}")
                # Two different problems wear the same red, and telling a planner to
                # "supply an item master" when they supplied one is how a real fix
                # gets looked for in the wrong place. A capability is missing either
                # because no document that could carry it arrived, or because one did
                # and the field behind it did not.
                held = [d for d in cap.suppliers if (d, cap.name) in self.withheld]
                if held:
                    for doc_type in held:
                        lines.append(
                            f"      {self.documents.get(doc_type, doc_type)} was read "
                            f"as a {doc_type}, and nothing in it mapped to the field "
                            f"this needs. The column is probably there under a name no "
                            f"alias matched — run "
                            f"`python -m inventory_planning.explain <file>` to see "
                            f"which headers went unmatched, then add the right one as "
                            f"an alias or map it in an adapter."
                        )
                else:
                    lines.append(
                        f"      No document supplied it. Provide one of: "
                        f"{' or '.join(cap.suppliers)}."
                    )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": dict(self.documents),
            "capabilities": {
                name: {
                    "satisfied": r.satisfied,
                    "supplied_by": r.supplied_by,
                    "source": r.source_name,
                    "required": r.capability.required,
                }
                for name, r in self.resolved.items()
            },
            "can_run": self.can_run,
            "degradations": self.degradations,
            "missing_required": [c.name for c in self.missing_required],
        }


class CapabilityResolver:
    """Turns a set of identified documents into an IntakePlan."""

    def resolve(
        self,
        documents: Dict[str, str],
        unrecognised: List[str] = None,
        withheld: Dict[str, Set[str]] = None,
    ) -> IntakePlan:
        """
        `documents` maps doc_type -> source name, as produced by routing.

        A capability is satisfied by the first supplier present, in the order the
        capability declares — so demand_timeseries outranks sales_history, because a
        planner who pre-aggregated the demand has already made that judgement and the
        pipeline should not silently prefer its own bucketing over theirs.

        `withheld` maps doc_type -> capabilities that document declares but cannot
        actually supply, because the fields behind them arrived empty. A purchase
        history with no goods-receipt date is the case this exists for: it is a real
        record of ordering behaviour and no record of lead time at all. Counting it as
        a lead-time source would report a measured lead time that was never measured,
        and suppress the item-master fallback that should have covered for it.
        """
        withheld = withheld or {}
        plan = IntakePlan(documents=dict(documents), unrecognised=list(unrecognised or []))

        for name, cap in CAPABILITIES.items():
            supplier = next(
                (s for s in cap.suppliers
                 if s in documents and name not in withheld.get(s, set())),
                None,
            )
            plan.resolved[name] = ResolvedCapability(
                capability=cap,
                satisfied=supplier is not None,
                supplied_by=supplier,
                source_name=documents.get(supplier) if supplier else None,
            )

        for doc_type, caps in sorted(withheld.items()):
            for cap_name in sorted(caps):
                plan.withheld.append((doc_type, cap_name))
        return plan

    @staticmethod
    def requirements_text() -> str:
        """Human-readable statement of what the pipeline needs, for the intake prompt."""
        lines = ["The planning run needs the following information:"]
        for cap in CAPABILITIES.values():
            if not cap.required:
                continue
            lines.append(
                f"  REQUIRED  {cap.name:<18} — {cap.description}\n"
                f"            supply any of: {', '.join(cap.suppliers)}"
            )
        for cap in CAPABILITIES.values():
            if cap.required:
                continue
            lines.append(
                f"  optional  {cap.name:<18} — {cap.description}\n"
                f"            without it: {cap.fallback}"
            )
        return "\n".join(lines)
