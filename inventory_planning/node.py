"""
The planning node: which location the run is planning for.

Every output carries `location_id` so the pipeline is ready for more than one
warehouse. Where a source names its own plant, that value is the answer and nothing
may overwrite it. Where none does, the configured node is stamped instead — and that
stamp is the part worth being careful about, because it is a value the pipeline
invented rather than read.

`DC-01` is what ships in `config/node_config.json`. It is a placeholder, not a
warehouse, and until someone edits that file it appears on every row of every output
looking exactly like data. A planner reading `DC-01` next to their stock has no way
to tell whether it came from their ERP or from a file nobody opened.

So the placeholder is tracked as a placeholder. `PlanningNode.is_placeholder` says
whether the value in force is still the shipped one, and callers report it rather than
passing it off as a location.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# What `config/node_config.json` ships with. Not a location — a stand-in for the one
# the user is meant to enter.
PLACEHOLDER_LOCATION = "DC-01"


@dataclass(frozen=True)
class PlanningNode:
    """The configured planning location, and whether it was ever configured."""

    location_id: str
    is_placeholder: bool
    config: Dict[str, Any]

    def stamp_note(self, doc_types) -> Optional[str]:
        """
        A warning for documents that carried no location and took the placeholder.

        None when the node has been configured, or when every source named its own
        plant — in both cases `location_id` says something true and needs no caveat.
        """
        doc_types = sorted(doc_types)
        if not self.is_placeholder or not doc_types:
            return None
        return (
            f"  ⚠ location_id is {self.location_id!r} on {', '.join(doc_types)} — that is "
            f"the placeholder config/node_config.json ships with, not a warehouse. "
            f"Neither the export nor the config named one. Set location_id in "
            f"config/node_config.json, or map the source's plant column, before reading "
            f"the location on any output as real."
        )


def load_planning_node(config_dir: Union[str, Path]) -> PlanningNode:
    """Read `node_config.json`, defaulting to the placeholder when it is absent."""
    path = Path(config_dir) / "node_config.json"
    config: Dict[str, Any] = {}
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    location = str(config.get("location_id") or PLACEHOLDER_LOCATION)
    return PlanningNode(
        location_id=location,
        is_placeholder=location == PLACEHOLDER_LOCATION,
        config=config,
    )


# ── Stock status per storage location ────────────────────────────────────────
#
# A warehouse export is one row per SKU x storage location, and the codes mean
# different things: `01` is sellable, `02` is under inspection, another is blocked or
# consignment. Summing them produces a position that is overstated by exactly the
# quantity that cannot be sold — and the run then *under-orders*, which is the
# expensive direction to be wrong in.
#
# Nothing in any extract says what a code means. `MARD-INSME` / `SPEME` would, where
# a replicated source carries them, but the semantics of a storage location is
# master data the planner holds and the ERP report does not print. So it is declared,
# in `config/location_status.json`, and the declaration is **required** rather than
# defaulted: a default of "everything is sellable" is the bug, written down.
#
# This is the same discipline `PLACEHOLDER_LOCATION` above follows and for the same
# reason. The difference is the consequence. An unconfigured `location_id` puts a
# wrong label on the output and every figure is still right; an undeclared location
# code puts a wrong *number* in the position and every figure downstream of it moves.
# A label is reported. A number refuses.

STOCK_STATUS_FILE = "location_status.json"

# Only `sellable` nets into the planning position. The rest are carried as reported
# quantities so the stock is visible without being counted on.
SELLABLE = "sellable"
STOCK_STATUSES: Dict[str, str] = {
    SELLABLE: "on hand and available to sell — nets into the planning position",
    "quarantine": "under quality inspection — present, not available",
    "blocked": "blocked stock — present, not available",
    "consignment": "on site but not owned until consumed",
    "in_transit_staging": "staged for a transfer, not available at this node",
}

# Declaring every code sellable in one keystroke. Offered because a bin-level export
# can carry dozens of codes that genuinely are all sellable, and refusing to run until
# forty lines are typed would get the file filled in carelessly, which is worse than
# the wildcard. It is reported on every run that uses it: the planner should be able
# to see, from the console, that the question was answered by a blanket rather than by
# an inventory of codes.
WILDCARD = "*"


class UndeclaredStockLocation(ValueError):
    """
    A stock snapshot holds location codes whose meaning has not been declared.

    Raised rather than defaulted. The alternative is to sum them, which is what the
    pipeline did until this existed, and it is wrong in one direction: quarantine and
    blocked stock counted as available overstates the position and under-orders.
    """


@dataclass(frozen=True)
class StockStatusMap:
    """What each storage-location code means, as the planner declared it."""

    statuses: Dict[str, str]
    source: str
    wildcard: Optional[str] = None

    @property
    def declared(self) -> bool:
        return bool(self.statuses) or self.wildcard is not None

    def status_of(self, code: Any) -> Optional[str]:
        """The declared status of one code, or None where it was never declared."""
        key = str(code or "").strip().upper()
        if not key:
            # A row with no location at all is the single-node case: there is nothing
            # to separate, and the snapshot is the position by construction.
            return SELLABLE
        if key in self.statuses:
            return self.statuses[key]
        return self.wildcard

    def undeclared(self, codes) -> List[str]:
        """Those of `codes` with no declared meaning, in a stable order."""
        seen = {str(c or "").strip().upper() for c in codes}
        return sorted(c for c in seen if c and self.status_of(c) is None)


def load_stock_status(config_dir: Union[str, Path]) -> StockStatusMap:
    """
    Read `location_status.json`.

    An absent file and an empty one are the same state — nothing declared — and are
    not an error here. The refusal belongs at the point where undeclared codes are
    actually about to be summed, because that is where the quantities are known and
    where a message can name them.
    """
    path = Path(config_dir) / STOCK_STATUS_FILE
    if not path.exists():
        return StockStatusMap(statuses={}, source=f"no {STOCK_STATUS_FILE}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    declared = raw.get("statuses") or {}
    if not isinstance(declared, dict):
        raise UndeclaredStockLocation(
            f"{path}: 'statuses' must be a map of location code to status, "
            f"not {type(declared).__name__}.")

    statuses: Dict[str, str] = {}
    wildcard: Optional[str] = None
    for code, status in declared.items():
        name = str(status or "").strip().lower()
        if name not in STOCK_STATUSES:
            raise UndeclaredStockLocation(
                f"{path}: location {code!r} is declared {status!r}, which is not one "
                f"of {', '.join(sorted(STOCK_STATUSES))}. A status the pipeline does "
                f"not know would be silently treated as not sellable.")
        key = str(code).strip().upper()
        if key == WILDCARD:
            wildcard = name
        else:
            statuses[key] = name
    return StockStatusMap(statuses=statuses, source=str(path), wildcard=wildcard)
