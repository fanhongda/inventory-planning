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
from typing import Any, Dict, Optional, Union

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
