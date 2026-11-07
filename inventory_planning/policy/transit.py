"""
The supplier transit table, edited through the interface.

A fourth editor beside the macro scalars, the rules and the location statuses, and it
is its own module for the same reason `locations.py` is: the thing being edited is a
different shape. `macro.py` replaces a scalar on the line it sits on; `locations.py`
replaces an object; this replaces the rows of a CSV.

What does not change is the contract. Propose, read the diff, approve those bytes —
with a reason and a name, recorded in the same append-only log as every other config
change.

Why this setting has an editor at all, when most config does not: it is the only place
the transport mode and the transit leg can come from. No purchase export carries them,
so `lead_time_days = (confirmed dispatch - order) + transit_days` has a term that
exists nowhere but here, and a supplier missing from the table gets no lead time from
that path at all. That makes it a question only a planner can answer — which is what
`/setup` is for — and it makes the answer worth collecting through something better
than a text editor.

Unlike the location statuses it cannot stop a run. An undeclared location puts a wrong
*number* in the position; a missing transit time withholds one, and the item falls back
to the planned delivery time in the item master, labelled as the stated value it is. So
the `/setup` item is not blocking, and the editor says what the absence costs rather
than refusing.

**The comment header is preserved byte for byte.** It carries the whole explanation of
what the file is for and what a stated transit time costs in accuracy, and a round trip
through a CSV writer would drop it. The same argument `locations.py` makes about the
blank lines between its prose blocks: the diff is the thing being approved, so anything
that makes the diff unreadable defeats the approval.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..readers.po_history_reader import TRANSIT_FILE, TRANSIT_WILDCARD, _text
from .edits import (
    EditRefused, check_basis, collapse, config_root as _config_root, digest as _digest,
    history as _history, record, require_attribution, unified,
)

TransitError = EditRefused

COLUMNS = ("supplier", "transport_mode", "transit_days")

# Modes offered by the form. Free text is still accepted — a business with a mode this
# list has never heard of should not be blocked by a dropdown — but an offered list is
# what stops the same mode arriving under four spellings.
MODES = ("AIR", "SEA", "ROAD", "RAIL", "COURIER")


@dataclass(frozen=True)
class TransitRow:
    """One supplier's declaration."""

    supplier: str
    transport_mode: Optional[str]
    transit_days: float

    def as_dict(self) -> Dict[str, Any]:
        return {"supplier": self.supplier, "transport_mode": self.transport_mode or "",
                "transit_days": self.transit_days}


@dataclass(frozen=True)
class TransitProposal:
    """A change that has not been made: what it would do, and to which bytes."""

    rows: List[TransitRow]
    diff: str
    basis: str
    added: List[str]
    changed: List[str]
    removed: List[str]

    @property
    def is_change(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows": [r.as_dict() for r in self.rows], "diff": self.diff,
            "basis": self.basis, "added": list(self.added),
            "changed": list(self.changed), "removed": list(self.removed),
            "is_change": self.is_change,
        }


def _path(config_dir=None) -> Path:
    return _config_root(config_dir) / TRANSIT_FILE


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _split(text: str) -> tuple:
    """
    The leading comment block and everything after it.

    Split on the first line that is neither blank nor a comment — the header row. What
    comes before is prose the planner reads and nothing here may rewrite.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return "".join(lines[:index]), "".join(lines[index:])
    return text, ""


def _parse(text: str) -> List[TransitRow]:
    """The rows as they stand, tolerating an absent or header-only file."""
    _, body = _split(text)
    if not body.strip():
        return []
    try:
        frame = pd.read_csv(io.StringIO(body), dtype=str)
    except (ValueError, pd.errors.ParserError):
        raise TransitError(
            f"{TRANSIT_FILE} is not readable as CSV. Fix it in an editor before "
            f"editing it here — a form that rewrote it would discard whatever is in "
            f"there now, including the part that is wrong and worth seeing.")
    if "supplier" not in frame.columns:
        return []
    out = []
    for row in frame.itertuples(index=False):
        days = pd.to_numeric(getattr(row, "transit_days", None), errors="coerce")
        if pd.isna(days):
            continue
        mode = _text(getattr(row, "transport_mode", None)).upper() or None
        out.append(TransitRow(_text(row.supplier).upper(), mode, float(days)))
    return out


def _clean(rows: Any) -> List[TransitRow]:
    """
    The submitted rows, validated and normalised.

    Refused before the file moves, the same reason `locations.py` gives: a transit time
    that is negative or unparseable would produce a lead time shorter than the dispatch
    leg alone, and a short lead time understates the reorder point — the error that
    stocks out.
    """
    if not isinstance(rows, list):
        raise TransitError("rows must be a list of {supplier, transport_mode, "
                           "transit_days}.")
    out: List[TransitRow] = []
    seen = set()
    for entry in rows:
        if not isinstance(entry, dict):
            raise TransitError("each row must be an object with a supplier and "
                               "transit_days.")
        supplier = str(entry.get("supplier") or "").strip().upper()
        if not supplier:
            raise TransitError("a supplier cannot be blank. Use "
                               f"{TRANSIT_WILDCARD!r} for the row that answers for "
                               f"every supplier not named.")
        if supplier in seen:
            raise TransitError(
                f"supplier {supplier!r} is declared twice. Suppliers are matched "
                f"without case or surrounding space, so this and an earlier entry are "
                f"the same row.")
        days = pd.to_numeric(entry.get("transit_days"), errors="coerce")
        if pd.isna(days):
            raise TransitError(
                f"supplier {supplier!r} has no usable transit_days. Leave the supplier "
                f"out altogether to say it is unknown — that falls back to the item "
                f"master's planned delivery time, where a blank here would be read as "
                f"zero transit and understate the lead time.")
        if days < 0:
            raise TransitError(f"supplier {supplier!r} has transit_days {days:g}, "
                               f"which is not a duration.")
        mode = str(entry.get("transport_mode") or "").strip().upper() or None
        seen.add(supplier)
        out.append(TransitRow(supplier, mode, float(days)))
    # The wildcard last, so the file reads as "these suppliers, and everyone else".
    return sorted(out, key=lambda r: (r.supplier == TRANSIT_WILDCARD, r.supplier))


def _render(text: str, rows: List[TransitRow]) -> str:
    """The file with its rows replaced and its comment header left alone."""
    header, _ = _split(text) if text.strip() else ("", "")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        days = row.transit_days
        writer.writerow([row.supplier, row.transport_mode or "",
                         f"{days:g}"])
    return header + buffer.getvalue()


def current(config_dir=None) -> Dict[str, Any]:
    """The table as it stands, with the vocabulary the form offers."""
    root = _config_root(config_dir)
    text = _read(_path(root))
    rows = _parse(text)
    named = [r for r in rows if r.supplier != TRANSIT_WILDCARD]
    wildcard = next((r for r in rows if r.supplier == TRANSIT_WILDCARD), None)
    return {
        "rows": [r.as_dict() for r in rows],
        "named": len(named),
        "wildcard": wildcard.as_dict() if wildcard else None,
        "declared": bool(rows),
        "file": TRANSIT_FILE,
        "basis": _digest(text),
        "wildcard_key": TRANSIT_WILDCARD,
        "modes": list(MODES),
    }


def propose(rows: Any, config_dir=None) -> TransitProposal:
    """What declaring these rows would do, without doing it."""
    root = _config_root(config_dir)
    text = _read(_path(root))
    wanted = _clean(rows)

    was = {r.supplier: r for r in _parse(text)}
    now = {r.supplier: r for r in wanted}
    added = sorted(k for k in now if k not in was)
    removed = sorted(k for k in was if k not in now)
    changed = sorted(k for k in now if k in was and (
        was[k].transit_days != now[k].transit_days
        or (was[k].transport_mode or "") != (now[k].transport_mode or "")))

    after = _render(text, wanted)
    return TransitProposal(rows=wanted, diff=unified(text, after, TRANSIT_FILE),
                           basis=_digest(text), added=added, changed=changed,
                           removed=removed)


def apply(rows: Any, *, reason: str, by: str, basis: str,
          config_dir=None) -> Dict[str, Any]:
    """
    Write the table, having been shown the diff it makes.

    `basis` is the digest the proposal reported. An apply that does not carry it, or
    carries one that no longer matches, is refused — otherwise someone approves one
    diff and a different edit lands, which is the failure the propose-then-approve
    shape exists to prevent.
    """
    root = _config_root(config_dir)
    path = _path(root)
    text = _read(path)
    require_attribution(reason, by, "a supplier transit declaration")
    check_basis(basis, _digest(text), TRANSIT_FILE)

    proposal = propose(rows, root)
    if not proposal.is_change:
        raise TransitError(
            "that is what the file already says. Nothing is written and nothing is "
            "logged, because a change log that records non-changes stops being a "
            "record of what moved.")

    after = _render(text, proposal.rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(after, encoding="utf-8")
    # Read back with the loader the pipeline uses, so a file this form writes cannot be
    # one the run then reads differently.
    try:
        _parse(_read(path))
    except Exception:
        path.write_text(text, encoding="utf-8") if text else path.unlink()
        raise

    record(root, {
        "kind": "supplier_transit", "setting": TRANSIT_FILE,
        "value": [r.as_dict() for r in proposal.rows],
        "added": proposal.added, "changed": proposal.changed,
        "removed": proposal.removed,
        "reason": collapse(reason), "by": collapse(by),
    })
    return {"rows": [r.as_dict() for r in proposal.rows], "added": proposal.added,
            "changed": proposal.changed, "removed": proposal.removed,
            "basis": _digest(after)}


def history(config_dir=None, limit: int = 50) -> List[Dict[str, Any]]:
    """The transit declarations, newest first — the shared log, narrowed to this one."""
    return _history(config_dir, limit=limit, kind="supplier_transit")
