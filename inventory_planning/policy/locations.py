"""
The location status map, edited through the interface.

A third editor on `edits.py`, beside the macro scalars and the rules, and it is a
separate module rather than another `MacroSetting` because the thing being edited is
a different shape. `macro.py` finds the one line a value sits on and replaces the value,
which is what keeps the prose and the formatting of a hand-maintained config file
intact. A map has no one line: declaring `02` is adding a row, and withdrawing it is
removing one.

What does not change is the contract. Propose, read the diff, approve those bytes —
with a reason and a name, recorded in the same append-only log as every other config
change. The one thing worth saying about why this editor exists at all: this is the
only setting in the product that can stop a run, so the surface that sets it has to be
better than a text editor rather than merely equivalent to one. A planner meeting the
refusal is one click from the form that answers it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..node import (
    SELLABLE,
    STOCK_STATUSES,
    STOCK_STATUS_FILE,
    WILDCARD,
    load_stock_status,
)
from .edits import (
    EditRefused, check_basis, collapse, config_root as _config_root, digest as _digest,
    history as _history, record, require_attribution, unified,
)

LocationError = EditRefused

# The keys the file carries besides `statuses`: prose, as every config file here does.
# Preserved across an edit, which is the whole reason the file is rewritten key by key
# rather than dumped from the parsed object.
_PROSE_PREFIX = "_"


@dataclass(frozen=True)
class LocationProposal:
    """A change that has not been made: what it would do, and to which bytes."""

    statuses: Dict[str, str]
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
            "statuses": dict(self.statuses), "diff": self.diff, "basis": self.basis,
            "added": list(self.added), "changed": list(self.changed),
            "removed": list(self.removed), "is_change": self.is_change,
        }


def _path(config_dir=None) -> Path:
    return _config_root(config_dir) / STOCK_STATUS_FILE


def _read(path: Path) -> tuple:
    """The file's text and its parsed object, tolerating its absence."""
    if not path.exists():
        return "", {}
    text = path.read_text(encoding="utf-8")
    try:
        return text, json.loads(text)
    except ValueError as exc:
        raise LocationError(
            f"{path} is not readable as JSON ({exc}). Fix it in an editor before "
            f"editing it here — a form that rewrote it would discard whatever is in "
            f"there now, including the part that is wrong and worth seeing."
        ) from exc


def _clean(statuses: Dict[str, Any], path: Path) -> Dict[str, str]:
    """
    The submitted map, validated and normalised.

    Validated here rather than only on load because the point of a form is to refuse
    before the file moves. A status the pipeline does not know would be treated as
    not-sellable, so a typo would withhold the stock it was meant to admit and the run
    would be quietly short of its own inventory.
    """
    if not isinstance(statuses, dict):
        raise LocationError("statuses must be a map of location code to status.")
    out: Dict[str, str] = {}
    for code, status in statuses.items():
        key = str(code or "").strip().upper()
        if not key:
            raise LocationError("a location code cannot be blank.")
        name = str(status or "").strip().lower()
        if name not in STOCK_STATUSES:
            raise LocationError(
                f"location {key!r} is declared {status!r}, which is not one of "
                f"{', '.join(sorted(STOCK_STATUSES))}.")
        if key in out:
            raise LocationError(
                f"location {key!r} is declared twice. Codes are matched without case "
                f"or surrounding space, so {code!r} and an earlier entry are the same "
                f"location.")
        out[key] = name
    return out


# The `statuses` object, found by its key at the start of a line. `_example` carries a
# nested block of exactly the same shape, so the match is disambiguated by indentation
# rather than by order: the top-level key is the least indented one.
_STATUSES_KEY = re.compile(r'^(?P<indent>[ \t]*)"statuses"[ \t]*:[ \t]*\{', re.M)


def _locate_statuses(text: str):
    """The character span of the top-level `statuses` object, or None."""
    found = list(_STATUSES_KEY.finditer(text))
    if not found:
        return None
    best = min(found, key=lambda m: len(m.group("indent")))
    depth, index, in_string, escaped = 0, best.end() - 1, False, False
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return best.start(), index + 1, best.group("indent")
        index += 1
    return None


def _render(text: str, statuses: Dict[str, str]) -> str:
    """
    The file with the `statuses` block replaced and every other byte left alone.

    The same discipline `macro.py` follows for a scalar, applied to an object: find
    where the value sits and replace it in place. A `json.load`/`json.dump` round trip
    keeps the data and reformats all of it — the blank lines between the prose blocks
    are how this file is read, and losing them makes a two-line change arrive as a
    forty-line diff that nobody can review. The diff is the thing being approved, so
    anything that makes it unreadable defeats the approval.
    """
    body = (json.dumps(dict(sorted(statuses.items())), indent=2, ensure_ascii=False)
            if statuses else "{}")
    located = _locate_statuses(text)
    if located is None:
        # No file yet, or one with no `statuses` key. Build the smallest valid file
        # rather than reconstructing the shipped prose, which would put words in the
        # planner's file that they did not write.
        existing = json.loads(text) if text.strip() else {}
        existing["statuses"] = dict(sorted(statuses.items()))
        return json.dumps(existing, indent=2, ensure_ascii=False) + "\n"

    start, end, indent = located
    # Re-indent the rendered object to sit where the key sits.
    block = body.replace("\n", "\n" + indent)
    return text[:start] + f'{indent}"statuses": {block}' + text[end:]


def current(config_dir=None) -> Dict[str, Any]:
    """The declaration as it stands, with the vocabulary it is written in."""
    root = _config_root(config_dir)
    loaded = load_stock_status(root)
    path = _path(root)
    text, _ = _read(path) if path.exists() else ("", {})
    return {
        "statuses": dict(sorted(loaded.statuses.items())),
        "wildcard": loaded.wildcard,
        "declared": loaded.declared,
        "source": loaded.source,
        "file": STOCK_STATUS_FILE,
        "basis": _digest(text),
        "sellable": SELLABLE,
        "wildcard_key": WILDCARD,
        "vocabulary": [{"name": name, "why": why, "nets": name == SELLABLE}
                       for name, why in STOCK_STATUSES.items()],
    }


def propose(statuses: Dict[str, Any], config_dir=None) -> LocationProposal:
    """What declaring this map would do, without doing it."""
    root = _config_root(config_dir)
    path = _path(root)
    text, _ = _read(path)
    before = load_stock_status(root)
    wanted = _clean(statuses, path)

    was = dict(before.statuses)
    if before.wildcard is not None:
        was[WILDCARD] = before.wildcard
    added = sorted(k for k in wanted if k not in was)
    removed = sorted(k for k in was if k not in wanted)
    changed = sorted(k for k in wanted if k in was and was[k] != wanted[k])

    after = _render(text, wanted)
    return LocationProposal(
        statuses=wanted, diff=unified(text, after, STOCK_STATUS_FILE),
        basis=_digest(text), added=added, changed=changed, removed=removed,
    )


def apply(statuses: Dict[str, Any], *, reason: str, by: str, basis: str,
          config_dir=None) -> Dict[str, Any]:
    """
    Write the declaration, having been shown the diff it makes.

    `basis` is the digest the proposal reported. An apply that does not carry it, or
    carries one that no longer matches, is refused — otherwise someone approves one
    diff and a different edit lands, which is the failure the propose-then-approve
    shape exists to prevent.
    """
    root = _config_root(config_dir)
    path = _path(root)
    text, _ = _read(path)
    require_attribution(reason, by, "a location status declaration")
    check_basis(basis, _digest(text), STOCK_STATUS_FILE)

    proposal = propose(statuses, root)
    if not proposal.is_change:
        raise LocationError(
            "that is what the file already says. Nothing is written and nothing is "
            "logged, because a change log that records non-changes stops being a "
            "record of what moved.")

    after = _render(text, proposal.statuses)
    # Validated by the loader the pipeline uses, not by a second opinion here, so a
    # file this form writes cannot be one the run then refuses.
    path.write_text(after, encoding="utf-8")
    try:
        load_stock_status(root)
    except Exception:
        path.write_text(text, encoding="utf-8") if text else path.unlink()
        raise

    entry = {
        "kind": "location_status", "setting": STOCK_STATUS_FILE,
        "value": dict(proposal.statuses), "added": proposal.added,
        "changed": proposal.changed, "removed": proposal.removed,
        "reason": collapse(reason), "by": collapse(by),
    }
    record(root, entry)
    return {"statuses": dict(proposal.statuses), "added": proposal.added,
            "changed": proposal.changed, "removed": proposal.removed,
            "basis": _digest(after)}


def history(config_dir=None, limit: int = 50) -> List[Dict[str, Any]]:
    """The location declarations, newest first — the shared log, narrowed to this one."""
    return _history(config_dir, limit=limit, kind="location_status")
