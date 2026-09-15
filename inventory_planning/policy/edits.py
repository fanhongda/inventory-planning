"""
What every edit through the interface has in common.

Two editors sit on this — the macro scalars and the rules — and they share a shape
rather than a subject. INTERFACE.md §7 fixed it as a contract so that it survives the
move to a server: propose a change and get the diff it would make, approve that diff,
and the approval names the bytes it was made against.

The three things kept here are the three that must not be reimplemented:

**The basis.** A proposal reports the digest of the file it read; an apply quotes it
back. Without it, approving means "change this thing" when what the person approved was
a specific diff against specific bytes, and a file that moved in between lands a
different change than the one on screen.

**The attribution.** A change made by clicking has only `reason` and `by` to say who
asserted it and why. Both are required, everywhere, for the same reason
`Declarations.write_override` requires them.

**The log.** `config/config_changes.jsonl`, append-only, beside the files it describes,
in the config directory rather than the output directory — a config directory copied to
another machine carries its own history of who changed what. The diff is deliberately
not stored: it is recoverable from the file's own history, and the reason is the part
that is nowhere else.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CHANGE_LOG_NAME = "config_changes.jsonl"


class EditRefused(ValueError):
    """A proposed change that cannot be made, with the reason a person needs."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_root(config_dir=None) -> Path:
    if config_dir is None:
        return Path(__file__).parents[2] / "config"
    return Path(config_dir)


def unified(before: str, after: str, filename: str) -> str:
    """The diff a person approves. Three lines of context, which is what fits a form."""
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{filename}", tofile=f"b/{filename}", n=3))


def require_attribution(reason: Any, by: Any, what: str) -> None:
    """
    Both, always. Refused before the proposal is even recomputed.

    The wording names what is being changed because these carry very different weight:
    a label on an output and a convention that restates every figure in the run are
    both "a change", and a message that said so identically would be telling the
    reader less than it knows.
    """
    if not str(reason or "").strip():
        raise EditRefused(
            f"{what} must carry a reason — it is the only account of why this moved, "
            f"and the file records what it says, never why it says it")
    if not str(by or "").strip():
        raise EditRefused(f"{what} must name who made it")


def check_basis(claimed: Optional[str], actual: str, filename: str) -> None:
    """
    Refuse an approval whose diff was made against different bytes.

    Required rather than optional, and an empty one fails: an approval with no basis is
    an approval of the setting rather than of the change, which is the distinction the
    whole propose-then-approve shape exists to preserve.
    """
    if not claimed or claimed != actual:
        raise EditRefused(
            f"{filename} has changed since that diff was produced. The change is not "
            f"applied: re-read it and approve the diff against the file as it stands "
            f"now.")


def record(config_dir, entry: Dict[str, Any]) -> Path:
    """Append one change to the log, and return where it went."""
    root = config_root(config_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / CHANGE_LOG_NAME
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(entry, at=entry.get("at") or _now()),
                            ensure_ascii=False) + "\n")
    return path


def history(config_dir=None, limit: int = 50, kind: str = "") -> List[Dict[str, Any]]:
    """
    The recorded changes, newest first. A malformed line is skipped, not fatal.

    A log that refused to be read because one line was truncated would take the whole
    history down with the one entry that was damaged, and the history is most wanted
    exactly when something has gone wrong.
    """
    path = config_root(config_dir) / CHANGE_LOG_NAME
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if kind and parsed.get("kind") != kind:
            continue
        entries.append(parsed)
    return list(reversed(entries))[:limit]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def collapse(text: Any) -> str:
    """Whitespace collapsed to single spaces — a reason typed into a form, tidied."""
    return " ".join(str(text or "").split())
