"""
The batch ledger.

One line per load. Nothing in the fact store is ever updated in place, so every
operation that looks like editing data is expressed here instead: a bad import is a
batch whose status becomes `void`, an amended document is a new batch that supersedes
an old one by carrying a later `valid_time` or `transaction_time`.

Two timestamps rather than one, because they are routinely different and the difference
is the whole point:

    valid_time        the moment the data describes — the stock snapshot date
    transaction_time  the moment it was loaded

An extract downloaded today may describe last week. Ordered by `transaction_time`, a
re-imported correction wins, which is right; but only `valid_time` can answer what was
believed *at* a past moment, which is the question asked when reviewing a decision after
the fact. One timestamp cannot do both.

`valid_time` is never inferred from a file's mtime. A copied file, a re-download, a
sync — all of them rewrite mtime while the data keeps describing whatever it described,
and a store that guesses here corrupts silently rather than loudly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LEDGER_NAME = "batches.jsonl"

STATUS_ACTIVE = "active"
STATUS_VOID = "void"


@dataclass
class BatchRecord:
    batch_id: str
    doc_type: str
    valid_time: str
    transaction_time: str
    rows: int
    source_name: str = ""
    source_sha: Optional[str] = None
    run_id: Optional[str] = None
    # From the contract tests. A batch loaded on a partial key is kept and marked, not
    # refused: it is a perfectly good record of what the file said. What it cannot do is
    # be superseded row by row, and that is what a later merge needs to know.
    key_verdict: Optional[str] = None
    storable: Optional[bool] = None
    status: str = STATUS_ACTIVE
    written_by: str = ""
    path: str = ""
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class VoidRecord:
    """A batch withdrawn. Appended, never applied by deleting the original line."""

    batch_id: str
    voided_at: str
    voided_by: str = ""
    reason: str = ""
    op: str = "void"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class BatchLedger:
    """Append-only record of every batch written under one store root."""

    def __init__(self, root: Path):
        self.path = Path(root) / LEDGER_NAME

    def append(self, record) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def _lines(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line — a process killed mid-write — must not take
                # the rest of the ledger with it.
                continue
        return out

    def batches(self, doc_type: str = None, include_void: bool = False) -> List[Dict[str, Any]]:
        """Every batch, with voids applied. Newest last."""
        voided = {e["batch_id"] for e in self._lines() if e.get("op") == "void"}
        out = []
        for entry in self._lines():
            if entry.get("op") == "void":
                continue
            if doc_type and entry.get("doc_type") != doc_type:
                continue
            if entry["batch_id"] in voided:
                entry = dict(entry, status=STATUS_VOID)
                if not include_void:
                    continue
            out.append(entry)
        return out

    def void(self, batch_id: str, reason: str = "", by: str = "") -> VoidRecord:
        record = VoidRecord(
            batch_id=batch_id,
            voided_at=datetime.now().isoformat(timespec="seconds"),
            voided_by=by,
            reason=reason,
        )
        self.append(record)
        return record

    def has_source(self, doc_type: str, source_sha: str) -> Optional[Dict[str, Any]]:
        """
        Whether these exact bytes were already loaded for this document type.

        Loading the same file twice is the most ordinary mistake there is, and without
        this it produces a duplicate batch that every later as-of read has to
        disambiguate for no reason.
        """
        if not source_sha:
            return None
        for entry in self.batches(doc_type=doc_type):
            if entry.get("source_sha") == source_sha:
                return entry
        return None
