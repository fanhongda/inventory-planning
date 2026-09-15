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

from ..attribution import SELF_ASSERTED, resolve_actor

LEDGER_NAME = "batches.jsonl"


def _without_empty_basis(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop `by_basis` when there is none to record.

    A marker on every record distinguishes nothing, and self-asserted is what every
    record in the store already is. The field appears the day a token fills a name,
    which is the only day it could have meant anything.
    """
    if body.get("by_basis") is None:
        body.pop("by_basis", None)
    return body

STATUS_ACTIVE = "active"
STATUS_VOID = "void"

# Which layer of the pipeline a batch's frame came from. Recorded rather than assumed,
# because two writers put two different things here before anyone noticed: the shadow
# write stored frames the bridge had already converted into the reporting currency and
# stamped with a location, while the interface stored the canonical frame the adapter
# produced. The money columns of the two are not comparable, and nothing said so.
LAYER_CANONICAL = "canonical"   # straight from the adapter — the frame the contract describes
LAYER_PREPARED = "prepared"     # after `ingest_bridge._prepare`: money converted, location stamped
LAYER_UNKNOWN = "unknown"       # written before the distinction was recorded


@dataclass
class BatchRecord:
    batch_id: str
    doc_type: str
    valid_time: str
    transaction_time: str
    rows: int
    source_name: str = ""
    source_sha: Optional[str] = None
    # What makes this batch *this* batch. Not the source bytes: the frame stored is the
    # canonical one, so the same file read under a different FX table or a different
    # incoterm rule is different content and has to be storable alongside. And the same
    # bytes observed to still hold a week later is a new observation, not a duplicate.
    # So: source bytes + the config that transformed them + the moment they describe.
    content_key: Optional[str] = None
    # The readable half of `content_key`: which parameter set transformed the source
    # into the frame stored here. Kept in the open so a batch can be joined back to the
    # run manifest that carries the same value, rather than only compared inside a hash.
    config_fingerprint: Optional[str] = None
    run_id: Optional[str] = None
    # From the contract tests. A batch loaded on a partial key is kept and marked, not
    # refused: it is a perfectly good record of what the file said. What it cannot do is
    # be superseded row by row, and that is what a later merge needs to know.
    key_verdict: Optional[str] = None
    storable: Optional[bool] = None
    status: str = STATUS_ACTIVE
    # See LAYER_* above. Absent on batches written before this was recorded, which read
    # back as `unknown` rather than as a guess about what they hold.
    frame_layer: str = LAYER_UNKNOWN
    written_by: str = ""
    path: str = ""
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RestateRecord:
    """
    A correction to what a batch says about *itself*, never to what it holds.

    The case it exists for: a writer's meaning changed while the store went on
    accumulating, so batches written before the change describe a layer they never
    named. Their frames are right and their marking is wrong, and the marking is what a
    reader uses to decide whether two batches may be added together.

    Appended like a void rather than applied by editing the original line, for the same
    reason: the original line is the record of what was believed when it was written,
    and a restatement is a later assertion about it. Both survive, both are attributable,
    and the restatement can itself be restated.

    It does not touch the parquet. A batch whose *contents* are wrong is voided and
    re-loaded; this is only for what the ledger says about them.
    """

    batch_id: str
    frame_layer: str
    restated_at: str
    restated_by: str = ""
    reason: str = ""
    op: str = "restate"
    # How the name was established, written only when it was not self-asserted — which
    # is every record so far, so this changes nothing on disk today. See `identity.py`.
    by_basis: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(_without_empty_basis(asdict(self)), ensure_ascii=False)


@dataclass
class VoidRecord:
    """A batch withdrawn. Appended, never applied by deleting the original line."""

    batch_id: str
    voided_at: str
    voided_by: str = ""
    reason: str = ""
    op: str = "void"
    by_basis: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(_without_empty_basis(asdict(self)), ensure_ascii=False)


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

    # Lines that say something *about* a batch rather than being one.
    _OPERATIONS = ("void", "restate")

    def batches(self, doc_type: str = None, include_void: bool = False) -> List[Dict[str, Any]]:
        """Every batch, with voids and restatements applied. Newest last."""
        voided = {e["batch_id"] for e in self._lines() if e.get("op") == "void"}
        # Last restatement wins, so a restatement can itself be restated.
        restated = {e["batch_id"]: e.get("frame_layer")
                    for e in self._lines() if e.get("op") == "restate"}
        out = []
        for entry in self._lines():
            if entry.get("op") in self._OPERATIONS:
                continue
            if doc_type and entry.get("doc_type") != doc_type:
                continue
            if entry["batch_id"] in restated:
                entry = dict(entry, frame_layer=restated[entry["batch_id"]],
                             frame_layer_restated=True)
            if entry["batch_id"] in voided:
                entry = dict(entry, status=STATUS_VOID)
                if not include_void:
                    continue
            out.append(entry)
        return out

    def restate(self, batch_id: str, frame_layer: str, reason: str = "",
                by=None) -> RestateRecord:
        """
        Say what layer a batch holds, after the fact. Appended, never applied.

        `by` has no default any more. It used to default to `""`, so enforcement lived
        entirely at the four entry points and any caller that went round one of them
        wrote an unattributed record in silence — the same shape as an argparse default
        that outranks a tenant: **the default is the hole.** A missing name is now a
        refusal here, where it cannot be got round.
        """
        actor = resolve_actor(by, what=f"restating batch {batch_id}")
        record = RestateRecord(
            batch_id=batch_id,
            frame_layer=frame_layer,
            restated_at=datetime.now().isoformat(timespec="seconds"),
            restated_by=actor.name,
            reason=reason,
            by_basis=None if actor.basis == SELF_ASSERTED else actor.basis,
        )
        self.append(record)
        return record

    def restatements(self) -> Dict[str, Dict[str, Any]]:
        """Every restatement, by batch id — the latest one for each."""
        return {e["batch_id"]: e for e in self._lines() if e.get("op") == "restate"}

    def voided(self) -> Dict[str, Dict[str, Any]]:
        """
        Every withdrawal, by batch id.

        Separate from `batches()` because a batch can be voided without ever having had
        a fact record: a load withdrawn while it is still only landed leaves a void and
        nothing for it to hide, and asking `batches()` about it answers about the wrong
        thing.
        """
        return {e["batch_id"]: e for e in self._lines() if e.get("op") == "void"}

    def void(self, batch_id: str, reason: str = "", by=None) -> VoidRecord:
        """Withdraw a batch. `by` is required, for the reason given on `restate`."""
        actor = resolve_actor(by, what=f"voiding batch {batch_id}")
        record = VoidRecord(
            batch_id=batch_id,
            voided_at=datetime.now().isoformat(timespec="seconds"),
            voided_by=actor.name,
            reason=reason,
            by_basis=None if actor.basis == SELF_ASSERTED else actor.basis,
        )
        self.append(record)
        return record

    def has_source(self, doc_type: str, source_sha: str) -> Optional[Dict[str, Any]]:
        """Any batch of this document type that came from these bytes."""
        if not source_sha:
            return None
        for entry in self.batches(doc_type=doc_type):
            if entry.get("source_sha") == source_sha:
                return entry
        return None

    def has_content(self, doc_type: str, content_key: str) -> Optional[Dict[str, Any]]:
        """
        Whether this exact batch is already stored.

        Deduping on the source bytes alone was wrong in two directions. It dropped a
        re-export of unchanged data at a later `valid_time`, which is a new observation
        — evidence the position still held — and it dropped a re-run of the same file
        after a config change, silently keeping the frame built under the old FX table
        while the run itself used the new one. Both are the same mistake: the stored
        frame is the canonical one, and the source bytes do not determine it.
        """
        if not content_key:
            return None
        for entry in self.batches(doc_type=doc_type):
            if entry.get("content_key") == content_key:
                return entry
        return None
