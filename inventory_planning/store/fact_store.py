"""
The fact store.

Append-only, typed, and read by nobody yet. This is deliberately phase one of three:
the pipeline still reads its files exactly as before, the store is written alongside,
and only once a run's outputs can be shown to be identical from either source does
anything start reading it. History that is not being collected cannot be recovered
later, so writing starts long before reading does.

Parquet rather than CSV, because the store sits exactly where the type damage happens.
A CSV round trip re-guesses every column: `000000000000123456` comes back as the integer
123456, a date comes back as a string. That is the failure `normalize: material_number`
exists to undo, and storing facts as CSV would reintroduce it at the one boundary every
future read has to cross.

What is *not* here, on purpose: any read path the planner depends on, and the merge
interface. Classifying an incoming row as append / correct / new version needs the key
to identify one row, which is why the natural-key work came first — and it needs a
reviewed plan in front of a human before anything is written, which is its own change.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ledger import BatchLedger, BatchRecord, STATUS_ACTIVE
from .location import resolve_store_root, warn_if_inside_repo

# Bumped when the on-disk layout changes in a way old code would misread. The check
# refuses rather than guessing: a store written by a newer schema may be readable by
# accident, and reading it wrongly is worse than not reading it at all.
SCHEMA_VERSION = 1
SCHEMA_NAME = "_schema_version.json"

FACTS_DIRNAME = "facts"
HISTORY_DIRNAME = "history"


class StoreSchemaError(RuntimeError):
    """The store on disk was written by a schema this code does not understand."""


class StoreUnavailable(RuntimeError):
    """The store cannot be written to. Never raised into the planning path."""


def _sha256_frame(frame) -> str:
    """A content hash of the canonical frame, for batches with no file behind them."""
    h = hashlib.sha256()
    h.update(",".join(map(str, frame.columns)).encode("utf-8"))
    h.update(str(len(frame)).encode("utf-8"))
    try:
        h.update(str(frame.head(50).to_dict()).encode("utf-8"))
    except Exception:
        pass
    return h.hexdigest()


def _as_iso_date(value) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


class FactStore:
    """
    One directory holding every batch ever loaded, plus the ledger describing them.

        <root>/_schema_version.json
        <root>/batches.jsonl
        <root>/facts/<doc_type>/<batch_id>.parquet
        <root>/history/<YYYY-MM>/snapshot_*.json
    """

    def __init__(self, root=None, create: bool = True):
        self.root, self.source = resolve_store_root(root)
        self.notes: List[str] = []
        warning = warn_if_inside_repo(self.root)
        if warning:
            self.notes.append(warning)
        self.ledger = BatchLedger(self.root)
        if create:
            self._ensure_schema()

    # ── Schema ───────────────────────────────────────────────────────────────

    @property
    def schema_path(self) -> Path:
        return self.root / SCHEMA_NAME

    def _ensure_schema(self) -> None:
        """
        Stamp the layout version, or refuse a store this code cannot read.

        Git cannot touch a store outside the working tree, but an upgrade still can:
        new code changes what the bytes mean while git moves nothing, and the failure
        mode is not a crash but a silent misreading. Hence the stamp, from version one
        rather than retrofitted onto data already written.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.schema_path.exists():
            self.schema_path.write_text(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
            return
        try:
            found = json.loads(self.schema_path.read_text(encoding="utf-8"))["schema_version"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise StoreSchemaError(
                f"{self.schema_path} is unreadable ({exc}). Refusing to write into a "
                f"store whose layout cannot be confirmed."
            ) from exc
        if int(found) != SCHEMA_VERSION:
            raise StoreSchemaError(
                f"store at {self.root} is schema v{found}; this code writes v{SCHEMA_VERSION}. "
                f"Migrate it, or point ${'INVENTORY_PLANNING_STORE'} elsewhere."
            )

    # ── Layout ───────────────────────────────────────────────────────────────

    @property
    def facts_dir(self) -> Path:
        return self.root / FACTS_DIRNAME

    @property
    def history_dir(self) -> Path:
        """
        Where planning snapshots go.

        `SnapshotSaver` used to derive this from `output_dir.parent`, which made the
        location depend on the shape of an unrelated CLI argument: `--output output`
        wrote to a git-tracked `history/` and `--output output/real` wrote to an ignored
        `output/history/`. Resolving it here instead makes it one place, outside the
        repository, regardless of where the CSVs were asked to go.
        """
        return self.root / HISTORY_DIRNAME

    def batch_path(self, doc_type: str, batch_id: str) -> Path:
        return self.facts_dir / doc_type / f"{batch_id}.parquet"

    # ── Writing ──────────────────────────────────────────────────────────────

    def write_batch(
        self,
        doc_type: str,
        frame,
        valid_time,
        source_name: str = "",
        source_sha: str = None,
        run_id: str = None,
        key_verdict: str = None,
        storable: bool = None,
        written_by: str = "",
    ) -> Optional[BatchRecord]:
        """
        Append one load. Returns None when these exact bytes are already stored.

        `valid_time` is required and never defaulted to now: the moment data describes
        is a fact about the data, and the moment it was loaded is already recorded
        separately.
        """
        if valid_time is None:
            raise ValueError(
                f"{doc_type}: valid_time is required — the moment the data describes "
                f"cannot be inferred from when it was read."
            )
        if frame is None or not len(frame):
            return None

        sha = source_sha or _sha256_frame(frame)
        already = self.ledger.has_source(doc_type, sha)
        if already:
            self.notes.append(
                f"{doc_type}: already stored as batch {already['batch_id']} "
                f"({already['transaction_time'][:16]}) — not written again"
            )
            return None

        now = datetime.now()
        batch_id = f"{now.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
        path = self.batch_path(doc_type, batch_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise StoreUnavailable(
                f"parquet support missing ({exc}). Install pyarrow, or the store cannot "
                f"keep the column types that are the reason it is not CSV."
            ) from exc

        record = BatchRecord(
            batch_id=batch_id,
            doc_type=doc_type,
            valid_time=_as_iso_date(valid_time),
            transaction_time=now.isoformat(timespec="seconds"),
            rows=len(frame),
            source_name=source_name,
            source_sha=sha,
            run_id=run_id,
            key_verdict=key_verdict,
            storable=storable,
            status=STATUS_ACTIVE,
            written_by=written_by,
            path=str(path.relative_to(self.root)),
        )
        self.ledger.append(record)
        return record

    # ── Reading ──────────────────────────────────────────────────────────────

    def read_batch(self, batch_id: str, doc_type: str = None):
        """One batch back as it was written. For verification, not for planning."""
        import pandas as pd

        for entry in self.ledger.batches(doc_type=doc_type, include_void=True):
            if entry["batch_id"] == batch_id:
                return pd.read_parquet(self.root / entry["path"])
        return None

    def batches(self, doc_type: str = None) -> List[Dict[str, Any]]:
        return self.ledger.batches(doc_type=doc_type)

    def summary(self) -> str:
        entries = self.batches()
        by_type: Dict[str, int] = {}
        for entry in entries:
            by_type[entry["doc_type"]] = by_type.get(entry["doc_type"], 0) + entry["rows"]
        lines = [f"  Fact store {self.root}  (from {self.source})",
                 f"    {len(entries)} batches, {len(by_type)} document types"]
        for doc_type, rows in sorted(by_type.items()):
            lines.append(f"      {doc_type:<18} {rows:>8,} rows")
        for note in self.notes:
            lines.append(f"    · {note}")
        return "\n".join(lines)
