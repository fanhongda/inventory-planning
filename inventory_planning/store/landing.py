"""
The landing layer — every source row exactly as it was read, before anything decides
what it means.

`FactStore` stores the *canonical* frame: the one the adapter produced after choosing
which column is the material, which is the open quantity, which is the order date.
That choice is the single largest source of wrong answers in this pipeline, and once
it is baked into the stored frame the only way to revise it is to find the original
file and read it again. On a real run that meant editing headers in Excel and
re-freezing an adapter, and the edit broke the next run outright.

So this module stores the layer underneath: one row per source row, verbatim, with no
type inference and no column selection. A mis-mapped column then costs a re-resolve
rather than a re-ingest, and the columns the adapter matched to nothing — `Planning
LT (D)`, `5052 Std Cost(CNY)`, `Old Material Number`, every one of which turned out to
matter — are still here to be recovered later.

Two decisions in here are not stylistic.

**Payload is keyed by column index, never by header.** Duplicate headers are not a
hypothetical: renaming several columns to `Material` in an export is exactly what a
planner does when told the SKU column is wrong, and pandas then hands back `Material`
and `Material.1` — a payload keyed by header would silently keep one of them. Real
exports also carry `5051 Std Cost(CNY)` beside `5052 Std Cost(CNY)`, and headers with
embedded newlines. Column position is the one thing in a spreadsheet that cannot
collide, so it is the key; the headers live once per batch in the header map.

**The sheet is read with `header=None`.** Letting pandas take the header row means
letting pandas de-duplicate it, which loses the original spelling before this module
ever sees it. The header row is located here and read as data, so what is stored is
what the file says.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .fact_store import StoreUnavailable
from .location import resolve_store_root, warn_if_inside_repo

RAW_DIRNAME = "raw"
REJECT_DIRNAME = "rejects"

# Reject reasons. Strings rather than an enum so a batch written by older code stays
# readable, and so a caller can add one without editing this module.
REJECT_UNTYPED = "type_conversion_failed"
REJECT_REQUIRED = "required_field_missing"
REJECT_IDENTITY = "identity_unresolved"


def _unnamed_share(values) -> float:
    """Share of a row's cells that are empty — the signal for a banner row."""
    if len(values) == 0:
        return 1.0
    blank = sum(1 for v in values if v is None or (isinstance(v, float) and pd.isna(v))
                or str(v).strip() == "" or str(v).startswith("Unnamed:"))
    return blank / len(values)


def find_header_row(raw: pd.DataFrame, limit: int = 4) -> int:
    """
    Index of the row holding the column headers, under any title banner above it.

    Mirrors `ingest.intake._read_excel_sheet`, which hunts the same rows with the same
    test. It has to agree with it: `row_no` in the landing layer is the offset into the
    data *below* the header, and the canonical frame counts from the same place. If the
    two disagreed, every pointer from a fact back to its source row would be off by the
    height of a banner — and off silently, since both would still be valid row numbers.
    """
    best, best_share = 0, _unnamed_share(list(raw.iloc[0])) if len(raw) else 1.0
    for candidate in range(1, min(limit, len(raw))):
        share = _unnamed_share(list(raw.iloc[candidate]))
        if share < best_share:
            best, best_share = candidate, share
        if best_share <= 0.5:
            break
    return best


def read_verbatim(path: Path, sheet: Any = 0) -> Tuple[List[str], pd.DataFrame]:
    """
    One sheet as (headers, body), with pandas never given the chance to rename a column.

    Returns the header cells in file order — duplicates included, exactly as written —
    and the rows beneath them, all as text.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        from ..ingest.encoding import sniff_encoding
        try:
            raw = pd.read_csv(path, encoding=sniff_encoding(path), dtype=str,
                              header=None, keep_default_na=False, na_values=[""])
        except pd.errors.EmptyDataError:
            # A zero-byte or header-only export is a real thing to receive, and the
            # useful response is a batch of no rows — which records that the file was
            # seen and was empty. Raising here would make an empty extract
            # indistinguishable from one that was never loaded.
            raw = pd.DataFrame()
    else:
        raw = pd.read_excel(path, sheet_name=sheet, dtype=str, header=None)

    if raw.empty:
        return [], raw

    header_row = find_header_row(raw)
    headers = ["" if pd.isna(v) else str(v) for v in raw.iloc[header_row]]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    return headers, body


def _payload_frame(body: pd.DataFrame) -> pd.DataFrame:
    """
    Body rows as (row_no, payload) where payload is JSON keyed by column index.

    Empty cells are omitted from the payload rather than stored as null. A wide export
    — the OTD report is 81 columns, most of them blank on most rows — would otherwise
    spend the bulk of the file on the string "null", and an absent key and a null value
    mean the same thing to every reader of this layer.
    """
    records = []
    for row_no, (_, row) in enumerate(body.iterrows()):
        payload = {
            str(idx): str(value)
            for idx, value in enumerate(row)
            if value is not None and not (isinstance(value, float) and pd.isna(value))
            and str(value).strip() != ""
        }
        records.append((row_no, json.dumps(payload, ensure_ascii=False)))
    return pd.DataFrame(records, columns=["row_no", "payload"])


class LandingStore:
    """
    Verbatim source rows, beside the fact store and under the same root.

        <root>/raw/<doc_type>/<batch_id>.parquet        row_no, payload
        <root>/raw/<doc_type>/<batch_id>.headers.json   header map + provenance
        <root>/rejects/<doc_type>/<batch_id>.parquet    row_no, stage, reason, field, detail

    Deliberately not schema-checked against `FactStore.SCHEMA_VERSION`: this layer is
    additive and a store written without it is not wrong, merely older. Refusing to
    open such a store would make landing a breaking change to something that already
    works, which is the opposite of the point.
    """

    def __init__(self, root=None):
        self.root, self.source = resolve_store_root(root)
        self.notes: List[str] = []
        warning = warn_if_inside_repo(self.root)
        if warning:
            self.notes.append(warning)

    # ── Layout ───────────────────────────────────────────────────────────────

    @property
    def raw_dir(self) -> Path:
        return self.root / RAW_DIRNAME

    @property
    def reject_dir(self) -> Path:
        return self.root / REJECT_DIRNAME

    def raw_path(self, doc_type: str, batch_id: str) -> Path:
        return self.raw_dir / (doc_type or "unrouted") / f"{batch_id}.parquet"

    def header_path(self, doc_type: str, batch_id: str) -> Path:
        return self.raw_path(doc_type, batch_id).with_suffix(".headers.json")

    def reject_path(self, doc_type: str, batch_id: str) -> Path:
        return self.reject_dir / (doc_type or "unrouted") / f"{batch_id}.parquet"

    # ── Writing ──────────────────────────────────────────────────────────────

    def land(
        self,
        path,
        sheet: Any = 0,
        doc_type: str = "",
        batch_id: str = None,
        source_sha: str = None,
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        Store one sheet verbatim. Returns the header-map record that was written.

        `batch_id` is accepted rather than generated so the landed rows carry the same
        identity as the canonical batch written by `FactStore`. That shared id is the
        whole pointer: `(batch_id, row_no)` on a fact resolves to a row here, and a
        landing layer with its own ids would be a second archive rather than the source
        of the first.
        """
        path = Path(path)
        headers, body = read_verbatim(path, sheet)
        batch_id = batch_id or f"{datetime.now():%Y%m%d_%H%M%S}-{uuid.uuid4().hex[:6]}"

        frame = _payload_frame(body)
        target = self.raw_path(doc_type, batch_id)
        self._write_parquet(frame, target)

        record = {
            "batch_id": batch_id,
            "doc_type": doc_type,
            "sheet": str(sheet),
            "source_name": path.name,
            "source_sha": source_sha,
            "run_id": run_id,
            "landed_at": datetime.now().isoformat(timespec="seconds"),
            "rows": len(frame),
            # Position first, because position is the key. `header` is the raw cell as
            # written — repeated spellings and all — so a duplicate is visible here
            # rather than resolved away.
            "columns": [{"idx": idx, "header": header}
                        for idx, header in enumerate(headers)],
            "duplicate_headers": sorted(
                {h for h in headers if h and headers.count(h) > 1}
            ),
        }
        self.header_path(doc_type, batch_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def reject(self, doc_type: str, batch_id: str, rows: List[Dict[str, Any]]) -> int:
        """
        Quarantine rows that could not be carried forward, with a reason each.

        Rows are isolated, not dropped and not fatal to the batch. The pipeline's
        current `dropna(subset=["sku", "supplier", "po_date"])` removes rows in
        silence: how many went, and which, is unanswerable afterwards. Everything about
        the row is already in the landing layer, so only the reason is stored here.
        """
        if not rows:
            return 0
        frame = pd.DataFrame([
            {
                "row_no": row.get("row_no"),
                "stage": row.get("stage", ""),
                "reason_code": row.get("reason_code", ""),
                "field": row.get("field", ""),
                "detail": json.dumps(row.get("detail", {}), ensure_ascii=False),
            }
            for row in rows
        ])
        self._write_parquet(frame, self.reject_path(doc_type, batch_id))
        return len(frame)

    @staticmethod
    def _write_parquet(frame: pd.DataFrame, target: Path) -> None:
        """Stage, then rename — a partial file no record points at is the one kind of
        litter a later glob would mistake for data."""
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_suffix(".parquet.partial")
        try:
            frame.to_parquet(staged, index=False)
        except ImportError as exc:
            staged.unlink(missing_ok=True)
            raise StoreUnavailable(
                f"parquet support missing ({exc}). Install pyarrow, or the landing "
                f"layer cannot store source rows without re-guessing their types."
            ) from exc
        except Exception:
            staged.unlink(missing_ok=True)
            raise
        staged.replace(target)

    # ── Reading ──────────────────────────────────────────────────────────────

    def header_map(self, doc_type: str, batch_id: str) -> Optional[Dict[str, Any]]:
        target = self.header_path(doc_type, batch_id)
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def rows(self, doc_type: str, batch_id: str, named: bool = True) -> pd.DataFrame:
        """
        The landed rows. With `named`, payload keys are turned back into headers.

        Turning them back is a read-time convenience and stays one: where a header is
        repeated, the later column keeps its index as a suffix rather than overwriting
        the earlier one, so nothing is lost on the way out either.
        """
        target = self.raw_path(doc_type, batch_id)
        if not target.exists():
            return pd.DataFrame(columns=["row_no", "payload"])
        frame = pd.read_parquet(target)
        if not named:
            return frame

        record = self.header_map(doc_type, batch_id) or {"columns": []}
        labels = {}
        seen: Dict[str, int] = {}
        for column in record["columns"]:
            header = column["header"] or f"col_{column['idx']}"
            seen[header] = seen.get(header, 0) + 1
            labels[str(column["idx"])] = (
                header if seen[header] == 1 else f"{header}#{column['idx']}")
        widened = pd.DataFrame(
            [{labels.get(k, k): v for k, v in json.loads(p).items()}
             for p in frame["payload"]]
        )
        return pd.concat([frame[["row_no"]], widened], axis=1)

    def source_row(self, doc_type: str, batch_id: str, row_no: int) -> Dict[str, str]:
        """One landed row as {header: value} — the end of every audit trail."""
        frame = self.rows(doc_type, batch_id, named=True)
        match = frame[frame["row_no"] == row_no]
        if match.empty:
            return {}
        row = match.iloc[0].drop(labels=["row_no"])
        return {k: v for k, v in row.items() if pd.notna(v)}

    def summary(self) -> str:
        if not self.raw_dir.exists():
            return f"Landing layer empty ({self.raw_dir})"
        lines = [f"Landing layer — {self.raw_dir}"]
        for target in sorted(self.raw_dir.rglob("*.headers.json")):
            record = json.loads(target.read_text(encoding="utf-8"))
            duplicates = record.get("duplicate_headers") or []
            lines.append(
                f"  {record['doc_type'] or '?':<16} {record['rows']:>7,} rows  "
                f"{len(record['columns']):>3} cols  {record['source_name']}"
                + (f"  ⚠ 重名列 {', '.join(duplicates)}" if duplicates else "")
            )
        return "\n".join(lines)
