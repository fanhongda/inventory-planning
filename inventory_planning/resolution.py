"""
Resolution — what routing decided about one file, as data rather than as a paragraph.

`explain.py` answers the same question for a person at a terminal and answers it in
prose. An interface needs the same finding in a shape it can lay out: which field each
column was taken to supply, how populated that column is, what the run would have done
without a declaration, and which of the file's columns matched nothing at all.

Nothing here re-implements routing. It calls `Intake.load_frame`, which is the path a
real run takes — declarations applied over the routed mapping, the adapter run, the
contract tests executed — and then reports what came back. That is deliberate and it is
the rule the whole interface layer rests on: a second implementation of the mapping
would be a second set of answers, and the one on screen would be the one nobody tested.

The per-field `source` is the part worth reading. A field the export supplies and a
field a person declared and a field that fell back to a default all arrive in the
canonical frame looking identical, and they are not the same claim at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SOURCE_MAPPED = "mapped"        # a column in the export supplies it
SOURCE_DECLARED = "declared"    # a person named the column it comes from
SOURCE_DEFAULT = "default"      # the export never carried it; a value was supplied
SOURCE_DERIVED = "derived"      # computed from other fields by the contract
SOURCE_ABSENT = "absent"        # not in the canonical frame at all


def _populated(series: pd.Series) -> int:
    if series is None:
        return 0
    values = series.astype("object")
    filled = values.notna() & (values.astype(str).str.strip() != "")
    return int(filled.sum())


@dataclass
class FieldResolution:
    """One canonical field, and the evidence for where it was taken from."""

    field: str
    source: str
    column: Optional[str] = None
    required: bool = False
    key: bool = False
    unit: Optional[str] = None
    type: str = "string"
    description: str = ""
    populated: int = 0
    rows: int = 0

    @property
    def fill_rate(self) -> float:
        return self.populated / self.rows if self.rows else 0.0

    @property
    def empty(self) -> bool:
        """
        Mapped to a column that carries nothing.

        Worth separating from absent, because the two need opposite responses: an
        absent field needs a column found, and an empty one has its column and the
        column is the wrong one — or the export was filtered to a scope that does not
        populate it.
        """
        return self.source in (SOURCE_MAPPED, SOURCE_DECLARED) and self.populated == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field, "source": self.source, "column": self.column,
            "required": self.required, "key": self.key, "unit": self.unit,
            "type": self.type, "description": self.description,
            "populated": self.populated, "rows": self.rows,
            "fill_rate": round(self.fill_rate, 4), "empty": self.empty,
        }


@dataclass
class Resolution:
    """Everything one file's routing decided, ready to be rendered."""

    source_name: str
    doc_type: str
    confidence: float
    reason: str
    adapter: str
    adapter_status: str
    is_draft: bool
    uncertain: bool
    rows: int
    sheet: str = ""
    fields: List[FieldResolution] = dc_field(default_factory=list)
    unmatched_columns: List[str] = dc_field(default_factory=list)
    tests_passed: bool = True
    test_failures: List[str] = dc_field(default_factory=list)
    transforms: List[str] = dc_field(default_factory=list)
    # Declarations that named a column this file does not have. Reported rather than
    # applied — applying one blanks the field, which reads downstream as an absent
    # measure rather than as a mistake. Believing a mapping is corrected when it is not
    # is worse than not having tried, because the looking stops.
    ignored_declarations: List[Dict[str, Any]] = dc_field(default_factory=list)

    @property
    def missing_required(self) -> List[FieldResolution]:
        return [f for f in self.fields
                if f.required and (f.source == SOURCE_ABSENT or f.empty)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name, "sheet": self.sheet,
            "doc_type": self.doc_type,
            "confidence": round(self.confidence, 4), "reason": self.reason,
            "adapter": self.adapter, "adapter_status": self.adapter_status,
            "is_draft": self.is_draft, "uncertain": self.uncertain,
            "rows": self.rows,
            "fields": [f.to_dict() for f in self.fields],
            "unmatched_columns": list(self.unmatched_columns),
            "missing_required": [f.field for f in self.missing_required],
            "tests_passed": self.tests_passed,
            "test_failures": list(self.test_failures),
            "transforms": list(self.transforms),
            "ignored_declarations": list(self.ignored_declarations),
        }


def from_document(doc, raw: pd.DataFrame, declarations=None) -> Resolution:
    """Structure a `LoadedDocument` — the object a real run produces — for an interface."""
    route = doc.route
    contract = route.contract
    adapter = route.adapter
    column_map = dict(adapter.column_map or {})
    defaults = dict(getattr(adapter, "defaults", {}) or {})
    frame = doc.frame
    rows = len(frame)

    declared_columns: Dict[str, str] = {}
    if declarations is not None:
        headers = [str(c) for c in raw.columns]
        declared_columns = declarations.column_map_for(
            contract.doc_type, headers, doc.source_name) or {}

    keys = {str(k) for k in (contract.natural_key or [])}
    fields: List[FieldResolution] = []
    for name, spec in contract.fields.items():
        column = column_map.get(name)
        # Declared only where the declaration is the mapping in force. One naming a
        # column the file does not have is left unapplied by intake, and labelling the
        # routed column `declared` would report a correction that did not happen.
        if column and declared_columns.get(name) == column:
            source = SOURCE_DECLARED
        elif column:
            source = SOURCE_MAPPED
        elif name in defaults:
            source = SOURCE_DEFAULT
        elif name in frame.columns:
            source = SOURCE_DERIVED
        else:
            source = SOURCE_ABSENT

        populated = (_populated(frame[name])
                     if source != SOURCE_ABSENT and name in frame.columns else 0)
        fields.append(FieldResolution(
            field=name, source=source, column=column,
            required=bool(spec.required), key=name in keys,
            unit=spec.unit, type=spec.type,
            description=" ".join((spec.description or "").split()),
            populated=populated, rows=rows,
        ))

    # Required first, then whatever the export actually supplies, then the rest. A
    # resolution screen is read from the top and a required field that found no column
    # is the only thing on it that stops a run.
    order = {SOURCE_ABSENT: 3, SOURCE_DEFAULT: 2, SOURCE_DERIVED: 2,
             SOURCE_DECLARED: 0, SOURCE_MAPPED: 1}
    fields.sort(key=lambda f: (not f.required, order.get(f.source, 4), f.field))

    used = {c for c in column_map.values() if c}
    unmatched = [str(c) for c in raw.columns if str(c) not in used]
    headers = {str(c) for c in raw.columns}
    ignored = [{"field": f, "column": c, "reason": "the file has no such column"}
               for f, c in sorted(declared_columns.items()) if c not in headers]

    report = doc.test_report
    failures = [str(line).strip() for line in (getattr(report, "failures", None) or [])]

    return Resolution(
        source_name=doc.source_name, sheet=doc.sheet_name,
        doc_type=doc.doc_type, confidence=route.confidence, reason=route.reason,
        adapter=adapter.name, adapter_status=adapter.status, is_draft=route.is_draft,
        uncertain=doc.route_uncertain, rows=rows, fields=fields,
        unmatched_columns=unmatched, ignored_declarations=ignored,
        tests_passed=bool(report.passed), test_failures=failures,
        transforms=[str(step) for step in doc.transform_log],
    )


def resolve_frame(raw: pd.DataFrame, source_name: str = "<frame>",
                  declarations=None, doc_type_hint: str = None,
                  sheet_name: str = "") -> Resolution:
    """Route one already-read frame and structure the outcome."""
    from .ingest.intake import Intake

    intake = Intake(verbose=False, declarations=declarations)
    doc = intake.load_frame(raw, source_name=source_name,
                            doc_type_hint=doc_type_hint, sheet_name=sheet_name)
    return from_document(doc, raw, declarations=declarations)


def resolve_file(path, declarations=None, doc_type_hint: str = None
                 ) -> List[Resolution]:
    """
    Every tabular sheet of a file, resolved.

    A list rather than one result: a workbook routinely carries a cover sheet and a
    pivot beside the export, and which of them is the document is exactly what the
    caller wants told rather than guessed at by taking the first.
    """
    from .ingest.intake import is_tabular, load_sheets
    from .ingest.templates import NON_DATA_SHEETS, read_meta

    path = Path(path)
    meta = read_meta(path)
    if meta is not None and not doc_type_hint:
        doc_type_hint = meta.doc_type

    out: List[Resolution] = []
    for sheet_name, raw in load_sheets(path):
        if sheet_name in NON_DATA_SHEETS:
            continue
        tabular, _ = is_tabular(raw)
        if not tabular:
            continue
        out.append(resolve_frame(raw, source_name=path.name, declarations=declarations,
                                 doc_type_hint=doc_type_hint, sheet_name=sheet_name))
    return out
