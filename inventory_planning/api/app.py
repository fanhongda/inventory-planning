"""
FastAPI application: upload, look at what routing decided, correct it, promote it.

The flow this serves is the one intake has always performed and never showed anybody:

    upload -> land verbatim -> route -> review the resolution -> declare a correction
           -> re-resolve from the landed rows -> promote to facts

The step that matters is the second arrow from the end. A correction re-resolves from
what was landed, not from the file, so fixing a mapping costs neither a re-upload nor a
hand-authored adapter — the two remedies that existed before and both of which have
broken a run. Landing exists precisely so that the bytes outlive the interpretation.

Every correction made here is written as a declaration, in the same syntax and the same
directory as one typed by hand. That is not tidiness: it is what keeps a headless run
reproducing a run driven from a browser, and it is why the interface can be trusted with
corrections at all.

Deferred deliberately, and answered with 501 rather than with something that half works:
querying facts as-of needs the fact table and the `current` view that do not exist yet,
and a run registry endpoint needs the same. Both say what is missing rather than
returning an empty list, which would read as "no data".
"""

# No `from __future__ import annotations` here, unlike the rest of the package. FastAPI
# reads the annotations to build its request model, and postponed evaluation turns
# `file: UploadFile` into a forward reference that pydantic cannot resolve — the name is
# local to `create_app`, because importing FastAPI at module scope would make it a hard
# dependency of a package that must install without it.

import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ingest.contract import default_registry
from ..ingest.templates import contract_fingerprint, emit
from ..resolution import resolve_frame
from ..store.declarations import (
    Declarations, DeclarationError, Override, SCOPE_MAPPING, SCOPE_VALUE,
)
from ..store.landing import LandingStore
from ..store.ledger import BatchLedger

_NOT_YET = (
    "Not built. Facts are written by the pipeline and read back only one whole batch at "
    "a time; there is no as-of view to query and no query engine in front of it. See "
    "INTERFACE.md §4 — this needs the fact table and the `current` view first. Returning "
    "an empty result here would read as 'no data', which is a different and worse claim."
)


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by absence, not by tests
        raise RuntimeError(
            "The HTTP surface needs FastAPI. Install the optional extra:\n"
            "    pip install -e '.[api]'\n"
            "The planning pipeline itself does not depend on it."
        ) from exc


class Service:
    """
    What the endpoints share: where config lives, and where the store is.

    Held on the app rather than as module state so a test can point one at a temporary
    store without the next test inheriting it — the store path is the isolation
    mechanism for data exactly as a branch is for code.
    """

    def __init__(self, config_dir=None, store_root=None):
        self.config_dir = Path(config_dir) if config_dir else None
        self.store_root = store_root
        self.contracts = default_registry()

    @property
    def landing(self) -> LandingStore:
        return LandingStore(self.store_root)

    @property
    def ledger(self) -> BatchLedger:
        return BatchLedger(self.landing.root)

    def declarations(self) -> Declarations:
        """
        Re-read every time rather than cached.

        A declaration written by one request has to be visible to the next one, and the
        file is also editable by hand while the server runs. A cached copy would make
        the interface disagree with the repository about what has been declared, which
        is the one disagreement this layer cannot afford.
        """
        return Declarations.load(self.config_dir)

    def find_batch(self, batch_id: str) -> Dict[str, Any]:
        record = self.landing.find(batch_id)
        if record is None:
            from fastapi import HTTPException
            raise HTTPException(404, f"no landed batch {batch_id!r}")
        return record

    def landed_frame(self, record: Dict[str, Any]):
        """The landed rows as a frame the router can read, without `row_no`."""
        frame = self.landing.rows(record.get("doc_type", ""), record["batch_id"],
                                  named=True)
        return frame.drop(columns=["row_no"], errors="ignore")

    def status_of(self, batch_id: str) -> str:
        """
        Where a batch has got to: landed, promoted to facts, or withdrawn.

        Withdrawal is checked first and against the void records themselves. A batch
        voided while it was still only landed has no fact record for the void to hide,
        so asking the batch list about it answers about the wrong thing and reports it
        as though nothing had happened.
        """
        ledger = self.ledger
        if batch_id in ledger.voided():
            return "void"
        if any(b["batch_id"] == batch_id for b in ledger.batches(include_void=True)):
            return "promoted"
        return "landed"


def create_app(config_dir=None, store_root=None):
    """Build the application. Importing this module does not require FastAPI; calling
    this does."""
    _require_fastapi()
    from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse

    service = Service(config_dir=config_dir, store_root=store_root)
    app = FastAPI(
        title="Inventory planning — intake",
        description=__doc__,
        version="0.1.0",
    )
    app.state.service = service

    # ── What the pipeline knows how to read ──────────────────────────────────

    @app.get("/health")
    def health() -> Dict[str, Any]:
        landing = service.landing
        return {
            "ok": True,
            "store_root": str(landing.root),
            "store_root_from": landing.source,
            "contracts": len(service.contracts.doc_types),
            "notes": landing.notes,
        }

    @app.get("/contracts")
    def contracts() -> List[Dict[str, Any]]:
        out = []
        for doc_type in service.contracts.doc_types:
            contract = service.contracts.get(doc_type)
            out.append({
                "doc_type": doc_type,
                "description": contract.description,
                "grain": contract.grain,
                "natural_key": list(contract.natural_key or []),
                "fields": len(contract.fields),
                "required": sum(1 for f in contract.fields.values() if f.required),
                "fingerprint": contract_fingerprint(contract),
            })
        return out

    @app.get("/contracts/{doc_type}")
    def contract(doc_type: str) -> Dict[str, Any]:
        try:
            spec = service.contracts.get(doc_type)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        keys = {str(k) for k in (spec.natural_key or [])}
        return {
            "doc_type": doc_type,
            "description": spec.description,
            "fingerprint": contract_fingerprint(spec),
            "fields": [
                {"field": name, "required": bool(f.required), "type": f.type,
                 "unit": f.unit, "key": name in keys,
                 "description": " ".join((f.description or "").split()),
                 "aliases": list(f.aliases)}
                for name, f in spec.fields.items()
            ],
        }

    @app.get("/contracts/{doc_type}/template")
    def template(doc_type: str):
        """The blank workbook for a document with no source system to export it."""
        try:
            path = emit(doc_type, Path(tempfile.mkdtemp()), registry=service.contracts)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(
            path, filename=path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── Getting a file in, and looking at what was made of it ────────────────

    @app.post("/uploads")
    async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
        """
        Land a file verbatim, then say what routing made of each sheet.

        Landing happens before anything is decided about the contents, which is the
        whole point of the layer: a column read as the wrong thing costs a re-resolve
        rather than asking the planner for the file again.
        """
        from ..ingest.intake import is_tabular, load_sheets
        from ..ingest.templates import NON_DATA_SHEETS, read_meta

        workdir = Path(tempfile.mkdtemp())
        target = workdir / Path(file.filename or "upload").name
        with open(target, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        try:
            sheets = load_sheets(target)
        except Exception as exc:
            raise HTTPException(400, f"could not read {target.name}: {exc}") from exc

        meta = read_meta(target)
        hint = meta.doc_type if meta else None
        declarations = service.declarations()

        landed: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []
        for sheet_name, raw in sheets:
            if sheet_name in NON_DATA_SHEETS:
                continue
            tabular, why = is_tabular(raw)
            if not tabular:
                skipped.append({"sheet": sheet_name, "reason": why})
                continue
            resolution = resolve_frame(raw, source_name=target.name,
                                       declarations=declarations,
                                       doc_type_hint=hint, sheet_name=sheet_name)
            record = service.landing.land(target, sheet=sheet_name or 0,
                                          doc_type=resolution.doc_type)
            landed.append({
                "batch_id": record["batch_id"],
                "doc_type": resolution.doc_type,
                "rows": record["rows"],
                "duplicate_headers": record.get("duplicate_headers", []),
                "resolution": resolution.to_dict(),
            })

        if not landed:
            # A blank template is the likeliest way to arrive here and it deserves its
            # own sentence: "no tabular sheet" is true and unhelpful about a file that
            # visibly has one, and the person who uploaded it is the person least able
            # to work out that the headers alone are not data.
            if meta is not None:
                raise HTTPException(
                    400, f"{target.name} is a {meta.doc_type} template with no rows "
                         f"filled in. Enter the data under the headers on the `data` "
                         f"sheet and upload it again.")
            raise HTTPException(
                400, f"no tabular sheet in {target.name}: {skipped or 'file is empty'}")
        return {"source_name": target.name,
                "template": meta.doc_type if meta else None,
                "stale_template": meta.staleness(service.contracts) if meta else None,
                "landed": landed, "skipped": skipped}

    @app.get("/batches")
    def batches(doc_type: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
        return [
            {"batch_id": r["batch_id"], "doc_type": r.get("doc_type", ""),
             "source_name": r.get("source_name", ""), "sheet": r.get("sheet", ""),
             "rows": r.get("rows", 0), "landed_at": r.get("landed_at", ""),
             "duplicate_headers": r.get("duplicate_headers", []),
             "status": service.status_of(r["batch_id"])}
            for r in service.landing.batches(doc_type)
        ]

    @app.get("/batches/{batch_id}")
    def batch(batch_id: str) -> Dict[str, Any]:
        record = service.find_batch(batch_id)
        return {**record, "status": service.status_of(batch_id)}

    @app.get("/batches/{batch_id}/rows")
    def batch_rows(batch_id: str, limit: int = Query(50, ge=1, le=1000),
                   offset: int = Query(0, ge=0)) -> Dict[str, Any]:
        """
        The landed rows, exactly as the file spelled them.

        The reason to serve this beside the resolution rather than instead of it: a
        mis-mapped column is found by looking at the two together, and it is found in
        seconds that way and not at all from a mapping log.
        """
        record = service.find_batch(batch_id)
        frame = service.landing.rows(record.get("doc_type", ""), batch_id, named=True)
        window = frame.iloc[offset:offset + limit]
        return {
            "batch_id": batch_id,
            "total": len(frame),
            "offset": offset,
            "columns": [str(c) for c in frame.columns],
            "rows": [{str(k): (None if v is None else str(v)) for k, v in row.items()}
                     for row in window.to_dict(orient="records")],
        }

    @app.get("/batches/{batch_id}/resolution")
    def resolution(batch_id: str) -> Dict[str, Any]:
        record = service.find_batch(batch_id)
        frame = service.landed_frame(record)
        resolved = resolve_frame(frame, source_name=record.get("source_name", batch_id),
                                 declarations=service.declarations(),
                                 doc_type_hint=record.get("doc_type") or None)
        return resolved.to_dict()

    @app.post("/batches/{batch_id}/declarations")
    def declare(batch_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Assert something about this document, then re-resolve from the landed rows.

        The response is the difference the assertion made, because that is the question
        a person asking it actually has. A declaration that changes nothing is usually a
        declaration that matched nothing, and it should be as visible as one that moved
        every number.
        """
        record = service.find_batch(batch_id)
        doc_type = record.get("doc_type", "")
        scope = str(body.get("scope", SCOPE_MAPPING))
        if scope not in (SCOPE_MAPPING, SCOPE_VALUE):
            raise HTTPException(
                400, f"scope must be {SCOPE_MAPPING!r} or {SCOPE_VALUE!r} here; "
                     f"identity and parameter declarations are not tied to one batch")
        if not body.get("field"):
            raise HTTPException(400, "`field` is required")
        if "value" not in body:
            raise HTTPException(400, "`value` is required")

        frame = service.landed_frame(record)
        before = resolve_frame(frame, source_name=record.get("source_name", batch_id),
                               declarations=service.declarations(),
                               doc_type_hint=doc_type or None)

        # Matched on headers, never on the file: a file is identified by its bytes and
        # next month's export of the same report is different bytes under a different
        # name, so a declaration tied to the file expires exactly when it starts being
        # useful.
        target = ({"doc_type": doc_type, "headers": [str(c) for c in frame.columns]}
                  if scope == SCOPE_MAPPING else {"doc_type": doc_type})
        override = Override(
            scope=scope, field=str(body["field"]), value=body["value"], target=target,
            reason=str(body.get("reason", "")), by=str(body.get("by", "")),
            at=date.today(),
        )
        try:
            written = Declarations.write_override(override, config_dir=service.config_dir)
        except DeclarationError as exc:
            raise HTTPException(400, str(exc)) from exc

        after = resolve_frame(frame, source_name=record.get("source_name", batch_id),
                              declarations=service.declarations(),
                              doc_type_hint=doc_type or None)
        return {
            "declaration": str(written),
            "changed": _diff(before, after),
            "resolution": after.to_dict(),
        }

    @app.post("/batches/{batch_id}/void")
    def void(batch_id: str, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        """
        Withdraw a batch. The rows are not touched — a void is appended, like everything
        else, so it is itself reversible and itself attributable.
        """
        service.find_batch(batch_id)
        reason = str(body.get("reason", "")).strip()
        by = str(body.get("by", "")).strip()
        if not reason or not by:
            raise HTTPException(400, "voiding a batch needs `reason` and `by`")
        record = service.ledger.void(batch_id, reason=reason, by=by)
        return {"batch_id": batch_id, "status": "void",
                "voided_at": record.voided_at, "voided_by": record.voided_by}

    # ── Deliberately not built yet ───────────────────────────────────────────

    @app.get("/facts/{doc_type}")
    def facts(doc_type: str):
        raise HTTPException(501, _NOT_YET)

    @app.get("/runs")
    def runs():
        raise HTTPException(501, _NOT_YET)

    return app


def _diff(before, after) -> List[Dict[str, Any]]:
    """Which fields resolve differently now, field by field."""
    was = {f.field: f for f in before.fields}
    out = []
    for field in after.fields:
        old = was.get(field.field)
        if old is None:
            out.append({"field": field.field, "from": None, "to": field.source})
            continue
        if old.source != field.source or old.column != field.column \
                or old.populated != field.populated:
            out.append({
                "field": field.field,
                "from": {"source": old.source, "column": old.column,
                         "populated": old.populated},
                "to": {"source": field.source, "column": field.column,
                       "populated": field.populated},
            })
    return out
