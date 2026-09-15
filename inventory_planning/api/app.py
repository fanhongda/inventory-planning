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

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ingest.contract import default_registry
from ..ingest.exposure import Assumption, BASIS_ASSUMED, BASIS_DECLARED, measure
from ..provenance import RunRegistry, sha256_file
from ..store.fact_store import FactStore, StoreUnavailable
from ..ingest.templates import contract_fingerprint, emit
from ..resolution import resolve_frame
from ..policy.edits import EditRefused
from ..policy.rules_edit import history as rule_history
from ..store.declarations import (
    Declarations, DeclarationError, Override, SCOPE_MAPPING, SCOPE_VALUE,
)
from ..store.landing import LandingStore
from ..store.query import (
    FactQuery, KeyIncomplete, MixedLayers, NoSuchColumn, QueryUnavailable,
)
from ..store.ledger import BatchLedger

# How far back /policy looks for a run under the rules on disk. Bounded because each
# candidate costs a manifest read, and named because the number appears in the sentence
# the screen shows when the search comes up empty — a bound the reader is not told about
# turns "not found" into "does not exist".
_REACH_SCAN = 50


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by absence, not by tests
        raise RuntimeError(
            "The HTTP surface needs FastAPI. Install the optional extra:\n"
            "    pip install -e '.[api]'\n"
            "The planning pipeline itself does not depend on it."
        ) from exc


# The three checkpoints that cannot run before the pipeline does, named so the screen
# can say what a clean intake does and does not promise. Each needs something that does
# not exist until the run builds it, which is why they are not merely "not implemented
# here" — they are not answerable here.
_LATER_GATES = [
    {"stage": "demand", "needs": "a compiled time series",
     "checks": "whether there is a demand signal at all, and how stale it is"},
    {"stage": "forecast", "needs": "a forecast",
     "checks": "whether every SKU with demand came out with one — a SKU planned "
               "against zero demand is never bought until it stocks out"},
    {"stage": "plan", "needs": "positions and recommendations",
     "checks": "whether the SKUs that should be stocked have a position to plan from"},
]


def _finding_dict(finding) -> Dict[str, Any]:
    """
    One finding as the screen needs it, which is the gate's own dict plus two readings.

    `waived` is derived from the evidence `Declarations.waive` stamps on rather than
    tracked separately, because a waived finding is a real finding that was downgraded
    — the record of that is already in the finding, and a second flag could disagree
    with it.
    """
    body = finding.to_dict()
    evidence = body.get("evidence") or {}
    body["waived"] = bool(evidence.get("waived_until"))
    body["waived_by"] = evidence.get("waived_by") or ""
    body["waived_until"] = evidence.get("waived_until") or ""
    # Which document the finding is about, where it is about one. The waiver form needs
    # it: a waiver scoped to a doc_type is the narrow kind, and one without is the kind
    # that turns a check off everywhere.
    body["doc_type"] = str(evidence.get("doc_type") or "")
    return body


class _IntakeView:
    """
    The documents, shaped as `gate_intake` expects to be handed them.

    `gate_intake` reads one attribute off an intake result and this supplies exactly
    that. Passing a real `IntakeResult` would mean building one from parts the API does
    not have — a plan, failures, supersessions, assumptions — and filling those with
    empties would state four things the API has not established.
    """

    def __init__(self, documents):
        self.documents = documents


@dataclass
class LandedIntake:
    """
    One reading of everything landed: the documents, the plan, and what was withheld.

    A structure rather than a tuple because three callers want different parts of it
    and a positional unpack is where "which of these five is the plan" becomes a bug
    nobody reads.
    """

    documents: Dict[str, Any]
    plan: Any
    records: Dict[str, Dict[str, Any]]
    declarations: Declarations
    # doc_type -> capabilities the document declares but cannot back. Carried from the
    # scan rather than recovered from the plan: the plan records what it decided, not
    # what it was told, and reconstructing the input from the output is how the two
    # drift.
    withheld: Dict[str, Any]


class Service:
    """
    What the endpoints share: where config lives, and where the store is.

    Held on the app rather than as module state so a test can point one at a temporary
    store without the next test inheriting it — the store path is the isolation
    mechanism for data exactly as a branch is for code.
    """

    def __init__(self, config_dir=None, store_root=None, output_dir=None):
        self.config_dir = Path(config_dir) if config_dir else None
        self.store_root = store_root
        # Where the runs are. The registry lives under the output directory the pipeline
        # writes to, so the interface reads the runs the CLI produced rather than
        # keeping a second record that could disagree with it.
        self.output_dir = Path(output_dir) if output_dir else Path("output")
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

    def landed_documents(self):
        """
        Every landed document, read the way the pipeline reads it, plus the plan.

        Shared by the requirements checklist and the quality gate because they are two
        readings of one thing. Two scans would be two answers about what is loaded, and
        the gate's answer is the one that decides whether a run may happen — a checklist
        that disagreed with it would be believed over it, because it is a checklist.

        Recomputed per call for the same reason `/requirements` recomputed it: a batch
        voided since the last call is not loaded any more, and a remembered list would
        not know.
        """
        from ..ingest.capabilities import CapabilityResolver
        from ..ingest.intake import Intake, unsupplied_capabilities

        declarations = self.declarations()
        intake = Intake(verbose=False, declarations=declarations)

        documents = {}
        names: Dict[str, str] = {}
        withheld: Dict[str, Any] = {}
        records: Dict[str, Dict[str, Any]] = {}
        # Newest first, so where two batches claim one document type the later upload is
        # the one described — the same rule a re-export follows everywhere else.
        for record in self.landing.batches():
            doc_type = record.get("doc_type") or ""
            if doc_type in records or self.status_of(record["batch_id"]) == "void":
                continue
            frame = self.landed_frame(record)
            doc = intake.load_frame(
                frame, source_name=record.get("source_name", record["batch_id"]))
            if doc.doc_type != doc_type:
                # Re-classified since it landed. Believe the reading, not the folder.
                doc_type = doc.doc_type
                if doc_type in records:
                    continue
            missing = unsupplied_capabilities(doc.frame, doc.route.contract)
            if missing:
                withheld[doc_type] = missing
            documents[doc_type] = doc
            names[doc_type] = record.get("source_name", record["batch_id"])
            records[doc_type] = record

        plan = CapabilityResolver().resolve(names, withheld=withheld)
        return LandedIntake(documents=documents, plan=plan, records=records,
                            declarations=declarations, withheld=withheld)

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


def create_app(config_dir=None, store_root=None, output_dir=None):
    """Build the application. Importing this module does not require FastAPI; calling
    this does."""
    _require_fastapi()
    from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse

    service = Service(config_dir=config_dir, store_root=store_root,
                      output_dir=output_dir)
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
        from starlette.background import BackgroundTask

        workdir = Path(tempfile.mkdtemp())
        try:
            path = emit(doc_type, workdir, registry=service.contracts)
        except KeyError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(404, str(exc)) from exc
        # Deleted after the response has been sent, not in a `finally`: FileResponse
        # streams the file once this handler has already returned, so removing it here
        # would truncate the download it was built for.
        return FileResponse(
            path, filename=path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
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
        # `.name` on its own, so a filename carrying a path cannot write outside the
        # temp directory — and a name that is empty after that leaves the directory
        # itself as the target, which `open` reports as a directory rather than
        # anything useful.
        name = Path(file.filename or "").name or "upload"
        target = workdir / name
        with open(target, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        # The bytes, hashed, carried onto the landing record and from there onto the
        # fact batch. Without it every batch's content key is (nothing, config, as-of),
        # so two *different* files promoted at the same as-of date collide and the
        # second is dropped as a duplicate of the first — silently, since a duplicate
        # is a legitimate no-op.
        source_sha = hashlib.sha256(target.read_bytes()).hexdigest()

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
                                          doc_type=resolution.doc_type,
                                          source_sha=source_sha)
            landed.append({
                "batch_id": record["batch_id"],
                "doc_type": resolution.doc_type,
                "rows": record["rows"],
                "duplicate_headers": record.get("duplicate_headers", []),
                "resolution": resolution.to_dict(),
            })

        # The landed parquet is the durable record; this copy was only ever needed for
        # the length of the request. Leaving it behind accumulated one copy of every
        # file ever uploaded in the temp directory — un-anonymised extracts among them,
        # outside both the repository and the store's retention.
        shutil.rmtree(workdir, ignore_errors=True)

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

    @app.get("/requirements")
    def requirements() -> Dict[str, Any]:
        """
        What a run still needs, against everything landed so far.

        Capability-shaped rather than file-shaped, because that is what the pipeline
        actually requires: a demand signal, supplied by either a pre-aggregated series
        or a sales history. Listing five filenames would state one of the two answers as
        though it were the only one, and would go stale the moment a contract gains a
        supplier.

        Recomputed on every call from the landed batches rather than accumulated as the
        uploads arrive. A checklist that remembers what it was told, rather than reading
        what is there, drifts from the store the first time a batch is voided — and it
        is a checklist, so a person will trust it over the store.
        """
        from ..ingest.capabilities import CAPABILITIES
        from ..resolution import SOURCE_ABSENT, from_document

        loaded = service.landed_documents()
        plan, withheld = loaded.plan, loaded.withheld

        landed: Dict[str, Dict[str, Any]] = {}
        for doc_type, record in loaded.records.items():
            resolved = from_document(loaded.documents[doc_type],
                                     service.landed_frame(record),
                                     declarations=loaded.declarations)
            landed[doc_type] = {
                "batch_id": record["batch_id"],
                "source_name": record.get("source_name", ""),
                "rows": record.get("rows", 0),
                "fields": [
                    {"field": f.field, "present": f.source != SOURCE_ABSENT and not f.empty,
                     "source": f.source, "column": f.column}
                    for f in resolved.fields if f.required
                ],
            }

        capabilities = []
        for name, cap in CAPABILITIES.items():
            entry = plan.resolved.get(name)
            capabilities.append({
                "name": name, "description": cap.description,
                "required": cap.required, "suppliers": list(cap.suppliers),
                "satisfied": bool(entry and entry.satisfied),
                "supplied_by": entry.supplied_by if entry and entry.satisfied else None,
                "withheld_by": sorted(d for d, caps in withheld.items() if name in caps),
                "fallback": cap.fallback, "degrades": list(cap.degrades),
            })

        wanted: Dict[str, Dict[str, Any]] = {}
        for cap in CAPABILITIES.values():
            for doc_type in cap.suppliers:
                slot = wanted.setdefault(doc_type, {"doc_type": doc_type,
                                                    "supplies": [], "required": False})
                slot["supplies"].append(cap.name)
                slot["required"] = slot["required"] or cap.required

        for doc_type, slot in wanted.items():
            contract = service.contracts.get(doc_type)
            slot["description"] = contract.description
            got = landed.get(doc_type)
            slot["landed"] = got
            slot["fields"] = got["fields"] if got else [
                {"field": name, "present": False, "source": SOURCE_ABSENT, "column": None}
                for name, spec in contract.fields.items() if spec.required
            ]

        return {
            "can_run": plan.can_run,
            "missing_required": [c.name for c in plan.missing_required],
            "degradations": plan.degradations,
            "capabilities": capabilities,
            "documents": sorted(wanted.values(),
                                key=lambda d: (not d["required"], d["doc_type"])),
        }

    @app.get("/gates")
    def gates() -> Dict[str, Any]:
        """
        The intake quality gate, over everything landed, before a run is attempted.

        This is the checkpoint worth having here: every silent failure this pipeline has
        produced was visible at intake and invisible in the report it went on to write.
        Running it before the plan means a mapping that would have produced a complete
        page of zeroes is found while the person who uploaded the file is still looking
        at it.

        **It is one of four, and the page has to say so.** `demand`, `forecast` and
        `plan` run during the run and cannot run here — they need a time series, a
        forecast and a position, none of which exist until the pipeline has done the
        work. So a clean answer here is "nothing at intake stops this", never "the run
        will pass", and the reply carries the other three by name so that distinction is
        not left to be inferred.

        Waivers are applied the way the run applies them: through `Declarations.waive`,
        which downgrades a blocking finding it has been told is a false positive here
        and leaves it visible, carrying who waived it and until when. The same call, so
        a finding waived on this screen is waived in the run and not merely hidden on
        the screen.
        """
        from ..quality import GateThresholds
        from ..quality.checks import gate_intake

        loaded = service.landed_documents()
        if not loaded.documents:
            return {
                "stage": "intake", "ran": False, "findings": [],
                "passed": None,
                "note": "Nothing is landed, so there is nothing to check. The intake "
                        "gate compares the documents against each other — whether "
                        "their item numbers meet, whether a column is a column — and "
                        "one document cannot disagree with itself.",
                "later_stages": _LATER_GATES,
            }

        report = gate_intake(_IntakeView(loaded.documents), loaded.plan,
                             GateThresholds.load(service.config_dir))
        report = loaded.declarations.waive(report)

        return {
            "stage": "intake",
            "ran": True,
            "passed": report.passed,
            "documents": sorted(loaded.documents),
            "counts": {
                "block": len(report.blocking),
                "severe": len(report.severe),
                "warn": len(report.warnings),
            },
            "findings": [_finding_dict(f) for f in report.ordered],
            "later_stages": _LATER_GATES,
        }

    @app.post("/gates/{check}/waivers")
    def waive_gate(check: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Declare one check a false positive on one document, until a date.

        Narrower than `allow_degraded` by construction, which is the whole reason it
        exists: waiving the SKU-agreement finding on a whole-warehouse stock snapshot
        does not also wave through an open PO quantity mapped to a money column.

        It lands as an ordinary `gate_waivers` entry in the config directory — the same
        statement, in the same syntax, in the same place as a hand-written one — so the
        headless run honours a waiver made by clicking. That is the governing rule of
        this whole interface, and a waiver stored anywhere else would break it.
        """
        from ..store.declarations import GateWaiver

        expires = payload.get("expires")
        try:
            when = date.fromisoformat(str(expires)) if expires else None
        except ValueError:
            raise HTTPException(400, f"expires must be a date (YYYY-MM-DD), "
                                     f"not {expires!r}") from None
        if when is None:
            raise HTTPException(400, "expires is required — a waiver without an end "
                                     "date is a permanently disabled check")
        if when < date.today():
            raise HTTPException(400, f"{when} is in the past, so the waiver would "
                                     f"never apply. Pick a date to review this by.")
        try:
            path = Declarations.write_waiver(
                GateWaiver(check=check, doc_type=str(payload.get("doc_type") or ""),
                           expires=when, reason=str(payload.get("reason") or ""),
                           by=str(payload.get("by") or "")),
                config_dir=service.config_dir)
        except DeclarationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"check": check, "expires": when.isoformat(),
                "doc_type": payload.get("doc_type") or "", "written_to": str(path)}

    @app.get("/batches/{batch_id}/summary")
    def summary(batch_id: str) -> Dict[str, Any]:
        """
        The totals a person would check by hand if they thought to, and what they rest on.

        Served beside the resolution because it is the question the reader is actually
        qualified to answer. Nobody who runs a warehouse can adjudicate `sku <- Material`
        against `sku <- PartNo.`; everybody who runs one knows whether the stock is
        twelve million or eighty-seven.
        """
        from ..ingest.intake import Intake
        from ..reporting.intake_summary import summarise_intake

        record = service.find_batch(batch_id)
        frame = service.landed_frame(record)
        declarations = service.declarations()
        # No `doc_type_hint`. Handing back the doc type the batch was landed under pins
        # the first decision permanently: the classifier never runs again, a misroute
        # can never be revised, and — because a hint routes at a flat 1.0 — every
        # re-read reports its type as stated where it should report what it measured.
        doc = Intake(verbose=False, declarations=declarations).load_frame(
            frame, source_name=record.get("source_name", batch_id))

        totals = summarise_intake({doc.doc_type: doc.frame})
        documents = [
            {"doc_type": d.doc_type, "rows": d.rows, "skus": d.skus,
             "implied_unit_value": d.implied_unit_value, "flags": list(d.flags),
             "readings": [
                 {"column": r.column, "kind": r.kind, "non_null": r.non_null,
                  "total": None if r.total != r.total else r.total,
                  "median": None if r.median != r.median else r.median,
                  "integer_share": (None if r.integer_share != r.integer_share
                                    else r.integer_share),
                  "earliest": str(r.earliest)[:10] if r.earliest is not None else None,
                  "latest": str(r.latest)[:10] if r.latest is not None else None,
                  "flags": list(r.flags)}
                 for r in d.readings]}
            for d in totals.documents
        ]

        return {"batch_id": batch_id, "doc_type": doc.doc_type,
                "documents": documents,
                "resting_on": _resting(service, doc, declarations).to_dict()}

    @app.get("/batches/{batch_id}/canonical")
    def canonical(batch_id: str, limit: int = Query(50, ge=1, le=1000),
                  offset: int = Query(0, ge=0)) -> Dict[str, Any]:
        """
        The batch after the adapter has run, in canonical field names.

        Served beside `/rows`, which is the same batch before anything decided what it
        meant. Reading the two side by side is how a person without the vocabulary to
        adjudicate a mapping still finds a mis-mapped column: the value is visibly the
        wrong kind of thing under a name that expects another.
        """
        from ..ingest.intake import Intake

        record = service.find_batch(batch_id)
        frame = service.landed_frame(record)
        doc = Intake(verbose=False, declarations=service.declarations()).load_frame(
            frame, source_name=record.get("source_name", batch_id))
        window = doc.frame.iloc[offset:offset + limit]
        return {"batch_id": batch_id, "doc_type": doc.doc_type,
                "total": len(doc.frame), "offset": offset,
                "columns": [str(c) for c in doc.frame.columns],
                "rows": _jsonable(window)}

    @app.get("/batches/{batch_id}/resolution")
    def resolution(batch_id: str) -> Dict[str, Any]:
        record = service.find_batch(batch_id)
        frame = service.landed_frame(record)
        # Deliberately un-hinted; see the note in `summary`.
        resolved = resolve_frame(frame, source_name=record.get("source_name", batch_id),
                                 declarations=service.declarations())
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
                               declarations=service.declarations())

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
                              declarations=service.declarations())
        return {
            "declaration": str(written),
            "changed": _diff(before, after),
            "resolution": after.to_dict(),
        }

    @app.post("/batches/{batch_id}/promote")
    def promote(batch_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Write the canonical frame of a landed batch into the fact store.

        `valid_time` is required and never inferred. It is the moment the data
        *describes* — a stock snapshot's date — and a file's timestamp is the moment it
        was copied, which a re-download rewrites while the data goes on describing
        whatever it described. Guessing it is how a store loses the ability to answer
        what was true last week.

        The fact batch keeps the landing batch's id, so a fact resolves back to the
        verbatim row it came from.
        """
        from ..ingest.intake import Intake

        record = service.find_batch(batch_id)
        valid_time = str(body.get("valid_time", "")).strip()
        by = str(body.get("by", "")).strip()
        if not valid_time:
            raise HTTPException(
                400, "`valid_time` is required — the date this data describes, which is "
                     "not the date the file was downloaded and cannot be inferred from it")
        if not by:
            raise HTTPException(400, "`by` is required")
        if service.status_of(batch_id) != "landed":
            raise HTTPException(
                409, f"batch {batch_id} is {service.status_of(batch_id)}, not landed")

        frame = service.landed_frame(record)
        doc = Intake(verbose=False, declarations=service.declarations()).load_frame(
            frame, source_name=record.get("source_name", batch_id))

        # The gate `KeyStatus.storable` was defined for and nothing had yet applied.
        # Storing a batch that cannot be keyed is not a smaller version of storing one
        # that can: a later correction lands as a second row, both are current, and
        # every as-of read of the document fails or double-counts. Refusing here, with
        # the remedy, beats accepting it and failing on every read afterwards.
        key = getattr(doc.test_report, "key_status", None)
        if key is not None and not key.storable:
            raise HTTPException(422, (
                f"{doc.doc_type} is keyed on {', '.join(key.declared)} and this export "
                f"supplies no {', '.join(key.missing) or 'part of it'}. It plans fine on "
                f"a partial key and cannot be stored on one, because a correction could "
                f"not be told from a new row. Declare the missing part for this document "
                f"— scope `value`, e.g. {(key.missing or ['location_id'])[0]} — and "
                f"promote it again."))

        try:
            written = FactStore(service.store_root).write_batch(
                doc_type=doc.doc_type, frame=doc.frame, valid_time=valid_time,
                source_name=record.get("source_name", ""),
                source_sha=record.get("source_sha"),
                key_verdict=key.verdict if key is not None else None,
                storable=key.storable if key is not None else None,
                written_by=by, batch_id=batch_id)
        except StoreUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        if written is None:
            # Same bytes, same parameters, same as-of: already stored. Not an error and
            # not a write — saying so is more use than a second identical batch.
            raise HTTPException(
                409, f"{doc.doc_type} with this content and as-of date is already in "
                     f"the store; nothing was written")
        return {"batch_id": written.batch_id, "doc_type": written.doc_type,
                "valid_time": written.valid_time, "rows": written.rows,
                "status": "promoted"}

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

    @app.get("/facts")
    def fact_types() -> List[Dict[str, Any]]:
        query = FactQuery(service.store_root, contracts=service.contracts)
        return [{"doc_type": doc_type,
                 "batches": len(query.select(doc_type).batches),
                 "describe": query.select(doc_type).describe()}
                for doc_type in query.doc_types()]

    @app.get("/facts/{doc_type}")
    def facts(doc_type: str,
              mode: str = Query("current", pattern="^(current|latest|history)$"),
              as_of: Optional[str] = Query(None),
              known_at: Optional[str] = Query(None),
              sku: Optional[str] = Query(None),
              location_id: Optional[str] = Query(None),
              layer: Optional[str] = Query(
                  None, pattern="^(canonical|prepared|unknown)$"),
              limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
        """
        Facts as of a moment. `mode` names the reading, because there is more than one.

        `current` is the newest observation of every key ever seen; `latest` is the
        newest batch and only it; `history` is every observation, and is the only one
        that must not be summed. Which of the first two is right is a property of the
        export rather than of this code — see `store/query.py` — so `carried_forward`
        comes back with `current`, sizing the disagreement instead of hiding it.
        """
        query = FactQuery(service.store_root, contracts=service.contracts)
        selection = query.select(doc_type, as_of=as_of, known_at=known_at, layer=layer)
        where = {k: v for k, v in (("sku", sku), ("location_id", location_id)) if v}
        try:
            frame = getattr(query, mode)(doc_type, as_of=as_of, known_at=known_at,
                                         where=where or None, limit=limit, layer=layer)
        except (KeyIncomplete, MixedLayers, NoSuchColumn) as exc:
            raise HTTPException(422, str(exc)) from exc
        except QueryUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

        body: Dict[str, Any] = {
            "doc_type": doc_type, "mode": mode,
            "as_of": as_of, "known_at": known_at, "layer": layer,
            "layers": selection.layers,
            "batches": [{"batch_id": b["batch_id"], "valid_time": b["valid_time"],
                         "transaction_time": b["transaction_time"], "rows": b["rows"],
                         "source_name": b.get("source_name", "")}
                        for b in selection.batches],
            "selection": selection.describe(),
            "columns": [str(c) for c in frame.columns],
            "rows": _jsonable(frame),
        }
        if mode == "current" and selection:
            body["carried_forward"] = query.carried_forward(
                doc_type, as_of=as_of, known_at=known_at, layer=layer)
        return body

    def _reach_of(rules_path: Path) -> Dict[str, Any]:
        """
        What these rules last reached, keyed by rule id, or an explanation of why not.

        Keyed rather than ordered: the caller holds the rules and does the join, which
        means a rule with no entry is visibly a rule with no entry instead of a row
        that silently slipped one position.

        The explanation is worth as much care as the counts. There are four reasons the
        page can be blank and they send someone to four different places, so the line
        says only what the search actually established. "The file has been edited" in
        particular is a claim about history, and it is only true when every recorded
        run was examined and none of them was under these bytes.
        """
        registry = RunRegistry(service.output_dir)
        digest = sha256_file(rules_path)
        run = registry.latest_under_rules(digest, with_hits=True, limit=_REACH_SCAN)
        if run is not None:
            return {
                "hits": {
                    "run_id": run.get("run_id"),
                    "run_at": run.get("run_at"),
                    "input_fingerprint": run.get("input_fingerprint"),
                    "rules": {h.get("rule_id"): h for h in run.get("rule_hits") or []},
                },
                "note": f"Counts are from run {run.get('run_id')} — the newest run "
                        f"under these exact rules. They move with the facts: a rule "
                        f"reaches whatever the SKUs of that run made it reach.",
            }
        recorded = registry.index()
        # These rules have run, and that run kept no reach: every run from before the
        # reach was retained lands here, which on the first look at any existing output
        # directory is all of them. Reporting that as an edit would send someone to
        # hunt for a change to a file nobody has touched.
        ran_without_reach = registry.latest_under_rules(digest, limit=_REACH_SCAN)
        if ran_without_reach is not None:
            why = (f"Run {ran_without_reach.get('run_id')} planned under these exact "
                   f"rules but recorded no per-rule reach — it ran before the reach "
                   f"was kept, or it stopped before resolving.")
        elif not recorded:
            why = f"No runs are recorded under {service.output_dir}."
        elif len(recorded) > _REACH_SCAN:
            # The scan is bounded, so a miss inside it is not a miss over the history.
            why = (f"None of the {_REACH_SCAN} most recent runs planned under these "
                   f"exact rules. Older runs were not searched, so this does not say "
                   f"the file has changed.")
        else:
            why = ("No run under these exact rules. Every recorded run was searched, "
                   "so the file has been edited since the last one — and counts from "
                   "before an edit belong to the rules as they were.")
        return {
            "hits": None,
            "note": why + " Which SKUs a rule reaches is decided against a frame of "
                          "SKUs, so it takes a run to answer; the next one fills this in.",
        }

    @app.get("/policy")
    def policy() -> Dict[str, Any]:
        """
        The parameter set in force, read-only.

        Read-only for now, and the value an interface adds here is not the editing
        anyway — it is showing which rule reached which SKUs, and what a change to one
        did to a run, neither of which a text editor can show.

        The reach comes from a run, because it has to: a rule's scope is a question
        about a frame of SKUs, and there is no frame here. So the newest run that
        planned under *these exact rule bytes* is found in the registry and its counts
        are shown against the rules they belong to. A run under a different rule set is
        not a near miss to fall back on — its counts would be attached to rules that
        never produced them — so when there is none the answer is that there is none.
        """
        from ..policy.parameters import PlanningParameters

        path = ((service.config_dir or Path(__file__).parents[2] / "config")
                / "planning_parameters.md")
        try:
            params = PlanningParameters(path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

        return {
            "source": str(params.path),
            "conventions": params.conventions,
            "defaults": params.defaults,
            "segmentation": params.segmentation,
            "rules": [
                {"rule_id": r.rule_id, "name": r.name, "scope": r.scope,
                 "sets": r.overrides, "rationale": " ".join(r.rationale.split()),
                 "owner": r.owner, "date": r.date}
                for r in params.rules
            ],
            "changes": rule_history(service.config_dir, limit=20),
            **_reach_of(params.path),
        }

    @app.put("/policy/rules")
    def edit_rules(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Propose a change to the rules, and apply it once the diff has been seen.

        One endpoint carrying the change rather than three shaped like HTTP verbs,
        which is the contract INTERFACE.md §7 fixed: `{change, reason, by}` in, the diff
        it would make out, approval applies it. The shape is what has to survive the
        move to a server, so it is worth more than the REST tidiness of a `DELETE`.

        Writing is opt-in (`apply: true`) and pinned to `basis`, exactly as the macro
        editor is — a client asking "what would this do" must not have already done it,
        and what gets approved is the diff that was shown rather than the rule it was
        shown for.

        The three actions differ in what they need and in what they cost:

        - `edit` takes `changes`, a subset of the rule's fields. Only those are
          rewritten; every other byte of the rule, including the comments beside its
          parameters, stays as it was.
        - `add` takes `rule` and appends it last, which is the only position with a
          statable meaning: it wins over everything above it.
        - `remove` takes only the id. Its reason goes to the log rather than the file,
          because afterwards there is no rule left to carry it.
        """
        from ..policy import rules_edit

        root = service.config_dir or Path(__file__).parents[2] / "config"
        action = str(payload.get("action") or "edit").strip()
        rule_id = str(payload.get("rule_id") or "").strip()

        if action == "edit":
            def proposal_for():
                return rules_edit.propose_edit(rule_id, payload.get("changes") or {},
                                               config_dir=root)
        elif action == "add":
            def proposal_for():
                return rules_edit.propose_add(payload.get("rule") or {},
                                              config_dir=root)
        elif action == "remove":
            def proposal_for():
                return rules_edit.propose_remove(rule_id, config_dir=root)
        else:
            raise HTTPException(400, f"action must be edit, add or remove, "
                                     f"not {action!r}")

        try:
            if not payload.get("apply"):
                return {**proposal_for().to_dict(), "applied": False}
            return rules_edit.apply(
                proposal_for, reason=payload.get("reason", ""),
                by=payload.get("by", ""), basis=str(payload.get("basis") or ""),
                config_dir=root)
        except EditRefused as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/policy/macro")
    def macro() -> Dict[str, Any]:
        """
        The scalars, each with the file it came from.

        Only what the engine actually reads. A settings page that lists a switch nothing
        honours is worse than no page, because a switch reads as a guarantee — which is
        why there is no working-day/calendar-day control here: the distinction does not
        exist anywhere in the pipeline yet.
        """
        import json as _json
        from ..policy.parameters import PlanningParameters

        root = service.config_dir or Path(__file__).parents[2] / "config"
        out: Dict[str, Any] = {"config_dir": str(root), "settings": []}

        from ..policy import macro as macro_edit

        def add(name, value, source, note=""):
            # Editability is attached by name from the one registry that decides it,
            # rather than inferred from the source file: `fx_rates.json` holds both an
            # editable scalar and a derived reading, and "it came from a file" is not
            # the same claim as "a form may write it".
            setting = macro_edit.BY_NAME.get(name)
            out["settings"].append({
                "name": name, "value": value, "source": source, "note": note,
                "editable": setting is not None,
                "kind": setting.kind if setting else None,
                "choices": list(setting.choices) if setting else [],
                "impact": setting.impact if setting else "",
                "file": setting.filename if setting else None,
            })

        try:
            node = _json.loads((root / "node_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            node = {}
        for key in ("location_id", "location_name", "currency", "planning_cycle"):
            if key in node:
                add(key, node[key], "node_config.json")

        try:
            params = PlanningParameters(root / "planning_parameters.md")
            for key, value in params.conventions.items():
                add(key, value, "planning_parameters.md", "convention — changes every figure")
        except (FileNotFoundError, ValueError):
            pass

        # Through `FxTable`, not by parsing the file here. Reading it directly reported
        # its top-level keys — `rates`, `reporting_currency` — as if they were currency
        # codes, which is the shape of every mistake this interface is supposed to make
        # impossible: a second implementation of a read the package already does.
        from ..fx import FxTable

        table = FxTable.load(root)
        add("reporting_currency", table.reporting_currency,
            "fx_rates.json" if table.source_path else "default",
            "every figure is restated into this")
        add("fx_currencies", table.currencies,
            str(table.source_path.name) if table.source_path else "no rate file",
            "money in any of these is converted on read"
            if table.currencies else "no rates configured — money is not converted")

        out["changes"] = macro_edit.history(root, limit=20)
        return out

    @app.put("/policy/macro")
    def edit_macro(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Propose a change to one scalar, and apply it once the diff has been seen.

        The two steps are the same request twice, which is deliberate: a stored
        proposal would be interface-only state, and INTERFACE.md rules that out for
        every screen here. What connects them instead is `basis` — the digest of the
        file the diff was produced against, returned by the proposal and required by
        the apply. Approving a diff therefore approves *those bytes*, and a file that
        moved in between is a refusal rather than a surprise.

        Writing is opt-in (`apply: true`). A PUT that writes by default would mean a
        client asking "what would this do" had already done it, and the one thing this
        endpoint exists to provide is the chance to look first.
        """
        from ..policy.macro import MacroError, apply as apply_macro, propose

        root = service.config_dir or Path(__file__).parents[2] / "config"
        name = str(payload.get("name") or "").strip()
        if "value" not in payload:
            raise HTTPException(status_code=400, detail="value is required")
        try:
            if not payload.get("apply"):
                proposal = propose(name, payload["value"], config_dir=root)
                return {**proposal.to_dict(), "applied": False}
            return apply_macro(
                name, payload["value"],
                reason=payload.get("reason", ""), by=payload.get("by", ""),
                basis=str(payload.get("basis") or ""), config_dir=root).to_dict()
        except MacroError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs")
    def runs(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
        registry = RunRegistry(service.output_dir)
        entries = registry.index()
        return {"output_dir": str(service.output_dir),
                "runs": list(reversed(entries))[:limit]}

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> Dict[str, Any]:
        manifest = RunRegistry(service.output_dir).get(run_id)
        if manifest is None:
            raise HTTPException(404, f"no run {run_id!r} under {service.output_dir}")
        return manifest

    @app.get("/runs/{run_a}/diff/{run_b}")
    def diff(run_a: str, run_b: str) -> Dict[str, Any]:
        """
        Why two runs differ, which is what decides how their outputs may be read.

        The point of a policy screen. A parameter change is not a state to inspect: it
        is the difference between two runs, and whether that difference is attributable
        to the change depends on whether anything else moved at the same time.
        """
        comparison = RunRegistry(service.output_dir).compare(run_a, run_b)
        if comparison is None:
            raise HTTPException(404, f"one of {run_a!r}, {run_b!r} is not in the registry")
        return {
            "a": run_a, "b": run_b,
            "basis": comparison.basis,
            "describe": comparison.describe(),
            "same_inputs": comparison.same_inputs,
            "same_config": comparison.same_config,
            "same_code": comparison.same_code,
            "moved": [name for name, same in (("facts", comparison.same_inputs),
                                              ("parameters", comparison.same_config),
                                              ("code", comparison.same_code))
                      if not same],
        }

    # Mounted last, and it matters: routes match in registration order and a mount at
    # "/" swallows everything after it. The screen is a client of this API and nothing
    # more — no state of its own, no second opinion about a mapping — which is what
    # makes it replaceable without touching anything above.
    web = Path(__file__).parent / "web"
    if web.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=web, html=True), name="web")

    return app


def _jsonable(frame) -> List[Dict[str, Any]]:
    """Rows as plain JSON. NaN is not valid JSON and pandas will happily emit it."""
    import numpy as np

    out = []
    for row in frame.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                clean[str(key)] = None
            elif isinstance(value, (np.integer,)):
                clean[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[str(key)] = float(value)
            elif isinstance(value, (np.bool_,)):
                clean[str(key)] = bool(value)
            else:
                clean[str(key)] = value if isinstance(value, (str, int, float, bool)) \
                    else str(value)
        out.append(clean)
    return out


def _reporting_currency(config_dir) -> str:
    """The currency the run reports in, from the node config the pipeline reads."""
    import json

    root = Path(config_dir) if config_dir else Path(__file__).parents[2] / "config"
    try:
        return str(json.loads((root / "node_config.json").read_text(
            encoding="utf-8")).get("currency") or "USD")
    except (OSError, ValueError):
        return "USD"


def _resting(service, doc, declarations):
    """
    What this document's figures rest on: what a person declared, and what defaulted.

    The defaulted case is read off the resolution rather than re-derived. A document
    with no currency column is taken by `fx.convert_money` to be in the reporting
    currency already, and the condition for that is exactly "the field found no column"
    — which the resolution has already established. Restating the rule here would give
    the interface its own opinion about when money is being assumed.
    """
    from ..resolution import SOURCE_ABSENT, from_document

    reporting = _reporting_currency(service.config_dir)
    # `overrides_for`, not `values_for`: the second reduces an override to its value and
    # drops who asserted it and why. An assumption listed without a name beside it is
    # exactly the unattributed statement this layer exists to replace, and the screen
    # rendered it as "someone declared".
    declared = [
        Assumption(doc_type=doc.doc_type, field=o.field, value=o.value,
                   basis=BASIS_DECLARED, reason=o.reason, by=o.by)
        for o in declarations.overrides_for("value", doc_type=doc.doc_type)
    ]
    resolved = from_document(doc, doc.frame, declarations=declarations)
    absent = {f.field for f in resolved.fields if f.source == SOURCE_ABSENT}
    assumed = ["currency"] if ("currency" in absent
                               and "currency" in doc.route.contract.fields) else []

    from ..ingest.exposure import assemble

    assumptions = assemble(declared, [doc.doc_type] if assumed else [], reporting)
    return measure(assumptions, {doc.doc_type: doc.frame},
                   {doc.doc_type: doc.route.contract}, reporting_currency=reporting)


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
