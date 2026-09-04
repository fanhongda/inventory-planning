"""
Intake — the single entry point for getting files into canonical form.

Hand it a list of paths in any order, with no declaration of what they are:

    intake = Intake()
    result = intake.load_files(["export1.xlsx", "planner_demand.xlsx", "stock.csv"])

Each file is profiled, routed to a contract, transformed by an adapter, and tested.
The result reports what the run can and cannot answer. Nothing here asks the caller
to name the document, because the profile already determines it — which is what makes
a planner-supplied time series work without anyone knowing a different loader exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .adapter import Adapter, TransformStep
from .capabilities import CapabilityResolver, IntakePlan
from .contract import ContractRegistry, DocContract, default_registry
from .contract_tests import ContractTester, ContractTestReport
from .encoding import describe_choice, sniff_encoding
from .profiler import TableProfile
from .registry import AdapterRegistry, RouteResult
from .supersede import SupersessionMap, SupersessionReport


def _unnamed_share(columns) -> float:
    if len(columns) == 0:
        return 1.0
    unnamed = sum(1 for c in columns
                  if str(c).startswith("Unnamed:") or str(c).strip() == "" or pd.isna(c))
    return unnamed / len(columns)


# Share of source headers two files must have in common before they can be treated as
# partitions of one export. Set low enough to tolerate an extra column in one country's
# tab, high enough that two different reports never qualify.
_PARTITION_HEADER_SIMILARITY = 0.75


def _same_layout(a: "LoadedDocument", b: "LoadedDocument") -> bool:
    """Whether two frames came out of the same report, judged on their raw headers."""
    pa, pb = a.route.profile, b.route.profile
    if pa.header_hash and pa.header_hash == pb.header_hash:
        return True
    tokens_a = {c.name for c in pa.columns}
    tokens_b = {c.name for c in pb.columns}
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
    return overlap >= _PARTITION_HEADER_SIMILARITY


def is_tabular(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Whether a sheet is a data table at all.

    Workbooks routinely carry pivot tables, cover sheets and scratch tabs alongside
    the export. A pivot in particular survives a naive read — it has rows and columns
    and will happily profile — so it has to be rejected explicitly rather than left to
    fail somewhere downstream with a confusing message.
    """
    if len(df) == 0 or len(df.columns) == 0:
        return False, "empty sheet"
    if _unnamed_share(df.columns) > 0.5:
        return False, "over half the columns have no header — pivot table or scratch sheet"

    header_text = " ".join(str(c).lower() for c in df.columns)
    for marker in ("sum of ", "count of ", "column labels", "row labels", "grand total"):
        if marker in header_text:
            return False, f"pivot-table marker {marker.strip()!r} in the header row"

    density = df.notna().mean().mean()
    if density < 0.15:
        return False, f"only {density:.0%} of cells populated — not a data table"
    return True, ""


def _read_excel_sheet(path: Path, sheet: Any) -> pd.DataFrame:
    """Read one sheet, hunting for the real header row under any title banner."""
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    for header_row in (1, 2, 3):
        if _unnamed_share(df.columns) <= 0.5:
            break
        try:
            candidate = pd.read_excel(path, sheet_name=sheet, dtype=str, header=header_row)
        except (ValueError, IndexError):
            break
        if _unnamed_share(candidate.columns) >= _unnamed_share(df.columns):
            break
        df = candidate
    return df


def load_sheets(path: Union[str, Path]) -> List[Tuple[str, pd.DataFrame]]:
    """
    Read every table in a file as raw strings — one entry per CSV, per sheet for Excel.

    Reading only the first sheet is not a simplification, it is a data-loss bug: a
    workbook whose stock is split into AUS and NZ tabs silently becomes one country,
    and one whose first tab is a pivot summary becomes no data at all. Both happen in
    real exports, so every sheet is read and each is routed on its own merits.

    Everything is read as text on purpose: pandas' type inference is the first place a
    European decimal or a leading-zero part number gets mangled, and the adapter's
    parsing rules — which know the source's locale — should be the only converter.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # `sniff_encoding` rather than a try-until-it-works ladder: latin-1 decodes
        # every possible byte, so the old loop could never reach a CJK codec and a GBK
        # export became mojibake without raising. See ingest/encoding.py.
        encoding = sniff_encoding(path)
        return [("", pd.read_csv(path, encoding=encoding, dtype=str,
                                 keep_default_na=False, na_values=[""]))]

    if suffix in (".xlsx", ".xls", ".xlsm"):
        book = pd.ExcelFile(path)
        return [(str(sheet), _read_excel_sheet(path, sheet)) for sheet in book.sheet_names]

    raise ValueError(f"Unsupported file type {suffix!r} for {path.name}. Use CSV or Excel.")


def load_file(path: Union[str, Path]) -> pd.DataFrame:
    """First tabular sheet of a file. Kept for callers that want a single frame."""
    for _, df in load_sheets(path):
        ok, _ = is_tabular(df)
        if ok:
            return df
    raise ValueError(f"No tabular sheet found in {Path(path).name}")



# At most this many columns per document are kept as possible item keys, and a column
# needs at least this many distinct values to qualify. Both exist to bound the memory
# a 127,000-row export would otherwise cost.
_MAX_KEY_CANDIDATES = 12
_MIN_KEY_DISTINCT = 10


def _key_candidates(raw: pd.DataFrame, profile: TableProfile) -> Dict[str, set]:
    """
    Source columns that could be an item identifier, as normalized value sets.

    Kept so that a document whose `sku` joins to nothing can be told which of its
    other columns *would* have joined. That is the difference between a report full
    of zeroes and a one-line answer: one planner master keyed on a local code in
    `Material` while carrying the ERP number in `Alternate material`, and nothing in
    the run could say so.
    """
    from .adapter import _normalize_material

    ranked = sorted(
        (c for c in profile.columns
         if not c.is_period_column and c.distinct >= _MIN_KEY_DISTINCT),
        key=lambda c: -c.distinct_rate,
    )[:_MAX_KEY_CANDIDATES]

    out: Dict[str, set] = {}
    for col in ranked:
        if col.name not in raw.columns:
            continue
        values = _normalize_material(raw[col.name].dropna().astype(str))
        out[col.name] = set(values.dropna())
    return out


@dataclass
class LoadedDocument:
    """One file, fully processed."""

    source_path: Optional[Path]
    source_name: str
    doc_type: str
    frame: pd.DataFrame
    route: RouteResult
    transform_log: List[TransformStep]
    test_report: ContractTestReport
    sheet_name: str = ""
    # Normalized values of the source columns that could plausibly be an item key,
    # kept so a document whose SKU joins to nothing can be told which of its other
    # columns would have joined. Sets of strings, capped — not the frames.
    key_candidates: Dict[str, set] = dc_field(default_factory=dict)

    @property
    def adapter(self) -> Adapter:
        return self.route.adapter

    @property
    def profile(self) -> TableProfile:
        return self.route.profile

    @property
    def passed(self) -> bool:
        return self.test_report.passed

    @property
    def row_count(self) -> int:
        return len(self.frame)

    # A misroute passes every contract test, so confidence is the only handle on it.
    # Named once because two callers must agree on what "uncertain" means: the summary
    # that asks a human to confirm, and the supersession guard that refuses to rewrite
    # item numbers on a document it is not sure about.
    _CONFIDENT = 0.75

    @property
    def route_uncertain(self) -> bool:
        return (self.route.confidence < self._CONFIDENT
                or "close call" in (self.route.reason or ""))

    def explain(self) -> str:
        """Full provenance for this document — routing, every transform, every test."""
        lines = [
            f"  {self.source_name}  ->  {self.doc_type}  ({self.row_count:,} rows)",
            f"    adapter    : {self.adapter.name} [{self.adapter.status}]",
            f"    confidence : {self.route.confidence:.0%} — {self.route.reason}",
            "    transforms :",
        ]
        lines.extend(f"      {step}" for step in self.transform_log)
        lines.append(self.test_report.summary())
        return "\n".join(lines)


@dataclass
class IntakeResult:
    """Everything loaded in one intake pass."""

    documents: Dict[str, LoadedDocument] = dc_field(default_factory=dict)
    plan: IntakePlan = dc_field(default_factory=IntakePlan)
    failures: List[Tuple[str, str]] = dc_field(default_factory=list)
    # What the declared material renumberings did to the frames above. Empty on a run
    # with no substitution list — which is not the same as a run where every number
    # is its own material, and the plan says which of the two happened.
    supersessions: SupersessionReport = dc_field(default_factory=SupersessionReport)
    # Observations worth stating that are not problems. Kept apart from `failures`
    # so the run does not cry wolf about things the planner already knows.
    notes: List[str] = dc_field(default_factory=list)

    def frame(self, doc_type: str) -> Optional[pd.DataFrame]:
        doc = self.documents.get(doc_type)
        return doc.frame if doc else None

    def get(self, doc_type: str) -> Optional[LoadedDocument]:
        return self.documents.get(doc_type)

    @property
    def can_run(self) -> bool:
        return self.plan.can_run

    @property
    def blocking_failures(self) -> List[LoadedDocument]:
        return [d for d in self.documents.values() if not d.passed]

    def unusable(self) -> List[Tuple[LoadedDocument, List[str]]]:
        """
        Documents missing a field the analytics will index by name, with the fields.

        Not every failed test should stop a run. A semantic assertion — 8% of lead
        times negative, a handful of negative quantities — is a data-quality finding
        that the report is designed to carry, and stopping on it would make the
        pipeline unusable on real exports. A *structurally* absent required field is
        different in kind: nothing downstream guards against it, so the run does not
        degrade, it raises `KeyError` several layers away from the cause.

        That is what happened to a sales history whose date column matched no alias:
        the contract test said so plainly, nothing consulted it, and the traceback
        surfaced in `to_time_series` with no mention of the file it came from.
        """
        out = []
        for doc in self.documents.values():
            absent = [
                r.name.split(":", 1)[1]
                for r in doc.test_report.results
                if not r.passed and r.name.startswith("required_field:")
            ]
            if absent:
                out.append((doc, absent))
        return out

    def summary(self) -> str:
        lines = ["", "=" * 62, "  DATA INTAKE", "=" * 62, ""]
        for doc in sorted(self.documents.values(), key=lambda d: d.doc_type):
            status = "OK" if doc.passed else doc.test_report.status
            draft = " [DRAFT ADAPTER]" if doc.route.is_draft else ""
            lines.append(
                f"  {doc.doc_type:<20} {doc.row_count:>7,} rows  {status:<9}"
                f"{draft}  <- {doc.source_name}"
            )
        for name, reason in self.failures:
            lines.append(f"  {'(failed)':<20} {'':>7}        {reason}  <- {name}")

        # A misroute passes every contract test — the numbers are computed correctly
        # from the wrong premise — so `OK` on the line above is not evidence that the
        # file is what the router thinks. Where the margin was thin, say so here rather
        # than only inside the routing reason nobody reads.
        uncertain = [doc for doc in self.documents.values() if doc.route_uncertain]
        if uncertain:
            lines.append("")
            lines.append("  ⚠ Routed on a thin margin — confirm before trusting the run:")
            for doc in sorted(uncertain, key=lambda d: d.route.confidence):
                lines.append(f"      {doc.source_name} -> {doc.doc_type} "
                             f"({doc.route.confidence:.0%})")
            lines.append("      python -m inventory_planning.explain <file>   "
                         "shows what each contract scored and why")

        # Above the notes, not among them. Merging two item numbers changes what every
        # figure below is counted over, so it is not an observation about the data —
        # it is a statement about what the run is planning.
        merge = self.supersessions.summary()
        if merge:
            lines.append("")
            lines.append(merge)

        for note in self.notes:
            lines.append("")
            lines.append(note)

        lines.append("")
        lines.append(self.plan.summary())

        problems = [d for d in self.documents.values() if not d.passed]
        if problems:
            lines.append("")
            lines.append("  Contract test failures:")
            for doc in problems:
                lines.append(doc.test_report.summary())
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": {
                doc_type: {
                    "source": doc.source_name,
                    "rows": doc.row_count,
                    "adapter": doc.adapter.name,
                    "adapter_status": doc.adapter.status,
                    "confidence": round(doc.route.confidence, 3),
                    "tests": doc.test_report.to_dict(),
                }
                for doc_type, doc in self.documents.items()
            },
            "plan": self.plan.to_dict(),
            "failures": [{"source": n, "reason": r} for n, r in self.failures],
            "supersessions": {
                "applied": [
                    {"old_sku": p.old_sku, "new_sku": p.new_sku, "ratio": p.ratio,
                     "source": p.source}
                    for p in self.supersessions.applied
                ],
                "phase_pairs": self.supersessions.phase_pairs,
                "problems": list(self.supersessions.problems),
                "challenges": list(self.supersessions.challenges),
            },
        }


class Intake:
    """Profiles, routes, transforms and verifies input files."""

    def __init__(
        self,
        contracts: ContractRegistry = None,
        adapters: AdapterRegistry = None,
        tenant: str = "default",
        baseline_path: Union[str, Path] = None,
        verbose: bool = True,
        declarations=None,
    ):
        self.contracts = contracts or default_registry()
        self.adapters = adapters or AdapterRegistry(contracts=self.contracts)
        self.tester = ContractTester()
        self.resolver = CapabilityResolver()
        self.tenant = tenant
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.verbose = verbose
        # What a person has declared about these documents. Applied over the routed
        # mapping so one column can be corrected without hand-authoring an adapter —
        # the setting between "guess" and "freeze" that did not exist, and whose
        # absence made editing headers in Excel the field remedy.
        self.declarations = declarations
        self._declaration_notes: List[str] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def load_files(
        self,
        paths: List[Union[str, Path]],
        hints: Dict[str, str] = None,
    ) -> IntakeResult:
        """
        Load every file, identifying each one automatically.

        `hints` maps a filename to a doc_type for the rare case where a file is
        genuinely ambiguous — two exports from the same system with near-identical
        headers, say. It is an override, not a requirement.
        """
        hints = hints or {}
        result = IntakeResult()
        unrecognised: List[str] = []
        # doc_type -> the documents claiming it, decided together once all are in
        claims: Dict[str, List[LoadedDocument]] = {}

        for path in paths:
            path = Path(path)
            try:
                sheets = load_sheets(path)
            except Exception as exc:
                result.failures.append((path.name, f"could not read: {exc}"))
                unrecognised.append(path.name)
                continue

            # A codec that had to be guessed decides what every supplier and customer
            # name in the file says, so say which one was used. Silence here is how a
            # GBK export became mojibake and the mojibake became the data.
            if path.suffix.lower() == ".csv":
                note = describe_choice(sniff_encoding(path), path.name)
                if note:
                    result.notes.append(note)

            hint = hints.get(path.name) or hints.get(str(path))
            # Only qualify the name when the workbook actually has several sheets;
            # `stock.xlsx[Sheet1]` is noise for the ordinary single-sheet case.
            multi_sheet = len(sheets) > 1
            for sheet_name, raw in sheets:
                label = f"{path.name}[{sheet_name}]" if multi_sheet and sheet_name else path.name

                tabular, reason = is_tabular(raw)
                if not tabular:
                    result.failures.append((label, f"skipped: {reason}"))
                    continue

                try:
                    doc = self.load_frame(raw, source_name=label, doc_type_hint=hint,
                                          source_path=path, sheet_name=sheet_name)
                except Exception as exc:
                    result.failures.append((label, f"could not process: {exc}"))
                    unrecognised.append(label)
                    continue

                claims.setdefault(doc.doc_type, []).append(doc)

        loaded: Dict[str, str] = {}
        withheld: Dict[str, set] = {}
        for doc_type, docs in claims.items():
            chosen = self._resolve_claims(doc_type, docs, result)
            result.documents[doc_type] = chosen
            loaded[doc_type] = chosen.source_name
            missing = self._unsupplied_capabilities(chosen, doc_type)
            if missing:
                withheld[doc_type] = missing

        # Before any cross-document check, because every one of them is about item
        # numbers: a renumbered part looks like two documents disagreeing on their key
        # until the two numbers have been made into one.
        self._apply_supersessions(result)

        result.plan = self.resolver.resolve(loaded, unrecognised, withheld=withheld)
        self._note_po_overlap(result)
        suspect = self._check_key_shape(result)
        self._check_sku_agreement(result, suspect)
        self._note_mixed_formats(result)
        # Declarations last, so what a person asked for and what became of it sit
        # together at the end of intake rather than scrolling past between files. A
        # declaration that matched nothing is reported here too — silence there is how
        # someone comes to believe a mapping is corrected when it is not.
        result.notes.extend(self._declaration_notes)
        if self.declarations is not None:
            result.notes.extend(self.declarations.notes())
        self._declaration_notes = []

        if self.verbose:
            print(result.summary())
        return result


    # ── Material identity ────────────────────────────────────────────────────

    def _apply_supersessions(self, result: IntakeResult) -> None:
        """
        Make one material out of two numbers, everywhere at once.

        "Everywhere at once" is the whole of it. Every join downstream is on `sku`, so
        rewriting the number in the demand history and not in the stock report is
        worse than leaving both alone: it produces a SKU with a forecast and no
        position and another with a position and no forecast, and neither of them
        raises anything.
        """
        # Routing is inferred, and for every other document a misroute is a wrong
        # premise that the numbers can still be checked against. Here it is a rewrite
        # of item identity that nothing downstream can see, so this is the one contract
        # that does not act on a guess: below the confidence the summary already asks a
        # human to confirm at, the pairs are reported and nothing is merged.
        doc = result.get("substitution")
        withheld = None
        if doc is not None and doc.route_uncertain:
            withheld = (
                f"{doc.source_name} was routed to a substitution list on a thin margin "
                f"({doc.route.confidence:.0%}). Merging item numbers rewrites every "
                f"document and leaves no trace downstream, so it is not done on a "
                f"guess. Confirm the file is a renumbering list and re-run."
            )

        smap = SupersessionMap.from_frames(
            substitution_df=result.frame("substitution"),
            item_master_df=result.frame("item_master"),
            withheld=withheld,
        )
        result.supersessions = smap.report
        if not smap:
            return

        # Asked before anything moves. Once the rewrite has happened the old number is
        # not in any frame to be checked against, and the one declaration worth
        # doubting is the one that has already been acted on.
        smap.challenge({
            dt: doc.frame for dt, doc in result.documents.items()
            if dt != "substitution"
        })

        for doc_type, doc in result.documents.items():
            if doc_type == "substitution":
                continue
            contract = self.contracts.get(doc_type)
            before = len(doc.frame)
            frame, records = smap.apply(doc.frame, contract, doc_type=doc_type)
            if not records:
                continue

            doc.frame = frame
            smap.report.records.extend(records)
            moved = sum(r.rows for r in records)
            doc.transform_log.append(TransformStep(
                "sku", "superseded",
                f"{moved:,} row(s) rewritten to their successor across "
                f"{len(records)} pair(s); {before:,} -> {len(frame):,} rows",
            ))

            # The recombination is the pipeline's own work, so nothing else will catch
            # it going wrong — and a broken grain is precisely the failure that stays
            # quiet while every quantity downstream doubles.
            grain = self.tester._grain_test(frame, contract)
            if not grain.passed:
                smap.report.problems.append(
                    f"{doc_type}: merging left duplicate rows on "
                    f"{contract.natural_key} — {grain.detail}"
                )

    # Fewer distinct item numbers than this, in a document of any real size, is not an
    # item key. SAP line numbers land here — a purchase order rarely runs past line 60,
    # so `Item` yields six values — and so does a status code or a plant.
    _SUSPICIOUS_KEY_DISTINCT = 10
    # Only asked of a source big enough for the count to mean anything. Ten SKUs in a
    # twelve-row file is a small file, not a broken key.
    _KEY_SHAPE_MIN_ROWS = 30

    @classmethod
    def _check_key_shape(cls, result: IntakeResult) -> set:
        """
        Name any document whose `sku` is too coarse to be an item number. Returns them.

        This is the check that was missing, and its absence is what let a line-number
        key through: `_check_sku_agreement` measures how much of a document's key the
        others recognise, and skips any document with fewer than ten distinct keys as
        too small a sample to judge. A document keyed on 10, 20, 30 has exactly that
        shape, so the one guard written for it declined to look — and then, because the
        corrupt key was still counted as evidence against everybody else, blamed the
        file that was mapped correctly.

        Cardinality is judged against the *source* row count rather than the frame's,
        because a rollup to sku grain collapses the document to one row per key and
        erases the very disproportion being looked for.

        Unlike the agreement check this one needs no second document, which matters:
        a lone sales history keyed on six line numbers is diagnosable on its own, and
        the old code went silent on it twice over.
        """
        keyed = {dt: doc for dt, doc in result.documents.items()
                 if "sku" in doc.frame.columns and len(doc.frame)}
        skus = {dt: set(doc.frame["sku"].dropna().astype(str)) for dt, doc in keyed.items()}

        # A coarse key is a symptom, not a diagnosis. Eight SKUs across eighty rows is
        # equally a small catalogue ordered repeatedly — the project's own sample data
        # is exactly that — so cardinality alone accuses honest files. Something else
        # has to corroborate: either the values are a series, or nothing else in the
        # run recognises a single one of them.
        coarse = {
            dt for dt, own in skus.items()
            if len(own) < cls._SUSPICIOUS_KEY_DISTINCT
            and max(keyed[dt].route.profile.row_count, len(keyed[dt].frame))
            >= cls._KEY_SHAPE_MIN_ROWS
        }
        columns = {dt: cls._key_column(keyed[dt]) for dt in coarse}
        suspect = {dt for dt, col in columns.items() if col is not None and col.is_small_ordinal}
        for doc_type in coarse - suspect:
            rest = set().union(*(s for dt, s in skus.items()
                                 if dt != doc_type and dt not in suspect), set())
            if rest and not (skus[doc_type] & rest):
                suspect.add(doc_type)

        # What the healthy documents key on. Where there is any, it is far better
        # evidence for naming the right column than cardinality is.
        others: set = set().union(
            *(s for dt, s in skus.items() if dt not in suspect), set()
        )

        for doc_type in sorted(suspect):
            doc, own = keyed[doc_type], skus[doc_type]
            source_rows = max(doc.route.profile.row_count, len(doc.frame))
            mapped = doc.route.adapter.column_map.get("sku", "(derived)")
            column = columns[doc_type]
            lines = [
                f"  ⚠ {doc_type} is keyed on only {len(own)} distinct item number(s) "
                f"across {source_rows:,} source rows.",
                f"      {doc.source_name}: sku <- {mapped!r}, values "
                f"{', '.join(sorted(own)[:6])}"
                + (" …" if len(own) > 6 else "") + ".",
            ]
            if column is not None and column.is_small_ordinal:
                lines.append(
                    f"      Those are a series, not identifiers — {mapped!r} is a "
                    f"document line number. SAP labels the line number `Item` in every "
                    f"module; the material is in a different column."
                )
            named = cls._alternative_keys(doc, mapped, others)
            lines.append(
                f"      Columns that could be the item key instead: {named}." if named
                else "      No other column in this export looks like an item key — "
                     "the material number may simply not be in the extract."
            )
            lines.append(
                "      Every join downstream is on sku. Map it in an adapter for this "
                "export before trusting any number in the run."
            )
            result.notes.append("\n".join(lines))
        return suspect

    @staticmethod
    def _key_column(doc: LoadedDocument):
        """Profile of the source column behind `sku`, or None when it was derived."""
        mapped = doc.route.adapter.column_map.get("sku")
        return doc.route.profile.column(mapped) if mapped else None

    @classmethod
    def _alternative_keys(cls, doc: LoadedDocument, mapped: str, others: set) -> str:
        """
        The columns in `doc` worth trying as the item key, best first.

        Overlap with the keys the other documents already use decides it where there
        are other documents; cardinality only stands in when this is the only file.
        Cardinality alone is a weak ranking and reads as nonsense — it recommended a
        net-value column, which is unique per row and therefore looks like a perfect
        key right up until someone tries to join on it.
        """
        scored = []
        for name, values in doc.key_candidates.items():
            if name == mapped or len(values) < _MIN_KEY_DISTINCT:
                continue
            col = doc.route.profile.column(name)
            # Dates and money are never item keys however distinct they are, and a
            # second line-number column is the mistake being diagnosed, not the fix.
            if col is not None and (col.inferred_type not in ("string", "integer")
                                    or col.is_small_ordinal):
                continue
            if not others:
                scored.append((len(values), f"{name!r} ({len(values):,} values)"))
            elif values & others:
                share = len(values & others) / len(values)
                scored.append((share, f"{name!r} (matches {share:.0%})"))
        return ", ".join(text for _, text in sorted(scored, reverse=True)[:3])

    # Below this share of its SKUs meeting any other document, a file is not joining.
    _SKU_AGREEMENT_FLOOR = 0.20
    # Held in step with `sku_superset_coverage` in quality/gates.py, which is the
    # blocking half of the same judgement. The two disagreeing would print a warning
    # the gate then contradicts.
    _SKU_SUPERSET_COVERAGE = 0.50

    @classmethod
    def _check_sku_agreement(cls, result: IntakeResult, suspect: set = frozenset()) -> None:
        """
        Warn when a document's SKUs meet nothing else, and name the column that would.

        Every join in the pipeline is on `sku`, and a key that matches nothing fails
        in total silence: the merges return empty, safety stock falls to zero, annual
        value is zero so every item lands in one ABC class, and the report is full of
        confident zeroes with no error anywhere. Three separate causes produced exactly
        that on one real dataset — a padded material number, a line number mapped as
        the item, and a master keyed on a local code — and none of them was visible
        until someone read the output and disbelieved it.

        The suggestion is the useful half. Knowing that `sku` matches nothing is a
        puzzle; knowing that `Alternate material` would have matched 82% is an answer.

        Documents `_check_key_shape` has already condemned are dropped rather than
        reported twice — and, more importantly, dropped from the evidence. A corrupt
        key agrees with nothing by construction, so leaving it in the comparison set
        makes every correctly-mapped file look like the one that disagrees. That is
        not hypothetical: on a two-file run the only warning printed named the file
        whose mapping was right.
        """
        keyed = {dt: doc for dt, doc in result.documents.items()
                 if dt not in suspect and "sku" in doc.frame.columns and len(doc.frame)}
        if len(keyed) < 2:
            return

        skus = {dt: set(doc.frame["sku"].dropna().astype(str)) for dt, doc in keyed.items()}
        for doc_type, own in skus.items():
            if len(own) < _MIN_KEY_DISTINCT:
                continue
            others = set().union(*(s for dt, s in skus.items() if dt != doc_type))
            if not others:
                continue
            shared = own & others
            share = len(shared) / len(own)
            if share >= cls._SKU_AGREEMENT_FLOOR:
                continue
            # A document whose grain is wider than the rest — a whole-warehouse stock
            # snapshot against the items that actually sell — has a low outward share
            # by construction. Saying it "keys on something the other documents do not
            # use" is then wrong, and it is the wording that sent a planner to
            # `allow_degraded=True`. The gate makes the same distinction on the same
            # two directions; see `sku_superset_coverage`.
            superset = len(shared) / len(others) >= cls._SKU_SUPERSET_COVERAGE

            doc = keyed[doc_type]
            mapped = doc.route.adapter.column_map.get("sku", "?")
            better = [
                (len(values & others) / max(len(values), 1), name)
                for name, values in doc.key_candidates.items()
                if len(values) >= _MIN_KEY_DISTINCT
            ]
            best = max(better, default=(0.0, None))
            if superset:
                result.notes.append("\n".join([
                    f"  ⓘ {doc_type} covers more items than the rest of the run.",
                    f"      {doc.source_name}: sku <- {mapped!r}. Only "
                    f"{share:.0%} of its {len(own):,} SKUs appear elsewhere, but it "
                    f"carries {len(shared) / len(others):.0%} of the SKUs the other "
                    f"documents use — a wider grain, not a different numbering system.",
                    "      The joins that matter still land. A rollup over its own row "
                    "count covers a different population from one over the planned "
                    "items.",
                ]))
                continue

            lines = [
                f"  ⚠ {doc_type} keys on something the other documents do not use.",
                f"      {doc.source_name}: sku <- {mapped!r}, and only "
                f"{share:.0%} of its {len(own):,} SKUs appear in any other file.",
            ]
            if best[1] and best[0] > max(share + 0.2, cls._SKU_AGREEMENT_FLOOR):
                lines.append(
                    f"      Column {best[1]!r} would match {best[0]:.0%}. If that is the "
                    f"identifier the rest of the business uses, map sku to it in an "
                    f"adapter for this export."
                )
            lines.append(
                "      Every join downstream is on sku, so until this agrees the "
                "numbers derived from it will be empty rather than wrong."
            )
            result.notes.append("\n".join(lines))

    @staticmethod
    def _note_mixed_formats(result: IntakeResult) -> None:
        """
        Name every source column that holds two conventions at once, for a human to look at.

        Raised even where the parser repaired it. The repair is a rescue of this run's
        numbers, not a fix — the file on the share drive is still broken, the next
        export will be broken the same way, and whoever produces it does not yet know.
        A note that disappears the moment the symptom is handled is how a source defect
        becomes permanent.

        It is also raised where the parser declined to repair, which is the case that
        actually needs the human: the column is split, the evidence did not support a
        correction, and some of it is being read one way and the rest another.
        """
        flagged: List[str] = []
        for doc in sorted(result.documents.values(), key=lambda d: d.doc_type):
            profile = doc.profile
            if profile is None:
                continue
            for col in profile.columns:
                if not col.representation_mix:
                    continue
                detail = ", ".join(f"{k.replace('_', ' ')} x{v:,}"
                                   for k, v in col.representation_mix.items())
                flagged.append(f"      {doc.source_name} · {col.name!r}: {detail}")
        if not flagged:
            return

        result.notes.append(
            "  ⚠ MIXED FORMATS — one column, two conventions. Check these by hand:\n"
            + "\n".join(flagged)
            + "\n      Usually a spreadsheet opened under a locale that disagreed with the "
              "file: it converts\n      the values it can read and silently swaps month and "
              "day, leaving the rest as text. Where\n      the evidence was conclusive the "
              "parser restored them — see the transform log — but the\n      source file is "
              "still wrong and the next export will be too. Fix it upstream: export dates\n"
              "      in ISO (YYYY-MM-DD), or as a real date column rather than text."
        )

    @staticmethod
    def _note_po_overlap(result: IntakeResult) -> None:
        """
        Report PO numbers appearing in both the open-PO and purchase-history extracts.

        This overlap is expected and correct, not a fault to repair. A PO that is not
        fully closed is still open supply *and* an order that was placed, so pulling
        both reports from the ERP returns it twice — once in each, by design.

        Nothing is deduplicated, because the two feed disjoint calculations: the open
        extract contributes an outstanding balance to the inventory position, the
        history contributes an order event and a receipt to lead time and ordering
        behaviour. The same line counted once in each is not the same quantity counted
        twice.

        It is stated anyway because the alternative — a planner noticing the overlap
        and wondering whether the run double-counted — costs more than one line of
        output. Which is also why it is a note and not a warning.
        """
        open_po, history = result.documents.get("open_po"), result.documents.get("po_history")
        if not open_po or not history:
            return
        if "po_number" not in open_po.frame.columns or "po_number" not in history.frame.columns:
            return

        open_pos = set(open_po.frame["po_number"].dropna().astype(str))
        hist_pos = set(history.frame["po_number"].dropna().astype(str))
        shared = open_pos & hist_pos
        if not shared:
            return

        result.notes.append(
            f"  ⓘ {len(shared):,} PO number(s) appear in both open_po and po_history "
            f"({len(shared) / max(len(open_pos), 1):.0%} of the open extract).\n"
            f"    Expected for partially-received orders, and left as-is: the open "
            f"extract supplies the outstanding balance, the history supplies the order\n"
            f"    and receipt behind lead time and ordering behaviour. Different "
            f"measures, so nothing is double-counted."
        )

    def _unsupplied_capabilities(self, doc: LoadedDocument, doc_type: str) -> set:
        """
        Capabilities the contract declares but this particular extract cannot back.

        The contract says what a document type *can* provide; only the frame says what
        arrived. An SAP purchase-record export with no goods-receipt date declares
        `lead_time_signal` by virtue of being a po_history, and has not one lead time
        in it. Left unchecked the plan reports the capability satisfied, the analytics
        find nothing, and the item-master fallback that should have covered for it is
        never consulted.
        """
        contract = self.contracts.get(doc_type)
        if not contract.capability_requires:
            return set()

        missing = set()
        for capability, needed in contract.capability_requires.items():
            if capability not in contract.capabilities:
                continue
            has_data = any(
                field in doc.frame.columns and doc.frame[field].notna().any()
                for field in needed
            )
            if not has_data:
                missing.add(capability)
        return missing

    # ── Reconciling several claims on one document type ──────────────────────

    def _resolve_claims(
        self, doc_type: str, docs: List[LoadedDocument], result: IntakeResult
    ) -> LoadedDocument:
        """
        Decide whether several frames claiming one document type are *partitions* of it
        or *duplicates* of it.

        This is the difference between an AUS tab plus an NZ tab — which must be
        concatenated, or half the business disappears — and a 200-row sample beside the
        full export, which must not be, or every quantity is counted twice. Row count
        cannot tell them apart; key overlap can. Partitions describe different things
        and their natural keys barely intersect; duplicates describe the same things
        and theirs coincide.
        """
        if len(docs) == 1:
            return docs[0]

        contract = self.contracts.get(doc_type)
        groups = self._group_partitions(docs, contract)

        if len(groups) == 1 and len(groups[0]) > 1:
            return self._concatenate(groups[0], contract, result)

        # Several groups: only one can be this document. Rank on how well each was
        # identified before how big it is — a file routed here at 60% losing to one
        # routed at 88% is the right outcome even when the weaker one has more rows,
        # and row count alone would let a mis-identified extract displace the real
        # thing purely by being longer.
        def rank(group: List[LoadedDocument]):
            return (max(d.route.confidence for d in group),
                    sum(d.row_count for d in group))

        groups.sort(key=rank, reverse=True)
        winner = self._concatenate(groups[0], contract, result) if len(groups[0]) > 1 else groups[0][0]

        for group in groups[1:]:
            for doc in group:
                # Two files claiming one document type are not necessarily the same
                # document twice. Where the layouts differ they are two different
                # reports, at least one of which is routed wrongly — a materially
                # different problem from a sample sitting beside a full export, and one
                # the planner has to resolve rather than merely be told about.
                same_shape = _same_layout(doc, winner)
                if same_shape:
                    detail = (f"duplicate {doc_type} ({doc.row_count:,} rows); kept "
                              f"{winner.source_name} ({winner.row_count:,} rows)")
                else:
                    detail = (
                        f"DROPPED — a different report also claimed {doc_type} and was "
                        f"identified more confidently ({winner.source_name}, "
                        f"{winner.route.confidence:.0%} vs {doc.route.confidence:.0%}). "
                        f"Their columns differ, so they are two documents, not two parts "
                        f"of one. Run `python -m inventory_planning.explain` on this file, "
                        f"or force it with load_all(..., hints={{'{doc.source_name}': "
                        f"'<doc_type>'}})"
                    )
                result.failures.append((doc.source_name, detail))
        return winner

    def _group_partitions(
        self, docs: List[LoadedDocument], contract: DocContract
    ) -> List[List[LoadedDocument]]:
        """Cluster documents so that mutually-disjoint ones share a group."""
        key = [k for k in contract.natural_key if all(k in d.frame.columns for d in docs)]
        groups: List[List[LoadedDocument]] = []

        for doc in docs:
            placed = False
            for group in groups:
                if all(self._is_partition_of(doc, other, key) for other in group):
                    group.append(doc)
                    placed = True
                    break
            if not placed:
                groups.append([doc])
        return groups

    @staticmethod
    def _is_partition_of(a: LoadedDocument, b: LoadedDocument, key: List[str]) -> bool:
        """
        True when two frames cover different populations rather than the same one.

        Provenance decides first: two different sheets of one workbook are essentially
        never copies of each other, and that is exactly how a country split gets
        published. Key overlap cannot substitute for this, because the same SKU
        legitimately appears in both an AU and an NZ tab — it is the same product sold
        in two places, not the same row twice — so overlap would call a genuine
        partition a duplicate and silently drop a country.

        Across separate files two things must hold. The layouts must match, and only
        then does key overlap decide: a sample and a full export share their keys,
        partitions do not.

        The layout test is what stops two *different documents* being welded together.
        A purchase-order history and an open-PO extract that both land on `open_po`
        have barely-overlapping PO numbers, so key overlap alone reads them as a clean
        partition and concatenates them — silently doubling the inbound quantity and
        flattering the stock position into never needing to buy. Genuine partitions of
        one export are produced by one report and carry the same columns; two different
        reports do not.
        """
        if a.source_path == b.source_path and a.sheet_name != b.sheet_name:
            return True
        if not _same_layout(a, b):
            return False
        if not key or not len(a.frame) or not len(b.frame):
            return False

        keys_a = set(map(tuple, a.frame[key].astype(str).drop_duplicates().values))
        keys_b = set(map(tuple, b.frame[key].astype(str).drop_duplicates().values))
        if not keys_a or not keys_b:
            return False
        overlap = len(keys_a & keys_b) / min(len(keys_a), len(keys_b))
        return overlap < 0.20

    def _concatenate(
        self, docs: List[LoadedDocument], contract: DocContract, result: IntakeResult
    ) -> LoadedDocument:
        """Stack partitions into one document, keeping where each row came from."""
        frames = []
        for doc in docs:
            frame = doc.frame.copy()
            frame["source_sheet"] = doc.sheet_name or doc.source_name
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        primary = max(docs, key=lambda d: d.row_count)

        parts = ", ".join(f"{d.sheet_name or d.source_name} ({d.row_count:,})" for d in docs)
        log = list(primary.transform_log)
        log.append(TransformStep(
            "<combine>", "rolled_up",
            f"merged {len(docs)} partitions into one {contract.doc_type}: {parts}",
            rows_affected=len(combined),
        ))

        report = self.tester.run(combined, contract, primary.profile,
                                 adapter_name=primary.adapter.name)
        return LoadedDocument(
            source_path=primary.source_path,
            source_name=" + ".join(d.source_name for d in docs),
            doc_type=contract.doc_type,
            frame=combined,
            route=primary.route,
            transform_log=log,
            test_report=report,
            sheet_name=primary.sheet_name,
        )

    def load_frame(
        self,
        raw: pd.DataFrame,
        source_name: str = "<frame>",
        doc_type_hint: str = None,
        source_path: Path = None,
        sheet_name: str = "",
    ) -> LoadedDocument:
        """Route, transform and verify a single already-read frame."""
        route = self.adapters.route(
            raw, source_name=source_name, doc_type_hint=doc_type_hint, tenant=self.tenant
        )
        route = self._apply_declared_mapping(route, raw, source_name)
        route = self._apply_declared_values(route, source_name)

        if route.contract.doc_type == "demand_timeseries" and route.profile.shape == "wide_periods":
            frame, log = self._melt_wide(raw, route)
        else:
            frame, log = route.adapter.apply(raw, route.contract, route.profile)

        # Baselines are per *source*, not per document type. Keying on doc_type alone
        # makes a different ERP's previous load the comparison point, so every first
        # run against a new system reports drift on every measure — noise that trains
        # the reader to ignore the one signal that catches a genuine field redefinition.
        baseline_key = f"{route.contract.doc_type}::{route.adapter.fingerprint.header_hash}"
        baseline = (
            ContractTester.load_baseline(self.baseline_path, baseline_key)
            if self.baseline_path else None
        )
        report = self.tester.run(
            frame, route.contract, route.profile,
            adapter_name=route.adapter.name, baseline=baseline,
        )
        if self.baseline_path and report.passed:
            ContractTester.save_baseline(
                self.baseline_path, baseline_key,
                ContractTester.build_baseline(frame, route.contract),
            )

        return LoadedDocument(
            source_path=source_path,
            source_name=source_name,
            doc_type=route.contract.doc_type,
            frame=frame,
            route=route,
            transform_log=log,
            test_report=report,
            sheet_name=sheet_name,
            key_candidates=_key_candidates(raw, route.profile),
        )

    def _apply_declared_mapping(self, route, raw: pd.DataFrame, source_name: str):
        """
        Merge any declared column mapping over the one routing produced.

        A copy, never a mutation: adapters are cached by the registry and shared across
        every document of their type, so editing one in place would carry a
        declaration written for the purchase history into the open PO report — a
        correction becoming a corruption, in the one place nobody would look.

        A declaration naming a column that is not in the file is reported rather than
        applied. Applying it would blank the field, which reads downstream as an
        absent measure rather than as a mistake, and the person who wrote it would
        have no way to learn it never matched.
        """
        if self.declarations is None:
            return route
        from dataclasses import replace as dc_replace

        doc_type = route.contract.doc_type
        headers = [str(c) for c in raw.columns]
        declared = self.declarations.column_map_for(doc_type, headers, source_name)
        if not declared:
            return route

        present = {c: col for c, col in declared.items() if col in raw.columns}
        for field in sorted(set(declared) - set(present)):
            column = declared[field]
            self._declaration_notes.append(
                f"  ⚠ The declared mapping {field} <- {column!r} names a column "
                f"{source_name} does not have. Left unapplied — the routed mapping "
                f"stands, and nothing was corrected.")
        if not present:
            return route

        for field, column in sorted(present.items()):
            was = route.adapter.column_map.get(field)
            self._declaration_notes.append(
                f"  ⓘ {doc_type}: {field} <- {column!r} by declaration"
                + (f" (routing had chosen {was!r})" if was and was != column else ""))
        adapter = dc_replace(
            route.adapter, column_map={**route.adapter.column_map, **present})
        return dc_replace(route, adapter=adapter)

    def _apply_declared_values(self, route, source_name: str):
        """
        Supply a field the export never carried, from what a person has declared.

        The case this exists for is currency. A document with no currency column is
        taken to be in the reporting currency already — the ordinary single-entity
        export, and a silent 7x error when it is not. The warning names the remedy as
        `defaults: {currency: XXX}` in the adapter, which on these extracts means
        hand-authoring a draft adapter that the next run regenerates: the exact
        practice the declaration layer was built to replace, still being prescribed
        because nothing was reading a `value` declaration.

        Applied as an adapter default, so it fills only where the field is genuinely
        absent or empty and never overwrites a value the export did carry. A document
        that has a currency column keeps it, one line at a time.
        """
        if self.declarations is None:
            return route
        from dataclasses import replace as dc_replace

        doc_type = route.contract.doc_type
        declared = self.declarations.values_for("value", doc_type=doc_type)
        if not declared:
            return route

        known = set(route.contract.fields)
        applied = {f: v for f, v in declared.items() if f in known}
        for field in sorted(set(declared) - set(applied)):
            self._declaration_notes.append(
                f"  ⚠ The declared value {field}={declared[field]!r} names a field the "
                f"{doc_type} contract does not have. Left unapplied.")
        if not applied:
            return route

        for field, value in sorted(applied.items()):
            self._declaration_notes.append(
                f"  ⓘ {doc_type}: {field} = {value!r} by declaration "
                f"(supplied for {source_name}, which does not carry it)")
        adapter = dc_replace(
            route.adapter, defaults={**route.adapter.defaults, **applied})
        return dc_replace(route, adapter=adapter)

    # ── Wide-format handling ─────────────────────────────────────────────────

    def _melt_wide(
        self, raw: pd.DataFrame, route: RouteResult
    ) -> Tuple[pd.DataFrame, List[TransformStep]]:
        """
        Reshape a SKU x period matrix into the long sku/period/qty grain.

        Reshaping happens before the adapter runs, so the adapter's mapping, parsing
        and assertions operate on the same canonical long shape as every other
        document. That is what keeps the wide case from being a parallel code path
        with its own bugs.
        """
        profile = route.profile
        period_cols = profile.period_columns
        id_cols = [c for c in profile.id_columns if c in raw.columns]

        log = [TransformStep(
            "<reshape>", "rolled_up",
            f"melted {len(period_cols)} period columns "
            f"({profile.period_range[0]} -> {profile.period_range[1]}) into sku_period grain"
            if profile.period_range else f"melted {len(period_cols)} period columns",
            rows_affected=len(raw) * len(period_cols),
        )]

        long = raw.melt(
            id_vars=id_cols,
            value_vars=period_cols,
            var_name="__period_header",
            value_name="__qty",
        )

        from .profiler import parse_period_header

        header_to_period = {h: parse_period_header(h) for h in period_cols}
        long["period"] = long["__period_header"].map(
            lambda h: str(header_to_period.get(h)) if header_to_period.get(h) else None
        )
        long = long.drop(columns=["__period_header"])
        long = long.rename(columns={"__qty": "qty"})

        # Map the identifier columns through the adapter, then re-attach period/qty.
        adapter = route.adapter
        contract = route.contract
        mapped, apply_log = adapter.apply(long, contract, profile)
        log.extend(apply_log)

        for col in ("period", "qty"):
            if col not in mapped.columns and col in long.columns:
                mapped[col] = long[col].values

        mapped["qty"] = pd.to_numeric(
            mapped["qty"].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
        # A blank cell in a demand matrix means "no demand", not "unknown" — the
        # distinction decides whether a SKU reads as intermittent or as short-history.
        filled = int(mapped["qty"].isna().sum())
        mapped["qty"] = mapped["qty"].fillna(0.0)
        if filled:
            log.append(TransformStep("qty", "defaulted",
                                     "blank periods read as zero demand", rows_affected=filled))

        mapped = mapped.dropna(subset=["sku", "period"])
        return mapped.reset_index(drop=True), log

    # ── Convenience ──────────────────────────────────────────────────────────

    def to_pivot(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Long sku/period/qty back to the SKU x period matrix the forecaster expects."""
        pivot = frame.pivot_table(
            index="period", columns="sku", values="qty", aggfunc="sum", fill_value=0.0
        )
        pivot.index = pd.PeriodIndex(pivot.index, freq="M")
        return pivot.sort_index()
