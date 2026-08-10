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
from .profiler import TableProfile
from .registry import AdapterRegistry, RouteResult


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
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return [("", pd.read_csv(path, encoding=encoding, dtype=str,
                                         keep_default_na=False, na_values=[""]))]
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode {path.name} with any known encoding")

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
        uncertain = [
            doc for doc in self.documents.values()
            if doc.route.confidence < 0.75 or "close call" in (doc.route.reason or "")
        ]
        if uncertain:
            lines.append("")
            lines.append("  ⚠ Routed on a thin margin — confirm before trusting the run:")
            for doc in sorted(uncertain, key=lambda d: d.route.confidence):
                lines.append(f"      {doc.source_name} -> {doc.doc_type} "
                             f"({doc.route.confidence:.0%})")
            lines.append("      python -m inventory_planning.explain <file>   "
                         "shows what each contract scored and why")

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
    ):
        self.contracts = contracts or default_registry()
        self.adapters = adapters or AdapterRegistry(contracts=self.contracts)
        self.tester = ContractTester()
        self.resolver = CapabilityResolver()
        self.tenant = tenant
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.verbose = verbose

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

        result.plan = self.resolver.resolve(loaded, unrecognised, withheld=withheld)

        if self.verbose:
            print(result.summary())
        return result

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
        )

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
