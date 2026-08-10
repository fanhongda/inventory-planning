"""
Contract tests — the verification half of "program synthesis + verification".

An adapter, whether hand-written or drafted, is a *hypothesis* about what a file
means. These tests are what turn it into a checked hypothesis. Three layers, in
increasing subtlety:

  structural     Are the required fields present? Does the declared grain match an
                 actual unique key? (Catches a PO-header extract passed off as PO
                 lines — the double-counting failure that is invisible downstream.)

  semantic       Do the contract's invariants hold? (open_qty >= 0, lead time within
                 a plausible band, receipt after order.) Catches swapped columns and
                 misparsed dates.

  reconciliation Does the load agree with the file's own control totals, and with the
                 previous load's distribution? Catches the nastiest failure of all —
                 the source quietly changing a column's meaning while keeping its name.

Only a result with no errors may be promoted. Warnings travel with the data into the
report so the planner sees what was uncertain about their own numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .contract import DocContract, SEVERITY_ERROR, SEVERITY_WARN
from .expressions import Expression, ExpressionError
from .profiler import TableProfile

LAYER_STRUCTURAL = "structural"
LAYER_SEMANTIC = "semantic"
LAYER_RECONCILIATION = "reconciliation"


@dataclass
class TestResult:
    """One assertion's outcome."""

    layer: str
    name: str
    passed: bool
    severity: str = SEVERITY_ERROR
    detail: str = ""
    failed_rows: int = 0
    total_rows: int = 0
    examples: List[Dict[str, Any]] = dc_field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return not self.passed and self.severity == SEVERITY_ERROR

    @property
    def failure_rate(self) -> float:
        return self.failed_rows / self.total_rows if self.total_rows else 0.0

    def __str__(self) -> str:
        if self.passed:
            return f"    ✓ [{self.layer}] {self.name}"
        icon = "✗" if self.severity == SEVERITY_ERROR else "⚠"
        rows = f" ({self.failed_rows:,}/{self.total_rows:,} rows)" if self.total_rows else ""
        return f"    {icon} [{self.layer}] {self.name}{rows} — {self.detail}"


@dataclass
class ContractTestReport:
    """All results for one document load."""

    doc_type: str
    source_name: str
    row_count: int
    results: List[TestResult] = dc_field(default_factory=list)
    adapter_name: str = ""
    checked_at: str = dc_field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def errors(self) -> List[TestResult]:
        return [r for r in self.results if r.is_error]

    @property
    def warnings(self) -> List[TestResult]:
        return [r for r in self.results if not r.passed and r.severity == SEVERITY_WARN]

    @property
    def passed(self) -> bool:
        """Promotable to the analytics layer."""
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "FAILED"
        return "WARNINGS" if self.warnings else "OK"

    def summary(self) -> str:
        lines = [
            f"  Contract tests — {self.doc_type} ({self.source_name})",
            f"    status: {self.status}   "
            f"{len(self.results) - len(self.errors) - len(self.warnings)} passed, "
            f"{len(self.warnings)} warned, {len(self.errors)} failed",
        ]
        for r in self.results:
            if not r.passed:
                lines.append(str(r))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "source_name": self.source_name,
            "adapter": self.adapter_name,
            "row_count": self.row_count,
            "status": self.status,
            "checked_at": self.checked_at,
            "failures": [
                {
                    "layer": r.layer,
                    "name": r.name,
                    "severity": r.severity,
                    "detail": r.detail,
                    "failed_rows": r.failed_rows,
                    "failure_rate": round(r.failure_rate, 4),
                    "examples": r.examples,
                }
                for r in self.results if not r.passed
            ],
        }


class ContractTester:
    """Runs the three layers against a canonicalised frame."""

    MAX_EXAMPLES = 3
    # A drifting mean beyond this ratio suggests the column's meaning changed
    DRIFT_MEAN_RATIO = 0.35
    DRIFT_ROWCOUNT_RATIO = 0.50

    def run(
        self,
        df: pd.DataFrame,
        contract: DocContract,
        profile: TableProfile = None,
        adapter_name: str = "",
        baseline: Dict[str, Any] = None,
    ) -> ContractTestReport:
        report = ContractTestReport(
            doc_type=contract.doc_type,
            source_name=profile.source_name if profile else "<frame>",
            row_count=len(df),
            adapter_name=adapter_name,
        )
        report.results.extend(self._structural(df, contract, profile))
        report.results.extend(self._semantic(df, contract))
        report.results.extend(self._reconciliation(df, contract, profile, baseline))
        return report

    # ── Layer 1: structural ──────────────────────────────────────────────────

    def _structural(
        self, df: pd.DataFrame, contract: DocContract, profile: TableProfile
    ) -> List[TestResult]:
        out: List[TestResult] = []
        total = len(df)

        out.append(TestResult(
            layer=LAYER_STRUCTURAL,
            name="non_empty",
            passed=total > 0,
            detail="" if total else "Adapter produced zero rows — check filters and column_map",
            total_rows=total,
        ))
        if total == 0:
            return out

        for field_name in contract.required_fields:
            present = field_name in df.columns
            populated = present and df[field_name].notna().any()
            nulls = int(df[field_name].isna().sum()) if present else total
            out.append(TestResult(
                layer=LAYER_STRUCTURAL,
                name=f"required_field:{field_name}",
                passed=populated,
                detail=(
                    "field absent after mapping and derivation"
                    if not present else
                    "field present but entirely null"
                ) if not populated else "",
                failed_rows=nulls,
                total_rows=total,
            ))

            if populated and nulls:
                rate = nulls / total
                out.append(TestResult(
                    layer=LAYER_STRUCTURAL,
                    name=f"required_field_nulls:{field_name}",
                    passed=rate <= 0.05,
                    severity=SEVERITY_WARN,
                    detail=f"{rate:.1%} null — rows will be dropped or defaulted downstream",
                    failed_rows=nulls,
                    total_rows=total,
                ))

        out.append(self._grain_test(df, contract))
        return out

    def _grain_test(self, df: pd.DataFrame, contract: DocContract) -> TestResult:
        """
        The declared grain must be real: row count should equal distinct natural-key
        count. When it does not, the file is at a finer grain than declared and needs
        a rollup — silently summing it later would double-count supply.
        """
        key = [k for k in contract.natural_key if k in df.columns]
        if not key:
            return TestResult(
                layer=LAYER_STRUCTURAL,
                name=f"grain:{contract.grain}",
                passed=True,
                severity=SEVERITY_WARN,
                detail=f"natural key {contract.natural_key} unavailable — grain unverified",
            )

        distinct = len(df.drop_duplicates(subset=key))
        duplicates = len(df) - distinct
        ratio = duplicates / len(df) if len(df) else 0.0

        return TestResult(
            layer=LAYER_STRUCTURAL,
            name=f"grain:{contract.grain}",
            passed=duplicates == 0,
            severity=SEVERITY_WARN,
            detail=(
                f"{duplicates:,} rows share a natural key ({'+'.join(key)}) — "
                f"actual grain is finer than declared '{contract.grain}'. "
                f"Set rollup_to in the adapter, or measures will be double-counted."
            ) if duplicates else "",
            failed_rows=duplicates,
            total_rows=len(df),
        )

    # ── Layer 2: semantic ────────────────────────────────────────────────────

    def _semantic(self, df: pd.DataFrame, contract: DocContract) -> List[TestResult]:
        out: List[TestResult] = []
        if len(df) == 0:
            return out

        for assertion in contract.all_assertions():
            expr = assertion.expr
            if not expr.can_evaluate(df):
                # Not a failure: the assertion simply does not apply to this source.
                continue
            try:
                mask = expr.evaluate(df)
            except (ExpressionError, TypeError, ValueError) as exc:
                out.append(TestResult(
                    layer=LAYER_SEMANTIC,
                    name=assertion.source,
                    passed=False,
                    severity=SEVERITY_WARN,
                    detail=f"could not evaluate: {exc}",
                ))
                continue

            if not isinstance(mask, pd.Series):
                continue

            # A comparison against NaN yields False in pandas, not NaN — so a row with
            # a missing operand would otherwise be reported as violating the
            # invariant. Rows that cannot be judged are excluded instead; whether a
            # field is allowed to be null is the structural layer's question, and
            # conflating the two turns every unknown into a spurious error.
            judgeable = pd.Series(True, index=df.index)
            for col in expr.columns:
                if col in df.columns:
                    judgeable &= df[col].notna()

            violated = (~mask.fillna(False).astype(bool)) & judgeable
            failed = int(violated.sum())
            if not judgeable.any():
                continue

            out.append(TestResult(
                layer=LAYER_SEMANTIC,
                name=assertion.source,
                passed=failed == 0,
                severity=assertion.severity,
                detail=assertion.description or f"{failed:,} rows violate the invariant",
                failed_rows=failed,
                total_rows=int(judgeable.sum()),
                examples=self._examples(df, violated, expr.columns),
            ))
        return out

    def _examples(self, df: pd.DataFrame, mask: pd.Series, columns: set) -> List[Dict[str, Any]]:
        """A few offending rows, restricted to the columns the assertion referenced."""
        if not mask.any():
            return []
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return []
        sample = df.loc[mask, cols].head(self.MAX_EXAMPLES)
        out = []
        for _, row in sample.iterrows():
            out.append({k: _jsonable(v) for k, v in row.items()})
        return out

    # ── Layer 3: reconciliation ──────────────────────────────────────────────

    def _reconciliation(
        self,
        df: pd.DataFrame,
        contract: DocContract,
        profile: Optional[TableProfile],
        baseline: Optional[Dict[str, Any]],
    ) -> List[TestResult]:
        out: List[TestResult] = []
        if len(df) == 0:
            return out

        if profile is not None and profile.suspected_total_rows:
            out.append(TestResult(
                layer=LAYER_RECONCILIATION,
                name="control_rows_excluded",
                passed=len(df) <= profile.row_count - profile.suspected_total_rows,
                severity=SEVERITY_WARN,
                detail=(
                    f"source had {profile.suspected_total_rows} suspected total/subtotal "
                    f"row(s); {len(df):,} rows survived from {profile.row_count:,} — "
                    f"verify they were excluded, not summed in"
                ),
                total_rows=profile.row_count,
            ))

        if not baseline:
            return out

        prior_rows = baseline.get("row_count")
        if prior_rows:
            change = abs(len(df) - prior_rows) / prior_rows
            out.append(TestResult(
                layer=LAYER_RECONCILIATION,
                name="row_count_drift",
                passed=change <= self.DRIFT_ROWCOUNT_RATIO,
                severity=SEVERITY_WARN,
                detail=(
                    f"row count moved {prior_rows:,} -> {len(df):,} ({change:+.0%}) "
                    f"since the last load — check the extract filter did not change"
                ),
                total_rows=len(df),
            ))

        prior_means: Dict[str, float] = baseline.get("column_means", {}) or {}
        for col, prior_mean in prior_means.items():
            if col not in df.columns or not prior_mean:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            current = float(series.mean())
            shift = abs(current - prior_mean) / abs(prior_mean)
            out.append(TestResult(
                layer=LAYER_RECONCILIATION,
                name=f"distribution_drift:{col}",
                passed=shift <= self.DRIFT_MEAN_RATIO,
                severity=SEVERITY_WARN,
                detail=(
                    f"mean moved {prior_mean:,.1f} -> {current:,.1f} ({shift:+.0%}). "
                    f"A large shift with an unchanged column name is the signature of "
                    f"a redefined field (UoM, currency, or filter change)."
                ),
                total_rows=len(df),
            ))
        return out

    # ── Baselines ────────────────────────────────────────────────────────────

    @staticmethod
    def build_baseline(df: pd.DataFrame, contract: DocContract) -> Dict[str, Any]:
        """Snapshot the measures of this load so the next one can be drift-checked."""
        means: Dict[str, float] = {}
        for name, spec in contract.fields.items():
            if spec.role != "measure" or name not in df.columns:
                continue
            series = pd.to_numeric(df[name], errors="coerce").dropna()
            if len(series):
                means[name] = float(series.mean())
        return {
            "doc_type": contract.doc_type,
            "row_count": len(df),
            "column_means": means,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def load_baseline(path: Path, doc_type: str) -> Optional[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return store.get(doc_type)

    @staticmethod
    def save_baseline(path: Path, doc_type: str, baseline: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        store: Dict[str, Any] = {}
        if path.exists():
            try:
                store = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                store = {}
        store[doc_type] = baseline
        path.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value if isinstance(value, (str, int, float, bool)) else str(value)
