"""
Base reader: file loading, schema detection, column mapping, data quality checks.
All five document readers inherit from this.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..ingest.contract import default_registry
from ..ingest.encoding import sniff_encoding
from ..ingest.profiler import Profiler
from ..node import load_planning_node
from ..schema import ALL_SCHEMAS, REQUIRED_FIELDS


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace/underscores for fuzzy matching."""
    return re.sub(r"[\s_\-]+", " ", str(s).lower().strip())


def _disqualified_item_keys(raw: pd.DataFrame, doc_type: str) -> Dict[str, str]:
    """
    Columns that must not become `sku`, mapped to the reason why.

    The rules and the header lists are the contracts' — imported rather than restated,
    because two copies of "which spellings mean a line number" is how the two paths
    came to disagree in the first place. `ingest/` learned that `Item` is SAP's
    position within a document and never the material; this path had not, and its
    first-match-wins loop had `item` ranked second for `sku`, ahead of `material`.

    Returns an empty mapping when the frame cannot be profiled. A guard that raises on
    an odd file would be worse than the bug it prevents.
    """
    try:
        profile = Profiler().profile(raw, source_name="")
        spec = default_registry().get(doc_type).fields.get("sku")
    except Exception:
        return {}
    if spec is None:
        return {}

    banned, banned_tokens = spec.forbidden_headers(profile.system)
    out: Dict[str, str] = {}
    for col in profile.columns:
        if banned and (col.normalized in banned
                       or frozenset(col.normalized.split()) in banned_tokens):
            out[col.name] = f"{profile.system.upper()} uses it for the document line number"
        elif col.is_small_ordinal:
            out[col.name] = "its values are a series (10, 20, 30 …), not identifiers"
    return out


def _detect_mapping(
    df_cols: List[str],
    schema: Dict[str, List[str]],
    disqualified: Dict[str, str] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Auto-detect column mapping from df columns to canonical field names.
    Returns (mapping: {canonical -> df_col}, unmapped_canonicals).

    First match wins, so alias order is a ranking — see `schema.py`. `disqualified`
    names columns `sku` may not take whatever their name suggests; passing it is what
    stops an SAP `Item` column from keying the document when no better one exists.
    """
    norm_to_original = {_normalize(c): c for c in df_cols}
    disqualified = disqualified or {}
    mapping: Dict[str, str] = {}

    for canonical, aliases in schema.items():
        for alias in aliases:
            col = norm_to_original.get(_normalize(alias))
            if col is None:
                continue
            if canonical == "sku" and col in disqualified:
                continue
            mapping[canonical] = col
            break

    unmapped = [f for f in schema if f not in mapping]
    return mapping, unmapped


def load_file(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Not a try-until-it-works ladder: latin-1 decodes every possible byte, so the
        # loop this replaces could never reach a CJK codec and a GBK export became
        # mojibake without raising anything. See ingest/encoding.py.
        return pd.read_csv(path, encoding=sniff_encoding(path), dtype=str)
    elif suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def print_mapping_preview(doc_type: str, mapping: Dict[str, str], unmapped: List[str], df_cols: List[str]) -> None:
    """Print a human-readable mapping preview for user confirmation."""
    print(f"\n{'='*60}")
    print(f"  Column Mapping Preview — {doc_type.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    print(f"{'Canonical Field':<30} {'Detected Column':<30}")
    print(f"{'-'*30} {'-'*30}")
    for canonical, detected in mapping.items():
        print(f"  {canonical:<28} {detected:<30}")
    if unmapped:
        print(f"\n  [OPTIONAL/MISSING]")
        for f in unmapped:
            print(f"  {f:<28} (not found — will be skipped)")
    print(f"{'='*60}\n")


class BaseReader(ABC):
    """Abstract base for all document readers."""

    doc_type: str  # set by subclass

    def __init__(self, config_dir: Union[str, Path] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parents[2] / "config"
        self.schema = ALL_SCHEMAS[self.doc_type]
        self.required = REQUIRED_FIELDS[self.doc_type]
        self.node = load_planning_node(self.config_dir)
        self.location_id = self.node.location_id

    def read(self, path: Union[str, Path], interactive: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Full read pipeline: load → detect → (confirm) → clean → validate.
        Returns (clean_df, quality_report).
        interactive=False skips user confirmation prompts (for tests).
        """
        path = Path(path)
        raw = load_file(path)
        disqualified = _disqualified_item_keys(raw, self.doc_type)
        mapping, unmapped = _detect_mapping(list(raw.columns), self.schema, disqualified)
        refused = {c: why for c, why in disqualified.items() if c in raw.columns}
        if refused and "sku" not in mapping:
            names = ", ".join(f"{c!r} ({why})" for c, why in sorted(refused.items()))
            raise ValueError(
                f"{path.name}: no column can serve as the item key. Refused {names}. "
                f"The material number is in a column this reader does not recognise — "
                f"name it, or load the file through Intake and freeze an adapter."
            )

        if interactive:
            print_mapping_preview(self.doc_type, mapping, unmapped, list(raw.columns))
            mapping = self._confirm_mapping(mapping, unmapped, list(raw.columns))

        df = self._apply_mapping(raw, mapping)
        df = self._clean(df)
        quality = self._quality_report(df, path)
        df = self._post_process(df)
        # Only stamp the configured node where the source did not carry a location of
        # its own. Overwriting a real warehouse code destroys the one column that
        # explains why a SKU appears more than once.
        #
        # Which of the two happened is recorded, because the stamped value is invented
        # and the read one is not. Until every document declared `location_id`, four of
        # the five could only ever be stamped — the plant column was dropped at mapping
        # time and `DC-01` went out on every row regardless of what the export said.
        if "location_id" in df.columns:
            quality["location_source"] = "source"
            df["location_id"] = df["location_id"].fillna(self.location_id)
        else:
            quality["location_source"] = "configured node"
            df["location_id"] = self.location_id
            if self.node.is_placeholder:
                quality.setdefault("issues", []).append(
                    f"location_id set to the placeholder {self.location_id!r} — neither "
                    f"this export nor config/node_config.json names a warehouse"
                )
        return df, quality

    def _confirm_mapping(self, mapping: Dict[str, str], unmapped: List[str], df_cols: List[str]) -> Dict[str, str]:
        """Interactive mapping confirmation. User can override individual fields."""
        answer = input("Accept this mapping? [Y/n/edit]: ").strip().lower()
        if answer in ("", "y", "yes"):
            return mapping
        if answer in ("e", "edit"):
            for canonical in list(self.schema.keys()):
                current = mapping.get(canonical, "(not mapped)")
                user_in = input(f"  {canonical} [{current}] → (enter new col or blank to keep): ").strip()
                if user_in and user_in in df_cols:
                    mapping[canonical] = user_in
                elif user_in:
                    print(f"    Column '{user_in}' not found — keeping '{current}'")
        return mapping

    def _apply_mapping(self, raw: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Rename detected columns to canonical names, drop unmapped originals."""
        rename = {v: k for k, v in mapping.items()}
        df = raw.rename(columns=rename)
        keep = list(mapping.keys())
        existing = [c for c in keep if c in df.columns]
        return df[existing].copy()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common cleaning: strip whitespace, normalize SKU, parse dates/numerics."""
        # Strip text columns. Not select_dtypes(include="object"): pandas 3 stores
        # text as `str`, and pandas 4 drops it from the `object` selector.
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].str.strip()

        # Normalize SKU: uppercase, no leading/trailing spaces
        if "sku" in df.columns:
            df["sku"] = df["sku"].str.upper().str.strip()

        # Parse date columns (any column ending in _date or named date/eta)
        date_cols = [c for c in df.columns if "date" in c or c in ("eta", "snapshot_date")]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

        # Parse numeric columns
        numeric_cols = [c for c in df.columns if any(
            kw in c for kw in ("qty", "amount", "value")
        )]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        # Drop fully empty rows
        df.dropna(how="all", inplace=True)

        return df

    def _quality_report(self, df: pd.DataFrame, path: Path) -> Dict:
        """Generate data quality summary."""
        total_rows = len(df)
        issues = []
        null_summary = {}

        for col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                null_summary[col] = int(n_null)
                if col in self.required:
                    issues.append(f"REQUIRED field '{col}' has {n_null} nulls ({n_null/total_rows:.0%})")

        # Duplicate SKU check for inventory
        if self.doc_type == "inventory" and "sku" in df.columns:
            dups = df["sku"].duplicated().sum()
            if dups:
                issues.append(f"{dups} duplicate SKUs in inventory (will be summed)")

        # Negative qty check
        for col in [c for c in df.columns if "qty" in c]:
            neg = (df[col] < 0).sum()
            if neg:
                issues.append(f"{neg} negative values in '{col}'")

        report = {
            "file": str(path.name),
            "doc_type": self.doc_type,
            "rows_loaded": total_rows,
            "null_by_column": null_summary,
            "issues": issues,
            "status": "OK" if not issues else "WARNINGS",
        }
        self._print_quality_report(report)
        return report

    def _print_quality_report(self, report: Dict) -> None:
        print(f"\n  Data Quality — {report['doc_type'].upper()}")
        print(f"  Rows loaded : {report['rows_loaded']}")
        if report["null_by_column"]:
            print(f"  Nulls       : {report['null_by_column']}")
        if report["issues"]:
            print(f"  Issues:")
            for iss in report["issues"]:
                print(f"    ⚠  {iss}")
        else:
            print(f"  Status      : OK — no issues")

    @abstractmethod
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Doc-specific transformations after common cleaning."""
        ...
