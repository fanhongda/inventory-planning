"""
Adapters: how one specific (tenant, system, report) satisfies a contract.

An adapter is a data artefact, not code. That is the whole point — it can be
generated, diffed, reviewed, version-pinned and rolled back, and adding a new ERP
means adding a YAML file rather than editing a reader. It carries the four things
an alias dict cannot express:

  column_map    which source column supplies which canonical field
  derivations   how to synthesise fields the source does not carry
  value_maps    how this source's status/incoterm codes map to canonical values
  grain/rollup  what one row means, and how to get to the grain the contract wants

Applying an adapter is deterministic and produces a transform log, so every canonical
column can be traced back to the rule that produced it. That log is what makes an
LLM-drafted adapter reviewable instead of merely plausible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .contract import DocContract, ValueDomain
from .expressions import Expression, ExpressionError, evaluate_first_satisfiable
from .profiler import TableProfile, normalize_header

ADAPTERS_DIR = Path(__file__).parent / "adapters"

# Aggregation used when rolling several source rows up to the contract grain.
# Measures sum; everything else takes the first non-null.
_MEASURE_AGG = "sum"


@dataclass
class TransformStep:
    """One traceable operation performed while applying an adapter."""

    field: str
    action: str          # mapped | derived | value_mapped | defaulted | rolled_up | filtered | parsed
    detail: str
    rows_affected: Optional[int] = None

    def __str__(self) -> str:
        suffix = f" [{self.rows_affected:,} rows]" if self.rows_affected is not None else ""
        return f"{self.field:<22} {self.action:<13} {self.detail}{suffix}"


@dataclass
class Fingerprint:
    """
    How a file is recognised as belonging to this adapter.

    `header_hash` is exact and cheap. `header_contains` is the resilient fallback:
    a source that adds one column changes its hash but still contains its signature
    columns, so the adapter keeps matching instead of silently falling back to
    generic detection.
    """

    header_hash: Optional[str] = None
    header_contains: List[str] = dc_field(default_factory=list)
    header_absent: List[str] = dc_field(default_factory=list)
    filename_pattern: Optional[str] = None
    min_confidence: float = 0.6

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> "Fingerprint":
        raw = raw or {}
        return cls(
            header_hash=raw.get("header_hash"),
            header_contains=list(raw.get("header_contains", []) or []),
            header_absent=list(raw.get("header_absent", []) or []),
            filename_pattern=raw.get("filename_pattern"),
            min_confidence=float(raw.get("min_confidence", 0.6)),
        )

    def match(self, profile: TableProfile) -> Tuple[bool, float, str]:
        """Return (matched, confidence, reason)."""
        if self.header_hash and profile.header_hash == self.header_hash:
            return True, 1.0, "exact header hash"

        tokens = set(profile.header_tokens)

        for absent in self.header_absent:
            if normalize_header(absent) in tokens:
                return False, 0.0, f"disqualified: header contains {absent!r}"

        if not self.header_contains:
            return False, 0.0, "no signature columns declared"

        wanted = [normalize_header(h) for h in self.header_contains]
        hits = sum(1 for w in wanted if w in tokens)
        confidence = hits / len(wanted)

        if confidence >= self.min_confidence:
            missing = [h for h, w in zip(self.header_contains, wanted) if w not in tokens]
            reason = f"{hits}/{len(wanted)} signature columns"
            if missing:
                reason += f" (missing: {', '.join(missing[:3])})"
            return True, confidence, reason
        return False, confidence, f"only {hits}/{len(wanted)} signature columns"


@dataclass
class ParsingRules:
    """Source-level parsing quirks — the things that corrupt data silently."""

    date_format: Optional[str] = None
    dayfirst: bool = False
    decimal_sep: str = "."
    thousands_sep: str = ","
    qty_sign: Optional[str] = None      # negate | negate_if_credit
    strip_chars: Optional[str] = None
    na_values: List[str] = dc_field(default_factory=lambda: ["", "-", "N/A", "n/a", "NULL", "#N/A"])

    @classmethod
    def parse(cls, raw: Dict[str, Any]) -> "ParsingRules":
        raw = raw or {}
        return cls(
            date_format=raw.get("date_format"),
            dayfirst=bool(raw.get("dayfirst", False)),
            decimal_sep=raw.get("decimal_sep", "."),
            thousands_sep=raw.get("thousands_sep", ","),
            qty_sign=raw.get("qty_sign"),
            strip_chars=raw.get("strip_chars"),
            na_values=list(raw.get("na_values", []) or
                           ["", "-", "N/A", "n/a", "NULL", "#N/A"]),
        )


@dataclass
class Adapter:
    """A frozen, versioned mapping from one source format to one contract."""

    name: str
    doc_type: str
    version: int = 1
    tenant: str = "default"
    system: str = "unknown"
    description: str = ""
    fingerprint: Fingerprint = dc_field(default_factory=Fingerprint)
    grain: Optional[str] = None
    rollup_to: Optional[str] = None
    column_map: Dict[str, str] = dc_field(default_factory=dict)
    derivations: Dict[str, str] = dc_field(default_factory=dict)
    value_maps: Dict[str, Dict[str, str]] = dc_field(default_factory=dict)
    defaults: Dict[str, Any] = dc_field(default_factory=dict)
    filters: List[str] = dc_field(default_factory=list)
    parsing: ParsingRules = dc_field(default_factory=ParsingRules)
    drop_rows_matching: Dict[str, str] = dc_field(default_factory=dict)
    status: str = "draft"                 # draft | verified | frozen
    source_path: Optional[Path] = None
    notes: List[str] = dc_field(default_factory=list)

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "Adapter":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.parse(raw, source_path=path)

    @classmethod
    def parse(cls, raw: Dict[str, Any], source_path: Path = None) -> "Adapter":
        derivations = raw.get("derivations", {}) or {}
        for target, expr_src in derivations.items():
            try:
                Expression(expr_src)   # validate now, not mid-run
            except ExpressionError as exc:
                raise ExpressionError(
                    f"Adapter {raw.get('name')!r} derivation for {target!r}: {exc}"
                ) from exc
        for filt in raw.get("filters", []) or []:
            Expression(filt)

        return cls(
            name=raw["name"],
            doc_type=raw["doc_type"],
            version=int(raw.get("version", 1)),
            tenant=raw.get("tenant", "default"),
            system=raw.get("system", "unknown"),
            description=raw.get("description", ""),
            fingerprint=Fingerprint.parse(raw.get("fingerprint", {})),
            grain=raw.get("grain"),
            rollup_to=raw.get("rollup_to"),
            column_map=dict(raw.get("column_map", {}) or {}),
            derivations=dict(derivations),
            value_maps={k: dict(v) for k, v in (raw.get("value_maps", {}) or {}).items()},
            defaults=dict(raw.get("defaults", {}) or {}),
            filters=list(raw.get("filters", []) or []),
            parsing=ParsingRules.parse(raw.get("parsing", {})),
            drop_rows_matching=dict(raw.get("drop_rows_matching", {}) or {}),
            status=raw.get("status", "draft"),
            source_path=Path(source_path) if source_path else None,
            notes=list(raw.get("notes", []) or []),
        )

    def to_yaml(self) -> str:
        body: Dict[str, Any] = {
            "name": self.name,
            "doc_type": self.doc_type,
            "version": self.version,
            "tenant": self.tenant,
            "system": self.system,
            "status": self.status,
        }
        if self.description:
            body["description"] = self.description
        fp: Dict[str, Any] = {}
        if self.fingerprint.header_hash:
            fp["header_hash"] = self.fingerprint.header_hash
        if self.fingerprint.header_contains:
            fp["header_contains"] = self.fingerprint.header_contains
        if self.fingerprint.header_absent:
            fp["header_absent"] = self.fingerprint.header_absent
        if fp:
            body["fingerprint"] = fp
        if self.grain:
            body["grain"] = self.grain
        if self.rollup_to:
            body["rollup_to"] = self.rollup_to
        if self.column_map:
            body["column_map"] = self.column_map
        if self.derivations:
            body["derivations"] = self.derivations
        if self.value_maps:
            body["value_maps"] = self.value_maps
        if self.defaults:
            body["defaults"] = self.defaults
        if self.filters:
            body["filters"] = self.filters
        # Only emit parsing rules that deviate from the default, so a reviewer sees
        # the locale decisions and nothing else.
        parsing: Dict[str, Any] = {}
        if self.parsing.date_format:
            parsing["date_format"] = self.parsing.date_format
        if self.parsing.dayfirst:
            parsing["dayfirst"] = True
        if self.parsing.decimal_sep != ".":
            parsing["decimal_sep"] = self.parsing.decimal_sep
        if self.parsing.thousands_sep != ",":
            parsing["thousands_sep"] = self.parsing.thousands_sep
        if self.parsing.qty_sign:
            parsing["qty_sign"] = self.parsing.qty_sign
        if parsing:
            body["parsing"] = parsing
        if self.notes:
            body["notes"] = self.notes
        return yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)

    @property
    def slug(self) -> str:
        return f"{self.tenant}__{self.system}/{self.doc_type}.v{self.version}"

    # ── Application ──────────────────────────────────────────────────────────

    def apply(
        self, raw: pd.DataFrame, contract: DocContract, profile: TableProfile = None
    ) -> Tuple[pd.DataFrame, List[TransformStep]]:
        """
        Transform a raw frame into the contract's canonical shape.

        Order matters and is fixed: map -> parse -> value-map -> derive -> default ->
        filter -> roll up. Derivations run after parsing so arithmetic sees numbers,
        and after value mapping so a filter can test a canonical status.
        """
        log: List[TransformStep] = []
        df = pd.DataFrame(index=raw.index)

        # 1. Column mapping ---------------------------------------------------
        available = {normalize_header(c): c for c in raw.columns}
        for canonical, source_col in self.column_map.items():
            actual = available.get(normalize_header(source_col))
            if actual is None:
                log.append(TransformStep(canonical, "mapped",
                                         f"SKIPPED — source column {source_col!r} absent"))
                continue
            df[canonical] = raw[actual]
            log.append(TransformStep(canonical, "mapped", f"<- {actual!r}"))

        # 2. Parsing / typing --------------------------------------------------
        df = self._apply_parsing(df, contract, log)

        # 3. Value domain translation -----------------------------------------
        df = self._apply_value_maps(df, contract, log)

        # 4. Derivations -------------------------------------------------------
        df = self._apply_derivations(df, contract, log)

        # 5. Defaults ----------------------------------------------------------
        for field_name, value in self.defaults.items():
            if field_name not in df.columns or df[field_name].isna().all():
                df[field_name] = value
                log.append(TransformStep(field_name, "defaulted", f"= {value!r}"))

        # 6. Row filters -------------------------------------------------------
        df = self._apply_filters(df, log, contract)

        # 7. Grain rollup ------------------------------------------------------
        df = self._apply_rollup(df, contract, log)

        return df, log

    # ── Stages ───────────────────────────────────────────────────────────────

    def _apply_parsing(
        self, df: pd.DataFrame, contract: DocContract, log: List[TransformStep]
    ) -> pd.DataFrame:
        p = self.parsing
        na_set = {str(v).strip().lower() for v in p.na_values}

        for col in df.columns:
            spec = contract.field(col)
            if spec is None:
                continue

            series = df[col]
            if series.dtype == object:
                series = series.astype(str).str.strip()
                series = series.mask(series.str.lower().isin(na_set), np.nan)
                if p.strip_chars:
                    series = series.str.strip(p.strip_chars)

            if spec.is_numeric:
                text = series.astype(str)
                if p.thousands_sep:
                    text = text.str.replace(p.thousands_sep, "", regex=False)
                if p.decimal_sep != ".":
                    text = text.str.replace(p.decimal_sep, ".", regex=False)
                # Accounting-style negatives: (1,234) means -1234
                paren = text.str.match(r"^\(.*\)$", na=False)
                text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
                parsed = pd.to_numeric(text, errors="coerce")
                parsed = parsed.where(~paren, -parsed)
                if p.qty_sign == "negate" and "qty" in col:
                    parsed = -parsed
                df[col] = parsed
                log.append(TransformStep(col, "parsed", f"numeric ({spec.type})"))

            elif spec.is_temporal:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    if p.date_format:
                        df[col] = pd.to_datetime(series, format=p.date_format, errors="coerce")
                    else:
                        df[col] = pd.to_datetime(series, errors="coerce", dayfirst=p.dayfirst)
                fmt = p.date_format or ("dayfirst" if p.dayfirst else "inferred")
                log.append(TransformStep(col, "parsed", f"date ({fmt})"))

            else:
                if spec.normalize == "upper_strip":
                    series = series.str.upper().str.strip() if series.dtype == object else series
                elif spec.normalize == "lower_strip":
                    series = series.str.lower().str.strip() if series.dtype == object else series
                elif spec.normalize == "strip":
                    series = series.str.strip() if series.dtype == object else series
                df[col] = series

        return df

    def _apply_value_maps(
        self, df: pd.DataFrame, contract: DocContract, log: List[TransformStep]
    ) -> pd.DataFrame:
        """
        Translate raw codes to canonical values. The adapter's own map wins over the
        contract's shared domain, so a source with a bizarre local code can override
        without polluting the shared vocabulary.
        """
        for field_name, spec in contract.fields.items():
            if not spec.value_domain or field_name not in df.columns:
                continue

            domain = contract.value_domains.get(spec.value_domain)
            override = self.value_maps.get(field_name) or self.value_maps.get(spec.value_domain)

            if override:
                merged = dict(domain.map) if domain else {}
                merged.update({str(k).strip().lower(): v for k, v in override.items()})
                domain = ValueDomain(
                    name=spec.value_domain,
                    canonical=domain.canonical if domain else sorted(set(override.values())),
                    map=merged,
                    default=domain.default if domain else None,
                )
            if domain is None:
                continue

            unknown = domain.unknown_codes(df[field_name])
            target = field_name[:-4] if field_name.endswith("_raw") else f"{field_name}_canonical"
            df[target] = df[field_name].map(domain.translate)

            detail = f"{spec.value_domain} -> {target!r}"
            if unknown:
                # Loud on purpose: an unmapped status code is how closed POs get
                # counted as open supply.
                detail += f"  ⚠ UNMAPPED CODES: {unknown[:6]}"
            log.append(TransformStep(field_name, "value_mapped", detail))

        return df

    def _apply_derivations(
        self, df: pd.DataFrame, contract: DocContract, log: List[TransformStep]
    ) -> pd.DataFrame:
        """
        Adapter-declared derivations first (source-specific and explicit), then the
        contract's `derivable_from` fallbacks for anything still missing. The second
        pass is what lets a brand-new export work with no adapter at all when its
        columns happen to be conventionally named.
        """
        for target, expr_src in self.derivations.items():
            expr = Expression(expr_src)
            if not expr.can_evaluate(df):
                missing = sorted(expr.missing_columns(df))
                log.append(TransformStep(target, "derived",
                                         f"SKIPPED — needs {missing} ({expr_src})"))
                continue
            df[target] = expr.evaluate(df)
            log.append(TransformStep(target, "derived", f"= {expr_src}"))

        for field_name in contract.derivable_fields:
            already_present = field_name in df.columns and df[field_name].notna().any()
            if already_present or field_name in self.derivations:
                continue
            value, used = evaluate_first_satisfiable(
                contract.field(field_name).derivable_from, df
            )
            if used is not None:
                df[field_name] = value
                log.append(TransformStep(field_name, "derived",
                                         f"= {used}  (contract fallback)"))

        return df

    def _apply_filters(
        self, df: pd.DataFrame, log: List[TransformStep], contract: DocContract = None
    ) -> pd.DataFrame:
        # Contract-level filters express what the document *is* — an "open PO" file
        # containing closed lines is not a filtering preference, it is a violation of
        # the document's definition. They run first, and only when their columns
        # resolved, so a source with no status column is unaffected.
        contract_filters = list(contract.default_filters) if contract else []
        for filt in contract_filters + self.filters:
            expr = Expression(filt)
            if not expr.can_evaluate(df):
                log.append(TransformStep("<filter>", "filtered",
                                         f"SKIPPED — needs {sorted(expr.missing_columns(df))}"))
                continue
            mask = expr.evaluate(df)
            if not isinstance(mask, pd.Series):
                continue
            mask = mask.fillna(False).astype(bool)
            removed = int((~mask).sum())
            df = df[mask].copy()
            log.append(TransformStep("<filter>", "filtered", filt, rows_affected=removed))

        for col, pattern in self.drop_rows_matching.items():
            if col not in df.columns:
                continue
            hit = df[col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            removed = int(hit.sum())
            if removed:
                df = df[~hit].copy()
                log.append(TransformStep(col, "filtered",
                                         f"dropped rows matching /{pattern}/",
                                         rows_affected=removed))
        return df

    def _apply_rollup(
        self, df: pd.DataFrame, contract: DocContract, log: List[TransformStep]
    ) -> pd.DataFrame:
        """
        Aggregate from the source grain up to the contract grain.

        Declaring this explicitly is what stops a PO *schedule line* extract from
        being silently double-counted as PO lines — the failure the architecture doc
        flags as the one LLMs get wrong most often.
        """
        if not self.rollup_to or self.rollup_to == self.grain:
            return df
        if len(df) == 0:
            # A frame emptied by the contract filters has nothing to aggregate, and
            # pandas raises rather than returning an empty group. "Zero rows survived"
            # is a legitimate outcome the contract tests already report on.
            return df

        key = [k for k in contract.natural_key if k in df.columns]
        if not key:
            log.append(TransformStep("<rollup>", "rolled_up",
                                     f"SKIPPED — natural key {contract.natural_key} not available"))
            return df

        before = len(df)
        agg: Dict[str, Any] = {}
        for col in df.columns:
            if col in key:
                continue
            spec = contract.field(col)
            if spec is not None and spec.role == "measure" and spec.is_numeric:
                agg[col] = _MEASURE_AGG
            else:
                agg[col] = "first"

        df = df.groupby(key, as_index=False, dropna=False).agg(agg)
        log.append(TransformStep(
            "<rollup>", "rolled_up",
            f"{self.grain} -> {self.rollup_to} on {'+'.join(key)} "
            f"({before:,} -> {len(df):,} rows)",
        ))
        return df

    def __repr__(self) -> str:
        return f"Adapter({self.slug!r}, status={self.status})"
