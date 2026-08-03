"""
Canonical data contracts.

This replaces the alias-dict in ``schema.py``. The difference that matters:

  schema.py answered only "what other names might this column have?"
  A contract also answers "how can this field be *reconstructed* when it is absent?",
  "at what grain is it meaningful?", and "what must be true of it once built?"

Those three questions are the failure modes 2 and 3 in
``projects/sc-data-platform-architecture.md`` — the ones that used to require a code
change per ERP. Encoding them as data means a new ERP needs a new YAML, not a new
branch in a reader.

A contract is loaded once and is immutable at run time. Every expression it contains
is parsed (and therefore validated) at load, so a malformed contract fails fast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .expressions import Expression, ExpressionError

CONTRACTS_DIR = Path(__file__).parent / "contracts"

# Severity drives whether a failed assertion blocks promotion or just warns.
SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"


@dataclass
class Assertion:
    """A semantic invariant over the canonical frame."""

    expr: Expression
    severity: str = SEVERITY_ERROR
    description: str = ""

    @classmethod
    def parse(cls, raw: Any) -> "Assertion":
        if isinstance(raw, str):
            return cls(expr=Expression(raw))
        return cls(
            expr=Expression(raw["expr"]),
            severity=raw.get("severity", SEVERITY_ERROR),
            description=raw.get("description", ""),
        )

    @property
    def source(self) -> str:
        return self.expr.source

    def __repr__(self) -> str:
        return f"Assertion({self.source!r}, {self.severity})"


@dataclass
class FieldContract:
    """Contract for a single canonical field."""

    name: str
    type: str = "string"
    unit: Optional[str] = None
    grain: Optional[str] = None
    role: Optional[str] = None            # identifier | measure | dimension | timestamp
    required: bool = False
    description: str = ""
    aliases: List[str] = dc_field(default_factory=list)
    derivable_from: List[str] = dc_field(default_factory=list)
    assertions: List[Assertion] = dc_field(default_factory=list)
    required_by: List[str] = dc_field(default_factory=list)
    normalize: Optional[str] = None       # upper_strip | strip | lower_strip
    value_domain: Optional[str] = None    # name of a shared value_domain

    @classmethod
    def parse(cls, name: str, raw: Dict[str, Any]) -> "FieldContract":
        raw = raw or {}
        derivable = raw.get("derivable_from", []) or []
        # Parse now so a bad expression surfaces at contract-load, not mid-run
        for expr_src in derivable:
            try:
                Expression(expr_src)
            except ExpressionError as exc:
                raise ExpressionError(
                    f"Field {name!r} has invalid derivable_from entry: {exc}"
                ) from exc
        return cls(
            name=name,
            type=raw.get("type", "string"),
            unit=raw.get("unit"),
            grain=raw.get("grain"),
            role=raw.get("role"),
            required=bool(raw.get("required", False)),
            description=raw.get("description", ""),
            aliases=list(raw.get("aliases", []) or []),
            derivable_from=list(derivable),
            assertions=[Assertion.parse(a) for a in (raw.get("assertions", []) or [])],
            required_by=list(raw.get("required_by", []) or []),
            normalize=raw.get("normalize"),
            value_domain=raw.get("value_domain"),
        )

    @property
    def is_numeric(self) -> bool:
        return self.type in ("decimal", "integer", "number")

    @property
    def is_temporal(self) -> bool:
        return self.type in ("date", "datetime", "timestamp")


@dataclass
class ValueDomain:
    """
    A controlled vocabulary plus the raw codes that map into it.

    Broken out as a named, reusable object because — per the architecture doc's
    "known traps" — status/incoterm/UoM code differences cause more rework than
    column-name differences, and the same mapping recurs across sources.
    """

    name: str
    canonical: List[str]
    map: Dict[str, str] = dc_field(default_factory=dict)
    default: Optional[str] = None
    case_sensitive: bool = False

    @classmethod
    def parse(cls, name: str, raw: Dict[str, Any]) -> "ValueDomain":
        raw_map = raw.get("map", {}) or {}
        case_sensitive = bool(raw.get("case_sensitive", False))
        norm_map = {
            (str(k) if case_sensitive else str(k).strip().lower()): v
            for k, v in raw_map.items()
        }
        return cls(
            name=name,
            canonical=list(raw.get("canonical", []) or []),
            map=norm_map,
            default=raw.get("default"),
            case_sensitive=case_sensitive,
        )

    def translate(self, value: Any) -> Optional[str]:
        """Map one raw code to its canonical value; unknown codes return the default."""
        import pandas as pd

        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return self.map.get("", self.default)
        key = str(value) if self.case_sensitive else str(value).strip().lower()
        return self.map.get(key, self.default)

    def unknown_codes(self, values) -> List[str]:
        """Raw codes present in the data that the domain does not cover."""
        import pandas as pd

        seen = pd.Series(values).dropna().astype(str).unique()
        out = []
        for raw in seen:
            key = raw if self.case_sensitive else raw.strip().lower()
            if key not in self.map:
                out.append(raw)
        return sorted(out)


@dataclass
class DocContract:
    """Contract for one document type (open_po, inventory, ...)."""

    doc_type: str
    description: str = ""
    grain: str = "row"
    natural_key: List[str] = dc_field(default_factory=list)
    fields: Dict[str, FieldContract] = dc_field(default_factory=dict)
    value_domains: Dict[str, ValueDomain] = dc_field(default_factory=dict)
    table_assertions: List[Assertion] = dc_field(default_factory=list)
    capabilities: List[str] = dc_field(default_factory=list)
    default_filters: List[str] = dc_field(default_factory=list)
    # Content test used only to break a tie between documents whose headers are
    # indistinguishable. One ERP export filtered two ways — open orders and shipped
    # history — has byte-identical columns, so no header-based rule can separate them;
    # only what is *in* the rows can.
    discriminator: Optional[str] = None
    source_path: Optional[Path] = None

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "DocContract":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.parse(raw, source_path=Path(path))

    @classmethod
    def parse(cls, raw: Dict[str, Any], source_path: Path = None) -> "DocContract":
        fields = {
            name: FieldContract.parse(name, spec)
            for name, spec in (raw.get("fields", {}) or {}).items()
        }
        domains = {
            name: ValueDomain.parse(name, spec)
            for name, spec in (raw.get("value_domains", {}) or {}).items()
        }
        default_filters = list(raw.get("default_filters", []) or [])
        for filt in default_filters:
            Expression(filt)   # validate at load

        discriminator = raw.get("discriminator")
        if discriminator:
            Expression(str(discriminator))

        return cls(
            doc_type=raw["doc_type"],
            description=raw.get("description", ""),
            grain=raw.get("grain", "row"),
            natural_key=list(raw.get("natural_key", []) or []),
            fields=fields,
            value_domains=domains,
            table_assertions=[Assertion.parse(a) for a in (raw.get("assertions", []) or [])],
            capabilities=list(raw.get("capabilities", []) or []),
            default_filters=default_filters,
            discriminator=str(discriminator) if discriminator else None,
            source_path=source_path,
        )

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def required_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.required]

    @property
    def derivable_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.derivable_from]

    def field(self, name: str) -> Optional[FieldContract]:
        return self.fields.get(name)

    def alias_index(self) -> Dict[str, str]:
        """Normalized alias -> canonical field name, for lexical detection."""
        from .profiler import normalize_header

        index: Dict[str, str] = {}
        for canonical, spec in self.fields.items():
            for alias in [canonical] + spec.aliases:
                index.setdefault(normalize_header(alias), canonical)
        return index

    def all_assertions(self) -> List[Assertion]:
        """Field-level assertions followed by table-level ones."""
        out: List[Assertion] = []
        for spec in self.fields.values():
            out.extend(spec.assertions)
        out.extend(self.table_assertions)
        return out

    def __repr__(self) -> str:
        return (
            f"DocContract({self.doc_type!r}, grain={self.grain!r}, "
            f"{len(self.fields)} fields, {len(self.derivable_fields)} derivable)"
        )


class ContractRegistry:
    """Loads and caches every contract in a directory."""

    def __init__(self, contracts_dir: Path = None):
        self.contracts_dir = Path(contracts_dir) if contracts_dir else CONTRACTS_DIR
        self._cache: Dict[str, DocContract] = {}
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        if not self.contracts_dir.exists():
            raise FileNotFoundError(f"Contracts directory not found: {self.contracts_dir}")
        for path in sorted(self.contracts_dir.glob("*.yaml")):
            contract = DocContract.load(path)
            self._cache[contract.doc_type] = contract
        self._loaded = True

    def get(self, doc_type: str) -> DocContract:
        self._load_all()
        if doc_type not in self._cache:
            raise KeyError(
                f"No contract for doc_type {doc_type!r}. "
                f"Known: {sorted(self._cache)}"
            )
        return self._cache[doc_type]

    def all(self) -> Dict[str, DocContract]:
        self._load_all()
        return dict(self._cache)

    @property
    def doc_types(self) -> List[str]:
        self._load_all()
        return sorted(self._cache)

    def capability_map(self) -> Dict[str, List[str]]:
        """capability -> doc_types that can supply it."""
        self._load_all()
        out: Dict[str, List[str]] = {}
        for doc_type, contract in self._cache.items():
            for cap in contract.capabilities:
                out.setdefault(cap, []).append(doc_type)
        return {k: sorted(v) for k, v in out.items()}

    def summary(self) -> str:
        self._load_all()
        lines = [f"Contracts loaded from {self.contracts_dir}"]
        for doc_type in sorted(self._cache):
            c = self._cache[doc_type]
            lines.append(
                f"  {doc_type:<20} grain={c.grain:<20} "
                f"fields={len(c.fields):<3} required={len(c.required_fields)} "
                f"derivable={len(c.derivable_fields)} caps={','.join(c.capabilities) or '-'}"
            )
        return "\n".join(lines)


_default_registry: Optional[ContractRegistry] = None


def default_registry() -> ContractRegistry:
    """Process-wide contract registry over the bundled contracts directory."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ContractRegistry()
    return _default_registry
