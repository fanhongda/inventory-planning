"""
The declaration layer — what a person has said, as data.

Correcting a mapping had exactly two settings before this, and the gap between them is
where the operational pain lived. Either the pipeline guessed, or someone hand-wrote
and froze an adapter. There was no way to say "in this export the SKU is the `Material`
column, work the rest out yourself" — so the actual field remedy became editing headers
in Excel, which broke the next run outright: a frozen adapter reads its source column
by name, and renaming that column removed the field it was pointing at.

Two kinds of statement live here.

**Overrides** say what a value is, in the four places a person legitimately knows
better than the inference: which column supplies a field, which item a code names, what
a fact actually is where the ERP was never told, and what a planner-owned parameter
should be. They are applied deterministically, they are recorded on the resolution that
used them, and they never edit a fact — a `value` override is new evidence sitting
beside the ERP's record, not a correction of it.

**Gate waivers** say that one check, on one document, is a false positive here. They
replace `allow_degraded=True`, which is a single switch over the whole run: waiving the
SKU-agreement check on a whole-warehouse stock snapshot should not also wave through a
purchase order whose open quantity was mapped to a money column. Every waiver carries
an expiry, because a waiver without one is how a temporary judgement becomes permanent
and how a checked pipeline quietly becomes an unchecked one.

Both are loaded from a file meant to live in git. That is not incidental. A mapping
correction should be reviewed, diffed and attributable, and a version-control system
does that better than a database does — while the loaded copy is what the pipeline
joins against.

Two failure modes are treated as findings rather than as silence:

  a declaration that matched nothing — someone believes they fixed something and did
  not, which is worse than not having tried, because they have stopped looking

  a waiver past its expiry — the check starts blocking again, and the message says the
  waiver expired rather than presenting it as a fresh discovery
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

DECLARATIONS_NAME = "declarations.yaml"

SCOPE_MAPPING = "mapping"
SCOPE_IDENTITY = "identity"
SCOPE_VALUE = "value"
SCOPE_PARAMETER = "parameter"
_SCOPES = (SCOPE_MAPPING, SCOPE_IDENTITY, SCOPE_VALUE, SCOPE_PARAMETER)


class DeclarationError(ValueError):
    """A declaration file that cannot be trusted to mean what it says."""


def _as_date(value: Any, where: str) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError as exc:
        raise DeclarationError(f"{where}: {value!r} is not a date (use YYYY-MM-DD)") from exc


@dataclass(frozen=True)
class Override:
    """One thing a person has declared to be so."""

    scope: str
    field: str
    value: Any
    target: Dict[str, Any] = dc_field(default_factory=dict)
    reason: str = ""
    by: str = ""
    at: Optional[date] = None
    expires: Optional[date] = None

    def active_on(self, when: date) -> bool:
        return self.expires is None or when <= self.expires

    def matches_document(self, doc_type: str, headers: Sequence[str],
                         source_name: str = "") -> bool:
        """
        Whether this mapping override applies to the document in front of us.

        Matched on the headers the export carries, not on the file. A file is
        identified by its bytes, and next month's export of the same report is
        different bytes under a different name — a declaration tied to it expires the
        moment it becomes useful. `source` remains available for the genuinely
        one-off correction, and says so by being the narrower of the two.
        """
        target = self.target
        if target.get("doc_type") and target["doc_type"] != doc_type:
            return False
        wanted = target.get("headers")
        if wanted:
            present = {str(h).strip().lower() for h in headers}
            if not {str(w).strip().lower() for w in wanted} <= present:
                return False
        source = target.get("source")
        if source and source.strip().lower() != str(source_name).strip().lower():
            return False
        return bool(wanted or source or target.get("doc_type"))


@dataclass(frozen=True)
class GateWaiver:
    """One check, on one document, declared a false positive here — until a date."""

    check: str
    doc_type: str
    expires: date
    reason: str = ""
    by: str = ""

    def active_on(self, when: date) -> bool:
        return when <= self.expires

    def applies_to(self, finding) -> bool:
        if finding.check != self.check:
            return False
        if self.doc_type in ("", "*"):
            return True
        return str(finding.evidence.get("doc_type", "")) == self.doc_type


class Declarations:
    """
    Everything a person has declared, loaded once and asked many times.

    Usage is deliberately query-shaped rather than mutation-shaped: nothing here edits
    a stored fact. A caller asks what was declared and applies it while building its
    own result, so the declaration and the thing it changed stay separable — which is
    what lets `resolution.decisions` say a field came from an override rather than
    from inference.
    """

    def __init__(self, overrides: Iterable[Override] = (),
                 waivers: Iterable[GateWaiver] = (), source: Path = None,
                 today: date = None):
        self.overrides: List[Override] = list(overrides)
        self.waivers: List[GateWaiver] = list(waivers)
        self.source = source
        self.today = today or date.today()
        self._used: Set[int] = set()

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_dir=None, today: date = None) -> "Declarations":
        """
        Read `<config_dir>/declarations.yaml`, or return an empty set if there is none.

        Absent is not an error — most runs declare nothing. Present but malformed *is*
        an error, and loudly: a declaration file the pipeline half-understands is worse
        than none, because the half it dropped is the correction someone is relying on.
        """
        if config_dir is None:
            config_dir = Path(__file__).parents[2] / "config"
        target = Path(config_dir) / DECLARATIONS_NAME
        if not target.exists():
            return cls(source=target, today=today)

        import yaml
        try:
            raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise DeclarationError(f"{target} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise DeclarationError(f"{target} must be a mapping, not {type(raw).__name__}")

        return cls(
            overrides=[cls._parse_override(item, target, i)
                       for i, item in enumerate(raw.get("overrides") or [])],
            waivers=[cls._parse_waiver(item, target, i)
                     for i, item in enumerate(raw.get("gate_waivers") or [])],
            source=target, today=today,
        )

    @staticmethod
    def _parse_override(item: Dict[str, Any], target: Path, index: int) -> Override:
        where = f"{target}: overrides[{index}]"
        if not isinstance(item, dict):
            raise DeclarationError(f"{where} must be a mapping")
        scope = str(item.get("scope", "")).strip()
        if scope not in _SCOPES:
            raise DeclarationError(
                f"{where}: scope {scope!r} is not one of {', '.join(_SCOPES)}")
        if not item.get("field"):
            raise DeclarationError(f"{where}: `field` is required")
        if "value" not in item:
            raise DeclarationError(f"{where}: `value` is required")
        if not str(item.get("reason", "")).strip():
            # A declaration overrules the pipeline's own evidence. Whoever reads it in
            # six months needs to know on what grounds, and the person writing it is
            # the only one who can say.
            raise DeclarationError(f"{where}: `reason` is required — say why")
        return Override(
            scope=scope, field=str(item["field"]), value=item["value"],
            target=dict(item.get("target") or {}),
            reason=str(item["reason"]), by=str(item.get("by", "")),
            at=_as_date(item.get("at"), where), expires=_as_date(item.get("expires"), where),
        )

    @staticmethod
    def _parse_waiver(item: Dict[str, Any], target: Path, index: int) -> GateWaiver:
        where = f"{target}: gate_waivers[{index}]"
        if not isinstance(item, dict):
            raise DeclarationError(f"{where} must be a mapping")
        if not item.get("check"):
            raise DeclarationError(f"{where}: `check` is required")
        if not str(item.get("reason", "")).strip():
            raise DeclarationError(f"{where}: `reason` is required — say why")
        expires = _as_date(item.get("expires"), where)
        if expires is None:
            # The whole point of per-check waiving is that it stays reviewable. A
            # waiver with no end is `allow_degraded=True` with extra steps, and it is
            # how a checked pipeline becomes an unchecked one without a decision.
            raise DeclarationError(
                f"{where}: `expires` is required — a waiver without an end date is a "
                f"permanently disabled check")
        return GateWaiver(
            check=str(item["check"]), doc_type=str(item.get("doc_type", "")),
            expires=expires, reason=str(item["reason"]), by=str(item.get("by", "")),
        )

    # ── Mapping overrides ────────────────────────────────────────────────────

    def column_map_for(self, doc_type: str, headers: Sequence[str],
                       source_name: str = "") -> Dict[str, str]:
        """
        Field → source column, for every mapping override that applies here.

        The caller merges this over whatever the adapter or the scorer produced, so a
        declaration corrects one field and leaves the rest inferred. That partial shape
        is the whole point: the alternative on offer today is authoring every column of
        a frozen adapter to fix one of them.
        """
        out: Dict[str, str] = {}
        for i, override in enumerate(self.overrides):
            if override.scope != SCOPE_MAPPING or not override.active_on(self.today):
                continue
            if override.matches_document(doc_type, headers, source_name):
                out[override.field] = str(override.value)
                self._used.add(i)
        return out

    def values_for(self, scope: str, **target) -> Dict[str, Any]:
        """Field → value for `identity`, `value` and `parameter` overrides at a target."""
        out: Dict[str, Any] = {}
        for i, override in enumerate(self.overrides):
            if override.scope != scope or not override.active_on(self.today):
                continue
            if all(override.target.get(k) == v for k, v in target.items()):
                out[override.field] = override.value
                self._used.add(i)
        return out

    # ── Gate waivers ─────────────────────────────────────────────────────────

    def waive(self, report):
        """
        Downgrade the blocking findings this run has declared false positives.

        Returns a new report; `GateReport` and `Finding` are left as they were, because
        a waived finding is still a finding. It is shown, it is recorded, and it travels
        into the manifest — it simply does not stop the run. A waiver that hid the
        finding would be indistinguishable from a check that never fired.
        """
        from ..quality.gates import BLOCK, WARN, Finding, GateReport

        if not self.waivers or not report.findings:
            return report

        findings, applied = [], []
        for finding in report.findings:
            waiver = self._waiver_for(finding)
            if waiver is None or finding.severity != BLOCK:
                findings.append(finding)
                continue
            applied.append((finding.check, waiver))
            findings.append(Finding(
                stage=finding.stage, check=finding.check, severity=WARN,
                what=finding.what,
                why=finding.why + (
                    f"  — waived until {waiver.expires} by {waiver.by or 'someone'}: "
                    f"{waiver.reason}"),
                fix=finding.fix,
                evidence=dict(finding.evidence, waived_by=waiver.by or "",
                              waived_until=str(waiver.expires),
                              waiver_reason=waiver.reason),
                impacts=list(finding.impacts),
            ))
        if not applied:
            return report
        waived = GateReport(stage=report.stage, findings=findings)
        waived.overridden = getattr(report, "overridden", False)
        return waived

    def _waiver_for(self, finding) -> Optional[GateWaiver]:
        for waiver in self.waivers:
            if waiver.applies_to(finding) and waiver.active_on(self.today):
                return waiver
        return None

    # ── Hygiene ──────────────────────────────────────────────────────────────

    def expired(self) -> List[Any]:
        """Waivers and overrides past their date. Reported, never silently re-applied."""
        return (
            [w for w in self.waivers if not w.active_on(self.today)]
            + [o for o in self.overrides
               if o.expires is not None and not o.active_on(self.today)]
        )

    def unused(self) -> List[Override]:
        """
        Overrides that matched nothing this run.

        The failure this exists for: a declaration is written against a header that has
        since been renamed, it silently applies to nothing, and the person who wrote it
        believes the mapping is corrected. Nothing errors — the pipeline simply goes on
        inferring, which is the state the declaration was written to end.
        """
        return [o for i, o in enumerate(self.overrides)
                if i not in self._used and o.active_on(self.today)]

    def notes(self) -> List[str]:
        """Lines for the intake report — empty when there is nothing to say."""
        lines = []
        for item in self.expired():
            if isinstance(item, GateWaiver):
                lines.append(
                    f"  ⚠ The waiver on [{item.check}] for "
                    f"{item.doc_type or 'every document'} expired {item.expires}. "
                    f"The check applies again — this is an expiry, not a new finding.")
            else:
                lines.append(
                    f"  ⚠ The override on {item.field!r} expired {item.expires} and was "
                    f"not applied.")
        for override in self.unused():
            lines.append(
                f"  ⚠ The {override.scope} override on {override.field!r} matched no "
                f"document in this run — check the headers it targets "
                f"({', '.join(str(h) for h in override.target.get('headers', [])) or 'none named'}). "
                f"Nothing was corrected.")
        return lines

    def summary(self) -> str:
        if not self.overrides and not self.waivers:
            return ""
        active_o = sum(1 for o in self.overrides if o.active_on(self.today))
        active_w = sum(1 for w in self.waivers if w.active_on(self.today))
        return (f"  Declarations — {active_o} override(s), {active_w} gate waiver(s) "
                f"from {self.source}")
