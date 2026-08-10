"""
Planning parameters — segmentation boundaries, conventions and override rules.

Read from `config/planning_parameters.md`. The knowledge of *which SKU gets which
policy* is business judgment that changes far more often than the arithmetic does, so
it lives in a file a planner can edit without touching Python — the same principle as
the ingest contracts.

Two things make this more than a config loader:

  rationale   Every rule must state why it exists. Three months on, the reason is the
              only thing that lets anyone judge whether the rule still applies — and
              it is what lets the system ask whether a rule's scope should extend to a
              newly-appeared segment, rather than matching a column blindly.

  audit       Every application records which rule changed which parameter on which
              SKUs. A parameter that silently differs from the default is how a
              planning system loses its reader's trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ..ingest.expressions import Expression, ExpressionError

DEFAULT_PARAMS_FILE = Path(__file__).parents[2] / "config" / "planning_parameters.md"

# Sections whose fenced yaml blocks are merged into flat settings rather than rules.
_SETTING_SECTIONS = {"口径约定", "conventions", "默认参数", "defaults",
                     "分档边界", "segmentation"}


@dataclass
class Rule:
    """One scoped parameter override."""

    rule_id: str
    name: str
    scope: str
    overrides: Dict[str, Any]
    rationale: str = ""
    owner: str = ""
    date: str = ""
    _expr: Optional[Expression] = None

    def __post_init__(self):
        if not self.rationale.strip():
            raise ValueError(
                f"Rule {self.rule_id} has no rationale. Every rule must state why it "
                f"exists — without it nobody can judge later whether it still applies."
            )
        try:
            self._expr = Expression(self.scope)
        except ExpressionError as exc:
            raise ValueError(f"Rule {self.rule_id} has an invalid scope: {exc}") from exc

    @property
    def columns(self) -> set:
        return self._expr.columns

    def match(self, skus: pd.DataFrame) -> pd.Series:
        """Boolean mask of the SKUs this rule applies to."""
        missing = self.columns - set(skus.columns)
        if missing:
            # A rule that references an unavailable attribute matches nothing, rather
            # than raising. The gap is reported in the audit so it is visible without
            # aborting a run over an optional dimension.
            return pd.Series(False, index=skus.index)
        mask = self._expr.evaluate(skus)
        if not isinstance(mask, pd.Series):
            return pd.Series(bool(mask), index=skus.index)
        return mask.fillna(False).astype(bool)

    def __str__(self) -> str:
        sets = ", ".join(f"{k}={v}" for k, v in self.overrides.items())
        return f"{self.rule_id} · {self.name} — [{self.scope}] → {sets}"


@dataclass
class RuleHit:
    """Audit record: one rule's effect on one run."""

    rule: Rule
    matched: int
    sample_skus: List[str]
    unavailable_columns: List[str] = dc_field(default_factory=list)
    # param -> the earlier rule whose value this rule replaced
    overrides_earlier: Dict[str, str] = dc_field(default_factory=dict)

    def __str__(self) -> str:
        if self.unavailable_columns:
            return (f"    {self.rule.rule_id:<8} SKIPPED — scope needs unavailable "
                    f"column(s) {self.unavailable_columns}")
        if self.matched == 0:
            return f"    {self.rule.rule_id:<8} matched 0 SKUs — scope may be wrong"
        sample = ", ".join(self.sample_skus[:3])
        more = f" +{self.matched - len(self.sample_skus[:3])}" if self.matched > 3 else ""
        line = (f"    {self.rule.rule_id:<8} {self.matched:>4} SKUs  "
                f"{self.rule.name}  ({sample}{more})")
        if self.overrides_earlier:
            supers = ", ".join(f"{p} (was {r}'s)" for p, r in self.overrides_earlier.items())
            line += f"\n{'':<14}⚠ overrides {supers}"
        return line


@dataclass
class ParameterSet:
    """Resolved parameters plus the audit trail that produced them."""

    frame: pd.DataFrame                     # per-SKU resolved parameters
    hits: List[RuleHit] = dc_field(default_factory=list)
    conventions: Dict[str, Any] = dc_field(default_factory=dict)
    defaults: Dict[str, Any] = dc_field(default_factory=dict)
    segmentation: Dict[str, Any] = dc_field(default_factory=dict)

    def convention(self, key: str, fallback: Any = None) -> Any:
        return self.conventions.get(key, fallback)

    @property
    def conflicts(self) -> List[RuleHit]:
        return [h for h in self.hits if h.overrides_earlier]

    def summary(self) -> str:
        lines = ["  Planning parameters", "  " + "-" * 58]
        lines.append(f"    cycle stock basis   : {self.convention('cycle_stock_basis')}")
        lines.append(f"    SS exposure period  : {self.convention('safety_stock_exposure')}")
        lines.append(f"    pipeline basis      : {self.convention('pipeline_basis')}")
        lines.append("")
        lines.append(f"    Rules applied ({len(self.hits)}):")
        for hit in self.hits:
            lines.append(str(hit))

        if "review_period_days" in self.frame.columns:
            dist = self.frame["review_period_days"].value_counts().sort_index()
            spread = ", ".join(f"{int(k)}d: {v}" for k, v in dist.items())
            lines.append(f"\n    Review periods      : {spread}")
        if "service_level" in self.frame.columns:
            dist = self.frame["service_level"].value_counts().sort_index()
            spread = ", ".join(f"{k:.0%}: {v}" for k, v in dist.items())
            lines.append(f"    Service levels      : {spread}")
        return "\n".join(lines)


class PlanningParameters:
    """Loads the markdown parameter file and resolves per-SKU parameters."""

    def __init__(self, path: Path = None):
        self.path = Path(path) if path else DEFAULT_PARAMS_FILE
        self.conventions: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.segmentation: Dict[str, Any] = {}
        self.rules: List[Rule] = []
        self._load()

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Planning parameters not found at {self.path}. This file holds the "
                f"segmentation boundaries and policy overrides — the pipeline cannot "
                f"assume them."
            )
        text = self.path.read_text(encoding="utf-8")

        for heading, body in self._sections(text, level=2):
            key = self._section_key(heading)
            blocks = self._yaml_blocks(body)
            if not blocks:
                continue
            merged: Dict[str, Any] = {}
            for block in blocks:
                merged.update(block)
            if key in ("口径约定", "conventions"):
                self.conventions.update(merged)
            elif key in ("默认参数", "defaults"):
                self.defaults.update(merged)
            elif key in ("分档边界", "segmentation"):
                self.segmentation.update(merged)

        for heading, body in self._sections(text, level=3):
            blocks = self._yaml_blocks(body)
            if not blocks:
                continue
            self.rules.append(self._build_rule(heading, blocks[0]))

        self._validate()

    @staticmethod
    def _fenced_spans(text: str) -> List[Tuple[int, int]]:
        """
        Character ranges covered by fenced code blocks.

        Needed because a YAML comment (`# cycle stock basis`) is indistinguishable from
        a markdown H1 by regex alone. Without this, every section body is truncated at
        its first commented parameter — and since comments are exactly where the
        rationale for a setting lives, that is where they belong.
        """
        return [(m.start(), m.end())
                for m in re.finditer(r"```.*?```", text, re.DOTALL)]

    @classmethod
    def _sections(cls, text: str, level: int) -> List[Tuple[str, str]]:
        """Split markdown into (heading, body) at exactly the given heading level."""
        spans = cls._fenced_spans(text)

        def in_fence(pos: int) -> bool:
            return any(start <= pos < end for start, end in spans)

        heading = re.compile(rf"^{'#' * level} +(.+?)$", re.MULTILINE)
        # A body ends at the next heading of this level *or shallower*, so a level-3
        # body does not run past the level-2 section that contains it.
        boundary = re.compile(rf"^#{{1,{level}}} +", re.MULTILINE)

        matches = [m for m in heading.finditer(text) if not in_fence(m.start())]
        out = []
        for m in matches:
            start = m.end()
            end = len(text)
            for b in boundary.finditer(text, start):
                if not in_fence(b.start()):
                    end = b.start()
                    break
            out.append((m.group(1).strip(), text[start:end]))
        return out

    @staticmethod
    def _section_key(heading: str) -> str:
        """`口径约定 (Conventions)` -> `口径约定`."""
        return re.sub(r"\s*\(.*?\)\s*$", "", heading).strip().lower()

    @staticmethod
    def _yaml_blocks(body: str) -> List[Dict[str, Any]]:
        out = []
        for match in re.finditer(r"```ya?ml\s*\n(.*?)```", body, re.DOTALL):
            try:
                parsed = yaml.safe_load(match.group(1))
            except yaml.YAMLError as exc:
                raise ValueError(f"Malformed YAML block in planning parameters: {exc}") from exc
            if isinstance(parsed, dict):
                out.append(parsed)
        return out

    @staticmethod
    def _build_rule(heading: str, block: Dict[str, Any]) -> Rule:
        # `R-001 · A 类物料周度 review`
        parts = re.split(r"\s*[·•|]\s*", heading, maxsplit=1)
        rule_id = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else heading

        missing = {"scope", "set"} - set(block)
        if missing:
            raise ValueError(f"Rule {rule_id} is missing required key(s): {sorted(missing)}")

        return Rule(
            rule_id=rule_id,
            name=name,
            scope=str(block["scope"]),
            overrides=dict(block["set"] or {}),
            rationale=str(block.get("rationale", "")),
            owner=str(block.get("owner", "")),
            date=str(block.get("date", "")),
        )

    def _validate(self) -> None:
        required_conventions = {"cycle_stock_basis", "safety_stock_exposure", "pipeline_basis"}
        missing = required_conventions - set(self.conventions)
        if missing:
            raise ValueError(
                f"planning_parameters.md is missing convention(s): {sorted(missing)}. "
                f"These decide how should-be inventory is computed and cannot be defaulted."
            )
        if self.conventions["cycle_stock_basis"] not in ("peak", "average"):
            raise ValueError("cycle_stock_basis must be 'peak' or 'average'")

        ids = [r.rule_id for r in self.rules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate rule id(s) in planning parameters: {sorted(dupes)}")

    # ── Resolution ───────────────────────────────────────────────────────────

    def resolve(self, skus: pd.DataFrame) -> ParameterSet:
        """
        Apply defaults then rules to a per-SKU attribute frame.

        Rules apply in file order, later winning — which makes the file readable
        top-to-bottom as "general policy, then the exceptions". Where a later rule
        overwrites an earlier one on the same SKU and parameter, both are recorded so
        the override is visible rather than merely effective.
        """
        if "sku" not in skus.columns:
            raise ValueError("SKU attribute frame must have a 'sku' column")

        params = skus.copy()
        for key, value in self.defaults.items():
            if key in params.columns:
                # A default fills gaps; it does not overwrite. MOQ, order multiple and
                # unit cost legitimately arrive from the ERP per SKU, and clobbering
                # them with a global default would silently discard the real constraint
                # while leaving every number looking plausible.
                params[key] = params[key].where(params[key].notna(), value)
            else:
                params[key] = value

        # param -> rule_id that last set it, per row; used to detect real conflicts
        claimed: Dict[str, pd.Series] = {}
        hits: List[RuleHit] = []

        for rule in self.rules:
            unavailable = sorted(rule.columns - set(skus.columns))
            if unavailable:
                hits.append(RuleHit(rule=rule, matched=0, sample_skus=[],
                                    unavailable_columns=unavailable))
                continue

            mask = rule.match(params)
            matched = int(mask.sum())
            overridden: Dict[str, str] = {}

            for key, value in rule.overrides.items():
                if key in claimed:
                    clash = mask & claimed[key].notna()
                    if clash.any():
                        prior = claimed[key][clash].mode()
                        if len(prior):
                            overridden[key] = str(prior.iloc[0])
                else:
                    claimed[key] = pd.Series(pd.NA, index=params.index, dtype="object")

                params.loc[mask, key] = value
                claimed[key].loc[mask] = rule.rule_id

            hits.append(RuleHit(
                rule=rule,
                matched=matched,
                sample_skus=params.loc[mask, "sku"].astype(str).head(3).tolist(),
                overrides_earlier=overridden,
            ))

        params["applied_rules"] = self._rule_trace(params.index, claimed)
        return ParameterSet(
            frame=params,
            hits=hits,
            conventions=dict(self.conventions),
            defaults=dict(self.defaults),
            segmentation=dict(self.segmentation),
        )

    @staticmethod
    def _rule_trace(index: pd.Index, claimed: Dict[str, pd.Series]) -> pd.Series:
        """Per-SKU list of the rules that touched it — carried into the report."""
        trace = pd.Series([[] for _ in range(len(index))], index=index, dtype=object)
        for series in claimed.values():
            for idx, rule_id in series.dropna().items():
                if rule_id not in trace[idx]:
                    trace[idx].append(rule_id)
        return trace.apply(lambda ids: ",".join(sorted(ids)) if ids else "")

    # ── Segmentation ─────────────────────────────────────────────────────────

    def assign_abc(self, skus: pd.DataFrame, value_col: str = "annual_value") -> pd.Series:
        """
        Pareto ABC by cumulative share of annual value, using the configured cut-offs.

        Segmentation lives here rather than in the analytics because the boundaries are
        a business decision, not a statistical one — and the planner said outright that
        there is no absolute right answer, so it must be easy to move.
        """
        thresholds = self.segmentation.get("abc_thresholds", {"A": 0.80, "B": 0.95})
        value = pd.to_numeric(skus[value_col], errors="coerce").fillna(0.0)
        total = value.sum()
        if total <= 0:
            return pd.Series("C", index=skus.index)

        order = value.sort_values(ascending=False)
        cumulative = order.cumsum() / total
        classes = pd.Series("C", index=skus.index, dtype=object)
        classes.loc[cumulative[cumulative <= thresholds.get("B", 0.95)].index] = "B"
        classes.loc[cumulative[cumulative <= thresholds.get("A", 0.80)].index] = "A"
        return classes

    def assign_volatility(self, cv: pd.Series) -> pd.Series:
        t = self.segmentation.get("volatility_thresholds", {"stable": 0.5, "variable": 1.0})
        out = pd.Series("erratic", index=cv.index, dtype=object)
        out[cv <= t.get("variable", 1.0)] = "variable"
        out[cv <= t.get("stable", 0.5)] = "stable"
        return out
