"""
Quality gates.

## Why a run stops

The pipeline's failure mode is not a crash. It is a complete report, correctly
formatted, every column populated, built on a join that matched nothing or a dimension
that arrived empty — and no error anywhere. Three real datasets produced exactly that:
a padded material number, a line number mapped as the item, and a master keyed on a
local code. Each returned a full set of numbers, all of them zero, and none of them
questioned until somebody read the output and disbelieved it.

A report that cannot be falsified by its reader is worse than no report, because it is
acted on. So at defined points the run asks whether what it holds can support the
statement it is about to make, and where it cannot, it stops and says why.

## What does *not* stop a run

Dirtiness is not a gate. Some negative quantities, a handful of lead times beyond the
plausible, a few rows with no request date — these are findings the report is built to
carry, and halting on them would make the pipeline unusable on every real export. The
distinction this module draws is not clean/dirty but *recoverable/not*:

    a field 8% wrong        the report says so and the number stands   → WARN
    a field 100% absent     nothing downstream guards against it       → BLOCK
    a join matching 3%      every figure derived from it is empty      → BLOCK

The test is whether a reader of the output could tell. If the damage shows up in the
report as an obvious hole, warn. If it shows up as a plausible number, block.

## Overriding

`allow_degraded=True` proceeds past BLOCK findings. It exists because a threshold is a
judgement and judgements are sometimes wrong on data nobody anticipated — not as a
convenience. Every overridden finding is recorded on the report and travels into the
run manifest, so an output produced under an override says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional

BLOCK = "block"
# Does not stop the run; leads every output that follows. The distinction from WARN is
# not how bad the data is but what a reader has to do about it: a WARN is context they
# can note and move past, a SEVERE means specific figures in front of them are not what
# they appear to be, and they have to know which before they act on any of them.
#
# Three levels rather than two because collapsing them loses the case this exists for.
# Made blocking, "no product family" stops a run that a planner legitimately needs;
# made an ordinary warning, it sits in a list of eleven things and the revenue-by-line
# table gets read as if it meant something.
SEVERE = "severe"
WARN = "warn"

_RANK = {BLOCK: 0, SEVERE: 1, WARN: 2}


# ── Findings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """
    One thing wrong, stated so it can be acted on without reading the code.

    The three-part shape is deliberate. A message saying "SKU agreement 3%" names a
    symptom; a planner cannot do anything with it. `what` is the symptom, `why` is what
    it does to the report they were about to read, and `fix` is the next action. All
    three, every time — a finding that cannot say what to do about it is a finding that
    should not stop a run.
    """

    stage: str
    check: str
    severity: str
    what: str
    why: str
    fix: str
    evidence: Dict[str, Any] = dc_field(default_factory=dict)
    # The named outputs this finding makes unreliable, one per line, each saying what
    # the figure will look like rather than that it is "affected". "Revenue by product
    # line groups on a guess" tells a reader whether to use the table; "product family
    # data quality issue" does not.
    impacts: List[str] = dc_field(default_factory=list)

    @property
    def mark(self) -> str:
        return {BLOCK: "✗", SEVERE: "!"}.get(self.severity, "⚠")

    def summary(self, indent: str = "    ") -> str:
        lines = [
            f"{indent}{self.mark} [{self.check}] {self.what}",
            f"{indent}    Why it matters: {self.why}",
        ]
        for item in self.impacts:
            lines.append(f"{indent}    → {item}")
        lines.append(f"{indent}    Fix: {self.fix}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage, "check": self.check, "severity": self.severity,
            "what": self.what, "why": self.why, "fix": self.fix,
            "impacts": list(self.impacts),
            "evidence": dict(self.evidence),
        }


@dataclass
class GateReport:
    """What one checkpoint found, and whether the run may go on."""

    stage: str
    findings: List[Finding] = dc_field(default_factory=list)
    overridden: bool = False

    @property
    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == BLOCK]

    @property
    def severe(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == SEVERE]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    @property
    def ordered(self) -> List[Finding]:
        """Worst first, then by check name so a run's output is stable."""
        return sorted(self.findings, key=lambda f: (_RANK.get(f.severity, 9), f.check))

    @property
    def passed(self) -> bool:
        return not self.blocking

    def summary(self) -> str:
        if not self.findings:
            return f"  Quality gate · {self.stage}: passed"
        lines = [f"  Quality gate · {self.stage}", "  " + "-" * 58]
        for f in self.ordered:
            lines.append(f.summary())
        if self.overridden and self.blocking:
            lines.append(
                "    OVERRIDDEN — the run was told to continue past the failures above "
                "(allow_degraded). Every number below rests on data that did not pass."
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "overridden": self.overridden,
            "findings": [f.to_dict() for f in self.findings],
        }


class DataQualityError(ValueError):
    """Raised when a gate finds something the run cannot honestly continue past."""

    def __init__(self, report: GateReport):
        self.report = report
        blocking = report.blocking
        head = (
            f"Cannot continue — {len(blocking)} quality "
            f"{'check' if len(blocking) == 1 else 'checks'} failed at the "
            f"'{report.stage}' stage.\n"
            "A report built on this data would be complete, plausible and wrong, which "
            "is worse than no report.\n"
        )
        body = "\n".join(f.summary(indent="  ") for f in blocking)
        tail = (
            "\n\n  Fix the cause and re-run. To proceed anyway — every figure then rests "
            "on data that did not pass, and the run records that it was overridden — "
            "construct the planner with allow_degraded=True."
        )
        super().__init__(head + body + tail)


# ── Thresholds ───────────────────────────────────────────────────────────────


_DEFAULTS: Dict[str, float] = {
    # A document whose SKUs meet no other document's. Below this share every join it
    # feeds returns empty, and empty reads as zero rather than as missing. The intake
    # already computes this agreement and warns; the gate is where it stops being
    # advice.
    "sku_agreement_floor": 0.20,
    # Product family coverage across the SKUs actually being planned. Partial coverage
    # is the dangerous case: `assemble._infer_family` fills the rest from the part
    # number, so a family rollup is part master data and part numbering scheme with
    # nothing marking the boundary.
    "product_family_coverage_floor": 0.90,
    # A semantic assertion failing on nearly every row is not dirt, it is the wrong
    # column. Below this the finding is reported and the run continues, which is what
    # makes the pipeline usable on real exports.
    "semantic_failure_ceiling": 0.50,
    # SKUs with a demand history that came out of the forecaster with no forecast.
    "forecast_coverage_floor": 0.80,
    # Forecast SKUs with no row in the inventory snapshot. These are planned against a
    # position of zero, so every one of them reads as a shortage.
    "position_coverage_floor": 0.80,
    # Months of history at the median SKU. Below this a backtest cannot rank anything
    # and every model falls back to the pattern route.
    "median_history_months": 6.0,
    # Days between the newest date in the data and the run. Past this the extract is
    # stale enough that open orders are scored against a "now" that has moved.
    "extract_staleness_days": 120.0,
}


@dataclass
class GateThresholds:
    """
    The numbers the gates judge against, overridable per site.

    They live in `config/quality_gates.json` rather than in code because every one of
    them is a judgement about a particular business — a 20% join floor is generous for
    a single-plant extract and far too loose for a consolidated one — and a site that
    disagrees should be able to say so without editing the pipeline.
    """

    values: Dict[str, float] = dc_field(default_factory=lambda: dict(_DEFAULTS))

    @classmethod
    def load(cls, config_dir: Optional[Path]) -> "GateThresholds":
        values = dict(_DEFAULTS)
        if config_dir:
            path = Path(config_dir) / "quality_gates.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                # Only known keys. A typo in a config file that silently introduces a
                # threshold nothing reads is the same class of bug as everything else
                # here: it looks like it was configured and it was not.
                for key, value in raw.items():
                    # `_name` entries are the file's own documentation, sitting beside
                    # the value they explain so the two cannot drift apart.
                    if key.startswith("_"):
                        continue
                    if key not in _DEFAULTS:
                        raise ValueError(
                            f"quality_gates.json: unknown threshold {key!r}. "
                            f"Known: {', '.join(sorted(_DEFAULTS))}"
                        )
                    values[key] = float(value)
        return cls(values=values)

    def __getitem__(self, key: str) -> float:
        return self.values[key]
