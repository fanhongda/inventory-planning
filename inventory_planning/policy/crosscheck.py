"""
Where each planning parameter came from, and what the other sources said.

Once an item master and a planner's worksheet are in play, the same parameter arrives
from several places at once and they will not agree. Lead time is measured at 62 days
from receipts, the ERP master says 45, the planner's spreadsheet says 90. Something
has to choose, and — more importantly — something has to say out loud that the other
two disagreed, because a stale master lead time is not a nuisance to be resolved
quietly: every MRP run in the ERP is planning on it.

## Authority

Sources are ranked by what kind of claim they are, not by how convenient they are:

    measured        derived from transactions in this run — what actually happened
    item_master     a standing parameter in the system of record — an intention
    planning_master a value a person maintains by hand — a decision
    config          a pipeline-wide default — a guess

Higher authority wins. The exception is thin evidence: a lead time "measured" from two
receipts is not a distribution, and a master value that reflects years of buying is
the better number. Below `min_samples` the measured value yields.

## Three jobs, one mechanism

  fill        a parameter with no measurement takes the best available stated value,
              and the row records which one
  cross-check where two or more sources speak, disagreement beyond a tolerance is
              recorded with both values and the relative gap
  compare     the `planning_master` tier is never silently consumed. Whatever else
              happens, the planner's own number stays on the frame under its own name
              so the suggestion engine can say "you set 500, the maths says 310"

The planner's values are deliberately *not* authoritative for anything the pipeline
can compute. Safety stock is the clearest case: it is fully determined by demand
variability, lead time and a service level, so a hand-set value is a position to be
compared, never an input. Treating it as an input would make the pipeline agree with
whatever it was shown, which is the one thing it must not do.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List

import numpy as np
import pandas as pd

MEASURED = "measured"
ITEM_MASTER = "item_master"
PLANNING_MASTER = "planning_master"
CONFIG = "config"

AUTHORITY_ORDER = (MEASURED, ITEM_MASTER, PLANNING_MASTER, CONFIG)

SOURCE_LABELS = {
    MEASURED: "measured from transactions",
    ITEM_MASTER: "ERP item master",
    PLANNING_MASTER: "planner worksheet",
    CONFIG: "config default",
}

# Below this many observations a "measured" value is a coincidence, not a distribution,
# and a stated master value is the better estimate.
MIN_SAMPLES_TO_TRUST_MEASUREMENT = 3

# Relative gap beyond which two sources are reported as disagreeing. Set where a
# difference starts to change a decision rather than where it becomes arithmetically
# detectable — flagging a 46-vs-45-day lead time trains the reader to skip the section.
DEFAULT_TOLERANCE = 0.25


@dataclass
class Disagreement:
    """Two sources gave materially different values for the same SKU and attribute."""

    attribute: str
    sku: str
    chosen_source: str
    chosen_value: float
    other_source: str
    other_value: float
    relative_gap: float

    def __str__(self) -> str:
        return (
            f"{self.sku:<18} {self.attribute:<22} "
            f"{SOURCE_LABELS[self.chosen_source]} {self.chosen_value:,.1f} vs "
            f"{SOURCE_LABELS[self.other_source]} {self.other_value:,.1f} "
            f"({self.relative_gap:+.0%})"
        )


@dataclass
class AttributeResolution:
    """How one attribute was settled across every SKU."""

    attribute: str
    counts_by_source: Dict[str, int] = dc_field(default_factory=dict)
    disagreements: List[Disagreement] = dc_field(default_factory=list)
    compared: int = 0          # source-pair comparisons made, not distinct SKUs

    @property
    def total(self) -> int:
        return sum(self.counts_by_source.values())

    @property
    def disagreement_rate(self) -> float:
        """Share of the comparisons that came out beyond tolerance."""
        return len(self.disagreements) / self.compared if self.compared else 0.0


@dataclass
class CrossCheckResult:
    """Per-attribute provenance and every disagreement worth a planner's attention."""

    resolutions: Dict[str, AttributeResolution] = dc_field(default_factory=dict)
    sources_present: List[str] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)

    @property
    def all_disagreements(self) -> List[Disagreement]:
        out: List[Disagreement] = []
        for res in self.resolutions.values():
            out.extend(res.disagreements)
        return sorted(out, key=lambda d: -abs(d.relative_gap))

    def frame(self) -> pd.DataFrame:
        """Every disagreement as a row, for the CSV a planner works through."""
        rows = [
            {
                "sku": d.sku,
                "attribute": d.attribute,
                "chosen_source": d.chosen_source,
                "chosen_value": d.chosen_value,
                "other_source": d.other_source,
                "other_value": d.other_value,
                "relative_gap": round(d.relative_gap, 3),
            }
            for d in self.all_disagreements
        ]
        return pd.DataFrame(rows, columns=["sku", "attribute", "chosen_source",
                                           "chosen_value", "other_source",
                                           "other_value", "relative_gap"])

    def summary(self, top: int = 8) -> str:
        if not self.resolutions:
            return "  Parameter sources: only the transaction history was supplied."

        lines = [
            "  Parameter sources and cross-check",
            "  " + "-" * 58,
            f"    Sources present: {', '.join(SOURCE_LABELS[s] for s in self.sources_present)}",
            "",
        ]
        for attribute, res in self.resolutions.items():
            spread = ", ".join(
                f"{n} {SOURCE_LABELS[s]}"
                for s, n in sorted(res.counts_by_source.items(), key=lambda kv: -kv[1])
                if n
            )
            lines.append(f"    {attribute:<22} {spread}")

        disagreements = self.all_disagreements
        if not disagreements:
            lines.append("")
            lines.append("    No source disagrees with another beyond tolerance.")
            return "\n".join(lines)

        lines.append("")
        lines.append(
            f"    {len(disagreements):,} disagreement(s) beyond tolerance. The value on "
            f"the left is what this run used; the one on the right is what the other "
            f"system is still planning on:"
        )
        for d in disagreements[:top]:
            lines.append(f"      {d}")
        if len(disagreements) > top:
            lines.append(f"      … and {len(disagreements) - top:,} more "
                         f"(see the cross-check CSV)")

        by_attr = {}
        for d in disagreements:
            by_attr.setdefault(d.attribute, []).append(d)
        for attribute, items in by_attr.items():
            res = self.resolutions.get(attribute)
            if res and res.disagreement_rate > 0.3:
                lines.append(
                    f"    ⚠ {attribute}: {res.disagreement_rate:.0%} of the SKUs that "
                    f"could be compared disagree. That is not per-item drift — the two "
                    f"sources are measuring different things, and the definition is "
                    f"worth settling before either number is used."
                )

        for note in self.notes:
            lines.append(f"    · {note}")
        return "\n".join(lines)


class SourceResolver:
    """
    Resolves one attribute across candidate sources and records what it found.

    Used by `assemble.build_sku_attributes`; kept separate so the precedence rule and
    the reporting of it live in one place rather than being re-implemented per column.
    """

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE,
                 min_samples: int = MIN_SAMPLES_TO_TRUST_MEASUREMENT):
        self.tolerance = float(tolerance)
        self.min_samples = int(min_samples)
        self.result = CrossCheckResult()

    def note(self, text: str) -> None:
        self.result.notes.append(text)

    def resolve(
        self,
        attribute: str,
        skus: pd.Series,
        candidates: Dict[str, pd.Series],
        sample_counts: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Pick a value per SKU by authority and record the provenance and disagreements.

        `candidates` maps source name -> a value Series aligned to `skus`. Absent or
        all-null sources may be passed; they are ignored. `sample_counts`, when given,
        demotes a measured value that rests on too few observations.

        Returns a frame with `value` and `source` columns, aligned to the input index.
        """
        usable = {
            name: pd.to_numeric(series, errors="coerce")
            for name, series in candidates.items()
            if series is not None and pd.to_numeric(series, errors="coerce").notna().any()
        }
        index = skus.index
        value = pd.Series(np.nan, index=index, dtype="float64")
        source = pd.Series(None, index=index, dtype="object")

        if not usable:
            return pd.DataFrame({"value": value, "source": source})

        # A measured value backed by too few observations drops below the stated ones.
        order = list(AUTHORITY_ORDER)
        thin = pd.Series(False, index=index)
        if MEASURED in usable and sample_counts is not None:
            thin = pd.to_numeric(sample_counts, errors="coerce").fillna(0) < self.min_samples

        for name in order:
            series = usable.get(name)
            if series is None:
                continue
            eligible = value.isna() & series.notna()
            if name == MEASURED:
                eligible &= ~thin
            value = value.where(~eligible, series)
            source = source.where(~eligible, name)

        # Second pass: a measurement demoted for thinness is still better than nothing.
        if MEASURED in usable:
            rescue = value.isna() & usable[MEASURED].notna()
            value = value.where(~rescue, usable[MEASURED])
            source = source.where(~rescue, MEASURED)

        self._record(attribute, skus, usable, value, source)
        return pd.DataFrame({"value": value, "source": source})

    # ── Recording ────────────────────────────────────────────────────────────

    def _record(
        self,
        attribute: str,
        skus: pd.Series,
        usable: Dict[str, pd.Series],
        value: pd.Series,
        source: pd.Series,
    ) -> None:
        res = self.result.resolutions.setdefault(
            attribute, AttributeResolution(attribute=attribute)
        )
        counts = source.value_counts().to_dict()
        for name in AUTHORITY_ORDER:
            res.counts_by_source[name] = res.counts_by_source.get(name, 0) + int(counts.get(name, 0))

        for name in usable:
            if name not in self.result.sources_present:
                self.result.sources_present.append(name)
        self.result.sources_present.sort(key=lambda s: AUTHORITY_ORDER.index(s))

        for other_name, other in usable.items():
            comparable = value.notna() & other.notna() & (source != other_name)
            res.compared += int(comparable.sum())
            if not comparable.any():
                continue

            denominator = value.abs().where(value.abs() > 0, np.nan)
            gap = (other - value) / denominator
            flagged = comparable & gap.abs().gt(self.tolerance)

            for idx in value.index[flagged]:
                res.disagreements.append(Disagreement(
                    attribute=attribute,
                    sku=str(skus.loc[idx]),
                    chosen_source=str(source.loc[idx]),
                    chosen_value=float(value.loc[idx]),
                    other_source=other_name,
                    other_value=float(other.loc[idx]),
                    relative_gap=float(gap.loc[idx]),
                ))
