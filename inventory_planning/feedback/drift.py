"""
Lead-time drift across monthly snapshots.

Every planning run measures lead time afresh from recent receipts and plans on the
result. That is correct, and it is also why drift is invisible: each run silently
absorbs the new number and produces an internally consistent plan around it. A
supplier sliding from 45 days to 62 over four months never triggers anything — the
reorder point, the safety stock and the exposure period all move with it, and nobody
is told that the supplier changed.

This module reads the lead time recorded in successive snapshots and reports the
movement. It is the reason those fields are persisted at all; a snapshot field nothing
reads back is a file that only looks like a record.

## Two things that are not the same

  drift          the measured lead time itself moved — the supplier is behaving
                 differently, and that is a supplier conversation
  source change  last month's figure came from an item master and this month's is
                 measured (or the reverse) — nothing about the supplier changed, only
                 what the pipeline knew

Reporting the second as drift would manufacture a supplier problem out of a data
improvement, so they are separated. Only measurement-to-measurement comparisons count
as drift.

## What counts as material

Both a relative and an absolute floor have to be cleared. A relative threshold alone
flags 2 days moving to 4; an absolute one alone misses a 90-day lead time moving to
110, which is the more expensive event. Requiring both keeps the list short enough to
read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# A move must clear both to be reported.
DEFAULT_RELATIVE_THRESHOLD = 0.20      # 20% of the earlier value
DEFAULT_ABSOLUTE_THRESHOLD = 5.0       # and at least 5 days

MEASURED = "measured"


@dataclass
class LeadTimeMove:
    """One SKU's lead time, then and now."""

    sku: str
    supplier: Optional[str]
    first_month: str
    first_lt: float
    latest_month: str
    latest_lt: float
    latest_samples: Optional[int]
    months_observed: int

    @property
    def change_days(self) -> float:
        return self.latest_lt - self.first_lt

    @property
    def change_pct(self) -> float:
        return self.change_days / self.first_lt if self.first_lt else 0.0

    @property
    def direction(self) -> str:
        return "lengthening" if self.change_days > 0 else "shortening"

    def __str__(self) -> str:
        supplier = f" [{self.supplier}]" if self.supplier else ""
        samples = f", {self.latest_samples} receipts" if self.latest_samples else ""
        return (
            f"{self.sku:<18}{self.first_lt:>6.0f}d ({self.first_month}) -> "
            f"{self.latest_lt:>6.0f}d ({self.latest_month})  "
            f"{self.change_days:+.0f}d / {self.change_pct:+.0%}{supplier}{samples}"
        )


@dataclass
class SourceChange:
    """The lead time came from somewhere else this month — not a supplier event."""

    sku: str
    first_month: str
    first_source: str
    latest_month: str
    latest_source: str
    first_lt: float
    latest_lt: float

    def __str__(self) -> str:
        return (
            f"{self.sku:<18}{self.first_source} {self.first_lt:.0f}d "
            f"({self.first_month}) -> {self.latest_source} {self.latest_lt:.0f}d "
            f"({self.latest_month})"
        )


@dataclass
class DriftResult:
    """Lead-time movement across every snapshot that recorded it."""

    moves: List[LeadTimeMove] = dc_field(default_factory=list)
    source_changes: List[SourceChange] = dc_field(default_factory=list)
    months: List[str] = dc_field(default_factory=list)
    skus_tracked: int = 0
    reason: str = ""

    @property
    def lengthening(self) -> List[LeadTimeMove]:
        return [m for m in self.moves if m.change_days > 0]

    @property
    def shortening(self) -> List[LeadTimeMove]:
        return [m for m in self.moves if m.change_days < 0]

    def frame(self) -> pd.DataFrame:
        rows = [
            {
                "sku": m.sku,
                "supplier": m.supplier,
                "first_month": m.first_month,
                "first_lt_days": m.first_lt,
                "latest_month": m.latest_month,
                "latest_lt_days": m.latest_lt,
                "change_days": round(m.change_days, 1),
                "change_pct": round(m.change_pct, 3),
                "direction": m.direction,
                "latest_samples": m.latest_samples,
                "months_observed": m.months_observed,
            }
            for m in self.moves
        ]
        return pd.DataFrame(rows, columns=[
            "sku", "supplier", "first_month", "first_lt_days", "latest_month",
            "latest_lt_days", "change_days", "change_pct", "direction",
            "latest_samples", "months_observed",
        ])

    def summary(self, top: int = 10) -> str:
        if self.reason:
            return f"  Lead-time drift: {self.reason}"

        span = f"{self.months[0]} -> {self.months[-1]}" if len(self.months) > 1 else self.months[0]
        lines = [
            "  Lead-time drift",
            "  " + "-" * 58,
            f"    {len(self.months)} snapshots ({span}), {self.skus_tracked:,} SKUs tracked",
        ]

        if not self.moves and not self.source_changes:
            lines.append("    No lead time moved beyond the reporting threshold.")
            return "\n".join(lines)

        if self.moves:
            longer, shorter = self.lengthening, self.shortening
            lines.append(
                f"    {len(self.moves):,} moved materially — "
                f"{len(longer):,} lengthening, {len(shorter):,} shortening"
            )
            if longer:
                lines.append("")
                lines.append("    Lengthening — reorder points and safety stock are chasing this:")
                for move in sorted(longer, key=lambda m: -m.change_days)[:top]:
                    lines.append(f"      {move}")
            if shorter:
                lines.append("")
                lines.append("    Shortening — stock held against the old lead time is now surplus:")
                for move in sorted(shorter, key=lambda m: m.change_days)[:top]:
                    lines.append(f"      {move}")

        if self.source_changes:
            lines.append("")
            lines.append(
                f"    {len(self.source_changes):,} SKUs changed lead-time *source*, not "
                f"lead time. Nothing happened at the supplier — only what the pipeline "
                f"knew about it:"
            )
            for change in self.source_changes[:top]:
                lines.append(f"      {change}")
        return "\n".join(lines)


class LeadTimeDriftTracker:
    """Reads the lead time recorded in each monthly snapshot and compares them."""

    def __init__(
        self,
        relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD,
        absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
    ):
        self.relative_threshold = float(relative_threshold)
        self.absolute_threshold = float(absolute_threshold)

    def track(self, history_dir: Union[str, Path]) -> DriftResult:
        """
        `history_dir` is `output/history/` — one folder per month, snapshots inside.

        Unlike the loss calculation this does not need actuals: a lead time is recorded
        at plan time, so drift can be read the moment a second month exists.
        """
        history_dir = Path(history_dir)
        if not history_dir.exists():
            return DriftResult(reason=f"no history at {history_dir}")

        by_month = self._read_months(history_dir)
        if len(by_month) < 2:
            return DriftResult(
                months=sorted(by_month),
                reason=(
                    "only one snapshot recorded a lead time so far — drift needs a "
                    "second month to compare against"
                ),
            )

        months = sorted(by_month)
        result = DriftResult(months=months)
        tracked = set()

        for sku in sorted({s for m in months for s in by_month[m]}):
            observations = [(m, by_month[m][sku]) for m in months if sku in by_month[m]]
            if len(observations) < 2:
                continue
            tracked.add(sku)
            self._compare(sku, observations, result)

        result.skus_tracked = len(tracked)
        return result

    # ── Reading ──────────────────────────────────────────────────────────────

    @staticmethod
    def _read_months(history_dir: Path) -> Dict[str, Dict[str, dict]]:
        """
        month -> sku -> the lead-time fields recorded that month.

        Where a month holds several runs the newest is taken, on the grounds that a
        re-run replaces rather than supplements the one before it.
        """
        out: Dict[str, Dict[str, dict]] = {}
        for month_dir in sorted(p for p in history_dir.iterdir() if p.is_dir()):
            for snap_file in sorted(month_dir.glob("snapshot_*.json"), reverse=True):
                try:
                    snap = json.loads(snap_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                entries = {
                    sku: data for sku, data in (snap.get("skus") or {}).items()
                    if data.get("lead_time_days") is not None
                }
                if entries:
                    out[month_dir.name] = entries
                    break   # newest run in the month wins
        return out

    # ── Comparing ────────────────────────────────────────────────────────────

    def _compare(self, sku: str, observations: List[tuple], result: DriftResult) -> None:
        (first_month, first), (latest_month, latest) = observations[0], observations[-1]
        first_lt = float(first["lead_time_days"])
        latest_lt = float(latest["lead_time_days"])
        first_source = first.get("lt_source") or MEASURED
        latest_source = latest.get("lt_source") or MEASURED

        if first_source != latest_source:
            # The number moved because the pipeline learned something, not because the
            # supplier did. Calling that drift would invent a supplier problem.
            result.source_changes.append(SourceChange(
                sku=sku, first_month=first_month, first_source=first_source,
                latest_month=latest_month, latest_source=latest_source,
                first_lt=first_lt, latest_lt=latest_lt,
            ))
            return

        if first_source != MEASURED:
            # Two stated values in a row: someone edited a master. Real, but it is a
            # master-data change, and the cross-check at plan time already reports the
            # gap between a stated lead time and the receipts.
            return

        change = latest_lt - first_lt
        relative = abs(change) / first_lt if first_lt else 0.0
        if abs(change) < self.absolute_threshold or relative < self.relative_threshold:
            return

        result.moves.append(LeadTimeMove(
            sku=sku,
            supplier=latest.get("supplier"),
            first_month=first_month,
            first_lt=first_lt,
            latest_month=latest_month,
            latest_lt=latest_lt,
            latest_samples=latest.get("lt_samples"),
            months_observed=len(observations),
        ))
