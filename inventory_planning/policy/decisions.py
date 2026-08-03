"""
Decision log — closing the loop between recommendation and what the planner did.

The valuable half is the **rejections**, not the acceptances.

An accepted recommendation confirms the model was right about something it already
modelled. A rejected one says the model is missing something — and that something is
usually a constraint that exists nowhere in the ERP: a supplier who will not reschedule,
a customer contract with a minimum stock clause, a quality hold, an allocation
agreement. None of it is in any extract, all of it lives in the planner's head.

So rejections are recorded with a reason, and repeated reasons are surfaced as
**constraint candidates** — patterns strong enough that the model should probably
learn them rather than keep proposing the same rejected action every month.

The log is JSONL: append-only, greppable, diffable, and safe to have several runs
writing to it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field as dc_field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ACCEPTED = "accepted"
REJECTED = "rejected"
DEFERRED = "deferred"
MODIFIED = "modified"

VALID_DECISIONS = {ACCEPTED, REJECTED, DEFERRED, MODIFIED}

# Reason codes keep the pattern analysis meaningful — free text alone cannot be
# aggregated. `other` is always available and its free text is what tells us when a
# new code is needed.
REASON_CODES = {
    "supplier_wont_reschedule": "Supplier will not accept a date change",
    "moq_constraint": "Minimum order quantity prevents the proposed lot size",
    "contract_minimum": "Customer or contract requires a minimum stock level",
    "quality_hold": "Stock is on hold and cannot be counted or consumed",
    "allocation": "Stock is allocated to a specific customer or project",
    "demand_known_wrong": "Planner knows demand will differ from the forecast",
    "npi_or_eol": "Item is in launch or end-of-life; normal policy does not apply",
    "freight_consolidation": "Would break container or milk-run consolidation",
    "already_actioned": "Action was already taken outside the system",
    "disagree_with_model": "Planner disagrees with the calculation",
    "other": "Other — see note",
}


@dataclass
class Decision:
    """One planner verdict on one recommendation."""

    decision_id: str
    run_id: str
    recorded_at: str
    action_kind: str
    action_label: str
    sku: Optional[str]
    decision: str
    reason_code: Optional[str] = None
    note: str = ""
    planner: str = ""
    value_at_stake: float = 0.0

    def __post_init__(self):
        if self.decision not in VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {sorted(VALID_DECISIONS)}, got {self.decision!r}"
            )
        if self.decision in (REJECTED, DEFERRED, MODIFIED) and not self.reason_code:
            raise ValueError(
                f"A {self.decision} decision needs a reason_code — the reason is the "
                f"only part of a rejection the model can learn from. "
                f"Options: {sorted(REASON_CODES)}"
            )
        if self.reason_code and self.reason_code not in REASON_CODES:
            raise ValueError(
                f"Unknown reason_code {self.reason_code!r}. Use one of "
                f"{sorted(REASON_CODES)}, or 'other' with a note."
            )

    @classmethod
    def create(cls, run_id: str, action_kind: str, action_label: str, decision: str,
               sku: str = None, reason_code: str = None, note: str = "",
               planner: str = "", value_at_stake: float = 0.0) -> "Decision":
        return cls(
            decision_id=uuid.uuid4().hex[:12],
            run_id=run_id,
            recorded_at=datetime.now().isoformat(timespec="seconds"),
            action_kind=action_kind,
            action_label=action_label,
            sku=sku,
            decision=decision,
            reason_code=reason_code,
            note=note,
            planner=planner,
            value_at_stake=float(value_at_stake),
        )


@dataclass
class ConstraintCandidate:
    """A rejection pattern strong enough that the model should probably encode it."""

    scope: str                  # e.g. "supplier == 'V100'"
    reason_code: str
    action_kind: str
    occurrences: int
    distinct_skus: int
    value_at_stake: float
    example_notes: List[str] = dc_field(default_factory=list)

    @property
    def suggested_rule(self) -> str:
        """A planning_parameters.md rule stub the planner can paste and edit."""
        return (
            f"### R-XXX · {REASON_CODES.get(self.reason_code, self.reason_code)}\n"
            f"```yaml\n"
            f"scope: {self.scope}\n"
            f"set:\n"
            f"  # parameter to change so this action stops being proposed\n"
            f"rationale: >\n"
            f"  Auto-suggested from the decision log: {self.occurrences} rejections of\n"
            f"  '{self.action_kind}' across {self.distinct_skus} SKUs, all coded\n"
            f"  '{self.reason_code}'. ${self.value_at_stake:,.0f} of recommendations affected.\n"
            f"```"
        )

    def __str__(self) -> str:
        return (f"{self.scope:<34} {self.reason_code:<26} "
                f"{self.occurrences:>3}x on {self.distinct_skus:>3} SKUs "
                f"(${self.value_at_stake:,.0f})")


class DecisionLog:
    """Append-only record of planner verdicts, plus pattern analysis over them."""

    # A pattern needs this many rejections before it is worth proposing as a rule.
    MIN_OCCURRENCES = 3

    def __init__(self, path: Path):
        self.path = Path(path)

    # ── Writing ──────────────────────────────────────────────────────────────

    def record(self, decision: Decision) -> Decision:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
        return decision

    def record_many(self, decisions: List[Decision]) -> int:
        for d in decisions:
            self.record(d)
        return len(decisions)

    # ── Reading ──────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=[f.name for f in Decision.__dataclass_fields__.values()])
        rows = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue      # a truncated write must not lose the rest of the history
        return pd.DataFrame(rows)

    # ── Pattern analysis ─────────────────────────────────────────────────────

    def constraint_candidates(
        self, sku_attributes: pd.DataFrame = None, min_occurrences: int = None
    ) -> List[ConstraintCandidate]:
        """
        Find rejection patterns worth encoding as rules.

        Patterns are sought on SKU attributes rather than on individual SKUs, because a
        rule that names one SKU is a workaround while a rule that names a supplier or a
        product family is a constraint. Attribute columns come from the SKU frame so
        the same vocabulary is available here as in planning_parameters.md.
        """
        min_occurrences = min_occurrences or self.MIN_OCCURRENCES
        log = self.load()
        if log.empty:
            return []

        rejected = log[log["decision"].isin([REJECTED, DEFERRED])].copy()
        if rejected.empty:
            return []

        if sku_attributes is not None and "sku" in sku_attributes.columns:
            rejected = rejected.merge(sku_attributes, on="sku", how="left")

        candidates: List[ConstraintCandidate] = []
        group_cols = [c for c in ("supplier", "product_family", "abc_class", "incoterm")
                      if c in rejected.columns]

        for attr in group_cols:
            grouped = rejected.dropna(subset=[attr]).groupby(
                [attr, "reason_code", "action_kind"], dropna=True
            )
            for (value, reason, kind), grp in grouped:
                if len(grp) < min_occurrences:
                    continue
                candidates.append(ConstraintCandidate(
                    scope=f'{attr} == "{value}"',
                    reason_code=str(reason),
                    action_kind=str(kind),
                    occurrences=len(grp),
                    distinct_skus=int(grp["sku"].nunique()),
                    value_at_stake=float(grp["value_at_stake"].sum()),
                    example_notes=[n for n in grp["note"].dropna().unique()[:3] if n],
                ))

        # A reason that recurs across every attribute is a global gap, not a segment one.
        for (reason, kind), grp in rejected.groupby(["reason_code", "action_kind"]):
            if len(grp) < min_occurrences * 2:
                continue
            candidates.append(ConstraintCandidate(
                scope="(all SKUs)",
                reason_code=str(reason),
                action_kind=str(kind),
                occurrences=len(grp),
                distinct_skus=int(grp["sku"].nunique()),
                value_at_stake=float(grp["value_at_stake"].sum()),
                example_notes=[n for n in grp["note"].dropna().unique()[:3] if n],
            ))

        return sorted(candidates, key=lambda c: (-c.occurrences, -c.value_at_stake))

    def acceptance_rate(self, by: str = "action_kind") -> pd.DataFrame:
        """
        How often each kind of recommendation is taken.

        A persistently low rate is a signal about the recommendation, not the planner —
        it means the pipeline keeps proposing something that cannot be done.
        """
        log = self.load()
        if log.empty or by not in log.columns:
            return pd.DataFrame()

        summary = (
            log.assign(accepted=log["decision"].eq(ACCEPTED).astype(int))
            .groupby(by)
            .agg(proposed=("decision", "size"),
                 accepted=("accepted", "sum"),
                 value_at_stake=("value_at_stake", "sum"))
            .reset_index()
        )
        summary["acceptance_rate"] = summary["accepted"] / summary["proposed"]
        return summary.sort_values("acceptance_rate")

    def summary(self, sku_attributes: pd.DataFrame = None) -> str:
        log = self.load()
        if log.empty:
            return ("  Decision log empty — no planner feedback recorded yet.\n"
                    "  Record acceptances and rejections to let the model learn the "
                    "constraints that are not in any ERP extract.")

        lines = [
            "  Decision log",
            "  " + "-" * 58,
            f"    {len(log)} decisions across {log['run_id'].nunique()} runs",
        ]
        counts = log["decision"].value_counts()
        lines.append("    " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

        rates = self.acceptance_rate()
        if not rates.empty:
            lines.append("")
            lines.append("    Acceptance by action:")
            for _, row in rates.iterrows():
                flag = "  ← rarely actionable" if row["acceptance_rate"] < 0.35 else ""
                lines.append(f"      {row['action_kind']:<32}"
                             f"{row['accepted']:>3}/{row['proposed']:<3} "
                             f"{row['acceptance_rate']:>5.0%}{flag}")

        candidates = self.constraint_candidates(sku_attributes)
        if candidates:
            lines.append("")
            lines.append("    Constraint candidates — the model is missing these:")
            for c in candidates[:6]:
                lines.append(f"      {c}")
            lines.append("")
            lines.append("    Add the strongest as a rule in planning_parameters.md; "
                         "call .suggested_rule for a stub.")
            if len({c.reason_code for c in candidates[:6]}) < len(candidates[:6]):
                lines.append("    ⚠ Attributes above may be confounded — if every V100 item "
                             "is also a valve, both patterns appear. Pick the one that is "
                             "the actual cause.")
        return "\n".join(lines)


def decisions_from_frontier(run_id: str, frontier, verdicts: Dict[str, Any]) -> List[Decision]:
    """
    Build decision records from a planner's verdicts on an action frontier.

    `verdicts` maps action kind to either a decision string, or a dict with
    `decision`, `reason_code`, `note`. Actions with no verdict are left unrecorded
    rather than assumed accepted — silence is not agreement.
    """
    out: List[Decision] = []
    for action in list(frontier.actions) + list(getattr(frontier, "too_slow", [])):
        verdict = verdicts.get(action.kind)
        if verdict is None:
            continue
        if isinstance(verdict, str):
            verdict = {"decision": verdict}
        out.append(Decision.create(
            run_id=run_id,
            action_kind=action.kind,
            action_label=action.label,
            decision=verdict["decision"],
            sku=verdict.get("sku"),
            reason_code=verdict.get("reason_code"),
            note=verdict.get("note", ""),
            planner=verdict.get("planner", ""),
            value_at_stake=action.value_released,
        ))
    return out
