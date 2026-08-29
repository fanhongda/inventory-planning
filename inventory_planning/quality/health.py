"""
One statement of what this run can and cannot be relied on for.

The findings exist already — the gates raise them, the capability plan lists its own
degradations — and they are printed as they occur, which is the wrong moment for all
but the first of them. By the time a planner reaches the purchase recommendations,
sixty lines of intake output have scrolled past, and the one sentence that would have
changed how they read the revenue table went by before there was a revenue table to
read.

So they are collected and restated at the end, ranked, with the affected outputs named.
The ranking is the whole point of the document: a run typically has a dozen things
worth saying and two that change what a reader should do, and a list that does not
separate them is a list that gets skimmed.

The impacts are written as *what the figure will look like*, never as "this is
affected". "Revenue by product line groups on the part-number prefix — do not present
these" is actionable in a way that "product family data quality issue" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

from .gates import BLOCK, SEVERE, WARN, Finding, GateReport


@dataclass
class RunHealth:
    """Every reservation attached to a run, in one place, worst first."""

    run_id: str
    gates: List[GateReport] = dc_field(default_factory=list)
    # From the capability plan: the specific consequences of an input nobody supplied.
    # A different kind of statement from a gate finding — nothing is *wrong*, something
    # is absent — but identical in what a reader needs it for, so they are reported
    # together rather than in two places neither of which is complete.
    degradations: List[str] = dc_field(default_factory=list)
    allow_degraded: bool = False

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def findings(self) -> List[Finding]:
        return [f for g in self.gates for f in g.findings]

    @property
    def critical(self) -> List[Finding]:
        """
        What a reader has to know before they read anything else.

        Blocking findings appear here only when the run was forced past them: an
        unoverridden block never produced an output for this document to describe.
        """
        out = [f for f in self.findings if f.severity == SEVERE]
        if self.allow_degraded:
            out = [f for f in self.findings if f.severity == BLOCK] + out
        return out

    @property
    def noted(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    @property
    def clean(self) -> bool:
        return not self.critical and not self.noted and not self.degradations

    @property
    def affected_outputs(self) -> List[str]:
        """Every impact line from every critical finding, de-duplicated, in order."""
        seen: Dict[str, None] = {}
        for finding in self.critical:
            for impact in finding.impacts:
                seen.setdefault(impact, None)
        return list(seen)

    # ── Reporting ────────────────────────────────────────────────────────────

    def console(self) -> str:
        """The block printed at the end of a run, after the recommendations."""
        if self.clean:
            return ("\n" + "=" * 60 + "\n  RUN HEALTH\n" + "=" * 60 +
                    "\n  Every quality gate passed and no input was missing. Nothing "
                    "in this\n  report rests on a fallback.")

        lines = ["", "=" * 60, "  RUN HEALTH — read before acting on the numbers above",
                 "=" * 60]

        if self.critical:
            lines.append("")
            lines.append(f"  \033[91m*** {len(self.critical)} SEVERE — specific figures "
                         f"in this report are not what they appear ***\033[0m")
            for finding in self.critical:
                lines.append("")
                lines.append(f"  {finding.mark} {finding.what}")
                lines.append(f"      {finding.why}")
                if finding.impacts:
                    lines.append("      Affected:")
                    for impact in finding.impacts:
                        lines.append(f"        • {impact}")
                lines.append(f"      Fix: {finding.fix}")

        if self.noted:
            lines.append("")
            lines.append(f"  {len(self.noted)} noted — applied or reported, the run "
                         f"stands:")
            for finding in self.noted:
                lines.append(f"    ⚠ {finding.what}")

        if self.degradations:
            lines.append("")
            lines.append("  What this run could not measure — an input nobody supplied:")
            for item in self.degradations:
                lines.append(f"    ○ {item}")

        if self.allow_degraded and any(f.severity == BLOCK for f in self.findings):
            lines.append("")
            lines.append("  \033[91mThis run was forced past a blocking gate "
                         "(allow_degraded). The findings above\n  marked ✗ stopped it "
                         "for a reason.\033[0m")
        return "\n".join(lines)

    def markdown(self) -> str:
        """The same thing as a file, for whoever reads the outputs rather than the run."""
        lines = [f"# Run health — {self.run_id}", ""]
        if self.clean:
            lines.append("Every quality gate passed and no input was missing. "
                         "Nothing in this report rests on a fallback.")
            return "\n".join(lines) + "\n"

        if self.critical:
            lines += [
                "## Do not rely on these",
                "",
                "Specific figures in this run's outputs are not what they appear to be. "
                "Each entry names what is wrong, what it does to the numbers, and which "
                "outputs it reaches.",
                "",
            ]
            for finding in self.critical:
                tag = "BLOCKING, OVERRIDDEN" if finding.severity == BLOCK else "SEVERE"
                lines += [f"### {tag} · {finding.check}", "",
                          f"**{finding.what}**", "", finding.why, ""]
                if finding.impacts:
                    # A list, not a table. Splitting each impact on its first colon to
                    # make two columns worked for the ones written as "output: effect"
                    # and mangled the ones that were not, which is a formatter deciding
                    # how a sentence is shaped. The visual report can lay these out
                    # however it likes from the JSON; here they are sentences.
                    lines.append("**What this reaches:**")
                    lines.append("")
                    for impact in finding.impacts:
                        lines.append(f"- {impact}")
                    lines.append("")
                lines += [f"**Fix:** {finding.fix}", ""]

        if self.noted:
            lines += ["## Noted", "",
                      "Reported or repaired; the numbers stand.", ""]
            for finding in self.noted:
                lines.append(f"- **{finding.check}** — {finding.what} {finding.fix}")
            lines.append("")

        if self.degradations:
            lines += ["## What this run could not measure", "",
                      "Not errors — inputs nobody supplied, and the specific "
                      "consequence of each.", ""]
            for item in self.degradations:
                lines.append(f"- {item}")
            lines.append("")

        passed = [g.stage for g in self.gates if g.passed and not g.findings]
        if passed:
            lines += ["## Gates that found nothing", "",
                      ", ".join(passed), ""]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "allow_degraded": self.allow_degraded,
            "clean": self.clean,
            "critical": [f.to_dict() for f in self.critical],
            "noted": [f.to_dict() for f in self.noted],
            "degradations": list(self.degradations),
            "affected_outputs": self.affected_outputs,
            "gates": [g.to_dict() for g in self.gates],
        }


def assess(run_id: str, gates: List[GateReport], plan=None,
           allow_degraded: bool = False) -> RunHealth:
    """Collect a run's reservations from the gates and the capability plan."""
    return RunHealth(
        run_id=run_id,
        gates=list(gates or []),
        degradations=list(getattr(plan, "degradations", []) or []),
        allow_degraded=allow_degraded,
    )
