"""
Parameter suggestions — what the data says the policy parameters should be.

`planning_parameters.md` holds the parameters a planner has *decided*. This module
produces the ones the data *supports*, so the two can be compared. It answers, per
SKU: should this be stocked at all, what is the lead time actually measured at, how
much safety stock does that lead time and this forecast error justify, how often is
it worth placing an order, and by which replenishment method.

Two outputs, because they have different readers:

  parameter_suggestions_<ts>.csv   one row per SKU, every suggested value next to the
                                   value in force, with the evidence behind it — the
                                   lead-time sample size, the sigma source, the demand
                                   pattern. A planner reviews this.

  suggested_rules_<ts>.md          the same conclusions expressed as scoped rules in
                                   exactly the format `planning_parameters.md` uses.
                                   Paste the blocks in, edit the rationale, done.

The rules file is not reverse-engineered from the per-SKU values. The suggestion
engine *is* a rule set — declared below in the same scope language the parameter file
uses, evaluated with the same expression evaluator, and validated by constructing the
same `Rule` objects the parser produces. What gets written out is therefore
guaranteed to parse, and to mean when re-read exactly what it meant here.

What this deliberately does not do is write to `planning_parameters.md`. A parameter
that changed because a script decided it should is a parameter nobody can defend in a
review. Every suggestion carries its reasoning and waits for a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .parameters import ParameterSet, Rule
from .should_be import ShouldBeCalculator, _z_from_service_level

DAYS_PER_MONTH = 30.0
DAYS_PER_YEAR = 365.0

# Review periods worth proposing. A continuous scale is false precision — nobody runs
# a 23-day review — and a short list keeps the resulting rule set readable.
REVIEW_LADDER = (7, 14, 30, 60, 90)


@dataclass
class SuggestionRule:
    """One scoped parameter proposal, in the parameter file's own grammar."""

    rule_id: str
    name: str
    scope: str
    overrides: Dict[str, Any]
    rationale: str

    def as_rule(self) -> Rule:
        """Build the parser's own `Rule` — validates the scope and the rationale."""
        return Rule(rule_id=self.rule_id, name=self.name, scope=self.scope,
                    overrides=dict(self.overrides), rationale=self.rationale,
                    owner="suggested", date=str(date.today()))


# ── The suggestion rule set ──────────────────────────────────────────────────
#
# Ordered, later winning, exactly like the parameter file. Read it top to bottom as
# "general policy, then the exceptions".

SUGGESTION_RULES: List[SuggestionRule] = [
    SuggestionRule(
        rule_id="S-001",
        name="Service level follows ABC",
        scope='abc_class in ["A", "B", "C"]',
        overrides={"service_level": 0.95},
        rationale=(
            "The baseline. Holding cost scales with value and stockout cost does not, "
            "so the class that ties up the most capital is the one whose service level "
            "has to be argued for rather than assumed."
        ),
    ),
    SuggestionRule(
        rule_id="S-002",
        name="A class buys the higher service level",
        scope='abc_class == "A"',
        overrides={"service_level": 0.98},
        rationale=(
            "A items are the ones customers notice missing. The extra safety stock is "
            "the cheapest insurance available at this margin, because the same z-step "
            "costs the same square-root of exposure whatever the class."
        ),
    ),
    SuggestionRule(
        rule_id="S-003",
        name="C class relaxes service level",
        scope='abc_class == "C"',
        overrides={"service_level": 0.90},
        rationale=(
            "The last few percent of service on a C item costs more in review effort "
            "and dead stock than the misses cost in revenue. Buy the coverage, not the "
            "decimal places."
        ),
    ),
    SuggestionRule(
        rule_id="S-004",
        name="Erratic demand goes to continuous review",
        scope='demand_pattern in ["erratic", "lumpy"] and stocking_class != "non-stocking"',
        overrides={"replenishment_method": "reorder_point"},
        rationale=(
            "Demand arrives in unpredictable bursts. A periodic review can miss a burst "
            "by almost a full cycle, and the safety stock needed to cover that gap is "
            "larger than the cost of watching the position continuously."
        ),
    ),
    SuggestionRule(
        rule_id="S-005",
        name="Low-value C items run on min-max",
        scope='abc_class == "C" and unit_cost < 20 and stocking_class != "non-stocking"',
        overrides={"replenishment_method": "min_max", "review_period_days": 90},
        rationale=(
            "Holding cost is far below the cost of managing these by hand. Let the "
            "lot size do the work and only look at exceptions."
        ),
    ),
    SuggestionRule(
        rule_id="S-006",
        name="Long lead times do not benefit from frequent review",
        scope="lead_time_days > 90",
        overrides={"review_period_days": 30},
        rationale=(
            "Safety stock scales with sqrt(R + LT). Once LT dominates R, shortening the "
            "review barely moves the stock and only adds order lines. The lever for "
            "these items is lead-time reliability, not review frequency."
        ),
    ),
    SuggestionRule(
        rule_id="S-007",
        name="Unreliable lead times need the review, not more stock",
        scope="lt_sigma_days > 0.5 * lead_time_days and lead_time_days > 0",
        overrides={"review_period_days": 14},
        rationale=(
            "Lead-time variability enters safety stock multiplied by mean demand, so "
            "it is the expensive term. Until the supplier is fixed, seeing the position "
            "sooner is cheaper than covering the spread with stock."
        ),
    ),
    SuggestionRule(
        rule_id="S-008",
        name="Order-on-demand items hold no policy stock",
        scope='stocking_class == "non-stocking"',
        overrides={"replenishment_method": "make_to_order", "service_level": 0.0},
        rationale=(
            "Demand appears in too few cycles to forecast, so any stock held is a bet "
            "on a specific order rather than on a rate. These are bought against firm "
            "backlog; a service level implies a stock policy that does not exist. Last "
            "in the list deliberately — it is the strongest exception and must win over "
            "the ABC service levels above."
        ),
    ),
]


@dataclass
class SuggestionResult:
    """Per-SKU suggested parameters, the evidence, and the rule set that produced them."""

    frame: pd.DataFrame
    rules: List[SuggestionRule]
    hits: Dict[str, int] = dc_field(default_factory=dict)
    notes: List[str] = dc_field(default_factory=list)

    # ── The gap that matters ─────────────────────────────────────────────────

    @property
    def changed(self) -> pd.DataFrame:
        """SKUs where at least one suggested parameter differs from the one in force."""
        return self.frame[self.frame["changes"] != ""]

    @property
    def safety_stock_delta_value(self) -> float:
        """Capital released (negative) or required (positive) by the suggested SS."""
        if "ss_delta_value" not in self.frame.columns:
            return 0.0
        return float(self.frame["ss_delta_value"].sum())

    def summary(self) -> str:
        f = self.frame
        lines = [
            "  Parameter suggestions",
            "  " + "-" * 58,
            f"    {len(f):,} SKUs assessed · {len(self.changed):,} would change under the "
            f"suggested rules",
        ]

        unmeasured_lt = int((f["lt_samples"].fillna(0) == 0).sum())
        thin_lt = int(f["lt_samples"].fillna(0).between(1, 2).sum())
        if unmeasured_lt:
            lines.append(
                f"    ⚠ {unmeasured_lt:,} SKUs have no PO history — lead time is unmeasured, "
                f"and their safety stock is a placeholder, not a calculation"
            )
        if thin_lt:
            lines.append(
                f"    ⚠ {thin_lt:,} SKUs rest on 1–2 lead-time observations — the sigma is "
                f"not yet a distribution"
            )

        delta = self.safety_stock_delta_value
        if abs(delta) > 1:
            verb = "more" if delta > 0 else "less"
            lines.append(
                f"    Safety stock under the suggested parameters: ${abs(delta):,.0f} {verb} "
                f"than the policy in force"
            )

        if "ss_verdict" in f.columns:
            lines.append("")
            lines.append("    Against the safety stock the planner set by hand:")
            counts = f["ss_verdict"].value_counts()
            for verdict in ("planner holds more than justified", "in line",
                            "planner holds less than justified"):
                n = int(counts.get(verdict, 0))
                if n:
                    lines.append(f"      {verdict:<40}{n:>6} SKUs")
            worst = f[f["ss_verdict"] == "planner holds more than justified"]
            worst = worst.nsmallest(5, "ss_vs_planner_value")
            if len(worst):
                lines.append("")
                lines.append("      Largest overstock against the justified level:")
                for _, row in worst.iterrows():
                    lines.append(
                        f"        {str(row['sku']):<18} set {row['planner_safety_stock']:>9,.0f}  "
                        f"justified {row['suggested_safety_stock']:>9,.0f}  "
                        f"${abs(row['ss_vs_planner_value']):>11,.0f} tied up"
                    )

        lines.append("")
        lines.append("    Rules that hit:")
        for rule in self.rules:
            n = self.hits.get(rule.rule_id, 0)
            flag = "" if n else "   (matched nothing — scope may not apply to this data)"
            lines.append(f"      {rule.rule_id:<8} {n:>5} SKUs  {rule.name}{flag}")

        for note in self.notes:
            lines.append(f"    · {note}")
        return "\n".join(lines)

    # ── Outputs ──────────────────────────────────────────────────────────────

    def to_csv(self, path: Path) -> Path:
        self.frame.to_csv(path, index=False)
        return Path(path)

    def to_rules_markdown(self, path: Path = None) -> str:
        """
        The suggestions as rule blocks in `planning_parameters.md` format.

        Written as a separate file rather than appended to the parameter file: a rule
        arrives with a rationale written by a machine that has seen the numbers but not
        the business, and that distinction should survive until someone signs off on it.
        """
        header = [
            "# Suggested planning parameters",
            "",
            f"Generated {date.today()} from the current run. **Nothing here is in force.**",
            "",
            "Each block below is valid `planning_parameters.md` syntax. To adopt one,",
            "paste it into the `覆盖规则 (Rules)` section of `config/planning_parameters.md`,",
            "renumber the `rule_id` into the `R-xxx` series, replace `owner: suggested`",
            "with your own name, and rewrite the rationale in your own words — a rule",
            "whose reasoning you cannot defend in a review is a rule you cannot keep.",
            "",
            "Rules apply in order, later winning, so keep the ordering when pasting.",
            "",
            "---",
            "",
            "## 建议规则 (Suggested rules)",
            "",
            "",
        ]

        blocks = []
        for rule in self.rules:
            n = self.hits.get(rule.rule_id, 0)
            sample = self._sample_skus(rule.rule_id)
            blocks.append(
                f"### {rule.rule_id} · {rule.name}\n\n"
                f"```yaml\n"
                f"scope: {rule.scope}\n"
                f"set:\n"
                + "".join(f"  {k}: {_yaml_scalar(v)}\n" for k, v in rule.overrides.items())
                + f"rationale: >\n  {rule.rationale}\n"
                f"owner: suggested\n"
                f"date: {date.today()}\n"
                f"```\n\n"
                f"Matched **{n:,} SKUs** in this run"
                + (f" (e.g. {', '.join(sample)})" if sample else "")
                + ".\n"
            )

        text = "\n".join(header) + "\n".join(blocks) + self._per_sku_appendix()
        if path is not None:
            Path(path).write_text(text)
        return text

    def _sample_skus(self, rule_id: str, n: int = 3) -> List[str]:
        if "applied_suggestions" not in self.frame.columns:
            return []
        hit = self.frame[self.frame["applied_suggestions"].str.contains(rule_id, na=False)]
        return hit["sku"].astype(str).head(n).tolist()

    def _per_sku_appendix(self) -> str:
        """The values that are computed per SKU and cannot be expressed as a rule."""
        f = self.frame
        unmeasured = int((f["lt_samples"].fillna(0) == 0).sum())
        return (
            "\n---\n\n"
            "## 逐 SKU 数值 (Per-SKU values)\n\n"
            "Lead time, safety stock and EOQ are measured or computed per SKU, not set "
            "by rule — they are in `parameter_suggestions_<ts>.csv` alongside the "
            "evidence for each one:\n\n"
            "| Column | What it is |\n"
            "|---|---|\n"
            "| `lead_time_days`, `lt_sigma_days`, `lt_samples` | Measured from PO receipts. "
            "`lt_samples` is the honest confidence signal — a sigma from two receipts is "
            "not a distribution. |\n"
            "| `suggested_safety_stock` | Recomputed at the suggested service level with "
            "exposure R + LT, using the same formula as should-be inventory. |\n"
            "| `sigma_source` | `forecast_rmse` where a forecast exists, `demand_std` "
            "otherwise. Demand std overstates safety stock by counting variation the "
            "forecast successfully predicted. |\n"
            "| `eoq_qty`, `eoq_review_days` | Economic order quantity and the review "
            "period it implies. `suggested_review_period_days` snaps this to the ladder "
            f"{list(REVIEW_LADDER)} and is then overridden by any rule above. |\n"
            "| `changes` | Which parameters differ from the ones in force, and by how much. |\n"
            "\n"
            + (f"⚠ {unmeasured:,} SKUs have no PO receipt history. Their lead time is "
               f"unmeasured and every number derived from it is a placeholder.\n"
               if unmeasured else "")
        )


class SuggestionBuilder:
    """Derives suggested planning parameters from what the run measured."""

    def __init__(self, config_dir: Path = None, rules: List[SuggestionRule] = None):
        self.config_dir = Path(config_dir) if config_dir else None
        self.rules = list(rules) if rules is not None else list(SUGGESTION_RULES)
        # Constructing the parser's own Rule validates every scope and rationale up
        # front, so a malformed suggestion fails here rather than in the planner's
        # parameter file a week later.
        self._compiled = [r.as_rule() for r in self.rules]

    def build(self, params: ParameterSet) -> SuggestionResult:
        """
        `params` is the resolved ParameterSet — it carries both the SKU attributes and
        the parameters currently in force, which is what the suggestions are compared
        against.
        """
        df = params.frame.copy()
        notes: List[str] = []

        for col, default in (("lead_time_days", np.nan), ("lt_sigma_days", 0.0),
                             ("demand_mean", 0.0), ("demand_sigma", 0.0),
                             ("unit_cost", np.nan)):
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["lt_samples"] = pd.to_numeric(df.get("lt_samples", np.nan), errors="coerce")
        df["lt_cv"] = np.where(
            df["lead_time_days"] > 0, df["lt_sigma_days"] / df["lead_time_days"], np.nan
        ).round(3)

        df = self._economics(df, params, notes)
        df = self._apply_rules(df)
        df = self._suggest_safety_stock(df, params)
        df = self._diff_against_current(df, params)
        df = self._compare_to_planner(df, notes)

        hits = {r.rule_id: int(df[f"_hit_{r.rule_id}"].sum()) for r in self.rules}
        df = df.drop(columns=[c for c in df.columns if c.startswith("_hit_")])

        return SuggestionResult(frame=self._select_columns(df), rules=self.rules,
                                hits=hits, notes=notes)

    # ── Economics: EOQ and the review period it implies ──────────────────────

    def _economics(self, df: pd.DataFrame, params: ParameterSet,
                   notes: List[str]) -> pd.DataFrame:
        """
        EOQ, and the review period whose cycle stock equals it.

        EOQ = sqrt(2 · D · S / (h · c)). Its real use here is not the lot size — it is
        the *ordering frequency* it implies. A review period much shorter than the EOQ
        time supply is paying order costs for no reduction in stock; much longer, and
        cycle stock is carrying the ordering department's convenience.
        """
        order_cost = float(params.defaults.get("order_cost_usd", 350))
        holding_rate = float(params.defaults.get("holding_cost_rate", 0.22))

        annual_demand = (df["demand_mean"].fillna(0.0) * 12).clip(lower=0)
        unit_cost = df["unit_cost"]
        holding_per_unit_year = unit_cost * holding_rate

        with np.errstate(divide="ignore", invalid="ignore"):
            eoq = np.sqrt(2 * annual_demand * order_cost / holding_per_unit_year)
        eoq = pd.Series(eoq, index=df.index).replace([np.inf, -np.inf], np.nan)
        df["eoq_qty"] = eoq.round(0)

        daily = annual_demand / DAYS_PER_YEAR
        df["eoq_review_days"] = (eoq / daily.replace(0, np.nan)).round(0)

        unpriced = int(unit_cost.isna().sum())
        if unpriced:
            notes.append(
                f"{unpriced:,} SKUs have no unit cost — EOQ and the review period it "
                f"implies are undefined for them, so they keep the default review period"
            )

        # Snap to the ladder; SKUs without an EOQ keep whatever is in force.
        current_review = pd.to_numeric(df.get("review_period_days"), errors="coerce")
        df["suggested_review_period_days"] = (
            df["eoq_review_days"].map(_snap_to_ladder).fillna(current_review)
        )
        df["suggested_replenishment_method"] = df.get(
            "replenishment_method", pd.Series("periodic", index=df.index)
        )
        df["suggested_service_level"] = pd.to_numeric(
            df.get("service_level"), errors="coerce"
        )
        return df

    # ── Rules ────────────────────────────────────────────────────────────────

    def _apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the suggestion rules onto the `suggested_*` columns.

        Same semantics as the parameter file: in order, later winning, and a rule whose
        scope needs a column this run does not have matches nothing rather than raising.
        """
        trace = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

        for suggestion, rule in zip(self.rules, self._compiled):
            mask = rule.match(df)
            df[f"_hit_{rule.rule_id}"] = mask
            for key, value in suggestion.overrides.items():
                df.loc[mask, f"suggested_{key}"] = value
            for idx in df.index[mask]:
                trace[idx].append(rule.rule_id)

        df["applied_suggestions"] = trace.apply(lambda ids: ",".join(ids))
        return df

    # ── Safety stock at the suggested parameters ─────────────────────────────

    def _suggest_safety_stock(self, df: pd.DataFrame, params: ParameterSet) -> pd.DataFrame:
        """
        Recompute safety stock under the suggested service level and review period.

        Deliberately routed through the same `ShouldBeCalculator._safety` the should-be
        engine uses. A suggestion computed by a second implementation of the formula is
        a suggestion that will disagree with the report it sits next to.
        """
        proposed = df.copy()
        proposed["service_level"] = df["suggested_service_level"]
        proposed["review_period_days"] = df["suggested_review_period_days"]
        proposed["replenishment_method"] = df["suggested_replenishment_method"]

        suggested_ss = ShouldBeCalculator._safety(proposed, params.conventions)
        current_ss = ShouldBeCalculator._safety(df, params.conventions)

        # A non-stocking SKU is ordered on demand — the business already decided to hold
        # none of it, and `ShouldBeCalculator.calculate` zeroes it for the same reason.
        # Printing a safety stock for it here would invite someone to go and buy one.
        non_stocking = ShouldBeCalculator._non_stocking_mask(df)
        suggested_ss = suggested_ss.where(~non_stocking, 0.0)
        current_ss = current_ss.where(~non_stocking, 0.0)

        df["suggested_safety_stock"] = suggested_ss
        df["suggested_z"] = df["suggested_service_level"].map(_z_from_service_level).round(3)

        daily = df["demand_mean"].fillna(0.0) / DAYS_PER_MONTH
        df["suggested_ss_days"] = (
            df["suggested_safety_stock"] / daily.replace(0, np.nan)
        ).round(0)

        df["current_safety_stock"] = current_ss
        df["ss_delta_qty"] = (df["suggested_safety_stock"] - current_ss).round(1)
        df["ss_delta_value"] = (df["ss_delta_qty"] * df["unit_cost"].fillna(0.0)).round(2)
        return df

    # ── Against what the planner set by hand ─────────────────────────────────

    @staticmethod
    def _compare_to_planner(df: pd.DataFrame, notes: List[str]) -> pd.DataFrame:
        """
        The planner's own parameters against the ones the data supports.

        This is the comparison the whole planning-master input exists for, and it is
        deliberately one-directional: the planner's number is never fed back into the
        calculation, only measured against it. A pipeline that consumed the safety
        stock it was shown would agree with whatever it was shown.

        Direction is stated in words as well as numbers. "SS 500 → 310" is ambiguous
        about who is holding the risk; "310 justified vs 500 held, −190 units of
        overstock" is not.
        """
        if "planner_safety_stock" not in df.columns:
            return df

        planner_ss = pd.to_numeric(df["planner_safety_stock"], errors="coerce")
        if planner_ss.notna().sum() == 0:
            return df

        justified = pd.to_numeric(df["suggested_safety_stock"], errors="coerce")
        df["ss_vs_planner_qty"] = (justified - planner_ss).round(1)
        df["ss_vs_planner_value"] = (
            df["ss_vs_planner_qty"] * df["unit_cost"].fillna(0.0)
        ).round(2)
        df["ss_vs_planner_pct"] = np.where(
            planner_ss > 0, (justified - planner_ss) / planner_ss, np.nan
        ).round(3)

        verdict = pd.Series("", index=df.index, dtype=object)
        comparable = planner_ss.notna() & justified.notna()
        # A 10% band. Below that the difference is inside the noise of the inputs and
        # calling it a finding wastes the planner's attention on rounding.
        material = comparable & (df["ss_vs_planner_pct"].abs() > 0.10)
        verdict[comparable & ~material] = "in line"
        verdict[material & (df["ss_vs_planner_qty"] < 0)] = "planner holds more than justified"
        verdict[material & (df["ss_vs_planner_qty"] > 0)] = "planner holds less than justified"
        df["ss_verdict"] = verdict

        over = df.loc[verdict == "planner holds more than justified", "ss_vs_planner_value"]
        under = df.loc[verdict == "planner holds less than justified", "ss_vs_planner_value"]
        if len(over):
            notes.append(
                f"{len(over):,} SKUs carry more safety stock than the measured lead time "
                f"and forecast error justify — ${abs(over.sum()):,.0f} of capital"
            )
        if len(under):
            notes.append(
                f"{len(under):,} SKUs carry less safety stock than justified — "
                f"${under.sum():,.0f} of exposure, and the service level in force is not "
                f"the one being achieved"
            )

        # Where a planner set a lead time or review period too, name those gaps as well.
        for planner_col, computed_col, label in (
            ("planner_review_period_days", "suggested_review_period_days", "review"),
            ("planner_service_level", "suggested_service_level", "service level"),
        ):
            if planner_col not in df.columns:
                continue
            stated = pd.to_numeric(df[planner_col], errors="coerce")
            computed = pd.to_numeric(df[computed_col], errors="coerce")
            differs = stated.notna() & computed.notna() & ~np.isclose(
                stated.fillna(-1), computed.fillna(-1), rtol=1e-3
            )
            if differs.any():
                notes.append(
                    f"{int(differs.sum()):,} SKUs have a {label} the planner set that "
                    f"differs from the suggested one"
                )
        return df

    # ── What actually changes ────────────────────────────────────────────────

    @staticmethod
    def _diff_against_current(df: pd.DataFrame, params: ParameterSet) -> pd.DataFrame:
        """A readable list of the parameters that differ, so the CSV can be scanned."""
        comparisons = [
            ("service_level", "service level", "{:.0%}"),
            ("review_period_days", "review", "{:.0f}d"),
            ("replenishment_method", "method", "{}"),
        ]
        changes = pd.Series("", index=df.index)
        for key, label, fmt in comparisons:
            if key not in df.columns:
                continue
            current, suggested = df[key], df[f"suggested_{key}"]
            differs = _differs(current, suggested)
            for idx in df.index[differs]:
                entry = (f"{label} {_fmt(current[idx], fmt)} → "
                         f"{_fmt(suggested[idx], fmt)}")
                changes[idx] = f"{changes[idx]}; {entry}" if changes[idx] else entry
        df["changes"] = changes
        return df

    @staticmethod
    def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "sku", "description", "abc_class", "volatility_class", "product_family",
            "stocking_class", "demand_pattern",
            "demand_mean", "demand_cv", "demand_sigma", "sigma_source",
            "unit_cost", "annual_value",
            # measured supply parameters
            "supplier", "incoterm", "lead_time_days", "lt_sigma_days", "lt_cv", "lt_samples",
            # in force vs suggested
            "service_level", "suggested_service_level", "suggested_z",
            "review_period_days", "suggested_review_period_days",
            "replenishment_method", "suggested_replenishment_method",
            "current_safety_stock", "suggested_safety_stock", "suggested_ss_days",
            "ss_delta_qty", "ss_delta_value",
            "eoq_qty", "eoq_review_days", "min_order_qty", "order_multiple",
            # what the planner set by hand, and the gap to what the data supports
            "planner_safety_stock", "ss_vs_planner_qty", "ss_vs_planner_pct",
            "ss_vs_planner_value", "ss_verdict",
            "planner_reorder_point", "planner_min_qty", "planner_max_qty",
            "planner_review_period_days", "planner_service_level", "planner_notes",
            # provenance
            "lead_time_days_source", "unit_cost_source",
            "applied_rules", "applied_suggestions", "changes",
        ]
        return df[[c for c in cols if c in df.columns]].copy()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _snap_to_ladder(days: Any) -> Any:
    """Nearest review period on the ladder, in log space so 7→14 counts like 30→60."""
    if days is None or pd.isna(days) or days <= 0:
        return np.nan
    return float(min(REVIEW_LADDER, key=lambda r: abs(np.log(r) - np.log(float(days)))))


def _differs(current: pd.Series, suggested: pd.Series) -> pd.Series:
    """Element-wise inequality that treats two nulls as equal and tolerates floats."""
    both_null = current.isna() & suggested.isna()
    if pd.api.types.is_numeric_dtype(current) and pd.api.types.is_numeric_dtype(suggested):
        close = np.isclose(
            pd.to_numeric(current, errors="coerce").fillna(np.nan),
            pd.to_numeric(suggested, errors="coerce").fillna(np.nan),
            rtol=1e-6, equal_nan=True,
        )
        return ~pd.Series(close, index=current.index) & ~both_null
    return (current.astype(str) != suggested.astype(str)) & ~both_null


def _fmt(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _yaml_scalar(value: Any) -> str:
    """Render an override value as YAML — strings quoted, numbers bare."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    return f'"{value}"'
