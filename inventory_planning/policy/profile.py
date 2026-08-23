"""
Replenishment policy profile — what the data says each SKU *is*, before anything is
ordered.

Which replenishment policy is correct is not a matter of taste. It is decided by where
an item sits on eight axes (`principles/sc-principles.md` §7, §9): how demand behaves,
how the lead time behaves, whether items are independent, whether review is continuous
or periodic, how many locations are in play, whether capacity constrains the order, what
happens to unmet demand, and how long the planning horizon is. Those axes select among
five methods — EOQ, single period, base stock, continuous review (s, Q), periodic review
(R, s, S) — and applying one arithmetic to every SKU regardless is how a pipeline ends
up under-ordering a long-lead-time item and over-managing a cheap one in the same run.

Most of these axes were already measurable from the five extracts, and several were
already measured — they were just never written down. A judgement that lives only inside
a branch of a formula cannot be reviewed, cannot be disagreed with, and cannot be shown
to a planner who wants to know why their item was treated the way it was. So it is
written down: one row per SKU, one column per axis, and next to each a plain statement
of the evidence and whether that evidence was **measured** from the transactions,
**stated** by a master file, or **defaulted** because nothing observed it.

The profile decides nothing on its own. It records two things and lets the difference
show:

  policy_in_force   what `config/planning_parameters.md` resolves to for this SKU —
                    the planner's own rules, which govern the order that gets raised
  policy_implied    what these axes imply on their own

Where they disagree the run says so and still uses the rule, because a planner's
deliberate exception outranks an inference drawn from twelve months of history. The
disagreement is the finding; overwriting the rule would destroy it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Policy vocabulary. `min_max` is the old name for the (s, Q) policy and is still
# parsed so an existing parameter file keeps working.
POLICY_PERIODIC = "periodic"            # (R, s, S)
POLICY_REORDER_POINT = "reorder_point"  # (s, Q)
POLICY_MAKE_TO_ORDER = "make_to_order"  # no policy stock; buy against firm orders
_POLICY_ALIASES = {"min_max": POLICY_REORDER_POINT}

POLICY_LABELS = {
    POLICY_PERIODIC: "(R, s, S)",
    POLICY_REORDER_POINT: "(s, Q)",
    POLICY_MAKE_TO_ORDER: "order on demand",
}

# CV below which demand is flat enough to call constant. Deliberately tighter than the
# `cv_intermittent_threshold` used for pattern classification: that one separates
# manageable variability from awkward variability, this one asks the much stronger
# question of whether the deterministic models apply at all.
CONSTANT_DEMAND_CV = 0.10

# A lead time whose spread is this share of its mean or more is not a constant, whatever
# the master file says it is.
VARIABLE_LT_RATIO = 0.20

# Below this many receipts a lead-time sigma is not a distribution. It is still the best
# figure available, but the profile says how thin it is rather than presenting it as
# measured.
MIN_LT_SAMPLES = 3

AXES = (
    "demand_variability", "demand_certainty", "demand_continuity",
    "lead_time_behaviour", "item_dependence", "review_mode",
    "location_scope", "capacity", "excess_demand", "planning_horizon",
)


def canonical_policy(method) -> str:
    """Resolve the configured method name, tolerating the legacy `min_max` spelling."""
    if method is None or (isinstance(method, float) and np.isnan(method)):
        return POLICY_PERIODIC
    key = str(method).strip().lower()
    return _POLICY_ALIASES.get(key, key)


@dataclass
class PolicyProfile:
    """One row per SKU: the axis readings, the evidence, and the policy they imply."""

    frame: pd.DataFrame

    @property
    def disagreements(self) -> pd.DataFrame:
        if "policy_agrees" not in self.frame.columns:
            return self.frame.iloc[0:0]
        return self.frame[~self.frame["policy_agrees"].astype(bool)]

    def summary(self) -> str:
        df = self.frame
        lines = ["  Replenishment policy profile", "  " + "-" * 58]
        lines.append(
            f"    {len(df)} SKUs profiled — {len(AXES)} readings across the eight axes "
            f"that decide the policy"
        )

        for axis in ("demand_variability", "demand_continuity", "lead_time_behaviour",
                     "capacity", "location_scope"):
            if axis not in df.columns:
                continue
            counts = df[axis].value_counts()
            spread = ", ".join(f"{k}: {v}" for k, v in counts.items())
            lines.append(f"    {axis:<20}: {spread}")

        if "policy_in_force" in df.columns:
            counts = df["policy_in_force"].value_counts()
            spread = ", ".join(
                f"{POLICY_LABELS.get(k, k)}: {v}" for k, v in counts.items()
            )
            lines.append(f"    {'policy in force':<20}: {spread}")

        disagree = self.disagreements
        if len(disagree):
            lines.append("")
            lines.append(
                f"    {len(disagree)} SKUs where the evidence implies a different policy "
                f"than the rules set."
            )
            lines.append("    The rule governs — this is reported, not applied.")
            for _, row in disagree.head(3).iterrows():
                lines.append(
                    f"      {str(row['sku']):<18} rules say "
                    f"{POLICY_LABELS.get(row['policy_in_force'], row['policy_in_force'])}"
                    f", history says "
                    f"{POLICY_LABELS.get(row['policy_implied'], row['policy_implied'])}"
                    f" — {row.get('policy_implied_because', '')}"
                )
            if len(disagree) > 3:
                lines.append(f"      +{len(disagree) - 3} more in the profile CSV")
        return "\n".join(lines)


def build_policy_profile(
    parameters: pd.DataFrame,
    backlog: Optional[pd.DataFrame] = None,
    inventory: Optional[pd.DataFrame] = None,
) -> PolicyProfile:
    """
    `parameters` is the resolved per-SKU frame from `PlanningParameters.resolve` — it
    already carries the demand statistics, the measured lead time, the segmentation and
    the parameters the rules produced.

    `backlog` and `inventory` are optional. Where they are absent the affected axis says
    "not observed" rather than guessing, because the difference between *no backorders*
    and *no visibility of backorders* is exactly the kind of thing a planner needs to
    know before trusting a service number.
    """
    df = parameters.copy()
    out = pd.DataFrame({"sku": df["sku"]}, index=df.index)

    num = lambda col, default=np.nan: (  # noqa: E731 — a local shorthand, used ~12 times
        pd.to_numeric(df[col], errors="coerce") if col in df.columns
        else pd.Series(default, index=df.index, dtype="float64")
    )

    # ── 1. Demand: constant or variable ──────────────────────────────────────
    cv = num("demand_cv")
    cycles = num("total_cycles_evaluated", 0).fillna(0)
    out["demand_variability"] = np.where(
        cv.isna(), "unknown", np.where(cv <= CONSTANT_DEMAND_CV, "constant", "variable")
    )
    out["demand_variability_evidence"] = _evidence(
        cv.isna(),
        "no CV — no demand in the window (default)",
        "CV " + cv.round(2).astype(str) + " over " + cycles.astype("Int64").astype(str)
        + " cycles (measured)",
    )

    # ── 2. Demand: known or random ───────────────────────────────────────────
    # Firm orders on the book are known demand; the rest is a forecast of a random
    # process. The share of the cycle already sold decides which description fits.
    monthly = num("demand_mean" if "demand_mean" in df.columns else "demand_mean_rolling",
                  0.0)
    firm = _backlog_due(df, backlog)
    covered = (firm / monthly.replace(0, np.nan)).fillna(0.0)
    observed_book = firm.notna() & (firm > 0)
    out["demand_certainty"] = np.select(
        [covered >= 1.0, observed_book],
        ["known", "mixed"],
        default="random",
    )
    out["demand_certainty_evidence"] = np.where(
        observed_book,
        "order book covers " + (covered * 100).round(0).astype(int).astype(str)
        + "% of a cycle (measured)",
        "no firm orders on the book (measured)" if backlog is not None
        else "no order book supplied (default)",
    )

    # ── 3. Demand: continuous or intermittent ────────────────────────────────
    active = num("active_cycles_rolling", 0).fillna(0)
    pattern = (df["demand_pattern"].astype(str) if "demand_pattern" in df.columns
               else pd.Series("unknown", index=df.index))
    out["demand_continuity"] = pattern
    out["demand_continuity_evidence"] = (
        "demand in " + active.astype("Int64").astype(str) + "/"
        + cycles.astype("Int64").astype(str) + " cycles (measured)"
    )

    # ── 4. Lead time: constant or variable, and how well known ───────────────
    lt = num("lead_time_days")
    lt_sigma = num("lt_sigma_days", 0.0).fillna(0.0)
    samples = num("lt_samples", 0).fillna(0)
    ratio = (lt_sigma / lt.replace(0, np.nan)).fillna(0.0)
    out["lead_time_behaviour"] = np.select(
        [lt.isna() | (lt <= 0), samples < MIN_LT_SAMPLES, ratio >= VARIABLE_LT_RATIO],
        ["unknown", "thinly-sampled", "variable"],
        default="constant",
    )
    out["lead_time_behaviour_evidence"] = np.select(
        [lt.isna() | (lt <= 0), samples < MIN_LT_SAMPLES],
        [
            "no lead time observed or stated (default)",
            lt.round(1).astype(str) + "d from " + samples.astype("Int64").astype(str)
            + " receipt(s) — too few to call a distribution (stated)",
        ],
        default=lt.round(1).astype(str) + "d ±" + lt_sigma.round(1).astype(str)
        + "d from " + samples.astype("Int64").astype(str) + " receipts (measured)",
    )

    # ── 5. Item dependence ───────────────────────────────────────────────────
    # Independent unless the intake merged this number with another one. Correlated and
    # indentured demand need a bill of materials, which no extract here carries.
    merged = (df["superseded_from"].notna() if "superseded_from" in df.columns
              else pd.Series(False, index=df.index))
    out["item_dependence"] = np.where(merged, "merged-supersession", "independent")
    out["item_dependence_evidence"] = np.where(
        merged,
        "history merged from a superseded number (measured)",
        "no BOM or substitution link in the extract (default)",
    )

    # ── 6. Review mode ───────────────────────────────────────────────────────
    review = num("review_period_days", 30).fillna(30)
    rules = (df["applied_rules"].fillna("").astype(str) if "applied_rules" in df.columns
             else pd.Series("", index=df.index))
    out["review_mode"] = "periodic"
    out["review_period_days"] = review
    out["review_mode_evidence"] = np.where(
        rules.str.len() > 0,
        "R = " + review.round(0).astype(int).astype(str) + "d, set by " + rules + " (stated)",
        "R = " + review.round(0).astype(int).astype(str) + "d, parameter default (default)",
    )

    # ── 7. Locations ─────────────────────────────────────────────────────────
    merged_locs = _merged_locations(df, inventory)
    out["location_scope"] = np.where(merged_locs > 1, "merged-multi", "single")
    out["location_scope_evidence"] = np.where(
        merged_locs > 1,
        merged_locs.astype("Int64").astype(str)
        + " storage locations summed into one position (measured)",
        "one storage location (measured)" if inventory is not None
        else "single planning node assumed (default)",
    )

    # ── 8. Capacity / lot constraint ─────────────────────────────────────────
    moq = num("min_order_qty", 0).fillna(0)
    multiple = num("order_multiple", 1).fillna(1)
    constrained = (moq > 0) | (multiple > 1)
    out["capacity"] = np.where(constrained, "lot-constrained", "unconstrained")
    out["capacity_evidence"] = np.where(
        constrained,
        "MOQ " + moq.round(0).astype(int).astype(str) + ", multiple of "
        + multiple.round(0).astype(int).astype(str) + " (stated)",
        "no MOQ or order multiple supplied (default)",
    )

    # ── 9. Excess demand ─────────────────────────────────────────────────────
    # Everything downstream — CSL as the service metric, the backlog carried into the
    # requirement — assumes unmet demand waits rather than walking away. Lost sales are
    # not measurable from any of these extracts and the profile says so plainly.
    out["excess_demand"] = "backordered" if backlog is not None else "unknown"
    out["excess_demand_evidence"] = (
        "unmet demand is carried as open order lines (measured)" if backlog is not None
        else "no open sales orders supplied — lost sales are not observable here (default)"
    )

    # ── 10. Planning horizon ─────────────────────────────────────────────────
    out["planning_horizon"] = "rolling"
    out["planning_horizon_evidence"] = "continuing replenishment, not a single buy (assumed)"

    # ── The policy ───────────────────────────────────────────────────────────
    out["policy_in_force"] = (
        df["replenishment_method"].map(canonical_policy) if "replenishment_method" in df.columns
        else POLICY_PERIODIC
    )
    implied, because = _implied_policy(df, out)
    out["policy_implied"] = implied
    out["policy_implied_because"] = because
    out["policy_agrees"] = out["policy_in_force"] == out["policy_implied"]
    out["policy_label"] = out["policy_in_force"].map(POLICY_LABELS).fillna(
        out["policy_in_force"])

    for carry in ("abc_class", "annual_value", "unit_cost", "lead_time_days",
                  "applied_rules"):
        if carry in df.columns and carry not in out.columns:
            out[carry] = df[carry]

    return PolicyProfile(frame=out.reset_index(drop=True))


def _evidence(mask: pd.Series, when_true: str, when_false: pd.Series) -> pd.Series:
    return pd.Series(np.where(mask, when_true, when_false), index=mask.index)


def _backlog_due(df: pd.DataFrame, backlog: Optional[pd.DataFrame]) -> pd.Series:
    """Firm quantity due inside the planning cycle, from whichever source carries it."""
    if backlog is not None and "sku" in backlog.columns:
        col = next((c for c in ("backlog_due_qty", "backlog_qty") if c in backlog.columns),
                   None)
        if col:
            due = backlog.drop_duplicates("sku").set_index("sku")[col]
            return pd.to_numeric(df["sku"].map(due), errors="coerce").fillna(0.0)
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _merged_locations(df: pd.DataFrame, inventory: Optional[pd.DataFrame]) -> pd.Series:
    """How many storage locations were summed into this SKU's position."""
    source = None
    if inventory is not None and "stock_locations" in inventory.columns:
        source = inventory.drop_duplicates("sku").set_index("sku")["stock_locations"]
    elif "stock_locations" in df.columns:
        source = df.drop_duplicates("sku").set_index("sku")["stock_locations"]
    if source is None:
        return pd.Series(1, index=df.index, dtype="float64")
    codes = df["sku"].map(source).fillna("").astype(str)
    counts = codes.str.count(",") + 1
    return counts.where(codes.str.len() > 0, 1).astype("float64")


def _implied_policy(df: pd.DataFrame, axes: pd.DataFrame):
    """
    What the axes imply on their own, and the one sentence that says why.

    Deliberately conservative: periodic review is the answer for a DC that plans in
    batches, and the two exceptions are the ones the textbook is unambiguous about — an
    item with too few demand cycles to hold stock against, and a cheap item whose
    ordering cost dominates its holding cost.
    """
    stocking = (df["stocking_class"].astype(str) if "stocking_class" in df.columns
                else pd.Series("", index=df.index))
    abc = (df["abc_class"].astype(str) if "abc_class" in df.columns
           else pd.Series("", index=df.index))
    unit_cost = (pd.to_numeric(df["unit_cost"], errors="coerce")
                 if "unit_cost" in df.columns else pd.Series(np.nan, index=df.index))

    is_mto = stocking == "non-stocking"
    is_cheap_tail = (abc == "C") & unit_cost.notna() & (unit_cost < 20)

    implied = pd.Series(POLICY_PERIODIC, index=df.index, dtype=object)
    implied[is_cheap_tail] = POLICY_REORDER_POINT
    implied[is_mto] = POLICY_MAKE_TO_ORDER

    because = pd.Series(
        "variable demand, stochastic lead time, periodic review", index=df.index,
        dtype=object)
    because[is_cheap_tail] = "C class under $20 — ordering cost dominates holding cost"
    because[is_mto] = "too few demand cycles to hold policy stock against"
    return implied, because
