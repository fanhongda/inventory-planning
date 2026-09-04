"""
Tests for the policy layer — should-be inventory, levers, targets, decisions.

The cases that matter here are the ones where naive arithmetic gives a confident wrong
answer: counting stock the buyer does not own, ranking a lever on the cash it frees
while ignoring what it costs to run, or promising a burn-down that demand cannot
deliver before the deadline.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.policy.decisions import (
    ACCEPTED, REJECTED, Decision, DecisionLog,
)
from inventory_planning.policy.levers import (
    LeverAnalyzer, LeverSpec, default_levers, economic_order_quantity,
)
from inventory_planning.policy.parameters import PlanningParameters
from inventory_planning.policy.should_be import ShouldBeCalculator, _z_from_service_level
from inventory_planning.policy.target import TargetPlanner

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def params():
    return PlanningParameters(CONFIG_DIR / "planning_parameters.md")


@pytest.fixture
def skus():
    """Five SKUs spanning the cases that behave differently."""
    return pd.DataFrame({
        "sku": ["ACT-1", "VAL-1", "BOLT-1", "LONG-1", "DDP-1"],
        "annual_value": [500_000, 120_000, 3_000, 200_000, 90_000],
        "unit_cost": [400.0, 90.0, 5.0, 150.0, 60.0],
        "lead_time_days": [45.0, 60.0, 30.0, 120.0, 50.0],
        "lt_sigma_days": [15.0, 10.0, 5.0, 25.0, 8.0],
        "demand_mean": [100.0, 100.0, 500.0, 100.0, 100.0],
        "demand_sigma": [30.0, 30.0, 150.0, 30.0, 30.0],
        "min_order_qty": [0.0] * 5,
        "product_family": ["actuator", "valve", "fastener", "valve", "valve"],
        "incoterm": ["FOB", "FOB", "FOB", "CIF", "DDP"],
    })


@pytest.fixture
def resolved(params, skus):
    skus = skus.copy()
    skus["abc_class"] = params.assign_abc(skus)
    return params.resolve(skus)


@pytest.fixture
def actual():
    return pd.DataFrame({
        "sku": ["ACT-1", "VAL-1", "BOLT-1", "LONG-1", "DDP-1"],
        "qty_on_hand": [900.0, 700.0, 9000.0, 1200.0, 500.0],
        "qty_in_transit": [200.0, 100.0, 0.0, 300.0, 300.0],
    })


# ── Parameters ───────────────────────────────────────────────────────────────

class TestPlanningParameters:
    def test_parses_conventions_containing_yaml_comments(self, params):
        """
        A YAML comment is indistinguishable from a markdown H1 by regex, so an
        un-fence-aware parser truncates every section at its first commented setting —
        which is exactly where the reasoning for that setting lives.
        """
        assert params.conventions["cycle_stock_basis"] == "peak"
        assert params.conventions["safety_stock_exposure"] == "review_plus_lt"
        assert params.conventions["pipeline_basis"] == "incoterm_aware"

    def test_loads_rules_with_rationale(self, params):
        assert len(params.rules) >= 4
        assert all(r.rationale.strip() for r in params.rules)

    def test_rule_without_rationale_is_rejected(self, tmp_path):
        (tmp_path / "p.md").write_text(
            "## Conventions\n```yaml\ncycle_stock_basis: peak\n"
            "safety_stock_exposure: review_plus_lt\npipeline_basis: none\n```\n"
            "## Rules\n### R-001 · no reason\n```yaml\nscope: abc_class == \"A\"\n"
            "set:\n  review_period_days: 7\n```\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="rationale"):
            PlanningParameters(tmp_path / "p.md")

    def test_applies_scoped_override(self, resolved):
        frame = resolved.frame.set_index("sku")
        assert frame.loc["ACT-1", "service_level"] == 0.98      # actuator rule
        assert frame.loc["VAL-1", "service_level"] == 0.95      # default

    def test_later_rule_wins_and_conflict_is_recorded(self, resolved):
        """An override that is merely effective, with no record, is how trust is lost."""
        assert resolved.conflicts, "actuator rule overrides the A-class rule"
        hit = resolved.conflicts[0]
        assert "review_period_days" in hit.overrides_earlier

    def test_records_which_rules_touched_each_sku(self, resolved):
        frame = resolved.frame.set_index("sku")
        assert "R-002" in frame.loc["ACT-1", "applied_rules"]
        assert frame.loc["VAL-1", "applied_rules"] == ""

    def test_rule_on_unavailable_column_matches_nothing_and_says_so(self, params, skus):
        skus = skus.drop(columns=["product_family"])
        skus["abc_class"] = params.assign_abc(skus)
        hits = {h.rule.rule_id: h for h in params.resolve(skus).hits}
        assert hits["R-002"].unavailable_columns == ["product_family"]

    def test_abc_follows_configured_cutoffs(self, params, skus):
        classes = params.assign_abc(skus)
        assert classes.loc[0] == "A"        # highest annual value
        assert set(classes) <= {"A", "B", "C"}


# ── Should-be inventory ──────────────────────────────────────────────────────

class TestShouldBe:
    def test_decomposes_into_three_levers(self, resolved, actual):
        result = ShouldBeCalculator(CONFIG_DIR).calculate(resolved, actual)
        f = result.frame
        assert np.allclose(
            f["should_be_qty"], f["cycle_qty"] + f["safety_qty"] + f["pipeline_qty"]
        )

    def test_ddp_stock_in_transit_belongs_to_neither_side(self, resolved, actual):
        """
        Under DDP the seller owns goods until delivery. Counting them in should-be but
        not in actual (or the reverse) manufactures a gap that does not exist.
        """
        result = ShouldBeCalculator(CONFIG_DIR).calculate(resolved, actual)
        row = result.frame.set_index("sku").loc["DDP-1"]
        assert row["pipeline_qty"] == 0.0
        assert row["actual_in_transit_owned"] == 0.0
        assert row["actual_qty"] == 500.0        # on-hand only, the 300 GIT excluded

    def test_fob_stock_in_transit_counts_on_both_sides(self, resolved, actual):
        result = ShouldBeCalculator(CONFIG_DIR).calculate(resolved, actual)
        row = result.frame.set_index("sku").loc["ACT-1"]
        assert row["pipeline_qty"] > 0
        assert row["actual_qty"] == 1100.0        # 900 on-hand + 200 owned GIT

    def test_cycle_basis_peak_is_double_average(self, params, skus, actual):
        skus["abc_class"] = params.assign_abc(skus)
        calc = ShouldBeCalculator(CONFIG_DIR)

        peak = calc.calculate(params.resolve(skus), actual).frame["cycle_qty"].sum()
        params.conventions["cycle_stock_basis"] = "average"
        avg = calc.calculate(params.resolve(skus), actual).frame["cycle_qty"].sum()

        assert peak == pytest.approx(avg * 2, rel=0.01)

    def test_exposure_is_review_plus_lead_time(self, params, skus, actual):
        """
        Using LT alone understates safety stock, because a shortfall found just after a
        review cannot be acted on until the next one.
        """
        skus["abc_class"] = params.assign_abc(skus)
        calc = ShouldBeCalculator(CONFIG_DIR)

        with_review = calc.calculate(params.resolve(skus), actual).frame["safety_qty"].sum()
        params.conventions["safety_stock_exposure"] = "lt_only"
        lt_only = calc.calculate(params.resolve(skus), actual).frame["safety_qty"].sum()

        assert with_review > lt_only

    def test_excess_is_gross_not_netted_against_shortfall(self, params, skus, actual):
        """
        Overstock on one SKU cannot offset a shortage on another — the units are not
        interchangeable and the cash is not either. Netting them reports "no excess"
        for a warehouse holding plenty of it.
        """
        skus["abc_class"] = params.assign_abc(skus)
        # One SKU far above policy, the rest stripped to nothing.
        lopsided = actual.copy()
        lopsided["qty_on_hand"] = [5000.0, 0.0, 0.0, 0.0, 0.0]
        lopsided["qty_in_transit"] = 0.0

        result = ShouldBeCalculator(CONFIG_DIR).calculate(params.resolve(skus), lopsided)

        assert result.excess_value > 0, "the overstocked SKU holds real capital"
        assert result.shortfall_value > 0, "the empty SKUs are genuinely short"
        assert result.excess_value != pytest.approx(max(result.gap_value, 0))
        assert "EXCESS" in result.summary() and "SHORTFALL" in result.summary()

    def test_an_order_on_demand_item_holds_no_speculative_stock(self, params, skus, actual):
        """The existing behaviour, unchanged: no cycle, no safety, pipeline only."""
        skus["abc_class"] = params.assign_abc(skus)
        skus["stocking_class"] = ["non-stocking"] + ["stocking-med"] * 4

        row = (ShouldBeCalculator(CONFIG_DIR)
               .calculate(params.resolve(skus), actual)
               .frame.set_index("sku").loc["ACT-1"])

        assert row["cycle_qty"] == 0.0
        assert row["safety_qty"] == 0.0

    def test_a_firm_order_book_is_not_excess(self, params, skus, actual):
        """
        Stock an order-on-demand item bought against a confirmed order, received and
        waiting to ship, used to read as excess above policy in full — while the
        recommender one stage earlier was the thing that asked for it. The two stages
        of one pipeline disagreed by construction.
        """
        skus["abc_class"] = params.assign_abc(skus)
        skus["stocking_class"] = ["non-stocking"] + ["stocking-med"] * 4
        # Nothing in transit: the goods have landed, which is exactly the moment the
        # old arithmetic priced the obligation at nothing.
        landed = actual.copy()
        landed["qty_on_hand"] = [800.0, 700.0, 9000.0, 1200.0, 500.0]
        landed["qty_in_transit"] = 0.0
        committed = pd.DataFrame({"sku": ["ACT-1"], "mto_actionable_qty": [800.0]})

        calc = ShouldBeCalculator(CONFIG_DIR)
        without = calc.calculate(params.resolve(skus), landed).frame.set_index("sku")
        with_book = calc.calculate(params.resolve(skus), landed,
                                   committed=committed).frame.set_index("sku")

        # Without the book, everything above the (demand-driven) pipeline reads as
        # excess; with it, the obligation is covered and the gap closes.
        assert without.loc["ACT-1", "gap_qty"] > 700.0           # nearly all called excess
        assert with_book.loc["ACT-1", "should_be_qty"] == 800.0
        assert with_book.loc["ACT-1", "gap_qty"] == 0.0          # owed, not excess

    def test_uncommitted_stock_on_the_same_item_is_still_excess(self, params, skus, actual):
        """The obligation is covered; the leftover above it is not spared."""
        skus["abc_class"] = params.assign_abc(skus)
        skus["stocking_class"] = ["non-stocking"] + ["stocking-med"] * 4
        landed = actual.copy()
        landed["qty_on_hand"] = [1000.0, 700.0, 9000.0, 1200.0, 500.0]
        landed["qty_in_transit"] = 0.0
        committed = pd.DataFrame({"sku": ["ACT-1"], "mto_actionable_qty": [600.0]})

        row = (ShouldBeCalculator(CONFIG_DIR)
               .calculate(params.resolve(skus), landed, committed=committed)
               .frame.set_index("sku").loc["ACT-1"])

        assert row["should_be_qty"] == 600.0
        assert row["gap_qty"] == 400.0

    def test_a_stocking_item_does_not_count_its_backlog_twice(self, params, skus, actual):
        """
        Its policy stock already covers the demand the backlog is part of, so adding
        the order book on top would buy the same demand twice — the error the
        recommender's `max(forecast, backlog)` exists to avoid, on the other side of
        the same pipeline.
        """
        skus["abc_class"] = params.assign_abc(skus)
        skus["stocking_class"] = ["stocking-med"] * 5
        committed = pd.DataFrame({"sku": ["ACT-1"], "mto_actionable_qty": [50.0]})

        calc = ShouldBeCalculator(CONFIG_DIR)
        without = calc.calculate(params.resolve(skus), actual).frame.set_index("sku")
        with_book = calc.calculate(params.resolve(skus), actual,
                                   committed=committed).frame.set_index("sku")

        assert (with_book.loc["ACT-1", "should_be_qty"]
                == without.loc["ACT-1", "should_be_qty"])

    def test_the_order_book_it_covers_is_named_in_the_summary(self, params, skus, actual):
        skus["abc_class"] = params.assign_abc(skus)
        skus["stocking_class"] = ["non-stocking"] + ["stocking-med"] * 4
        committed = pd.DataFrame({"sku": ["ACT-1"], "mto_actionable_qty": [800.0]})

        text = (ShouldBeCalculator(CONFIG_DIR)
                .calculate(params.resolve(skus), actual, committed=committed).summary())

        assert "firm order book" in text

    def test_unpriced_skus_are_reported_not_imputed(self, params, skus, actual):
        skus.loc[0, "unit_cost"] = np.nan
        skus["abc_class"] = params.assign_abc(skus)
        result = ShouldBeCalculator(CONFIG_DIR).calculate(params.resolve(skus), actual)
        assert result.unpriced_skus == 1
        assert result.frame.set_index("sku").loc["ACT-1", "should_be_value"] == 0.0

    def test_higher_service_level_needs_more_stock(self, params, skus, actual):
        skus["abc_class"] = params.assign_abc(skus)
        calc = ShouldBeCalculator(CONFIG_DIR)
        base = calc.calculate(params.resolve(skus), actual).frame["safety_qty"].sum()

        raised = params.resolve(skus)
        raised.frame["service_level"] = 0.99
        assert calc.calculate(raised, actual).frame["safety_qty"].sum() > base

    @pytest.mark.parametrize("sl,expected", [(0.95, 1.645), (0.98, 2.054), (0.90, 1.282)])
    def test_z_lookup(self, sl, expected):
        assert _z_from_service_level(sl) == pytest.approx(expected, abs=0.01)

    def test_z_interpolates_between_table_points(self):
        z = _z_from_service_level(0.965)
        assert 1.751 < z < 1.881


# ── Levers ───────────────────────────────────────────────────────────────────

class TestLevers:
    def test_ranks_on_net_not_on_stock_freed(self, resolved, actual):
        """
        The balance-sheet reduction is one-off; the ordering cost repeats every year.
        Ranking on the one-off systematically over-recommends frequent ordering.
        """
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(resolved, actual)
        weekly = analysis.results["review_weekly"]

        assert weekly.total_saving > 0, "it does free stock"
        assert weekly.is_value_destroying, "but costs more in ordering than it frees"
        assert analysis.ranked[0].spec.key != "review_weekly"

    def test_lead_time_variability_beats_lead_time_mean(self, resolved, actual):
        """
        Once pipeline is measured on the ownership boundary, cutting the mean lead time
        acts only through √(R+LT). Cutting its variability acts on the D̄²·σ_LT² term
        directly — so the supplier ask is reliability, not speed.
        """
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(resolved, actual)
        reliability = analysis.results["lt_reliability"]
        assert reliability.total_saving > 0
        assert reliability.net_annual > 0

    def test_unpriceable_cost_is_flagged_not_assumed_free(self, resolved, actual):
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(resolved, actual)
        assert analysis.results["lt_speed"].cost_known is False
        assert analysis.results["service_level_down"].cost_known is False
        assert analysis.results["lt_reliability"].cost_known is True

    def test_best_lever_differs_across_skus(self, resolved, actual):
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(resolved, actual)
        best = analysis.best_lever_per_sku()
        assert len(best) == 5
        assert best["lever"].nunique() > 1, "a single global ranking would hide this"

    def test_eoq_guard_fires_when_cadence_is_uneconomic(self, resolved, actual):
        """EOQ appears only when violated — planners order to a cadence, not to EOQ."""
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(resolved, actual)
        guards = analysis.guards
        assert len(guards)
        assert any("EOQ" in f for f in guards["finding"])

    def test_flags_lead_time_dominating_review_period(self, resolved, actual):
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(resolved, actual)
        findings = " ".join(analysis.guards["finding"])
        assert "shortening review barely moves safety stock" in findings

    def test_moq_blocks_the_cycle_stock_lever(self, params, skus, actual):
        skus["min_order_qty"] = 10_000.0
        skus["abc_class"] = params.assign_abc(skus)
        analysis = LeverAnalyzer(ShouldBeCalculator(CONFIG_DIR)).analyze(
            params.resolve(skus), actual
        )
        blocking = analysis.guards[analysis.guards["severity"] == "blocking"]
        assert len(blocking)

    def test_eoq_formula(self):
        eoq = economic_order_quantity(
            pd.Series([1200.0]), pd.Series([150.0]), order_cost=350, holding_rate=0.22
        )
        assert eoq.iloc[0] == pytest.approx(np.sqrt(2 * 1200 * 350 / 33), rel=0.01)


# ── Target frontier ──────────────────────────────────────────────────────────

class TestTargetFrontier:
    @pytest.fixture
    def frontier(self, resolved, actual):
        calc = ShouldBeCalculator(CONFIG_DIR)
        base = calc.calculate(resolved, actual)
        levers = LeverAnalyzer(calc).analyze(resolved, actual, base)
        open_po = pd.DataFrame({"sku": ["ACT-1", "LONG-1"], "open_qty": [200.0, 300.0]})
        return TargetPlanner().plan(
            base, target_value=350_000, deadline=date(2026, 12, 31),
            levers=levers, open_po=open_po, as_of=date(2026, 8, 2),
        )

    def test_burn_down_is_limited_by_demand_before_the_deadline(self, frontier):
        """Excess only becomes cash as fast as demand consumes it."""
        stop = next(a for a in frontier.actions if a.kind == "stop_buying")
        assert stop.time_limited
        assert stop.value_released < stop.value_theoretical

    def test_free_actions_rank_above_painful_ones(self, frontier):
        risks = [a.pain_rank for a in frontier.actions]
        assert risks == sorted(risks), "a bigger painful action must not outrank a free one"

    def test_slow_levers_are_excluded_from_a_dated_plan(self, frontier):
        """A supplier reliability programme is real work that will not land by December."""
        assert any(a.kind.endswith("lt_reliability") for a in frontier.too_slow)
        assert not any(a.kind.endswith("lt_reliability") for a in frontier.actions)

    def test_value_destroying_lever_is_excluded_with_its_reason(self, frontier):
        excluded = {a.kind: a for a in frontier.excluded}
        assert "excluded:review_weekly" in excluded
        assert "net loss" in excluded["excluded:review_weekly"].rationale

    def test_push_out_never_exceeds_the_excess(self, frontier):
        push = next((a for a in frontier.actions if a.kind == "push_out_po"), None)
        assert push is not None
        assert push.service_risk == "low"

    def test_reports_shortfall_when_target_is_unreachable(self, resolved, actual):
        calc = ShouldBeCalculator(CONFIG_DIR)
        base = calc.calculate(resolved, actual)
        frontier = TargetPlanner().plan(base, target_value=1000.0, deadline=None)
        assert not frontier.reachable
        assert "NOT reachable" in frontier.summary()

    def test_no_actions_when_already_below_target(self, resolved, actual):
        calc = ShouldBeCalculator(CONFIG_DIR)
        base = calc.calculate(resolved, actual)
        frontier = TargetPlanner().plan(base, target_value=99_000_000.0)
        assert frontier.gap < 0
        assert frontier.actions == []

    def test_cumulative_marks_where_the_target_is_met(self, frontier):
        cumulative = frontier.cumulative()
        assert "running_balance" in cumulative.columns
        assert cumulative["running_balance"].is_monotonic_decreasing


# ── Decision log ─────────────────────────────────────────────────────────────

class TestDecisionLog:
    def test_rejection_requires_a_reason_code(self):
        with pytest.raises(ValueError, match="reason_code"):
            Decision.create("run-1", "push_out_po", "Push out", REJECTED, sku="A-1")

    def test_acceptance_needs_no_reason(self):
        d = Decision.create("run-1", "stop_buying", "Stop", ACCEPTED, sku="A-1")
        assert d.decision == ACCEPTED

    def test_unknown_reason_code_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown reason_code"):
            Decision.create("run-1", "x", "x", REJECTED, sku="A", reason_code="made_up")

    def test_round_trips_through_the_log(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(Decision.create("run-1", "stop_buying", "Stop", ACCEPTED, sku="A-1"))
        assert len(log.load()) == 1

    def test_surfaces_repeated_rejections_as_a_constraint(self, tmp_path):
        """
        A rejection pattern is the model learning a constraint that exists in no ERP
        extract — a supplier who will not reschedule, for instance.
        """
        log = DecisionLog(tmp_path / "d.jsonl")
        for sku in ["A-1", "A-2", "A-3", "A-4"]:
            log.record(Decision.create(
                "run-1", "push_out_po", "Push out", REJECTED, sku=sku,
                reason_code="supplier_wont_reschedule", value_at_stake=10_000,
            ))
        attrs = pd.DataFrame({"sku": ["A-1", "A-2", "A-3", "A-4"], "supplier": ["V100"] * 4})

        candidates = log.constraint_candidates(attrs)
        assert candidates
        top = candidates[0]
        assert top.occurrences == 4
        assert "V100" in top.scope
        assert "scope:" in top.suggested_rule

    def test_ignores_patterns_below_the_threshold(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        log.record(Decision.create("run-1", "push_out_po", "Push", REJECTED,
                                   sku="A-1", reason_code="moq_constraint"))
        attrs = pd.DataFrame({"sku": ["A-1"], "supplier": ["V1"]})
        assert log.constraint_candidates(attrs) == []

    def test_acceptance_rate_identifies_unactionable_advice(self, tmp_path):
        log = DecisionLog(tmp_path / "d.jsonl")
        for sku in ["A-1", "A-2", "A-3"]:
            log.record(Decision.create("run-1", "push_out_po", "Push", REJECTED,
                                       sku=sku, reason_code="supplier_wont_reschedule"))
        log.record(Decision.create("run-1", "stop_buying", "Stop", ACCEPTED, sku="A-1"))

        rates = log.acceptance_rate().set_index("action_kind")
        assert rates.loc["push_out_po", "acceptance_rate"] == 0.0
        assert rates.loc["stop_buying", "acceptance_rate"] == 1.0

    def test_corrupt_line_does_not_lose_the_history(self, tmp_path):
        path = tmp_path / "d.jsonl"
        log = DecisionLog(path)
        log.record(Decision.create("run-1", "stop_buying", "Stop", ACCEPTED, sku="A-1"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{truncated write\n")
        log.record(Decision.create("run-2", "stop_buying", "Stop", ACCEPTED, sku="A-2"))
        assert len(log.load()) == 2
