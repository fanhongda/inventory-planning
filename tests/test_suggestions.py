"""
Tests for parameter suggestions.

The load-bearing property is that the generated rules file is not prose about rules —
it is rules. Anything written out must parse back through the same reader that loads
`planning_parameters.md`, and mean the same thing when it does.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.policy.parameters import PlanningParameters
from inventory_planning.policy.suggestions import (
    REVIEW_LADDER,
    SUGGESTION_RULES,
    SuggestionBuilder,
    _snap_to_ladder,
)

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def params():
    return PlanningParameters(CONFIG_DIR / "planning_parameters.md")


@pytest.fixture
def skus():
    """Spans the cases the suggestion rules distinguish."""
    return pd.DataFrame({
        "sku": ["A-1", "C-CHEAP", "ERRATIC-1", "LONG-1", "WOBBLY-1", "DEAD-1"],
        "annual_value": [900_000, 2_000, 150_000, 300_000, 120_000, 500],
        "unit_cost": [400.0, 5.0, 90.0, 150.0, 60.0, 10.0],
        "lead_time_days": [45.0, 30.0, 40.0, 120.0, 40.0, 30.0],
        "lt_sigma_days": [5.0, 3.0, 4.0, 20.0, 30.0, 2.0],
        "lt_samples": [12, 8, 6, 4, 9, 0],
        "demand_mean": [200.0, 400.0, 80.0, 100.0, 90.0, 2.0],
        "demand_sigma": [40.0, 100.0, 90.0, 30.0, 25.0, 3.0],
        "demand_cv": [0.2, 0.25, 1.2, 0.3, 0.28, 1.5],
        "stocking_class": ["stocking-high", "stocking-high", "stocking-high",
                           "stocking-high", "stocking-high", "non-stocking"],
        "demand_pattern": ["smooth", "smooth", "erratic", "smooth", "smooth",
                           "non-stocking"],
        "min_order_qty": [0.0] * 6,
        "product_family": ["actuator", "fastener", "valve", "valve", "valve", "valve"],
        "incoterm": ["FOB"] * 6,
    })


@pytest.fixture
def resolved(params, skus):
    skus = skus.copy()
    skus["abc_class"] = params.assign_abc(skus)
    return params.resolve(skus)


@pytest.fixture
def suggestions(resolved):
    return SuggestionBuilder(CONFIG_DIR).build(resolved)


class TestRuleSet:

    def test_every_rule_compiles_as_a_real_rule(self):
        """Scope grammar and the mandatory rationale are checked by the parser's Rule."""
        for suggestion in SUGGESTION_RULES:
            rule = suggestion.as_rule()
            assert rule.rule_id == suggestion.rule_id
            assert rule.columns, "a scope that references no column matches everything"

    def test_rule_ids_are_unique(self):
        ids = [r.rule_id for r in SUGGESTION_RULES]
        assert len(ids) == len(set(ids))

    def test_non_stocking_exception_wins_over_the_abc_baseline(self, suggestions):
        """
        Ordering is semantic: later rules win, so the strongest exception goes last.
        With it ordered first, an order-on-demand SKU picked up a C-class service level.
        """
        row = suggestions.frame.set_index("sku").loc["DEAD-1"]
        assert row["suggested_replenishment_method"] == "make_to_order"
        assert row["suggested_service_level"] == 0.0

    def test_a_class_gets_the_higher_service_level(self, suggestions):
        assert suggestions.frame.set_index("sku").loc["A-1", "suggested_service_level"] == 0.98

    def test_erratic_demand_goes_to_continuous_review(self, suggestions):
        row = suggestions.frame.set_index("sku").loc["ERRATIC-1"]
        assert row["suggested_replenishment_method"] == "reorder_point"

    def test_unreliable_lead_time_shortens_the_review(self, suggestions):
        """lt_sigma 30 against a 40-day lead time is a supplier problem, not a stock one."""
        row = suggestions.frame.set_index("sku").loc["WOBBLY-1"]
        assert row["suggested_review_period_days"] == 14

    def test_long_lead_time_keeps_a_monthly_review(self, suggestions):
        assert suggestions.frame.set_index("sku").loc["LONG-1",
                                                      "suggested_review_period_days"] == 30

    def test_hits_are_counted(self, suggestions):
        assert suggestions.hits["S-008"] == 1
        assert sum(suggestions.hits.values()) > 0


class TestSuggestedValues:

    def test_safety_stock_tracks_the_suggested_service_level(self, suggestions):
        """A-1 moves 0.95 → 0.98, so its safety stock must rise, not fall."""
        row = suggestions.frame.set_index("sku").loc["A-1"]
        assert row["suggested_safety_stock"] > row["current_safety_stock"]
        assert row["ss_delta_qty"] > 0

    def test_non_stocking_carries_no_safety_stock(self, suggestions):
        row = suggestions.frame.set_index("sku").loc["DEAD-1"]
        assert row["suggested_safety_stock"] == 0.0
        assert row["current_safety_stock"] == 0.0

    def test_lead_time_evidence_travels_with_the_suggestion(self, suggestions):
        row = suggestions.frame.set_index("sku").loc["DEAD-1"]
        assert row["lt_samples"] == 0, "no PO history must be visible, not silently zero-filled"

    def test_eoq_is_computed_where_cost_is_known(self, suggestions):
        assert suggestions.frame["eoq_qty"].notna().all()

    def test_changes_column_names_what_moved(self, suggestions):
        """
        A-1 is an actuator, so R-002 already put it at 0.98 — the suggestion agrees and
        says nothing about service level. The disagreement is the review period: the
        parameter file reviews A items weekly, the EOQ says monthly is enough.
        """
        row = suggestions.frame.set_index("sku").loc["A-1"]
        assert row["changes"] == "review 7d → 30d"
        assert row["service_level"] == row["suggested_service_level"] == 0.98

    def test_changes_are_only_reported_where_values_really_differ(self, suggestions):
        for _, row in suggestions.frame.iterrows():
            if "service level" in row["changes"]:
                assert row["service_level"] != row["suggested_service_level"]
            if "method" in row["changes"]:
                assert row["replenishment_method"] != row["suggested_replenishment_method"]

    def test_unchanged_parameters_produce_no_noise(self, resolved):
        """A SKU whose parameters already match must not be reported as a change."""
        result = SuggestionBuilder(CONFIG_DIR).build(resolved)
        unchanged = result.frame[result.frame["changes"] == ""]
        for _, row in unchanged.iterrows():
            assert row["service_level"] == row["suggested_service_level"]


class TestReviewLadder:

    @pytest.mark.parametrize("days,expected", [(6, 7), (9, 7), (13, 14), (25, 30), (200, 90)])
    def test_snaps_to_the_ladder(self, days, expected):
        assert _snap_to_ladder(days) == expected

    def test_undefined_input_stays_undefined(self):
        assert np.isnan(_snap_to_ladder(np.nan))
        assert np.isnan(_snap_to_ladder(0))

    def test_ladder_values_snap_to_themselves(self):
        for rung in REVIEW_LADDER:
            assert _snap_to_ladder(rung) == rung


class TestRulesMarkdown:

    def test_output_parses_as_planning_parameters(self, suggestions, tmp_path):
        """
        The whole point of the format. Generated rules are pasted into the parameter
        file, so they must survive a round trip through its parser unchanged.
        """
        source = (CONFIG_DIR / "planning_parameters.md").read_text(encoding="utf-8")
        # Keep the conventions/defaults the parser requires, drop the rules in force,
        # and append the suggested ones in their place.
        head = source.split("## 覆盖规则")[0]
        rules_md = suggestions.to_rules_markdown()
        suggested_blocks = rules_md.split("## 建议规则 (Suggested rules)")[1]
        suggested_blocks = suggested_blocks.split("## 逐 SKU 数值")[0]

        merged = tmp_path / "merged.md"
        merged.write_text(head + "## 覆盖规则 (Rules)\n" + suggested_blocks, encoding="utf-8")

        reloaded = PlanningParameters(merged)
        assert [r.rule_id for r in reloaded.rules] == [r.rule_id for r in SUGGESTION_RULES]
        for original, parsed in zip(SUGGESTION_RULES, reloaded.rules):
            assert parsed.scope == original.scope
            assert parsed.overrides == original.overrides
            assert parsed.rationale.strip()

    def test_reparsed_rules_select_the_same_skus(self, suggestions, resolved, tmp_path):
        """Parsing must preserve meaning, not just syntax."""
        source = (CONFIG_DIR / "planning_parameters.md").read_text(encoding="utf-8")
        head = source.split("## 覆盖规则")[0]
        blocks = (suggestions.to_rules_markdown()
                  .split("## 建议规则 (Suggested rules)")[1]
                  .split("## 逐 SKU 数值")[0])
        merged = tmp_path / "merged.md"
        merged.write_text(head + "## 覆盖规则 (Rules)\n" + blocks, encoding="utf-8")

        reparsed = PlanningParameters(merged).resolve(resolved.frame)
        rebuilt = reparsed.frame.set_index("sku")
        direct = suggestions.frame.set_index("sku")
        for sku in direct.index:
            assert rebuilt.loc[sku, "service_level"] == direct.loc[sku, "suggested_service_level"]

    def test_file_is_written_and_names_the_hit_counts(self, suggestions, tmp_path):
        path = tmp_path / "suggested_rules.md"
        suggestions.to_rules_markdown(path)
        text = path.read_text(encoding="utf-8")
        assert "Nothing here is in force" in text
        assert "Matched **1 SKUs**" in text

    def test_csv_round_trips(self, suggestions, tmp_path):
        path = suggestions.to_csv(tmp_path / "suggestions.csv")
        reloaded = pd.read_csv(path)
        assert len(reloaded) == len(suggestions.frame)
        for col in ("sku", "lead_time_days", "suggested_safety_stock", "changes"):
            assert col in reloaded.columns


class TestSummary:

    def test_summary_flags_unmeasured_lead_times(self, suggestions):
        text = suggestions.summary()
        assert "no PO history" in text

    def test_summary_lists_rules_that_matched_nothing(self, resolved):
        result = SuggestionBuilder(CONFIG_DIR).build(resolved)
        empty = [rid for rid, n in result.hits.items() if n == 0]
        if empty:
            assert "matched nothing" in result.summary()
