"""
Editing the rules through a form, and what the file has to keep.

INTERFACE.md §7 revised the position this replaces: read-only was argued as the design
because a rule wants review, a diff, a rationale and an owner and markdown in git gives
all four — an argument about *storage* that was allowed to decide *who may operate it*.
All four survive a form because the storage does not move, and that is what these pin.

Mostly they are about the file. One value changes and the comment above it does not; a
rationale nobody touched is not reflowed; the rule's own `rationale`, `owner` and `date`
move with the change, because that is where the next reader looks. And the file still
loads afterwards — checked by loading it with the loader the pipeline uses, never by a
second copy of its rules here.
"""

import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.policy import rules_edit as R
from inventory_planning.policy.parameters import PlanningParameters
from inventory_planning.policy.rules_edit import RuleEditError

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def config(tmp_path):
    """A copy of the real rules, because the point is that this file survives."""
    target = tmp_path / "config"
    target.mkdir()
    for path in CONFIG_DIR.glob("*"):
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _rules(config):
    return PlanningParameters(config / "planning_parameters.md").rules


def _by_id(config, rule_id):
    return {r.rule_id: r for r in _rules(config)}[rule_id]


def _apply(config, propose, **kw):
    return R.apply(propose, reason=kw.get("reason", "a recorded reason"),
                   by=kw.get("by", "jfanhon"), basis=propose().basis,
                   config_dir=config)


class TestTheFileSurvives:

    def test_changing_one_value_changes_two_lines(self, config):
        """The value, and the date that says when it moved. Nothing else."""
        path = config / "planning_parameters.md"
        before = path.read_text(encoding="utf-8")
        _apply(config, lambda: R.propose_edit(
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config))

        after = path.read_text(encoding="utf-8")
        changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines())
                   if a != b]
        assert len(changed) == 2
        assert ("  review_period_days: 7", "  review_period_days: 14") in changed
        assert len(before.splitlines()) == len(after.splitlines())

    def test_a_rationale_nobody_touched_is_not_reflowed(self, config):
        """
        Re-rendering the rule from its parsed form would rewrap the paragraph and put
        six lines in the diff for a change to one number. The diff is what is being
        approved, so the smallest true diff is the product.
        """
        before = _by_id(config, "R-001").rationale
        proposal = R.propose_edit("R-001", {"set": {"review_period_days": 14}},
                                  config_dir=config)
        rationale_lines = [l for l in proposal.diff.splitlines()
                           if l.startswith(("+", "-")) and "cycle stock" in l]
        assert rationale_lines == []
        _apply(config, lambda: R.propose_edit(
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config))
        assert _by_id(config, "R-001").rationale == before

    def test_the_comment_beside_an_untouched_parameter_stays(self, config):
        """
        A note written next to a number is often the only record of why that number.
        Reconciling the `set` block keeps the line, comment and all, when the value is
        unchanged.
        """
        path = config / "planning_parameters.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(
            "  review_period_days: 7\n  service_level: 0.98",
            "  review_period_days: 7   # agreed with the DC manager\n"
            "  service_level: 0.98"), encoding="utf-8")

        _apply(config, lambda: R.propose_edit(
            "R-002", {"set": {"review_period_days": 7, "service_level": 0.99}},
            config_dir=config))
        assert "# agreed with the DC manager" in path.read_text(encoding="utf-8")

    def test_removing_a_rule_leaves_the_spacing_the_file_had(self, config):
        """
        A removal that took both blank lines would close the gap between the rules
        either side; one that took neither would open a widening hole every time.
        """
        path = config / "planning_parameters.md"
        _apply(config, lambda: R.propose_remove("R-002", config_dir=config))
        text = path.read_text(encoding="utf-8")
        assert "```\n\n### R-003" in text
        assert "\n\n\n" not in text

    def test_add_then_remove_leaves_the_file_as_it_was(self, config):
        path = config / "planning_parameters.md"
        before = path.read_text(encoding="utf-8")
        rule = {"rule_id": "R-009", "name": "temporary", "scope": 'incoterm == "DDP"',
                "set": {"service_level": 0.92}, "rationale": "a reason", "owner": "FHD"}
        _apply(config, lambda: R.propose_add(rule, config_dir=config))
        _apply(config, lambda: R.propose_remove("R-009", config_dir=config))
        assert path.read_text(encoding="utf-8") == before


class TestWhatTheEngineReadsAfterwards:

    def test_an_edited_value_is_what_the_loader_returns(self, config):
        _apply(config, lambda: R.propose_edit(
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config))
        assert _by_id(config, "R-001").overrides == {"review_period_days": 14}

    def test_a_number_stays_a_number(self, config):
        """
        `0.98` written as the string "0.98" loads as a string, and the arithmetic
        downstream compares it to a float and never matches — a rule that appears to
        apply and decides nothing.
        """
        _apply(config, lambda: R.propose_edit(
            "R-002", {"set": {"review_period_days": 7, "service_level": 0.99}},
            config_dir=config))
        assert _by_id(config, "R-002").overrides["service_level"] == 0.99
        assert isinstance(_by_id(config, "R-002").overrides["service_level"], float)

    def test_an_added_rule_is_last_and_therefore_wins(self, config):
        """
        Rules apply in file order and later ones win, so where a rule sits is part of
        what it does. Last is the only position with a statable meaning.
        """
        proposal = R.propose_add(
            {"rule_id": "R-009", "name": "later", "scope": 'abc_class == "A"',
             "set": {"review_period_days": 21}, "rationale": "a reason"},
            config_dir=config)
        assert proposal.order[-1] == "R-009"
        assert "wins over every rule above it" in proposal.note

        _apply(config, lambda: R.propose_add(
            {"rule_id": "R-009", "name": "later", "scope": 'abc_class == "A"',
             "set": {"review_period_days": 21}, "rationale": "a reason"},
            config_dir=config))
        assert [r.rule_id for r in _rules(config)][-1] == "R-009"

    def test_a_removed_rule_is_gone_from_the_engine(self, config):
        _apply(config, lambda: R.propose_remove("R-002", config_dir=config))
        assert "R-002" not in [r.rule_id for r in _rules(config)]


class TestTheRationaleAndTheOwnerMoveWithTheRule:
    """
    The four properties INTERFACE.md names. Three of them live in the file itself, and
    an edit that left them saying what the rule used to be for would be worse than no
    edit — it would be a rule whose stated reason belongs to a different rule.
    """

    def test_the_date_is_stamped_rather_than_asked_for(self, config):
        assert _by_id(config, "R-001").date == "2026-08-02"
        _apply(config, lambda: R.propose_edit(
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config))
        assert _by_id(config, "R-001").date == date.today().isoformat()

    def test_a_new_rationale_and_owner_land_in_the_file(self, config):
        _apply(config, lambda: R.propose_edit(
            "R-001", {"rationale": "  the A class shrank after the  reclassification ",
                      "owner": "LMH"},
            config_dir=config))
        rule = _by_id(config, "R-001")
        assert "reclassification" in rule.rationale
        assert "  " not in rule.rationale.strip()          # collapsed on the way in
        assert rule.owner == "LMH"

    def test_the_change_is_logged_with_who_and_why(self, config):
        _apply(config, lambda: R.propose_edit(
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config),
            reason="ordering  cost   rose", by="jfanhon")
        entry, = R.history(config)
        assert entry["action"] == "edit" and entry["rule_id"] == "R-001"
        assert entry["by"] == "jfanhon"
        assert entry["reason"] == "ordering cost rose"      # whitespace collapsed
        assert entry["from"]["set"] == {"review_period_days": 7}
        assert entry["to"]["set"] == {"review_period_days": 14}

    def test_a_removal_is_the_one_case_the_log_is_the_only_record(self, config):
        """Afterwards there is no rule left to carry its own reason."""
        _apply(config, lambda: R.propose_remove("R-002", config_dir=config),
               reason="actuators moved to the other DC")
        entry, = R.history(config)
        assert entry["action"] == "remove"
        assert entry["from"]["rule_id"] == "R-002"
        assert entry["reason"] == "actuators moved to the other DC"

    def test_the_rule_log_and_the_macro_log_are_one_file_told_apart(self, config):
        from inventory_planning.policy import macro

        proposal = macro.propose("days_per_year", 250, config_dir=config)
        macro.apply("days_per_year", 250, reason="working days", by="jfanhon",
                    basis=proposal.basis, config_dir=config)
        _apply(config, lambda: R.propose_edit(
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config))

        assert [e["rule_id"] for e in R.history(config)] == ["R-001"]
        assert [e["setting"] for e in macro.history(config)] == ["days_per_year"]


class TestWhatIsRefused:
    """
    All of it before anything is written, so an approved diff is one already shown to
    work. Most of it by the loader rather than by a rule restated here.
    """

    def test_a_scope_that_does_not_parse(self, config):
        with pytest.raises(RuleEditError, match="invalid scope"):
            R.propose_edit("R-001", {"scope": 'abc_class === "A"'}, config_dir=config)

    def test_a_rule_left_without_a_rationale(self, config):
        """Enforced in `Rule.__post_init__`, which is where it should be."""
        with pytest.raises(RuleEditError, match="no rationale"):
            R.propose_edit("R-001", {"rationale": "   "}, config_dir=config)

    def test_a_rule_that_would_set_nothing(self, config):
        with pytest.raises(RuleEditError, match="decides nothing"):
            R.propose_edit("R-001", {"set": {}}, config_dir=config)

    def test_the_rule_id_is_not_editable(self, config):
        """
        The manifest records each rule's hits against it, so renaming one detaches
        every count ever recorded for it from the rule it belongs to.
        """
        with pytest.raises(RuleEditError, match="not editable"):
            R.propose_edit("R-001", {"rule_id": "R-099"}, config_dir=config)

    def test_a_field_the_file_does_not_have(self, config):
        with pytest.raises(RuleEditError, match="Not editable"):
            R.propose_edit("R-001", {"priority": 3}, config_dir=config)

    def test_a_rule_that_does_not_exist(self, config):
        with pytest.raises(RuleEditError, match="No rule R-099"):
            R.propose_edit("R-099", {"set": {"a": 1}}, config_dir=config)

    def test_adding_one_that_already_exists(self, config):
        with pytest.raises(RuleEditError, match="already exists"):
            R.propose_add({"rule_id": "R-001", "name": "x", "scope": "a == 1",
                           "set": {"b": 1}, "rationale": "y"}, config_dir=config)

    def test_a_new_rule_without_an_id_a_name_or_a_reason(self, config):
        base = {"rule_id": "R-009", "name": "x", "scope": 'abc_class == "A"',
                "set": {"review_period_days": 7}, "rationale": "y"}
        for missing in ("rule_id", "name", "rationale", "scope", "set"):
            with pytest.raises(RuleEditError):
                R.propose_add({**base, missing: ""}, config_dir=config)

    def test_a_refused_change_writes_nothing(self, config):
        path = config / "planning_parameters.md"
        before = path.read_text(encoding="utf-8")
        for bad in ({"scope": "=="}, {"set": {}}, {"rationale": ""}):
            with pytest.raises(RuleEditError):
                R.propose_edit("R-001", bad, config_dir=config)
        assert path.read_text(encoding="utf-8") == before

    def test_proposing_writes_nothing(self, config):
        path = config / "planning_parameters.md"
        before = path.read_text(encoding="utf-8")
        R.propose_edit("R-001", {"set": {"review_period_days": 14}}, config_dir=config)
        R.propose_remove("R-002", config_dir=config)
        R.propose_add({"rule_id": "R-009", "name": "x", "scope": 'abc_class == "A"',
                       "set": {"review_period_days": 7}, "rationale": "y"},
                      config_dir=config)
        assert path.read_text(encoding="utf-8") == before


class TestApprovingADiffAndNotAWish:

    def test_a_file_that_moved_since_the_diff_is_refused(self, config):
        path = config / "planning_parameters.md"
        propose = lambda: R.propose_edit(                        # noqa: E731
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config)
        basis = propose().basis
        path.write_text(path.read_text(encoding="utf-8").replace(
            "date: 2026-08-02", "date: 2026-08-03", 1), encoding="utf-8")

        with pytest.raises(RuleEditError, match="has changed since that diff"):
            R.apply(propose, reason="r", by="jfanhon", basis=basis, config_dir=config)
        assert _by_id(config, "R-001").overrides == {"review_period_days": 7}

    def test_an_apply_with_no_basis_is_refused(self, config):
        propose = lambda: R.propose_edit(                        # noqa: E731
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config)
        with pytest.raises(RuleEditError, match="has changed since that diff"):
            R.apply(propose, reason="r", by="jfanhon", basis="", config_dir=config)

    def test_a_change_that_moves_nothing_is_not_logged_as_one(self, config):
        propose = lambda: R.propose_edit(                        # noqa: E731
            "R-001", {"name": "A 类物料周度 review"}, config_dir=config)
        # Only `date` would move, which on its own is not a change to the rule.
        proposal = propose()
        if not proposal.unchanged:
            _apply(config, propose)
            assert len(R.history(config)) == 1
            return
        with pytest.raises(RuleEditError, match="nothing to apply"):
            R.apply(propose, reason="r", by="jfanhon", basis=proposal.basis,
                    config_dir=config)
        assert R.history(config) == []

    def test_it_must_say_why_and_who(self, config):
        propose = lambda: R.propose_edit(                        # noqa: E731
            "R-001", {"set": {"review_period_days": 14}}, config_dir=config)
        basis = propose().basis
        with pytest.raises(RuleEditError, match="must carry a reason"):
            R.apply(propose, reason="  ", by="jfanhon", basis=basis, config_dir=config)
        with pytest.raises(RuleEditError, match="name who made it"):
            R.apply(propose, reason="r", by="", basis=basis, config_dir=config)


class TestWhatTheScreenIsToldAboutReach:
    """
    A scope is a question about a frame of SKUs and there is no frame in an interface,
    so an edited rule's reach cannot be stated here. Saying so is the point: a wrong
    count beside a scope is an invitation to write the scope around it.
    """

    def test_an_edit_says_the_counts_beside_it_are_now_stale(self, config):
        proposal = R.propose_edit("R-001", {"scope": 'abc_class == "B"'},
                                  config_dir=config)
        assert "last run under the rules as they were" in proposal.note
        assert "the next run measures the new scope" in proposal.note

    def test_every_proposal_says_where_the_rule_sits_afterwards(self, config):
        """Half of what a rule does is which rules come after it."""
        edit = R.propose_edit("R-001", {"set": {"review_period_days": 14}},
                              config_dir=config)
        removal = R.propose_remove("R-002", config_dir=config)
        assert edit.order == ["R-001", "R-002", "R-003", "R-004"]
        assert removal.order == ["R-001", "R-003", "R-004"]
