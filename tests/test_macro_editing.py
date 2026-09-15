"""
Editing a macro scalar through a form — and the properties the file has to keep.

The claim being tested is INTERFACE.md's: git stays the store, and the form is a
validator with a nicer keyboard. So the tests are mostly about the file. One line
changes and no other byte moves; the comments that explain a setting survive; the value
the pipeline reads afterwards is the one that was approved; and a change that would
leave the file unloadable never reaches it.

The second half is the propose-then-approve contract. What is approved is a diff against
specific bytes, not a wish about a setting, and the tests hold that: a file that moved in
between is a refusal, and an approval with nothing to approve against is one too.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.policy import macro
from inventory_planning.policy.macro import MacroError

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def config(tmp_path):
    """A copy of the real config, because the point is that these files survive."""
    target = tmp_path / "config"
    target.mkdir()
    for path in CONFIG_DIR.glob("*"):
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _apply(config, name, value, **kw):
    proposal = macro.propose(name, value, config_dir=config)
    return macro.apply(name, value, reason=kw.get("reason", "a recorded reason"),
                       by=kw.get("by", "jfanhon"), basis=proposal.basis,
                       config_dir=config)


class TestTheFileSurvives:
    """
    A round trip through `json.load`/`json.dump` would have passed a test that only
    checked the value. It would also have reformatted sixty lines of documentation the
    files carry in their own keys, and turned a one-line review into an unreadable one.
    """

    def test_one_line_changes_and_nothing_else_does(self, config):
        before = (config / "planning_parameters.md").read_text(encoding="utf-8")
        _apply(config, "days_per_year", 250)
        after = (config / "planning_parameters.md").read_text(encoding="utf-8")

        changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines())
                   if a != b]
        assert changed == [("days_per_year: 365", "days_per_year: 250")]
        assert len(before.splitlines()) == len(after.splitlines())

    def test_the_comment_explaining_the_setting_is_still_there(self, config):
        _apply(config, "cycle_stock_basis", "average")
        text = (config / "planning_parameters.md").read_text(encoding="utf-8")
        assert "peak    = D × R" in text
        assert "cycle_stock_basis: average" in text

    def test_a_json_file_keeps_its_prose_keys_and_its_shape(self, config):
        before = (config / "fx_rates.json").read_text(encoding="utf-8")
        _apply(config, "reporting_currency", "CNY")
        after = (config / "fx_rates.json").read_text(encoding="utf-8")

        assert after.count("\n") == before.count("\n")
        assert "_what_this_is" in after and "_placeholder_note" in after
        assert json.loads(after)["reporting_currency"] == "CNY"
        assert json.loads(after)["rates"] == json.loads(before)["rates"]

    def test_the_diff_is_the_size_a_person_can_read(self, config):
        proposal = macro.propose("days_per_year", 250, config_dir=config)
        added = [l for l in proposal.diff.splitlines() if l.startswith("+")]
        removed = [l for l in proposal.diff.splitlines() if l.startswith("-")]
        # One of each, plus the `+++`/`---` file headers.
        assert len(added) == 2 and len(removed) == 2


class TestWhatTheEngineReadsAfterwards:

    def test_the_pipeline_loader_sees_the_new_value(self, config):
        from inventory_planning.policy.parameters import PlanningParameters

        _apply(config, "safety_stock_exposure", "lt_only")
        params = PlanningParameters(config / "planning_parameters.md")
        assert params.conventions["safety_stock_exposure"] == "lt_only"

    def test_the_fx_table_sees_the_new_reporting_currency(self, config):
        from inventory_planning.fx import FxTable

        _apply(config, "reporting_currency", "GBP")
        assert FxTable.load(config).reporting_currency == "GBP"

    def test_a_number_is_written_as_a_number_not_a_string(self, config):
        _apply(config, "transit_share_of_lt", "0.6")
        assert macro.current(config)["transit_share_of_lt"] == 0.6
        assert isinstance(macro.current(config)["transit_share_of_lt"], float)


class TestWhatIsRefused:
    """
    Each of these is refused before anything is written, so an approved diff is one
    that has already been shown to work. An apply that could still fail would make the
    diff a suggestion rather than a promise.
    """

    def test_a_setting_nothing_reads_is_not_editable(self, config):
        with pytest.raises(MacroError, match="not an editable setting"):
            macro.propose("echelon_level", 2, config_dir=config)

    def test_a_value_outside_the_branches_the_engine_takes(self, config):
        """
        `pipeline_basis: incoterm_awre` parses, loads, and quietly sends every SKU down
        the other branch. Nothing downstream would report it, so it is stopped here.
        """
        with pytest.raises(MacroError, match="must be one of"):
            macro.propose("pipeline_basis", "incoterm_awre", config_dir=config)

    def test_a_convention_the_loader_itself_rejects(self, config, monkeypatch):
        """
        Checked by loading rather than by a rule restated here. `quantity_rounding` is
        validated in `analytics.rounding`, which on a real run is an hour of work away
        from where the value is set.
        """
        import dataclasses

        monkeypatch.setitem(
            macro.BY_NAME, "quantity_rounding",
            dataclasses.replace(macro.BY_NAME["quantity_rounding"],
                                choices=("integer", "none", "ceil")))
        with pytest.raises(MacroError, match="refused by the loader"):
            macro.propose("quantity_rounding", "ceil", config_dir=config)
        assert macro.current(config)["quantity_rounding"] == "integer"

    def test_text_where_a_number_belongs(self, config):
        with pytest.raises(MacroError, match="must be a number"):
            macro.propose("days_per_year", "three hundred", config_dir=config)

    def test_a_refused_change_leaves_the_file_untouched(self, config):
        before = (config / "planning_parameters.md").read_text(encoding="utf-8")
        for bad in (("cycle_stock_basis", "pekk"), ("days_per_year", "x")):
            with pytest.raises(MacroError):
                macro.propose(*bad, config_dir=config)
        assert (config / "planning_parameters.md").read_text(encoding="utf-8") == before

    def test_a_setting_that_appears_twice_is_a_refusal_not_a_guess(self, config):
        """
        Two matches means either the file says the same thing in two places — where
        editing one and not the other is worse than editing neither — or the anchor has
        caught something that is not the setting.
        """
        path = config / "planning_parameters.md"
        path.write_text(path.read_text(encoding="utf-8")
                        + "\n```yaml\ndays_per_year: 365\n```\n", encoding="utf-8")
        with pytest.raises(MacroError, match="appears 2 times"):
            macro.propose("days_per_year", 250, config_dir=config)


class TestApprovingADiffAndNotAWish:

    def test_the_apply_is_pinned_to_the_bytes_that_were_diffed(self, config):
        proposal = macro.propose("days_per_year", 250, config_dir=config)
        path = config / "planning_parameters.md"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "transit_share_of_lt: 0.45", "transit_share_of_lt: 0.5"), encoding="utf-8")

        with pytest.raises(MacroError, match="has changed since that diff"):
            macro.apply("days_per_year", 250, reason="r", by="jfanhon",
                        basis=proposal.basis, config_dir=config)
        assert macro.current(config)["days_per_year"] == 365

    def test_an_apply_with_no_diff_behind_it_is_refused(self, config):
        with pytest.raises(MacroError, match="has changed since that diff"):
            macro.apply("days_per_year", 250, reason="r", by="jfanhon", basis="",
                        config_dir=config)

    def test_proposing_changes_nothing(self, config):
        before = (config / "planning_parameters.md").read_text(encoding="utf-8")
        macro.propose("days_per_year", 250, config_dir=config)
        macro.propose("cycle_stock_basis", "average", config_dir=config)
        assert (config / "planning_parameters.md").read_text(encoding="utf-8") == before

    def test_setting_a_value_to_what_it_already_is_is_not_a_change(self, config):
        """
        Not an error at proposal time — a form resubmitting the value on screen is not
        making a mistake — but not a log entry either, because nothing moved.
        """
        proposal = macro.propose("days_per_year", 365, config_dir=config)
        assert proposal.unchanged and proposal.diff == ""
        with pytest.raises(MacroError, match="nothing to apply"):
            macro.apply("days_per_year", 365, reason="r", by="jfanhon",
                        basis=proposal.basis, config_dir=config)
        assert macro.history(config) == []


class TestTheRationaleAndTheOwner:
    """
    The two things the file cannot hold. A diff is recoverable from the file's own
    history and the value is in the file; why someone moved a convention that restates
    every figure in the run is nowhere unless it is written down.
    """

    def test_a_change_without_a_reason_is_refused(self, config):
        proposal = macro.propose("days_per_year", 250, config_dir=config)
        with pytest.raises(MacroError, match="must carry a reason"):
            macro.apply("days_per_year", 250, reason="   ", by="jfanhon",
                        basis=proposal.basis, config_dir=config)

    def test_a_change_without_a_name_is_refused(self, config):
        proposal = macro.propose("days_per_year", 250, config_dir=config)
        with pytest.raises(MacroError, match="name who made it"):
            macro.apply("days_per_year", 250, reason="counted in working days", by="",
                        basis=proposal.basis, config_dir=config)

    def test_the_log_says_who_what_and_why(self, config):
        _apply(config, "days_per_year", 250, reason="finance  counts working  days",
               by="jfanhon")
        entry, = macro.history(config)
        assert entry["setting"] == "days_per_year"
        assert (entry["from"], entry["to"]) == (365, 250)
        assert entry["by"] == "jfanhon"
        assert entry["reason"] == "finance counts working days"   # whitespace collapsed
        assert entry["file"] == "planning_parameters.md"

    def test_the_log_is_append_only_and_newest_first(self, config):
        _apply(config, "days_per_year", 250, reason="first")
        _apply(config, "cycle_stock_basis", "average", reason="second")
        assert [e["reason"] for e in macro.history(config)] == ["second", "first"]

    def test_a_malformed_line_does_not_lose_the_rest(self, config):
        _apply(config, "days_per_year", 250, reason="kept")
        log = config / macro.CHANGE_LOG_NAME
        log.write_text("{not json\n" + log.read_text(encoding="utf-8"),
                       encoding="utf-8")
        assert [e["reason"] for e in macro.history(config)] == ["kept"]

    def test_the_log_travels_with_the_config_directory(self, config):
        """
        Beside the files it describes rather than under the output directory: a config
        directory copied to another machine carries its own history of who changed what.
        """
        _apply(config, "days_per_year", 250)
        assert (config / macro.CHANGE_LOG_NAME).exists()
