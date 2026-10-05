"""
The standing inventory target, and the way in.

`TargetPlanner` has existed as long as the should-be engine, and nothing could start
it. `target_value` appeared in two orchestrator signatures and in the KPI report's
renderer, and in no CLI flag, no config file and no form — so every run computed
`frontier = None`, and the ordered set of moves the module exists to produce was
unreachable from any surface a person uses.

What is tested here is mostly the difference between *null* and *zero*. Null is nobody
having said what the balance should be, and the run has no opinion. Zero is a target,
and a reachable one for a catalogue being discontinued. A reader that conflated them
would turn "we have not decided" into "cut everything", which is the one wrong answer
that still looks like an answer.
"""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.cli import _target_date, build_parser  # noqa: E402
from inventory_planning.policy import macro  # noqa: E402
from inventory_planning.policy.macro import MacroError  # noqa: E402
from inventory_planning.policy.target import (  # noqa: E402
    StatedTarget, StatedTargetError, stated_target,
)

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def config(tmp_path):
    target = tmp_path / "config"
    target.mkdir()
    for path in CONFIG_DIR.glob("*"):
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _write(config, **fields):
    (config / "targets.json").write_text(json.dumps(fields), encoding="utf-8")


def _apply(config, name, value):
    proposal = macro.propose(name, value, config_dir=config)
    macro.apply(name, value, reason="a recorded reason", by="jfanhon",
                basis=proposal.basis, config_dir=config)


class TestNullIsNotZero:

    def test_the_shipped_file_states_no_target(self, config):
        """
        The default has to be "nobody has said", not a number somebody would have to
        notice and undo. A seeded target would be a figure every run planned towards
        that no planner chose.
        """
        assert not stated_target(config)

    def test_zero_is_a_target(self, config):
        _write(config, inventory_target_value=0)
        stated = stated_target(config)
        assert bool(stated) is True and stated.value == 0.0

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert stated_target(tmp_path / "nowhere") == StatedTarget(
            source="no targets.json")


class TestWhatItRefuses:
    """
    Every other reading in this package degrades: a missing rate blanks a column, a
    missing adapter is drafted from the profile. There is no degraded reading of "cut
    inventory to $5M" — a target nobody can parse, silently dropped, produces a run
    indistinguishable from a run nobody set a target on.
    """

    def test_a_date_with_no_value_names_the_way_out(self, config):
        _write(config, inventory_target_date="2026-12-31")
        with pytest.raises(StatedTargetError) as refused:
            stated_target(config)
        # The state is reachable by clearing the value first, so the refusal has to say
        # which order gets out of it rather than only why it is wrong.
        assert "clear the date first" in str(refused.value)

    def test_a_date_it_would_misread_is_refused_rather_than_guessed(self, config):
        _write(config, inventory_target_value=5e6, inventory_target_date="31/12/2026")
        with pytest.raises(StatedTargetError):
            stated_target(config)

    def test_a_value_that_is_not_money_is_refused(self, config):
        _write(config, inventory_target_value="five million")
        with pytest.raises(StatedTargetError):
            stated_target(config)

    def test_true_is_not_a_number(self, config):
        """`isinstance(True, int)` — the one that gets through a plain numeric check."""
        _write(config, inventory_target_value=True)
        with pytest.raises(StatedTargetError):
            stated_target(config)


class TestTheFormCanSetAndWithdrawIt:
    """
    The first two settings on the macro screen that are not conventions, and the first
    whose correct state can be "not set". A convention has no unset value — the engine
    takes one branch or the other either way.
    """

    def test_it_round_trips_through_the_form(self, config):
        _apply(config, "inventory_target_value", 5_000_000)
        _apply(config, "inventory_target_date", "2026-12-31")
        stated = stated_target(config)
        assert stated.value == 5_000_000.0
        assert stated.deadline == date(2026, 12, 31)

    def test_a_target_can_be_withdrawn(self, config):
        _apply(config, "inventory_target_value", 5_000_000)
        _apply(config, "inventory_target_date", "2026-12-31")
        _apply(config, "inventory_target_date", "")
        _apply(config, "inventory_target_value", "none")
        assert not stated_target(config)
        assert macro.current(config)["inventory_target_value"] is None

    def test_cleared_is_null_rather_than_zero_in_the_file(self, config):
        _apply(config, "inventory_target_value", 5_000_000)
        _apply(config, "inventory_target_value", "")
        raw = json.loads((config / "targets.json").read_text(encoding="utf-8"))
        assert raw["inventory_target_value"] is None

    def test_the_loader_refuses_a_date_the_run_would_refuse(self, config):
        _apply(config, "inventory_target_value", 5_000_000)
        with pytest.raises(MacroError):
            _apply(config, "inventory_target_date", "2026-13-40")

    def test_the_prose_keys_survive_an_edit(self, config):
        """
        The file explains itself in `_`-prefixed keys, as the others here do. A
        `json.load`/`json.dump` round trip would keep the value and reformat all of it.
        """
        _apply(config, "inventory_target_value", 5_000_000)
        raw = json.loads((config / "targets.json").read_text(encoding="utf-8"))
        assert "_what_this_is" in raw and "_not_here_yet" in raw


class TestTheCommandLine:

    def test_a_target_and_a_date_are_read(self):
        args = build_parser().parse_args(["exports/", "--target", "5000000",
                                          "--target-by", "2026-12-31"])
        assert args.target == 5_000_000.0
        assert _target_date(args) == date(2026, 12, 31)

    def test_a_date_on_its_own_is_refused(self):
        args = build_parser().parse_args(["exports/", "--target-by", "2026-12-31"])
        with pytest.raises(SystemExit):
            _target_date(args)

    def test_a_date_it_would_misread_is_refused(self):
        args = build_parser().parse_args(["exports/", "--target", "1",
                                          "--target-by", "31/12/2026"])
        with pytest.raises(SystemExit):
            _target_date(args)


class TestTheRunReachesIt:
    """
    The wiring, guarded the way `test_cli.py` guards the workspace seam — by reading the
    source. A full pipeline run is the only other way to assert it, and the defect being
    guarded against is not an arithmetic error but a line going missing: `target_value` was a
    parameter of both of these for months with nothing to fill it.
    """

    @staticmethod
    def _source(name):
        import inspect

        from inventory_planning.orchestrator import InventoryPlanner

        return inspect.getsource(getattr(InventoryPlanner, name))

    @pytest.mark.parametrize("stage", ["run_policy_analysis", "run_kpi_review"])
    def test_both_stages_fall_back_to_the_stated_target(self, stage):
        assert "stated_target" in self._source(stage), (
            f"{stage} no longer reads the standing target, so a target set in "
            f"targets.json would be in force and invisible")

    def test_a_run_with_no_target_says_so_rather_than_printing_nothing(self):
        """
        A run with no frontier and a run whose frontier was dropped look identical in
        the output. The second is what happens when a target is set somewhere the run
        does not read, which is exactly the state this change came out of.
        """
        assert "no inventory target stated" in self._source("run_policy_analysis")

    def test_an_undated_target_says_what_it_cost(self):
        assert "burn-down limit" in self._source("run_policy_analysis")


class TestTheSettingsAreGradedByWhoHasToDecide:
    """
    Thirteen settings in one flat table, sorted by which file they live in, asked a
    planner to triage the page the page existed to triage for them. `location_name` is
    a label and `cycle_stock_basis` restates every should-be figure in the workbook,
    and they were rendered identically — as were the two targets nobody had set.

    The grading lives in `macro.py` beside the settings, not in the page. A screen that
    decided for itself which settings matter would be a second opinion kept beside the
    engine's, and the hardcoded currency list is what that looks like after a few weeks.
    """

    def test_every_setting_is_graded(self):
        assert all(s.group in macro.GROUPS for s in macro.SETTINGS)

    def test_a_target_outranks_a_convention_which_outranks_a_label(self):
        order = {name: i for i, name in enumerate(macro.GROUPS)}
        by_name = {s.name: order[s.group] for s in macro.SETTINGS}
        assert by_name["inventory_target_value"] < by_name["cycle_stock_basis"]
        assert by_name["cycle_stock_basis"] < by_name["location_name"]

    def test_the_page_decides_no_membership_of_its_own(self):
        """
        The same guard the currency list needed. A page naming settings would drift
        from the registry silently, which is the defect this grading came out of.
        """
        import re

        page = (Path(__file__).parents[1]
                / "inventory_planning/api/web/policy.js").read_text(encoding="utf-8")
        named = [s.name for s in macro.SETTINGS if f'"{s.name}"' in page]
        assert named == [], f"settings named in the page: {named}"
