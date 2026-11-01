"""
The setup surface: what a run needs from the planner, and the editor that answers it.

The screen exists because of a finding about the previous one rather than about a
missing feature. `/policy/macro` already listed every scalar the engine reads, already
grouped them by how often each is a decision — and rendered a label that had a value in
the same typeface as a target that did not. Everything looked configured, so nothing
looked asked, and the two settings that most change what the run says were the ones
nobody had filled in.

So the tests here are mostly about *states* rather than about arithmetic: which items
are outstanding, which one blocks, and whether an edit through the form lands the same
bytes an editor would.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.node import STOCK_STATUS_FILE, load_stock_status
from inventory_planning.policy import locations
from inventory_planning.policy.locations import LocationError

CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.fixture
def config(tmp_path):
    target = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, target)
    return target


def _apply(config, statuses, reason="because", by="tester"):
    basis = locations.propose(statuses, config).basis
    return locations.apply(statuses, reason=reason, by=by, basis=basis,
                           config_dir=config)


class TestTheLocationEditor:

    def test_a_declaration_is_written_and_read_back(self, config):
        _apply(config, {"01": "sellable", "02": "quarantine"})
        loaded = load_stock_status(config)
        assert loaded.statuses == {"01": "sellable", "02": "quarantine"}

    def test_the_prose_survives_an_edit(self, config):
        """
        The file explains itself in `_`-prefixed keys — what each status does, why the
        declaration is required at all — and those keys are what a planner opening it
        reads first. A `json.load`/`json.dump` round trip keeps the data and reformats
        all of it, so the diff becomes the whole file and the change becomes
        unreviewable. The diff is the thing being approved.
        """
        _apply(config, {"01": "sellable"})
        raw = json.loads((config / STOCK_STATUS_FILE).read_text(encoding="utf-8"))
        prose = [k for k in raw if k.startswith("_")]
        assert "_why_this_is_required" in prose
        assert len(prose) >= 4, prose

    def test_the_diff_is_only_the_statuses_block(self, config):
        """Blank lines between the prose blocks are how the file is read."""
        proposal = locations.propose({"01": "sellable"}, config)
        changed = [line for line in proposal.diff.splitlines()
                   if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        assert len(changed) <= 5, proposal.diff

    def test_the_nested_example_block_is_not_the_one_edited(self, config):
        """
        `_example` carries a `statuses` object of exactly the same shape. Editing it
        would leave the real declaration untouched while the form reported success.
        """
        _apply(config, {"07": "blocked"})
        raw = json.loads((config / STOCK_STATUS_FILE).read_text(encoding="utf-8"))
        assert raw["statuses"] == {"07": "blocked"}
        assert raw["_example"]["statuses"] == {"01": "sellable", "02": "quarantine",
                                               "05": "blocked"}

    def test_an_unknown_status_is_refused_before_the_file_moves(self, config):
        before = (config / STOCK_STATUS_FILE).read_text(encoding="utf-8")
        with pytest.raises(LocationError, match="sellabel"):
            _apply(config, {"01": "sellabel"})
        assert (config / STOCK_STATUS_FILE).read_text(encoding="utf-8") == before

    def test_a_stale_basis_is_refused(self, config):
        """
        Approving a diff approves *those bytes*. A file that moved in between lands a
        different change than the one on screen.
        """
        stale = locations.propose({"01": "sellable"}, config).basis
        _apply(config, {"02": "quarantine"})
        with pytest.raises(LocationError, match="changed since"):
            locations.apply({"01": "sellable"}, reason="r", by="b", basis=stale,
                            config_dir=config)

    def test_both_a_reason_and_a_name_are_required(self, config):
        basis = locations.propose({"01": "sellable"}, config).basis
        with pytest.raises(LocationError):
            locations.apply({"01": "sellable"}, reason="", by="b", basis=basis,
                            config_dir=config)
        with pytest.raises(LocationError):
            locations.apply({"01": "sellable"}, reason="r", by="", basis=basis,
                            config_dir=config)

    def test_a_non_change_is_refused(self, config):
        """A change log that records non-changes stops being a record of what moved."""
        _apply(config, {"01": "sellable"})
        with pytest.raises(LocationError, match="already says"):
            _apply(config, {"01": "sellable"})

    def test_the_change_is_logged_with_who_and_why(self, config):
        _apply(config, {"01": "sellable"}, reason="01 is finished goods", by="hongda")
        entries = locations.history(config)
        assert len(entries) == 1
        assert entries[0]["by"] == "hongda"
        assert entries[0]["reason"] == "01 is finished goods"
        assert entries[0]["added"] == ["01"]

    def test_withdrawing_a_code_is_visible_as_a_removal(self, config):
        _apply(config, {"01": "sellable", "02": "quarantine"})
        proposal = locations.propose({"01": "sellable"}, config)
        assert proposal.removed == ["02"]

    def test_the_wildcard_round_trips(self, config):
        _apply(config, {"*": "sellable", "02": "quarantine"})
        loaded = load_stock_status(config)
        assert loaded.wildcard == "sellable"
        assert loaded.statuses == {"02": "quarantine"}


class TestTheSetupEndpoint:

    @pytest.fixture
    def client(self, config, tmp_path):
        from fastapi.testclient import TestClient

        from inventory_planning.api.app import create_app

        return TestClient(create_app(config_dir=config,
                                     output_dir=tmp_path / "out",
                                     store_root=tmp_path / "store"))

    def test_it_names_three_things_and_counts_them(self, client):
        body = client.get("/setup").json()
        assert body["total"] == 3
        assert {i["key"] for i in body["items"]} == {
            "locations", "dioh_target_days", "otd_target_crsd"}

    def test_the_shipped_state_is_one_of_three(self, client):
        """
        The OTD target ships at 0.95 and the other two ship unstated — which is the
        state the screen was built for, and the state the old one made invisible.
        """
        body = client.get("/setup").json()
        assert body["set"] == 1
        assert not body["complete"]

    def test_only_the_locations_item_can_block(self, client):
        body = client.get("/setup").json()
        blocking = [i["key"] for i in body["items"] if i["blocking"]]
        assert blocking == ["locations"]

    def test_nothing_blocks_until_a_second_location_appears(self, client):
        """
        Mirrors `consolidate_to_planning_grain` exactly, and has to: a screen asking
        for more than the run requires trains planners to ignore it. With nothing
        landed there is no code to declare, so nothing is held up.
        """
        body = client.get("/setup").json()
        assert not body["blocked"]
        assert not next(i for i in body["items"] if i["key"] == "locations")["blocking_now"]

    def test_setting_a_target_moves_the_count(self, client, config):
        _apply(config, {"01": "sellable"})
        body = client.get("/setup").json()
        assert body["set"] == 2
        assert next(i for i in body["items"]
                    if i["key"] == "locations")["state"] == "set"

    def test_each_item_says_what_leaving_it_unset_costs(self, client):
        """
        The reason anyone stops and answers. A category — "target" — is not one.
        """
        for item in client.get("/setup").json()["items"]:
            assert len(item["why"]) > 80, item["key"]
            assert item["question"].endswith("?"), item["key"]

    def test_a_malformed_targets_file_is_reported_not_swallowed(self, client, config):
        (config / "targets.json").write_text(
            json.dumps({"dioh_target_days": 30, "inventory_target_value": 5_000_000}),
            encoding="utf-8")
        body = client.get("/setup").json()
        targets = [i for i in body["items"] if i["key"].endswith(("days", "crsd"))]
        assert all(i["state"] == "refused" for i in targets)
        assert all("twice" in i["note"] for i in targets)

    def test_the_editor_is_reachable_over_http(self, client):
        shown = client.put("/setup/locations",
                           json={"statuses": {"01": "sellable"}}).json()
        assert shown["is_change"] and shown["added"] == ["01"]
        done = client.put("/setup/locations", json={
            "statuses": {"01": "sellable"}, "apply": True,
            "basis": shown["basis"], "reason": "finished goods", "by": "tester"})
        assert done.status_code == 200
        assert done.json()["statuses"] == {"01": "sellable"}

    def test_a_refused_edit_comes_back_as_a_conflict(self, client):
        refused = client.put("/setup/locations", json={"statuses": {"01": "nope"}})
        assert refused.status_code == 409
        assert "nope" in refused.json()["detail"]

    def test_statuses_are_required(self, client):
        assert client.put("/setup/locations", json={}).status_code == 400


class TestTheRefusalReachesThePlanner:
    """
    How the one blocking check presents itself when it fires.

    A refusal is only worth having if it is answered, and the message is written to be
    answerable — it names the codes, the stock behind each, the file and what every
    status does. Delivered as a traceback none of that is read: the planner sees a
    crash, and a tool that crashes is broken rather than asking a question.
    """

    def test_the_message_is_printed_and_the_exit_code_is_not_zero(self, capsys,
                                                                  monkeypatch):
        """
        Exit 2, not 0. `main` returned a code the `__main__` block then threw away, so
        a scheduled run that declined to read the stock was recorded as a run that
        succeeded — the one way a refusal is worse than no check at all.
        """
        from inventory_planning import cli
        from inventory_planning.node import UndeclaredStockLocation

        def refuse():
            raise UndeclaredStockLocation("02 — 469 on hand across 4 rows")

        monkeypatch.setattr(cli, "_main", refuse)
        assert cli.main() == 2
        printed = capsys.readouterr().out
        assert "02 — 469 on hand" in printed
        assert "Nothing was written" in printed

    def test_other_failures_are_not_dressed_up_as_refusals(self, monkeypatch):
        """
        Scoped to the one exception. Everything else reaching there is a defect, and a
        defect printed as a tidy message is a defect nobody reports.
        """
        from inventory_planning import cli

        def boom():
            raise KeyError("some_column")

        monkeypatch.setattr(cli, "_main", boom)
        with pytest.raises(KeyError):
            cli.main()

    def test_the_module_entry_point_exits_with_the_code(self):
        source = (Path(__file__).parents[1]
                  / "inventory_planning" / "cli.py").read_text(encoding="utf-8")
        assert "sys.exit(main())" in source
