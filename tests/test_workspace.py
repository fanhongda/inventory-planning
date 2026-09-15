"""
One tenant's three directories, resolved in one place.

INTERFACE.md §7's first seam. What is pinned here is mostly one property and one
non-property.

The property is **isolation, and that it is now testable**. The store path could always
be pointed somewhere safe; config and output could not, so two tenants sharing a machine
would have shared them without anything being able to say so. `Workspace.contains` is
what the test asserts against, and it resolves both sides — a symlink into another
tenant's directory is precisely the case a string prefix check would pass.

The non-property is that **the default tenant moved nothing**. A seam cut for a future
user is not allowed to change a path for the one user there is, and the whole suite
passing is the wider version of that; these are the direct assertions.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.workspace import (
    BadTenant, DEFAULT_TENANT, ENV_CONFIG, ENV_OUTPUT, ENV_STORE, Workspace,
    check_tenant,
)

REPO = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch):
    """
    A developer's own $INVENTORY_PLANNING_STORE must not decide what these assert.

    The variable is the isolation mechanism for real work, which means it is usually
    set on the machine running the tests — and a test that silently resolved to the
    developer's dev store would pass here and fail in CI, or worse, pass in both while
    testing nothing.
    """
    for var in (ENV_CONFIG, ENV_OUTPUT, ENV_STORE, "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)


class TestTheDefaultTenantMovedNothing:
    """The seam is for a future user. It may not cost the present one a changed path."""

    def test_config_and_output_are_where_they_always_were(self):
        workspace = Workspace.resolve()
        assert workspace.tenant == DEFAULT_TENANT
        assert workspace.config_dir == REPO / "config"
        assert workspace.output_dir == REPO / "output"

    def test_the_store_still_resolves_through_its_own_rules(self, monkeypatch):
        from inventory_planning.store.location import resolve_store_root

        monkeypatch.setenv(ENV_STORE, "/tmp/some-dev-store")
        assert Workspace.resolve().store_root == resolve_store_root(None)[0]

    def test_an_explicit_argument_still_wins(self, tmp_path):
        workspace = Workspace.resolve(config_dir=tmp_path / "c", output_dir=tmp_path / "o",
                                      store_root=tmp_path / "s")
        assert workspace.config_dir == tmp_path / "c"
        assert workspace.output_dir == tmp_path / "o"
        assert workspace.store_root == tmp_path / "s"
        assert set(workspace.origins.values()) == {"argument"}

    def test_the_planner_resolves_the_same_paths_it_did_before(self, tmp_path):
        """
        The orchestrator used to compute these itself. Going through the workspace has
        to produce the same answer or every existing run reads a different config.
        """
        from inventory_planning.orchestrator import InventoryPlanner

        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        assert planner.config_dir == REPO / "config"
        assert planner.output_dir == tmp_path / "out"


class TestIsolation:
    """
    The half §7 called untestable. Two tenants on one machine must not be able to read
    or write each other's anything, and something has to be able to say so.
    """

    def test_no_directory_is_shared_between_two_tenants(self):
        a = Workspace.resolve("acme")
        b = Workspace.resolve("globex")
        assert set(a.paths.values()).isdisjoint(set(b.paths.values()))
        for path in b.paths.values():
            assert not a.contains(path)
        for path in a.paths.values():
            assert not b.contains(path)

    def test_a_named_tenant_keeps_nothing_in_the_working_tree(self):
        """
        The first version of this put a tenant's config and output at
        `<repo>/tenants/<name>/...` — untracked, one `git clean -fdx` from gone, and
        dirtying `git status` on every machine that pulls. That is what
        `store/location.py` already argues against for the store, and it applies to a
        tenant's config for the same reason.
        """
        from inventory_planning.store.location import inside_repo

        workspace = Workspace.resolve("acme")
        for key, path in workspace.paths.items():
            assert not inside_repo(path), f"{key} landed in the working tree"

    def test_the_default_tenant_still_uses_the_repository_config(self):
        """
        And must: the rules are markdown in git, which is what gives them review, a
        diff, a rationale and an owner. Only a *named* tenant moves out.
        """
        from inventory_planning.store.location import inside_repo

        assert inside_repo(Workspace.resolve().config_dir)

    def test_a_config_forced_into_the_working_tree_is_reported(self, tmp_path):
        workspace = Workspace.resolve("acme", config_dir=REPO / "tenants" / "acme")
        assert any("inside the working tree" in w for w in workspace.warnings)

    def test_a_tenant_is_not_inside_the_default_workspace(self):
        """
        The default tenant's config is `<repo>/config`; a tenant's is beside the store.
        Neither contains the other, which is what stops a tenant's rules from being
        picked up by a default run.
        """
        default = Workspace.resolve()
        tenant = Workspace.resolve("acme")
        for path in tenant.paths.values():
            assert not default.contains(path)
        for path in default.paths.values():
            assert not tenant.contains(path)

    def test_a_tenant_under_an_explicit_store_still_gets_its_own_subtree(self, tmp_path):
        """
        Pointing $INVENTORY_PLANNING_STORE at a dev store is how branch work is made
        safe. If a tenant named on top of it landed in that same root, every tenant's
        facts would merge into it — the isolation failing exactly where someone thought
        they were being careful.
        """
        a = Workspace.resolve("acme", store_root=tmp_path / "dev")
        b = Workspace.resolve("globex", store_root=tmp_path / "dev")
        assert a.store_root != b.store_root
        assert a.store_root.is_relative_to(tmp_path / "dev")
        assert "tenant" in a.origins["store_root"]

    def test_the_same_holds_for_config_and_output_from_the_environment(
            self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_CONFIG, str(tmp_path / "cfg"))
        monkeypatch.setenv(ENV_OUTPUT, str(tmp_path / "out"))
        a = Workspace.resolve("acme")
        b = Workspace.resolve("globex")
        assert a.config_dir != b.config_dir
        assert a.output_dir != b.output_dir
        assert a.config_dir.is_relative_to(tmp_path / "cfg")

    def test_contains_resolves_both_sides(self, tmp_path):
        """
        A symlink into another tenant's directory is the case a string prefix check
        passes and this must not.
        """
        workspace = Workspace.resolve("acme", base=tmp_path)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        workspace.config_dir.mkdir(parents=True)
        link = workspace.config_dir / "shortcut"
        link.symlink_to(outside, target_is_directory=True)

        assert workspace.contains(workspace.config_dir / "rules.md")
        assert not workspace.contains(link / "rules.md")

    def test_a_whole_workspace_can_be_resolved_into_a_temporary_directory(self, tmp_path):
        """
        `base` exists for this: a test can put config and output somewhere disposable
        and assert nothing reached outside it. Before the seam there was no way to ask.
        """
        workspace = Workspace.resolve("acme", base=tmp_path, store_root=tmp_path / "s")
        for path in workspace.paths.values():
            assert path.is_relative_to(tmp_path)


class TestATenantIdBecomesAPathSegment:
    """
    Today it comes from a flag; the moment identity lands it comes from a token. A
    request-supplied segment reaching a filesystem path is the oldest mistake there is,
    and refusing it now costs nothing.
    """

    @pytest.mark.parametrize("bad", [
        "../etc", "a/b", "..", ".hidden", "-lead", "with space", "x" * 65, "acme/../..",
    ])
    def test_it_is_refused(self, bad):
        with pytest.raises(BadTenant):
            check_tenant(bad)
        with pytest.raises(BadTenant):
            Workspace.resolve(bad)

    @pytest.mark.parametrize("ok", ["acme", "acme-eu", "acme_eu", "a", "t2026"])
    def test_a_usable_one_is_accepted(self, ok):
        assert check_tenant(ok) == ok

    def test_blank_and_the_word_default_both_mean_the_default_tenant(self):
        assert check_tenant("") == DEFAULT_TENANT
        assert check_tenant(None) == DEFAULT_TENANT
        assert check_tenant("  ") == DEFAULT_TENANT
        assert check_tenant("default") == DEFAULT_TENANT


class TestItSaysWhereEachChoiceCameFrom:
    """
    `resolve_store_root` already reported this, and it turned out to be the thing people
    ask first when a run reads the wrong config. The other two now report it too.
    """

    def test_each_path_carries_its_origin(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_CONFIG, str(tmp_path / "cfg"))
        workspace = Workspace.resolve(output_dir=tmp_path / "o")
        assert workspace.origins["config_dir"] == f"${ENV_CONFIG}"
        assert workspace.origins["output_dir"] == "argument"
        assert workspace.origins["store_root"] == "default"

    def test_the_summary_names_every_path_and_its_origin(self):
        summary = Workspace.resolve("acme").summary()
        assert "acme" in summary
        for key in ("config_dir", "store_root", "output_dir"):
            assert key in summary
        assert "(tenant)" in summary

    def test_a_store_inside_the_working_tree_is_reported_not_overridden(self, monkeypatch):
        """
        Said out loud every run, and never acted on: someone may be running a throwaway
        experiment, and silently relocating their data would be worse than the risk.
        """
        monkeypatch.setenv(ENV_STORE, str(REPO / "scratch-store"))
        workspace = Workspace.resolve()
        assert workspace.store_root == REPO / "scratch-store"
        assert any("inside the working tree" in w for w in workspace.warnings)

    def test_a_workspace_is_frozen(self):
        """
        A module that could repoint the output directory mid-run would write half a run
        to one place and half to another, and the manifest would describe neither.
        """
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            Workspace.resolve().output_dir = Path("/tmp")


class TestNothingQuietlyOutranksTheTenant:
    """
    The defect this class exists for, found by running a tenant plan and reading the
    workspace line it printed: `--output` defaulted to the string "output", argparse
    passes its default on every run, so it arrived at the resolver as an explicit
    argument and beat the tenant. Two of three paths were isolated and the third —
    the one the results screen reads — silently was not.

    An explicit argument beating a tenant is correct and stays. What is wrong is a
    default masquerading as one.
    """

    def test_the_cli_passes_no_directory_it_was_not_given(self):
        from inventory_planning.cli import build_parser

        args = build_parser().parse_args([
            "--sales", "s.csv", "--po-history", "p.csv", "--open-so", "o.csv",
            "--open-po", "q.csv", "--inventory", "i.csv", "--item-master", "m.csv",
        ])
        assert args.output is None
        assert args.config is None
        assert args.tenant is None

    @pytest.fixture
    def tenant_config(self, tmp_path, monkeypatch):
        """A tenant with a real rule set — the planner reads several files on build."""
        import shutil

        config = tmp_path / "cfg" / "tenants" / "acme"
        config.mkdir(parents=True)
        for path in (REPO / "config").glob("*"):
            if path.is_file():
                shutil.copy2(path, config / path.name)
        monkeypatch.setenv(ENV_CONFIG, str(tmp_path / "cfg"))
        monkeypatch.setenv(ENV_OUTPUT, str(tmp_path / "out"))
        return config

    def test_the_planner_isolates_output_when_it_was_not_given(self, tmp_path,
                                                               tenant_config):
        from inventory_planning.orchestrator import InventoryPlanner

        planner = InventoryPlanner(tenant="acme", interactive=False)
        assert planner.config_dir == tenant_config
        assert planner.output_dir == tmp_path / "out" / "tenants" / "acme"
        assert planner.output_dir.exists()          # created, and created there
        assert not (REPO / "output" / "tenants").exists()

    def test_an_explicit_argument_is_still_allowed_to_win(self, tmp_path,
                                                          tenant_config):
        """The distinction being drawn: a stated path beats a tenant; a default must not."""
        from inventory_planning.orchestrator import InventoryPlanner

        planner = InventoryPlanner(tenant="acme", output_dir=tmp_path / "chosen",
                                   interactive=False)
        assert planner.output_dir == tmp_path / "chosen"
