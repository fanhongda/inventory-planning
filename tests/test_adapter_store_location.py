"""
Where a frozen adapter is written, and which directories are read.

A stored adapter is two things at once, and the repository had been treating it as
only the first. It is production configuration: `route()` tries every stored
fingerprint before it classifies anything, so a stored adapter decides how a real
export is read, ahead of inference, whatever its status. It is also a description of
somebody's business — the column names their master data is keyed on, which of their
regions use a local code, which of their fields are not clean.

`save()` used to write it into the package, inside the working tree. That put
production configuration one `git clean -fdx` from gone, a fresh clone one step from
starting empty, and — on a public remote — a business's column map one `git add -A`
from published. None of those are hypothetical: `b5f125f` found adapters drafted into
the tree by an older build, and the mapping in one of them had been superseded five
days after it was frozen.

So the store is split. Shipped adapters stay in the package, reviewed in a diff like
any other change; local ones go outside the tree. Both are read, only the second is
written.
"""

import os
from pathlib import Path

import pytest

from inventory_planning.ingest import location
from inventory_planning.ingest.adapter import Adapter, Fingerprint
from inventory_planning.ingest.registry import AdapterRegistry


def _adapter(name="acme__sap__inventory", doc_type="inventory", version=1,
             system="sap", column_map=None, status="frozen"):
    return Adapter(
        name=name, doc_type=doc_type, version=version, tenant="acme", system=system,
        status=status, column_map=column_map or {"sku": "Material"},
        fingerprint=Fingerprint(header_contains=["Material"]),
    )


class TestAFrozenAdapterIsNotWrittenIntoThePackage:

    def test_save_writes_to_the_local_store(self, tmp_path):
        registry = AdapterRegistry(local_dir=tmp_path / "local")
        path = registry.save(_adapter())
        assert path.is_relative_to(tmp_path / "local")
        assert not path.is_relative_to(registry.adapters_dir)

    def test_the_package_directory_is_untouched_by_a_save(self, tmp_path):
        registry = AdapterRegistry(local_dir=tmp_path / "local")
        before = sorted(p.name for p in registry.adapters_dir.rglob("*.yaml"))
        registry.save(_adapter())
        assert sorted(p.name for p in registry.adapters_dir.rglob("*.yaml")) == before

    def test_a_saved_adapter_is_read_back(self, tmp_path):
        """
        The round trip is the point: writing somewhere the next run does not look is
        the same defect as not writing at all, and quieter.
        """
        registry = AdapterRegistry(local_dir=tmp_path / "local")
        registry.save(_adapter())
        assert "acme__sap__inventory" in {a.name for a in AdapterRegistry(
            local_dir=tmp_path / "local").adapters}


class TestBothDirectoriesAreRead:

    def test_shipped_comes_first(self, tmp_path):
        registry = AdapterRegistry(local_dir=tmp_path / "local")
        assert registry.source_dirs[0] == registry.adapters_dir
        assert registry.source_dirs[-1] == tmp_path / "local"

    def test_an_explicit_adapters_dir_is_the_whole_store(self, tmp_path):
        """
        What `tests/conftest.py` relies on. A suite whose routing depends on which
        files this machine has ingested cannot say whether the code is good, so when
        the caller names a directory it is the only one read — the local store is not
        added behind its back.
        """
        registry = AdapterRegistry(adapters_dir=tmp_path)
        assert registry.source_dirs == [tmp_path]
        assert registry.local_dir == tmp_path

    def test_a_local_adapter_displaces_a_shipped_one_of_the_same_version(self, tmp_path):
        """
        Otherwise routing would depend on which of the two `rglob` reached first.
        An installation that has reviewed and frozen its own answer for an export has
        said which one it wants.
        """
        shipped, local = tmp_path / "shipped", tmp_path / "local"
        for directory, column in ((shipped, "Material"), (local, "Artikel")):
            AdapterRegistry(adapters_dir=directory).save(
                _adapter(column_map={"sku": column}))

        registry = AdapterRegistry(local_dir=local)
        registry.adapters_dir = shipped
        matches = [a for a in registry.adapters if a.name == "acme__sap__inventory"]
        assert len(matches) == 1
        assert matches[0].column_map["sku"] == "Artikel"

    def test_a_different_version_coexists(self, tmp_path):
        """
        `version` is how a mapping change stays a reviewable diff — bumping it must not
        silently retire the version a past run is reproducible against.
        """
        shipped, local = tmp_path / "shipped", tmp_path / "local"
        AdapterRegistry(adapters_dir=shipped).save(_adapter(version=1))
        AdapterRegistry(adapters_dir=local).save(_adapter(version=2))

        registry = AdapterRegistry(local_dir=local)
        registry.adapters_dir = shipped
        assert {a.version for a in registry.adapters
                if a.name == "acme__sap__inventory"} == {1, 2}


class TestWhereTheLocalStoreResolvesFrom:

    def test_an_argument_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(location.ENV_VAR, str(tmp_path / "env"))
        assert location.resolve_adapter_store(tmp_path / "arg") == (
            tmp_path / "arg", "argument")

    def test_the_env_var_beats_the_default(self, tmp_path, monkeypatch):
        """
        The isolation mechanism, same as the store's: branch work that freezes an
        adapter would otherwise write into the store `main` reads.
        """
        monkeypatch.setenv(location.ENV_VAR, str(tmp_path / "env"))
        assert location.resolve_adapter_store() == (tmp_path / "env",
                                                    f"${location.ENV_VAR}")

    def test_xdg_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.delenv(location.ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        path, origin = location.resolve_adapter_store()
        assert path == tmp_path / "inventory-planning" / "adapters"
        assert origin == "$XDG_DATA_HOME"

    def test_the_default_is_outside_any_working_tree(self, monkeypatch):
        monkeypatch.delenv(location.ENV_VAR, raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        assert not location.inside_repo(location.default_adapter_store())


class TestAStoreInsideTheTreeIsReportedNotOverridden:
    """
    The judgement `store/location.py` already made: someone who pointed the variable
    at a path in the repo may be running a throwaway experiment, and silently
    relocating their configuration would be worse than the risk being described.
    """

    def test_a_path_outside_the_tree_says_nothing(self, tmp_path):
        assert location.warn_if_inside_repo(tmp_path) is None

    def test_a_path_inside_the_tree_names_both_consequences(self):
        root = location.repo_root()
        if root is None:
            pytest.skip("not a git checkout, so there is no working tree to be inside")
        message = location.warn_if_inside_repo(root / "inventory_planning")
        assert message is not None
        assert "git clean -fdx" in message
        assert "git add -A" in message

    def test_save_reports_it_without_refusing(self, tmp_path, capsys, monkeypatch):
        """
        Reported every run, never overridden — and the adapter is still written, because
        refusing to save is how a reviewed mapping gets lost instead of misplaced.
        """
        monkeypatch.setattr(location, "repo_root", lambda: tmp_path)
        registry = AdapterRegistry(local_dir=tmp_path / "inside")
        path = registry.save(_adapter())
        assert path.exists()
        assert "inside the working tree" in capsys.readouterr().out


class TestAdaptersAnOlderBuildLeftInTheTree:
    """
    `save()` wrote into the package until this change, so a mapping drafted from a real
    extract landed in the source tree, untracked. It is still read and still outranks
    inference, so it is named rather than migrated: moving a mapping nobody has looked
    at into the place the next run trusts would be the same defect with a new address.
    """

    def test_an_untracked_yaml_in_the_package_is_reported(self, tmp_path, monkeypatch):
        tree = tmp_path / "repo"
        shipped = tree / "inventory_planning" / "ingest" / "adapters" / "acme__sap"
        shipped.mkdir(parents=True)
        (tree / ".git").mkdir()
        (shipped / "inventory.v1.yaml").write_text("name: x\ndoc_type: inventory\n", encoding="utf-8")

        monkeypatch.setattr(location, "repo_root", lambda: tree)
        monkeypatch.setattr(location, "SHIPPED_ADAPTERS_DIR", shipped.parent)
        monkeypatch.setattr(location, "_tracked", lambda root, rel: set())

        stray = location.untracked_in_tree_adapters()
        assert [p.name for p in stray] == ["inventory.v1.yaml"]

        message = location.warn_about_in_tree_adapters()
        assert "acme__sap" in message
        assert "not under version control" in message

    def test_a_tracked_yaml_is_not_reported(self, tmp_path, monkeypatch):
        tree = tmp_path / "repo"
        shipped = tree / "inventory_planning" / "ingest" / "adapters" / "acme__sap"
        shipped.mkdir(parents=True)
        (tree / ".git").mkdir()
        yaml = shipped / "inventory.v1.yaml"
        yaml.write_text("name: x\ndoc_type: inventory\n", encoding="utf-8")

        monkeypatch.setattr(location, "repo_root", lambda: tree)
        monkeypatch.setattr(location, "SHIPPED_ADAPTERS_DIR", shipped.parent)
        monkeypatch.setattr(location, "_tracked",
                            lambda root, rel: {str(yaml.resolve())})

        assert location.untracked_in_tree_adapters() == []
        assert location.warn_about_in_tree_adapters() is None

    def test_an_install_that_is_not_a_checkout_reports_nothing(self, monkeypatch):
        """
        `pip install` puts the package somewhere with no working tree around it. There
        is nothing to be inside of, so there is nothing to say.
        """
        monkeypatch.setattr(location, "repo_root", lambda: None)
        assert location.untracked_in_tree_adapters() == []
        assert location.warn_about_in_tree_adapters() is None
