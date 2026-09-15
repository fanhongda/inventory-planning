"""
Where one tenant's three directories are — resolved once, in one place.

INTERFACE.md §7 named this as the first of three seams to cut while there is one user
and they are cheap. `config_dir`, `store_root` and `output_dir` are resolved separately
today and passed around independently — 230 occurrences across 30 modules. `Intake` and
the adapters already carry a `tenant`; nothing else does. Collapsing the three into one
object resolved from a tenant id makes multi-tenancy **"resolve a different workspace"**
rather than a refactor of every one of those call sites.

The three are not the same kind of thing, which is why this resolves them separately
rather than putting them under one root:

    config   versioned in git. It is the rules, and the whole argument for keeping the
             rule engine in markdown is that git gives review, diff, rationale and owner
             for nothing. A tenant gets its own rule set, not a copy of a shared one.
    store    outside the repository, and the one thing here that cannot be regenerated.
             Already tenant-aware in spirit: pointing $INVENTORY_PLANNING_STORE at a dev
             store is how branch work is made safe.
    output   regenerable per-run artefacts, and the run registry the interface reads.

## One rule for where a named tenant's directories go

    resolved from a base     default <repo>/<leaf>      tenant <data>/tenants/<name>/<leaf>
    named by a variable      default <$VAR>             tenant <$VAR>/tenants/<name>

The default tenant's config is the repository's own `config/`, which is the point: the
rules are markdown in git, and git is what gives them review, diff, rationale and owner.
A *named* tenant's config is not this repository's content, and the first version of this
put it at `<repo>/tenants/<name>/config` — inside the working tree, untracked, one
`git clean -fdx` from gone, and permanently dirtying `git status` on a machine that
pulls. That is the failure `store/location.py` already argues against for the store, and
it applies to a tenant's config for the same reason. So a named tenant's three
directories all sit together under the data directory, outside any working tree. A tenant
that wants its rules versioned points `--config` at a repository of its own.

Two shapes because a variable already names the directory itself — `$..._CONFIG` is the
config directory, the way `--config` is — so appending the leaf again would give
`<$VAR>/tenants/acme/config` for a variable that already ended in the config directory.
Worth stating rather than leaving to be inferred: I read it the other way once while
setting up a tenant run, and the failure was a missing file three frames deep.

Stated once, here, and it is the only thing that changes when real multi-tenancy
arrives. The default tenant resolves **exactly** what was resolved before this module
existed — this seam is not allowed to move a single path for the single user it has.

## What outranks what

Argument, then variable, then tenant, then default — and the distinction that cost an
hour: **a default is not an argument.** `cli.py` had `--output` defaulting to the string
`"output"`, argparse passed it on every run, and it therefore arrived here as an
explicit argument and beat the tenant. Two of three paths were isolated and the third —
the one the results screen reads — silently was not. Entry points pass `None` for what
they were not given.

## Why the tenant name is validated rather than trusted

Today it comes from a flag. The moment an identity layer lands it comes from a token,
and a request-supplied segment that reaches a filesystem path is the oldest mistake
there is. Refusing `../` here costs nothing now and is invisible to add later — which
is the same argument as cutting the seam at all.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .store.location import default_store_root, resolve_store_root, warn_if_inside_repo

DEFAULT_TENANT = "default"

ENV_CONFIG = "INVENTORY_PLANNING_CONFIG"
ENV_OUTPUT = "INVENTORY_PLANNING_OUTPUT"
# The store's variable predates this module and keeps its name — it is documented, it is
# in people's shells, and renaming it would break the one isolation mechanism that
# already works in order to make three of them look alike.
ENV_STORE = "INVENTORY_PLANNING_STORE"

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TENANTS_DIRNAME = "tenants"


class BadTenant(ValueError):
    """A tenant id that cannot be part of a path."""


def check_tenant(tenant: Optional[str]) -> str:
    """
    The tenant id, or a refusal. Empty means the default tenant.

    Alphanumeric plus `_` and `-`, starting alphanumeric, at most 64 characters. Narrow
    on purpose: this becomes a path segment, and the useful set of tenant names is much
    smaller than the set of strings a filesystem would accept.
    """
    name = str(tenant or "").strip()
    if not name or name == DEFAULT_TENANT:
        return DEFAULT_TENANT
    if not _TENANT_RE.match(name):
        raise BadTenant(
            f"{tenant!r} is not a usable tenant id. Letters, digits, `_` and `-`, "
            f"starting with a letter or digit, at most 64 characters — it becomes a "
            f"directory name, and everything else is a path that escapes its workspace.")
    return name


def _scoped(base: Path, leaf: str, tenant: str) -> Path:
    """The one rule. `<base>/<leaf>` for the default tenant, `tenants/<name>` under it."""
    if tenant == DEFAULT_TENANT:
        return base / leaf
    return base / _TENANTS_DIRNAME / tenant / leaf


@dataclass(frozen=True)
class Workspace:
    """
    One tenant's config, store and output, with where each choice came from.

    Frozen because a workspace is resolved once at an entry point and then read: a
    module that could repoint the output directory mid-run would be able to write half
    a run to one place and half to another, and the run manifest would describe neither.
    """

    tenant: str
    config_dir: Path
    store_root: Path
    output_dir: Path
    # field -> which rule decided it: "argument", "$VAR", "tenant", "default". Carried
    # because `resolve_store_root` already reported this and it turned out to be the
    # thing people ask first when a run reads the wrong config.
    origins: Dict[str, str] = dc_field(default_factory=dict)
    # Said, never acted on: a store under the working tree is one `git clean -fdx` from
    # gone, and silently relocating someone's data would be worse than the risk.
    warnings: Tuple[str, ...] = ()

    @classmethod
    def resolve(cls, tenant: str = None, *, config_dir=None, store_root=None,
                output_dir=None, base: Path = None) -> "Workspace":
        """
        Build a workspace, most explicit first: argument, environment, tenant, default.

        `base` is where the repo-relative defaults hang off, and exists so a test can
        resolve a whole workspace into a temporary directory and assert that nothing
        reached outside it. That is the half of the isolation §7 called untestable: the
        store path could already be pointed somewhere safe, config and output could not.
        """
        name = check_tenant(tenant)
        root = Path(base) if base else Path(__file__).parents[1]
        # Where a *named* tenant's directories hang off: beside the store, outside any
        # working tree. `base` overrides both so a test can put a whole workspace in a
        # temporary directory.
        data = Path(base) if base else default_store_root().parent
        origins: Dict[str, str] = {}

        config = cls._pick(config_dir, ENV_CONFIG, root, data, "config", name, origins,
                           "config_dir")
        output = cls._pick(output_dir, ENV_OUTPUT, root, data, "output", name, origins,
                           "output_dir")

        # The store keeps its own resolver: it has one more step than the others
        # ($XDG_DATA_HOME) and it is the one people have already configured.
        if store_root:
            store, origin = Path(store_root).expanduser(), "argument"
        elif os.environ.get(ENV_STORE):
            store, origin = Path(os.environ[ENV_STORE]).expanduser(), f"${ENV_STORE}"
        elif name != DEFAULT_TENANT:
            store, origin = _scoped(default_store_root().parent, "store", name), "tenant"
        else:
            store, origin = resolve_store_root(None)
        # A tenant named on top of an explicit store path still gets its own subtree —
        # otherwise pointing $INVENTORY_PLANNING_STORE at a dev store would silently
        # merge every tenant's facts into it, which is the isolation failing exactly
        # where someone thought they were being careful.
        if name != DEFAULT_TENANT and origin in ("argument", f"${ENV_STORE}"):
            store = store / _TENANTS_DIRNAME / name
            origin = f"{origin} + tenant"
        origins["store_root"] = origin

        # Said for all three now. The store always checked; config and output did not,
        # and a named tenant's config landing in the working tree is the case that
        # needed saying — `git clean -fdx` deletes it and `git status` is dirty until
        # somebody does.
        found = [warn_if_inside_repo(store)]
        if name != DEFAULT_TENANT:
            for path in (config, output):
                inside = warn_if_inside_repo(path)
                if inside:
                    found.append(
                        f"tenant {name}: {path} is inside the working tree — "
                        f"`git clean -fdx` deletes it and every pull sees it as "
                        f"untracked. Point it outside, or let it default.")
        warnings = tuple(w for w in found if w)
        return cls(tenant=name, config_dir=config, store_root=store, output_dir=output,
                   origins=origins, warnings=warnings)

    @staticmethod
    def _pick(explicit, env_var: str, root: Path, data: Path, leaf: str, tenant: str,
              origins: Dict[str, str], key: str) -> Path:
        if explicit:
            origins[key] = "argument"
            return Path(explicit).expanduser()
        from_env = os.environ.get(env_var)
        if from_env:
            path = Path(from_env).expanduser()
            if tenant != DEFAULT_TENANT:
                origins[key] = f"${env_var} + tenant"
                return path / _TENANTS_DIRNAME / tenant
            origins[key] = f"${env_var}"
            return path
        if tenant != DEFAULT_TENANT:
            origins[key] = "tenant"
            return _scoped(data, leaf, tenant)
        origins[key] = "default"
        return root / leaf

    # ── Reading ──────────────────────────────────────────────────────────────

    @property
    def parameters_file(self) -> Path:
        """The rule set this workspace plans under, unless a scenario overrides it."""
        return self.config_dir / "planning_parameters.md"

    @property
    def paths(self) -> Dict[str, Path]:
        return {"config_dir": self.config_dir, "store_root": self.store_root,
                "output_dir": self.output_dir}

    def contains(self, path) -> bool:
        """
        Whether a path lies inside this workspace. The isolation test's whole subject.

        Resolved on both sides before comparing, because a symlink into another tenant's
        directory is precisely the case a string prefix check would pass.
        """
        try:
            target = Path(path).resolve()
        except OSError:
            return False
        for base in self.paths.values():
            try:
                target.relative_to(Path(base).resolve())
                return True
            except (ValueError, OSError):
                continue
        return False

    def summary(self) -> str:
        lines = [f"  Workspace · {self.tenant}"]
        width = max(len(k) for k in self.paths)
        for key, path in self.paths.items():
            lines.append(f"    {key:<{width}}  {path}  "
                         f"({self.origins.get(key, 'default')})")
        for warning in self.warnings:
            lines.append(f"    ⚠ {warning}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, str]:
        body = {"tenant": self.tenant}
        body.update({k: str(v) for k, v in self.paths.items()})
        body["origins"] = dict(self.origins)
        return body
