"""
Where adapters live.

There are two kinds and they are not the same thing.

**Shipped** adapters sit in the package, next to the contracts, under version control.
They describe an export template that is not specific to one installation, they are
reviewed in a diff like any other change, and `pip install` puts them there. They are
read-only: nothing this package does writes into its own installation directory.

**Local** adapters are the frozen mappings one installation accumulates — this ERP, this
region, this export. They are production configuration. They are also, unavoidably,
a description of somebody's business: the column names of their master data, the codes
their regions are keyed on, which of their fields are not clean. That belongs outside
the working tree for two independent reasons, and either one is sufficient:

    1. Inside it, `git clean -fdx` deletes them, a fresh clone starts empty, and
       `git status` reports production configuration as untracked clutter. This is the
       argument `store/location.py` already makes for the store.
    2. Inside it, the only thing standing between a business's column map and a
       `git add -A` is whoever is typing. A public remote makes that irreversible.

So `save()` writes here, never into the package.

Resolution order, most explicit first:

    1. an argument passed by the caller
    2. $INVENTORY_PLANNING_ADAPTERS
    3. $XDG_DATA_HOME/inventory-planning/adapters
    4. ~/.local/share/inventory-planning/adapters

The env var is the isolation mechanism, same as the store's: branch work that drafts an
adapter would otherwise freeze it into the same store `main` reads.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

ENV_VAR = "INVENTORY_PLANNING_ADAPTERS"
_APP_DIRNAME = "inventory-planning"

# The package's own adapter directory. Imported here rather than from `adapter.py` so
# that module keeps its single definition and this one does not create a second.
from .adapter import ADAPTERS_DIR as SHIPPED_ADAPTERS_DIR  # noqa: E402


def default_adapter_store() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / _APP_DIRNAME / "adapters"


def resolve_adapter_store(explicit=None) -> Tuple[Path, str]:
    """Return the local adapter store and where the choice came from."""
    if explicit:
        return Path(explicit).expanduser(), "argument"
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser(), f"${ENV_VAR}"
    if os.environ.get("XDG_DATA_HOME"):
        return default_adapter_store(), "$XDG_DATA_HOME"
    return default_adapter_store(), "default"


def repo_root() -> Optional[Path]:
    """The working tree this package is installed from, when it is one."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _tracked(root: Path, relative: str) -> Optional[set]:
    """Resolved paths git tracks under `relative`, or None if git cannot be asked."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", relative],
            # `encoding` rather than bare `text=True`: a path with a non-ASCII
            # character decodes by the platform's locale otherwise, which on Windows
            # is not UTF-8 and raises on the one input this is reading for.
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return {str((root / p).resolve()) for p in done.stdout.split("\0") if p}


def untracked_in_tree_adapters() -> List[Path]:
    """
    Adapter files inside the working tree that git does not track.

    These are what an older build left behind: `save()` wrote into the package, so a
    mapping drafted from a real extract landed in the source tree. They are still read
    — a stored adapter takes routing ahead of inference, and this one is invisible to
    review — so they are worth naming out loud rather than migrating silently, which
    would move a mapping nobody has looked at into the place the next run trusts.

    Empty when this is not a git checkout, which is the normal case for an install:
    there is no working tree to be inside of, and nothing to report.
    """
    root = repo_root()
    if root is None or not SHIPPED_ADAPTERS_DIR.exists():
        return []
    try:
        relative = SHIPPED_ADAPTERS_DIR.resolve().relative_to(root)
    except ValueError:
        return []
    tracked = _tracked(root, str(relative))
    if tracked is None:
        return []
    return sorted(p for p in SHIPPED_ADAPTERS_DIR.rglob("*.yaml")
                  if str(p.resolve()) not in tracked)


def warn_about_in_tree_adapters() -> Optional[str]:
    """One message naming the adapters an older build left in the working tree."""
    stray = untracked_in_tree_adapters()
    if not stray:
        return None
    names = ", ".join(sorted({p.parent.name for p in stray}))
    store, _ = resolve_adapter_store()
    return (
        f"{len(stray)} adapter file(s) in the working tree are not under version "
        f"control ({names}). An older build wrote drafted adapters into the package. "
        f"They still take routing ahead of inference, and a mapping frozen against an "
        f"older contract is never updated — a field the contract gained since is read "
        f"as absent rather than as wrong. Review them, move what is still correct to "
        f"{store}, and delete the originals."
    )


def inside_repo(path: Path) -> bool:
    root = repo_root()
    if root is None:
        return False
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def warn_if_inside_repo(path: Path) -> Optional[str]:
    """
    A local store under the working tree is allowed and reported, never overridden.

    Same judgement `store/location.py` makes for the store: someone who pointed the
    variable at a path in the repo may be running a throwaway experiment, and silently
    relocating their configuration would be worse than the risk being described. The
    difference is what is at stake. A store inside the tree can be deleted; an adapter
    inside the tree can also be committed, and on a public remote that does not come
    back. So the message says both.
    """
    if not inside_repo(path):
        return None
    return (f"adapter store at {path} is inside the working tree — `git clean -fdx` "
            f"deletes it, a fresh clone starts empty, and `git add -A` publishes the "
            f"column names of your master data. Set ${ENV_VAR} to a path outside the "
            f"repo.")
