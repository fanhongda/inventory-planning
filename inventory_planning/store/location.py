"""
Where the store lives.

Outside the repository, always. The store is the one thing this project accumulates
that cannot be regenerated from its inputs — the outputs can, the derived parameters
can, the history of what was true last month cannot. Inside the repository it would be
one `git clean -fdx` from gone, a fresh clone would start empty, and a cloud session
would report that empty store as normal rather than as a missing dependency.

Resolution order, most explicit first:

    1. an argument passed by the caller
    2. $INVENTORY_PLANNING_STORE
    3. $XDG_DATA_HOME/inventory-planning/store
    4. ~/.local/share/inventory-planning/store

The env var is not a convenience. Branch work writes to the same store as `main`
otherwise, so pointing it at a dev store is the isolation mechanism — version control
isolates the code and does nothing at all for the data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

ENV_VAR = "INVENTORY_PLANNING_STORE"
_APP_DIRNAME = "inventory-planning"


def default_store_root() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / _APP_DIRNAME / "store"


def resolve_store_root(explicit=None) -> Tuple[Path, str]:
    """Return the store root and where the choice came from."""
    if explicit:
        return Path(explicit).expanduser(), "argument"
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser(), f"${ENV_VAR}"
    if os.environ.get("XDG_DATA_HOME"):
        return default_store_root(), "$XDG_DATA_HOME"
    return default_store_root(), "default"


def repo_root() -> Optional[Path]:
    """The working tree this package is installed from, when it is one."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


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
    A store under the working tree is allowed and reported, never overridden.

    Someone who set the variable to a path in the repo may be running a throwaway
    experiment, and silently relocating their data would be worse than the risk being
    described. But it is a real risk and it is invisible until the day it costs
    something, so it is said out loud every run.
    """
    if not inside_repo(path):
        return None
    return (f"store at {path} is inside the working tree — `git clean -fdx` deletes it "
            f"and a fresh clone starts empty. Set ${ENV_VAR} to a path outside the repo.")
