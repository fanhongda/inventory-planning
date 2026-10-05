"""
What the suite is allowed to see of the machine it runs on.

`AdapterRegistry` defaults to `inventory_planning/ingest/adapters/`, and `save()` writes
drafted and frozen adapters into that same directory — inside the installed package,
inside the working tree. So the adapter store a production run writes to is the adapter
store the tests read from, and a machine that has ever ingested a real extract has
adapters there that the repository does not ship.

Found on a Windows machine, three failures in `test_sop_and_review.py` that reproduce
nowhere else. A backlog adapter auto-drafted on 2026-08-26 carried
`customer: Sold-to Region` — the exact mapping `a43e40d` split into `country` and
`region` five days later. A frozen adapter is the whole column map rather than a set of
overrides, so it never picked the new fields up; and its fingerprint matched the
synthetic backlog in those tests on four of six columns, above the 0.6 default. A stored
adapter outranks inference, so the tests read a real export's mapping and failed.

Neither side was wrong. The adapter described that export correctly and the tests
describe the contract correctly. What was wrong is that they could see each other: a
suite whose result depends on which files this machine has ingested is not a suite that
can tell you whether the code is good.

So the tests run against the adapters the repository tracks, and nothing else. `git
ls-files` is the question asked literally — shipped means committed — because there is
no marker on disk that separates the one adapter this repo ships from the four a planner
accumulated. Where git cannot answer (an unpacked sdist, no git binary) the suite says
so and reads the ambient store rather than silently running against an empty one, which
would turn the routing tests green for the wrong reason.
"""

import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

import inventory_planning.ingest.adapter as _adapter
import inventory_planning.ingest.registry as _registry

REPO = Path(__file__).resolve().parents[1]
ADAPTERS = "inventory_planning/ingest/adapters"


def _tracked_adapter_files():
    """The adapter files under version control, or None if git cannot be asked."""
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--", ADAPTERS],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line for line in done.stdout.split("\0") if line]


@pytest.fixture(scope="session", autouse=True)
def adapter_store_holds_only_what_the_repository_ships(tmp_path_factory):
    """
    Point the adapter store at a copy of the tracked adapters, for the whole session.

    `registry.py` binds `ADAPTERS_DIR` by value at import (`from .adapter import ...`),
    so its own module attribute is the one that decides where `AdapterRegistry()` looks;
    patching only `adapter.ADAPTERS_DIR` would leave the default untouched. Both are set
    and both are restored.
    """
    tracked = _tracked_adapter_files()
    if tracked is None:
        warnings.warn(
            "git could not list the tracked adapters, so the suite is reading "
            f"{_registry.ADAPTERS_DIR}. Any adapter this machine has drafted from a "
            "real extract is visible to the tests, and routing results may not be "
            "reproducible.",
            RuntimeWarning,
        )
        yield None
        return

    root = tmp_path_factory.mktemp("adapter-store")
    for rel in tracked:
        source = REPO / rel
        if not source.is_file():
            continue
        destination = root / Path(rel).relative_to(ADAPTERS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    before = (_adapter.ADAPTERS_DIR, _registry.ADAPTERS_DIR)
    _adapter.ADAPTERS_DIR = root
    _registry.ADAPTERS_DIR = root
    try:
        yield root
    finally:
        _adapter.ADAPTERS_DIR, _registry.ADAPTERS_DIR = before
