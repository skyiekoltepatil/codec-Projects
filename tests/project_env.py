"""Project-scoped import isolation for the test suite.

The problem
-----------
Every project ships its own package literally named ``src``, because that is the
repository structure this suite is required to follow. Python caches modules by
name in ``sys.modules``, so once ``src.data`` from project 01 has been imported,
importing ``src.data`` again resolves to that same cached module instead of
project 02's. Running the whole suite in one ``pytest`` process would therefore
silently test the wrong project's code.

The solution
------------
Each test module calls :func:`use_project` *before* importing anything from
``src``. The helper evicts any cached ``src`` package and puts exactly one
project directory at the front of ``sys.path``. This is deterministic and
explicit: there is no import-time magic and no ambiguity about which project a
test is exercising.

The same collision cannot occur at runtime, because the dashboard launches each
project's Streamlit app in a separate process with its own working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repository root, derived from this file's location.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def project_paths() -> list[Path]:
    """Return every numbered project directory, sorted."""
    return sorted(
        path
        for path in REPO_ROOT.iterdir()
        if path.is_dir() and len(path.name) > 2 and path.name[:2].isdigit()
    )


def use_project(slug: str) -> Path:
    """Activate one project for the current test module.

    Drops any cached ``src`` package, removes every other project directory from
    ``sys.path``, and puts ``slug`` at the front so ``import src`` resolves to
    that project only.

    Args:
        slug: Project directory name, e.g. ``"01-stock-price-predictor"``.

    Returns:
        The absolute path to the activated project directory.

    Raises:
        FileNotFoundError: When the project directory does not exist.
    """
    project_dir = REPO_ROOT / slug
    if not project_dir.is_dir():
        raise FileNotFoundError(f"No such project directory: {project_dir}")

    # Evict the cached package so the next import re-resolves from disk.
    for module_name in [name for name in list(sys.modules) if name == "src" or name.startswith("src.")]:
        del sys.modules[module_name]

    # Remove every project directory (and this one) so ordering is unambiguous.
    stale = {str(project_dir), *(str(path) for path in project_paths())}
    sys.path[:] = [entry for entry in sys.path if entry not in stale]

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(project_dir))

    return project_dir