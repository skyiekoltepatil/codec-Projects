"""Platform-independent path resolution.

All paths are derived from the location of this file, so the repository can be
cloned anywhere (macOS, Windows, Linux) without editing a single absolute path.
"""

from __future__ import annotations

from pathlib import Path

# Repository root == the parent directory of the ``shared`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def project_dir(slug: str) -> Path:
    """Return the directory of a numbered project, e.g. ``01-stock-price-predictor``."""
    return PROJECT_ROOT / slug


def data_dir(slug: str) -> Path:
    """Return the ``data`` directory of a project, creating it when missing."""
    path = project_dir(slug) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(slug: str) -> Path:
    """Return the ``models`` directory of a project, creating it when missing."""
    path = project_dir(slug) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def screenshots_dir(slug: str) -> Path:
    """Return the ``screenshots`` directory of a project, creating it when missing."""
    path = project_dir(slug) / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of ``path`` if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def portable_display(path: Path) -> str:
    """Render a path relative to the repository root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)