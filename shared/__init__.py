"""Shared utilities used across every project in this repository.

The modules here deliberately avoid project-specific logic so that each of the
ten projects stays independently runnable while still sharing a consistent
visual language, data-acquisition layer, and evaluation vocabulary.
"""

from shared.errors import (
    DataDownloadError,
    DataNotFoundError,
    InvalidInputError,
    ModelNotFoundError,
    ProjectError,
    friendly_error,
)
from shared.paths import PROJECT_ROOT, data_dir, models_dir, project_dir, screenshots_dir

__all__ = [
    "DataDownloadError",
    "DataNotFoundError",
    "InvalidInputError",
    "ModelNotFoundError",
    "ProjectError",
    "friendly_error",
    "PROJECT_ROOT",
    "project_dir",
    "data_dir",
    "models_dir",
    "screenshots_dir",
]