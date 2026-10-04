"""Shared pytest configuration.

The ``tests`` directory is placed on ``sys.path`` so that every test module can
``from project_env import use_project`` without requiring the tests directory to
be an installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

for candidate in (str(TESTS_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
