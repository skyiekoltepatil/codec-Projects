"""Logging setup and environment-variable helpers.

Training scripts and Streamlit apps call :func:`configure_logging` so that
reviewers see consistent, timestamped progress output. Secrets are read from the
environment (optionally via a local ``.env``) and never committed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from shared.paths import PROJECT_ROOT

#: Log format used everywhere: concise, greppable, timestamped.
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

_CONFIGURED = False

#: Optional environment variables. Every one of them has a working default, so
#: a fresh clone runs without creating a ``.env`` file at all.
OPTIONAL_ENV_VARS: dict[str, str] = {
    "AI_ML_LOG_LEVEL": "INFO",
    "AI_ML_SEED": "42",
    "HF_HOME": "",  # Where faster-whisper caches downloaded speech models.
}


def configure_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configure root logging once per process.

    Args:
        level: Explicit level name or number. Falls back to ``AI_ML_LOG_LEVEL``
            and then ``INFO``.
        force: Reconfigure even when logging was already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = level or os.environ.get("AI_ML_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # Third-party libraries are chatty at INFO; keep their output actionable.
    for noisy in ("urllib3", "matplotlib", "PIL", "httpx", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an environment variable, optionally raising when it is required."""
    value = os.environ.get(name, default)
    if required and not value:
        from shared.errors import InvalidInputError

        raise InvalidInputError(
            f"Required environment variable '{name}' is not set.",
            hint=f"Copy .env.example to .env and set {name}, or export it in your shell.",
        )
    return value


def load_dotenv_if_present(path: Path | None = None) -> bool:
    """Load a local ``.env`` file when python-dotenv is installed.

    Returns:
        ``True`` when a ``.env`` file was found and loaded. This is intentionally
        optional: the repository works without it.
    """
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return False
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(env_path, override=False)
        return True
    except ImportError:
        logging.getLogger(__name__).debug(
            "python-dotenv not installed; relying on real environment variables only."
        )
        return False


def seed_from_env() -> int:
    """Return the configured random seed, defaulting to 42."""
    raw = os.environ.get("AI_ML_SEED", "42")
    try:
        return int(raw)
    except ValueError:
        return 42