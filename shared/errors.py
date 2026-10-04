"""Typed exceptions and user-facing error rendering.

Every project raises these instead of bare ``Exception`` so that the Streamlit
layer can present an actionable message rather than a raw traceback.
"""

from __future__ import annotations

from typing import Any


class ProjectError(Exception):
    """Base class for all recoverable, user-facing errors in this repository."""

    #: Short label used as the heading when the error reaches the UI.
    title: str = "Something went wrong"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation for logging or tests."""
        return {"type": type(self).__name__, "message": self.message, "hint": self.hint}


class DataNotFoundError(ProjectError):
    """A required dataset file is missing from disk."""

    title = "Dataset not found"


class DataDownloadError(ProjectError):
    """A dataset could not be downloaded from its public source."""

    title = "Dataset download failed"


class ModelNotFoundError(ProjectError):
    """A trained model artefact is missing; the user must train it first."""

    title = "Trained model not found"


class InvalidInputError(ProjectError):
    """User supplied input that cannot be used by the model."""

    title = "Invalid input"


class DependencyUnavailableError(ProjectError):
    """An optional third-party capability is not available in this environment."""

    title = "Feature unavailable"


def friendly_error(error: BaseException) -> tuple[str, str]:
    """Convert an exception into a ``(title, message)`` pair safe to display.

    Known project errors keep their curated wording. Anything else is mapped to
    a generic message so that a raw traceback is never shown to an evaluator.
    """
    if isinstance(error, ProjectError):
        message = error.message
        if error.hint:
            message = f"{message}\n\n**How to fix:** {error.hint}"
        return error.title, message
    return (
        "Unexpected error",
        f"{type(error).__name__}: {error}",
    )