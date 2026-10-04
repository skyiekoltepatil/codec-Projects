"""Shared model I/O, splitting strategy and compute-device selection.

These helpers exist so that every project saves and loads artefacts the same
way, and so that no project accidentally leaks future information into training.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from shared.errors import ModelNotFoundError
from shared.paths import ensure_parent

logger = logging.getLogger(__name__)

#: Default seed used by every project, so results are reproducible.
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed ``random`` and NumPy, and PyTorch when it is importable."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is always installed in this repo
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def resolve_device(prefer: str | None = None) -> str:
    """Return the compute device string for PyTorch models.

    Preference order is ``prefer`` (when it is genuinely available), then Apple
    Metal (MPS), then CPU. CPU is always a valid fallback, so no project ever
    requires a GPU.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"

    available = {"cpu": True, "mps": bool(torch.backends.mps.is_available()), "cuda": bool(torch.cuda.is_available())}
    if prefer and available.get(prefer):
        return prefer
    if available["cuda"]:
        return "cuda"
    if available["mps"]:
        return "mps"
    return "cpu"


def describe_device() -> str:
    """Return a short human-readable description of the active device."""
    device = resolve_device()
    if device == "cuda":  # pragma: no cover - no GPU in the reference environment
        import torch

        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


# --------------------------------------------------------------------------- #
# Joblib artefacts (classical ML)
# --------------------------------------------------------------------------- #


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist a Python object with joblib, creating parent directories."""
    import joblib

    target = ensure_parent(Path(path))
    joblib.dump(obj, target, compress=3)
    logger.info("Saved artefact to %s", target)
    return target


def load_joblib(path: str | Path, *, description: str = "model", train_command: str | None = None) -> Any:
    """Load a joblib artefact, raising an actionable error when it is missing.

    Args:
        path: Artefact location.
        description: Noun used in the error message, e.g. ``"sentiment model"``.
        train_command: Command the user should run to create the artefact.
    """
    import joblib

    target = Path(path)
    if not target.exists():
        hint = (
            f"Run the training script first: {train_command}"
            if train_command
            else "Run the project's train.py to create this artefact."
        )
        raise ModelNotFoundError(f"Trained {description} not found at '{target.name}'.", hint=hint)
    try:
        return joblib.load(target)
    except Exception as error:
        raise ModelNotFoundError(
            f"Trained {description} at '{target.name}' could not be read.",
            hint=(
                "The file may be corrupted or produced by an incompatible library version. "
                f"Delete it and re-run training. ({type(error).__name__})"
            ),
        ) from error


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON sidecar file (metadata, class names, feature order)."""
    target = ensure_parent(Path(path))
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def load_json(path: str | Path, *, description: str = "metadata") -> dict[str, Any]:
    """Read a JSON sidecar file, raising an actionable error when missing."""
    target = Path(path)
    if not target.exists():
        raise ModelNotFoundError(
            f"{description.capitalize()} file not found at '{target.name}'.",
            hint="Re-run the project's train.py to regenerate it.",
        )
    return json.loads(target.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Splitting strategies
# --------------------------------------------------------------------------- #


def chronological_split(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ordered indices into contiguous train/validation/test blocks.

    This is the correct split for time-series data: no future observation is
    ever used to fit the model. Random shuffling is deliberately *not* offered
    here to make accidental leakage harder.

    Args:
        n_samples: Number of ordered samples.
        train_fraction: Share of samples used for training.
        val_fraction: Share of samples used for validation.

    Returns:
        ``(train_idx, val_idx, test_idx)`` integer index arrays.

    Raises:
        ValueError: When the fractions are invalid or leave a split empty.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    train_end = int(n_samples * train_fraction)
    val_end = int(n_samples * (train_fraction + val_fraction))

    if train_end == 0 or val_end == train_end or val_end >= n_samples:
        raise ValueError(
            f"n_samples={n_samples} is too small for a {train_fraction:.0%}/{val_fraction:.0%} chronological split"
        )

    return (
        np.arange(0, train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n_samples),
    )


def train_test_split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified split for i.i.d. classification data."""
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


class Timer:
    """Context manager that records elapsed wall-clock seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.1f}s"


def progress_log(message: str, *, level: int = logging.INFO) -> None:
    """Log a progress line; used by training scripts for reviewer-visible output."""
    logging.getLogger("training").log(level, message)


def describe_array(name: str, array: np.ndarray) -> str:
    """Return a compact one-line shape/dtype summary for training logs."""
    return f"{name}: shape={tuple(array.shape)} dtype={array.dtype}"


def ensure_columns(frame: Any, required: Iterable[str], *, context: str) -> None:
    """Validate that a DataFrame contains every required column.

    Raises:
        ValueError: Listing the missing columns, which is far more useful than a
            downstream ``KeyError`` deep inside a pipeline.
    """
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required column(s) {missing}. Found columns: {list(frame.columns)[:20]}"
        )


def class_distribution(labels: Sequence[Any]) -> dict[str, int]:
    """Return a ``{label: count}`` mapping with string keys, preserving order."""
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts