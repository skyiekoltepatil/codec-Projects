"""Leakage-free feature engineering for next-day weather prediction.

Every feature for day *t* is built from observations available at the end of day
*t*: lagged temperatures, trailing rolling statistics and calendar terms. The
target is the mean temperature ``horizon`` days ahead, expressed as a *change*
from today's value, which is the default because tree ensembles cannot
extrapolate beyond the range of values seen during training.

The shifted future column is the only forward-looking quantity and it is dropped
before fitting, so no future observation can influence a training row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared.errors import InvalidInputError

#: Feature columns produced by :func:`build_features`, in a fixed order. The
#: order is stored with the model and re-checked at inference time.
FEATURE_COLUMNS: list[str] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_30",
    "roll_std_7",
    "roll_min_7",
    "roll_max_7",
    "temp_range",
    "precipitation",
    "wind_max",
    "day_of_year_sin",
    "day_of_year_cos",
    "month",
    "day_of_week",
]

#: Name of the supervised target column.
TARGET_COLUMN = "target"

#: Supported target definitions; see the module docstring for why ``delta`` wins.
TARGET_MODES: tuple[str, ...] = ("delta", "level")
DEFAULT_TARGET_MODE = "delta"

#: Default lag offsets (days) and trailing rolling windows (days).
LAGS: tuple[int, ...] = (1, 2, 3, 7)
ROLL_WINDOWS: tuple[int, ...] = (7, 14, 30)

#: Minimum usable rows before a model is worth training.
MIN_TRAINING_ROWS = 120


@dataclass
class FeatureConfig:
    """Configuration for feature construction.

    Attributes:
        lags: Lag offsets, in days, for the raw mean temperature.
        roll_windows: Trailing windows, in days, for the rolling statistics.
        horizon: Days ahead to predict (1 = tomorrow).
        target_mode: ``"delta"`` (default) or ``"level"``.
        feature_columns: Model input order; must stay stable between training and
            inference.
    """

    lags: tuple[int, ...] = LAGS
    roll_windows: tuple[int, ...] = ROLL_WINDOWS
    horizon: int = 1
    target_mode: str = DEFAULT_TARGET_MODE
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    @property
    def warmup_rows(self) -> int:
        """Rows consumed by the longest look-back before features are complete."""
        return max(max(self.lags), max(self.roll_windows))


@dataclass
class FeatureFrame:
    """An engineered frame plus the metadata needed to interpret it."""

    data: pd.DataFrame
    feature_columns: list[str]
    target_mode: str = DEFAULT_TARGET_MODE
    target_column: str = TARGET_COLUMN

    def __len__(self) -> int:
        """Number of rows in the underlying frame."""
        return len(self.data)

    def ready(self) -> pd.DataFrame:
        """Return only the rows with every feature and the target present."""
        return ready_rows(self)


def _engineer(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Build the engineered columns for a single city, sorted chronologically."""
    result = frame.sort_values("date").reset_index(drop=True).copy()
    mean = result["temp_mean"].astype(float)

    result["temp_range"] = result["temp_max"].astype(float) - result["temp_min"].astype(float)

    for lag in config.lags:
        result[f"lag_{lag}"] = mean.shift(lag)

    for window in config.roll_windows:
        result[f"roll_mean_{window}"] = mean.rolling(window).mean()

    short = min(config.roll_windows)
    trailing = mean.rolling(short)
    result[f"roll_std_{short}"] = trailing.std()
    result[f"roll_min_{short}"] = trailing.min()
    result[f"roll_max_{short}"] = trailing.max()

    day_of_year = result["date"].dt.dayofyear
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["month"] = result["date"].dt.month
    result["day_of_week"] = result["date"].dt.dayofweek

    future = mean.shift(-config.horizon)
    if config.target_mode == "delta":
        result[TARGET_COLUMN] = future - mean
    else:
        result[TARGET_COLUMN] = future

    return result


def build_features(
    frame: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
) -> FeatureFrame:
    """Build the supervised modelling table for one city.

    Raises:
        InvalidInputError: When required columns are missing, the target mode is
            unknown, or too few usable rows remain.
    """
    config = config or FeatureConfig()
    if config.target_mode not in TARGET_MODES:
        raise InvalidInputError(
            f"Unknown target mode '{config.target_mode}'.",
            hint=f"Choose one of: {', '.join(TARGET_MODES)}.",
        )
    if config.horizon < 1:
        raise InvalidInputError(
            f"Horizon must be at least 1 day, got {config.horizon}.",
            hint="Use a positive integer number of days.",
        )

    if frame is None or frame.empty:
        raise InvalidInputError(
            "No weather data supplied.",
            hint="Load a city's data before building features.",
        )

    required = ["date", "temp_mean", "temp_max", "temp_min", "precipitation", "wind_max"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise InvalidInputError(
            f"Weather data is missing column(s): {', '.join(missing)}.",
            hint="Reload the dataset; it may have been modified.",
        )

    engineered = _engineer(frame, config)

    missing_features = [column for column in config.feature_columns if column not in engineered.columns]
    if missing_features:
        raise InvalidInputError(
            f"Feature construction did not produce: {', '.join(missing_features)}.",
            hint="This indicates a bug in feature engineering.",
        )

    result = FeatureFrame(
        engineered,
        feature_columns=list(config.feature_columns),
        target_mode=config.target_mode,
    )
    usable = ready_rows(result)
    if len(usable) < MIN_TRAINING_ROWS:
        raise InvalidInputError(
            f"Only {len(usable)} complete rows are available.",
            hint=f"At least {MIN_TRAINING_ROWS} days of history are needed to train.",
        )
    return result


def ready_rows(frame: FeatureFrame) -> pd.DataFrame:
    """Return only the rows that are complete enough to train or predict on."""
    columns = [*frame.feature_columns, frame.target_column]
    return frame.data.dropna(subset=columns).reset_index(drop=True)


def make_feature_frame(frame: pd.DataFrame, *, config: FeatureConfig | None = None) -> FeatureFrame:
    """Convenience wrapper around :func:`build_features`."""
    return build_features(frame, config=config)


def summarise_features(frame: FeatureFrame) -> dict[str, object]:
    """Return a short description of an engineered frame for logs and the UI."""
    usable = ready_rows(frame)
    return {
        "total_rows": int(len(frame.data)),
        "usable_rows": int(len(usable)),
        "features": len(frame.feature_columns),
        "start": frame.data["date"].min().strftime("%Y-%m-%d") if len(frame.data) else "n/a",
        "end": frame.data["date"].max().strftime("%Y-%m-%d") if len(frame.data) else "n/a",
    }

