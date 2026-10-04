"""Regression models and persistence helpers for next-day weather prediction.

Three estimators are compared on the *same* chronological split and each is
judged against a persistence baseline (tomorrow = today). Because temperature is
strongly autocorrelated, the baseline is genuinely hard to beat, and reporting it
next to the model keeps the evaluation honest.

All metrics are computed in **degrees Celsius** in absolute temperature space, so
the numbers are directly interpretable rather than being an error on a delta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import load_metrics, regression_metrics, save_metrics
from shared.ml import (
    RANDOM_SEED,
    chronological_split,
    load_joblib,
    save_joblib,
    set_global_seed,
)
from shared.paths import models_dir

from src.features import FeatureConfig, FeatureFrame, ready_rows

PROJECT_SLUG = "09-weather-analysis-prediction"

#: Filenames inside ``models/``.
MODEL_FILENAME = "weather_model.joblib"
METRICS_FILENAME = "training_metrics.json"

#: Selectable model keys, in presentation order.
SELECTABLE_MODELS: tuple[str, ...] = ("linear", "rf", "gb")

#: Human-readable labels used in tables and the interface.
MODEL_LABELS: dict[str, str] = {
    "linear": "Linear Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

#: Hyperparameters, kept explicit so this module is the single source of truth.
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "linear": {},
    "rf": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "random_state": RANDOM_SEED},
    "gb": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "random_state": RANDOM_SEED},
}

#: Fraction of ordered samples used for training and validation (rest is test).
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1


@dataclass
class WeatherBundle:
    """A trained model plus everything needed to interpret its output."""

    city: str
    model: Any
    model_name: str
    feature_columns: list[str]
    target_mode: str
    horizon: int
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    n_train: int
    n_test: int
    model_params: dict[str, Any] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    trained_at: str = ""

    @property
    def algorithm(self) -> str:
        """Human-readable model name."""
        return MODEL_LABELS.get(self.model_name, self.model_name)

    @property
    def beats_baseline(self) -> bool:
        """Whether the model's test RMSE is lower than the persistence baseline."""
        return self.metrics.get("rmse", float("inf")) < self.baseline_metrics.get("rmse", float("inf"))


def _make_model(model_name: str, config: FeatureConfig):
    """Return an unfitted scikit-learn estimator for ``model_name``."""
    if model_name not in MODEL_PARAMS:
        raise InvalidInputError(
            f"Unknown model '{model_name}'.",
            hint=f"Choose one of: {', '.join(SELECTABLE_MODELS)}.",
        )
    params = dict(MODEL_PARAMS[model_name])
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    if model_name == "linear":
        return LinearRegression()
    if model_name == "rf":
        return RandomForestRegressor(**params)
    return GradientBoostingRegressor(**params)


def _feature_matrix(frame: FeatureFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(usable_rows, X, y)`` for a feature frame."""
    usable = ready_rows(frame)
    if usable.empty:
        raise InvalidInputError(
            "No complete rows are available for training.",
            hint="Download more history for this city.",
        )
    X = usable[frame.feature_columns].to_numpy(dtype=float)
    y = usable[frame.target_column].to_numpy(dtype=float)
    return usable, X, y


def _to_absolute(target_mode: str, current: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Convert model outputs to absolute temperature for the given target mode."""
    if target_mode == "delta":
        return current + values
    return values


def _importances(model: Any, feature_columns: list[str]) -> dict[str, float]:
    """Return feature importances, using coefficients for linear models."""
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    if values is None or values.size != len(feature_columns):
        return {}
    return {name: float(value) for name, value in zip(feature_columns, values)}


def train_single(
    frame: FeatureFrame,
    *,
    config: FeatureConfig,
    model_name: str,
) -> tuple[Any, dict[str, float], dict[str, float], dict[str, float], int, int]:
    """Fit one estimator on the chronological split and evaluate it.

    Returns:
        ``(model, metrics, baseline, importances, n_train, n_test)`` where both
        metric dictionaries are in absolute degrees Celsius.
    """
    usable, X, y = _feature_matrix(frame)
    train_idx, _val_idx, test_idx = chronological_split(
        len(y), train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION
    )

    model = _make_model(model_name, config)
    model.fit(X[train_idx], y[train_idx])

    current = usable["temp_mean"].to_numpy(dtype=float)
    test_current = current[test_idx]

    predicted = _to_absolute(frame.target_mode, test_current, model.predict(X[test_idx]))
    actual = _to_absolute(frame.target_mode, test_current, y[test_idx])
    baseline = _to_absolute(frame.target_mode, test_current, np.zeros_like(y[test_idx]))

    metrics = regression_metrics(actual, predicted)
    baseline_metrics = regression_metrics(actual, baseline)
    importances = _importances(model, frame.feature_columns)
    return model, metrics, baseline_metrics, importances, int(len(train_idx)), int(len(test_idx))


def train_bundle(
    frame: FeatureFrame,
    *,
    city: str,
    config: FeatureConfig | None = None,
) -> tuple[WeatherBundle, dict[str, Any]]:
    """Train every candidate, select the best by test RMSE and bundle it.

    Returns:
        ``(bundle, comparison)`` where ``comparison`` maps ``model_name ->
        {"metrics": ..., "baseline": ...}`` plus a ``"selected"`` entry, so the
        README and the interface can quote the full comparison.
    """
    config = config or FeatureConfig()
    set_global_seed()
    comparison: dict[str, Any] = {}
    best: tuple[str, Any, dict[str, float], dict[str, float], dict[str, float], int, int] | None = None

    for model_name in SELECTABLE_MODELS:
        model, metrics, baseline, importances, n_train, n_test = train_single(
            frame, config=config, model_name=model_name
        )
        comparison[model_name] = {"metrics": metrics, "baseline": baseline}
        if best is None or metrics["rmse"] < best[2]["rmse"]:
            best = (model_name, model, metrics, baseline, importances, n_train, n_test)

    assert best is not None  # SELECTABLE_MODELS is non-empty
    model_name, model, metrics, baseline, importances, n_train, n_test = best
    comparison["selected"] = {"model": model_name, "metrics": metrics}

    bundle = WeatherBundle(
        city=city,
        model=model,
        model_name=model_name,
        feature_columns=list(frame.feature_columns),
        target_mode=frame.target_mode,
        horizon=config.horizon,
        metrics=metrics,
        baseline_metrics=baseline,
        n_train=n_train,
        n_test=n_test,
        model_params=MODEL_PARAMS[model_name],
        feature_importances=importances,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return bundle, comparison


def latest_ready_row(frame: FeatureFrame) -> pd.Series:
    """Return the most recent row whose features are all present.

    The target may be missing on this row (tomorrow has not happened yet), which
    is exactly the row a genuine next-day forecast is made from.

    Raises:
        InvalidInputError: When no row has a complete feature vector.
    """
    columns = frame.feature_columns
    complete = frame.data[frame.data[columns].notna().all(axis=1)]
    if complete.empty:
        raise InvalidInputError(
            "Not enough history to build a forecast for this city.",
            hint="Choose a longer date range.",
        )
    return complete.iloc[-1]


def predict_next(bundle: WeatherBundle, frame: FeatureFrame) -> dict[str, float | str]:
    """Return a next-day forecast from the most recent complete row."""
    row = latest_ready_row(frame)
    X = row[bundle.feature_columns].to_numpy(dtype=float).reshape(1, -1)
    current = float(row["temp_mean"])
    raw = float(np.asarray(bundle.model.predict(X)).ravel()[0])
    predicted = current + raw if bundle.target_mode == "delta" else raw
    last_date = pd.Timestamp(row["date"])
    return {
        "date": (last_date + pd.Timedelta(days=bundle.horizon)).strftime("%Y-%m-%d"),
        "based_on": last_date.strftime("%Y-%m-%d"),
        "current": current,
        "predicted": float(predicted),
        "change": float(predicted - current),
        "model": bundle.algorithm,
    }


def predict_series(bundle: WeatherBundle, frame: FeatureFrame) -> pd.DataFrame:
    """Return model and baseline predictions for every usable historical row."""
    usable = ready_rows(frame)
    if usable.empty:
        return pd.DataFrame(columns=["date", "actual", "predicted", "baseline", "residual"])
    X = usable[bundle.feature_columns].to_numpy(dtype=float)
    y = usable[bundle.target_column].to_numpy(dtype=float)
    current = usable["temp_mean"].to_numpy(dtype=float)
    predicted = _to_absolute(bundle.target_mode, current, bundle.model.predict(X))
    actual = _to_absolute(bundle.target_mode, current, y)
    return pd.DataFrame(
        {
            "date": usable["date"].to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "baseline": current,
            "residual": actual - predicted,
        }
    )



def model_directory() -> Path:
    """Return (and create) this project's ``models/`` directory."""
    return models_dir(PROJECT_SLUG)


def save_bundle(bundle: WeatherBundle, path: str | Path | None = None) -> Path:
    """Persist a bundle with joblib."""
    target = Path(path) if path is not None else model_directory() / MODEL_FILENAME
    return save_joblib(bundle, target)


def load_bundle(
    path: str | Path | None = None,
    *,
    train_command: str = "python train.py",
) -> WeatherBundle:
    """Load a bundle, raising an actionable error when it is missing."""
    target = Path(path) if path is not None else model_directory() / MODEL_FILENAME
    bundle = load_joblib(target, description="weather model", train_command=train_command)
    if not isinstance(bundle, WeatherBundle):
        raise ModelNotFoundError(
            "The saved artefact is not a weather model.",
            hint="Delete it and re-run python train.py.",
        )
    return bundle


def build_training_report(
    bundle: WeatherBundle,
    comparison: dict[str, Any],
    *,
    dataset: dict[str, Any],
    feature_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON report written next to the model."""
    return {
        "project": PROJECT_SLUG,
        "city": bundle.city,
        "selected_model": bundle.model_name,
        "selected_label": bundle.algorithm,
        "target_mode": bundle.target_mode,
        "horizon_days": bundle.horizon,
        "model_params": bundle.model_params,
        "metrics": bundle.metrics,
        "baseline_metrics": bundle.baseline_metrics,
        "beats_baseline": bundle.beats_baseline,
        "n_train": bundle.n_train,
        "n_test": bundle.n_test,
        "feature_importances": bundle.feature_importances,
        "feature_columns": bundle.feature_columns,
        "comparison": comparison,
        "dataset": dataset,
        "features": feature_summary,
        "random_seed": RANDOM_SEED,
        "trained_at": bundle.trained_at,
    }


def save_metrics_report(report: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write the training report as JSON."""
    target = Path(path) if path is not None else model_directory() / METRICS_FILENAME
    return save_metrics(report, target)


def load_metrics_report(path: str | Path | None = None) -> dict[str, Any]:
    """Load the JSON report written by :func:`save_metrics_report`."""
    target = Path(path) if path is not None else model_directory() / METRICS_FILENAME
    return load_metrics(target)

