"""Model training, evaluation, persistence and inference.

Design decision: **one model per ticker**.

An earlier version pooled all five instruments into a single training table.
That produced a misleading R2 of 0.9998 because the pooled price level spans
roughly 10 to 2000, so between-ticker variance dominated the metric while the
model was still poor on each individual series. The honest reference (the
persistence baseline) scored just as well, which gave the flaw away.

Training one model per ticker fixes this: the metrics describe the model's
ability to forecast that specific series. The five small estimators are bundled
into a single ``dict`` saved with joblib, so the UI still loads one file.

Three regressors are compared on the same chronological split:

* ``linear`` - Ordinary least squares. The required baseline; interpretable.
* ``rf``     - Random Forest Regressor. Non-linear, no scaling required.
* ``gb``     - Gradient Boosting Regressor. Usually strongest on tabular data.

A ``persistence`` baseline (tomorrow's price equals today's) is always reported:
if a model cannot beat it, it has learned nothing useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import regression_metrics
from shared.ml import (
    RANDOM_SEED,
    chronological_split,
    load_joblib,
    resolve_device,
    save_joblib,
    save_json,
    set_global_seed,
)
from shared.paths import models_dir

from src.features import (
    FEATURE_COLUMNS,
    MIN_TRAINING_ROWS,
    TARGET_COLUMN,
    FeatureConfig,
    FeatureFrame,
    ready_rows,
)

PROJECT_SLUG = "01-stock-price-predictor"


def _actual_price(frame: FeatureFrame, base_close: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Recover the actual future price from the stored target.

    For the ``delta`` target mode the stored target is a price change, so it is
    added back to the base close. For ``level`` it is already a price.
    """
    if frame.target_mode == "level":
        return np.asarray(target, dtype=float)
    return np.asarray(base_close, dtype=float) + np.asarray(target, dtype=float)


#: Human-readable names for each supported estimator.
MODEL_LABELS: dict[str, str] = {
    "linear": "Linear Regression",
    "rf": "Random Forest Regressor",
    "gb": "Gradient Boosting Regressor",
    "persistence": "Persistence baseline (tomorrow = today)",
}

#: Estimators that can actually be fitted. ``persistence`` is a reference only.
SELECTABLE_MODELS: tuple[str, ...] = ("linear", "rf", "gb")

#: Hyperparameters, kept explicit so the README can document them.
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "linear": {},
    "rf": {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "gb": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": RANDOM_SEED,
    },
}

#: Default train/validation/test fractions for the chronological split.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1


def build_estimator(name: str) -> Any:
    """Instantiate a fresh estimator by key.

    Args:
        name: One of :data:`SELECTABLE_MODELS`.

    Raises:
        InvalidInputError: For an unknown model key.
    """
    if name == "linear":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**MODEL_PARAMS["rf"])

    if name == "gb":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(**MODEL_PARAMS["gb"])

    raise InvalidInputError(
        f"Unknown model '{name}'.",
        hint=f"Choose one of: {', '.join(SELECTABLE_MODELS)}.",
    )


@dataclass
class TickerModel:
    """A fitted estimator for one ticker plus its evaluation results."""

    ticker: str
    model: Any
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    residual_std: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Predict from an already-built ``(n_samples, n_features)`` matrix."""
        return np.asarray(self.model.predict(matrix), dtype=float)

    @property
    def beats_baseline(self) -> bool:
        """Whether this model beats the persistence baseline on test RMSE."""
        if not self.metrics or not self.baseline_metrics:
            return False
        return bool(self.metrics["rmse"] < self.baseline_metrics["rmse"])

    def interval_half_width(self, confidence: float = 0.95) -> float:
        """Return the empirical half-width of a prediction interval.

        Uses the normal approximation ``z * residual_std``. This describes the
        spread of model errors on the test block; it is not a market risk model.
        """
        if not np.isfinite(self.residual_std):
            return float(self.metrics.get("rmse", 0.0))
        z_scores = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        z = z_scores.get(round(confidence, 2), 1.9600)
        return float(z * self.residual_std)


@dataclass
class StockModelBundle:
    """All per-ticker models, saved and loaded as a single artefact."""

    models: dict[str, TickerModel] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    horizon: int = 1
    target_mode: str = "delta"
    trained_at: str = ""

    def __len__(self) -> int:
        return len(self.models)

    @property
    def tickers(self) -> list[str]:
        """Tickers covered by this bundle, sorted."""
        return sorted(self.models)

    def get(self, ticker: str) -> TickerModel:
        """Return the model for ``ticker``.

        Raises:
            InvalidInputError: When the ticker was not part of training.
        """
        if ticker not in self.models:
            raise InvalidInputError(
                f"No trained model for '{ticker}'.",
                hint=f"Available tickers: {', '.join(self.tickers) or 'none'}.",
            )
        return self.models[ticker]

    def best_model_name(self) -> str:
        """Return the estimator key used by the most tickers."""
        if not self.models:
            return "n/a"
        counts: dict[str, int] = {}
        for ticker_model in self.models.values():
            counts[ticker_model.model_name] = counts.get(ticker_model.model_name, 0) + 1
        return max(counts, key=lambda name: counts[name])

    def summary_table(self) -> pd.DataFrame:
        """Return a per-ticker metrics table for the UI and reports."""
        rows = []
        for ticker, ticker_model in sorted(self.models.items()):
            rows.append(
                {
                    "ticker": ticker,
                    "model": MODEL_LABELS.get(ticker_model.model_name, ticker_model.model_name),
                    "mae": ticker_model.metrics.get("mae", float("nan")),
                    "rmse": ticker_model.metrics.get("rmse", float("nan")),
                    "r2": ticker_model.metrics.get("r2", float("nan")),
                    "baseline_rmse": ticker_model.baseline_metrics.get("rmse", float("nan")),
                    "beats_baseline": ticker_model.beats_baseline,
                    "test_rows": ticker_model.n_test,
                }
            )
        return pd.DataFrame(rows)

    def mean_metrics(self) -> dict[str, float]:
        """Return the arithmetic mean of MAE/RMSE/R2 across tickers."""
        if not self.models:
            return {}
        keys = ("mae", "rmse", "r2")
        return {
            key: float(np.mean([ticker_model.metrics.get(key, np.nan) for ticker_model in self.models.values()]))
            for key in keys
        }


def train_ticker_model(
    frame: FeatureFrame,
    ticker: str,
    model_name: str = "linear",
    *,
    config: FeatureConfig | None = None,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> TickerModel:
    """Train and evaluate one estimator for one ticker.

    Args:
        frame: Engineered features covering all tickers.
        ticker: The ticker to train on.
        model_name: One of :data:`SELECTABLE_MODELS`.
        config: Feature configuration (supplies the prediction horizon).
        train_fraction: Share of the earliest rows used for training.
        val_fraction: Share reserved between the train and test blocks.

    Returns:
        A fitted :class:`TickerModel` with test metrics and baseline metrics.

    Raises:
        InvalidInputError: When the ticker is absent or has too little history.

    Note:
        The split is chronological. Random shuffling is never used, because it
        would place future observations in the training set and inflate scores.
    """
    config = config or FeatureConfig()

    if ticker not in frame.tickers():
        raise InvalidInputError(
            f"Ticker '{ticker}' is not present in the engineered features.",
            hint=f"Available tickers: {', '.join(frame.tickers()) or 'none'}.",
        )

    usable = ready_rows(frame.for_ticker(ticker)).sort_values("date").reset_index(drop=True)

    if len(usable) < MIN_TRAINING_ROWS:
        raise InvalidInputError(
            f"Ticker '{ticker}' has only {len(usable)} usable rows after indicator warm-up.",
            hint=f"At least {MIN_TRAINING_ROWS} are required. Choose a longer date range.",
        )

    train_idx, _val_idx, test_idx = chronological_split(
        len(usable), train_fraction=train_fraction, val_fraction=val_fraction
    )
    train_rows = usable.iloc[train_idx]
    test_rows = usable.iloc[test_idx]

    x_train = train_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_rows[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_rows[TARGET_COLUMN].to_numpy(dtype=float)

    estimator = build_estimator(model_name)
    estimator.fit(x_train, y_train)
    raw_predictions = np.asarray(estimator.predict(x_test), dtype=float)

    # Metrics are always computed on the *price*, never on the internal target.
    # With the default delta target the model predicts a price change, so it is
    # added back to the base close before scoring.
    base_close = test_rows["close"].to_numpy(dtype=float)
    predictions = frame.reconstruct_price(base_close, raw_predictions)
    actual_price = base_close + y_test if frame.target_mode == "delta" else y_test

    metrics = regression_metrics(actual_price, predictions)
    # Honest reference: "tomorrow equals today" is exactly the lag_1 feature.
    baseline_metrics = regression_metrics(actual_price, test_rows["lag_1"].to_numpy(dtype=float))

    residuals = actual_price - predictions
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")

    return TickerModel(
        ticker=ticker,
        model=estimator,
        model_name=model_name,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        residual_std=residual_std,
        n_train=int(len(train_rows)),
        n_test=int(len(test_rows)),
        train_start=train_rows["date"].min().strftime("%Y-%m-%d"),
        train_end=train_rows["date"].max().strftime("%Y-%m-%d"),
        test_start=test_rows["date"].min().strftime("%Y-%m-%d"),
        test_end=test_rows["date"].max().strftime("%Y-%m-%d"),
        hyperparameters=dict(MODEL_PARAMS.get(model_name, {})),
    )


def train_bundle(
    frame: FeatureFrame,
    *,
    model_names: tuple[str, ...] = SELECTABLE_MODELS,
    config: FeatureConfig | None = None,
) -> tuple[StockModelBundle, dict[str, dict[str, dict[str, float]]]]:
    """Train every candidate estimator for every ticker and pick the winner each.

    Args:
        frame: Engineered features for all tickers.
        model_names: Candidate estimator keys to compare.
        config: Feature configuration.

    Returns:
        ``(bundle, comparison)`` where ``comparison`` maps
        ``ticker -> model_name -> {"metrics": ..., "baseline": ...}`` so the
        README and final report can quote the full comparison.

    Raises:
        InvalidInputError: When no ticker has enough usable history.
    """
    set_global_seed()
    config = config or FeatureConfig()

    tickers = frame.tickers()
    if not tickers:
        raise InvalidInputError("No tickers found in the feature frame.", hint="Reload the dataset.")

    comparison: dict[str, dict[str, dict[str, float]]] = {}
    selected: dict[str, TickerModel] = {}

    for ticker in tickers:
        comparison[ticker] = {}
        candidates: list[TickerModel] = []

        for model_name in model_names:
            try:
                ticker_model = train_ticker_model(frame, ticker, model_name, config=config)
            except InvalidInputError:
                # Not enough history for this ticker: skip it rather than
                # failing the whole run, and record why in the comparison.
                continue
            candidates.append(ticker_model)
            comparison[ticker][model_name] = {
                "metrics": ticker_model.metrics,
                "baseline": ticker_model.baseline_metrics,
            }

        if not candidates:
            continue

        # Choose the lowest test RMSE, the metric that penalises large misses.
        winner = min(candidates, key=lambda candidate: candidate.metrics["rmse"])
        comparison[ticker]["selected"] = {
            "model": winner.model_name,
            "metrics": winner.metrics,
            "baseline": winner.baseline_metrics,
        }
        selected[ticker] = winner

    if not selected:
        raise InvalidInputError(
            "No ticker had enough usable history to train a model.",
            hint=f"At least {MIN_TRAINING_ROWS} rows per ticker are required. Choose a longer date range.",
        )

    bundle = StockModelBundle(
        models=selected,
        feature_columns=list(config.feature_columns),
        horizon=config.horizon,
        target_mode=config.target_mode,
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    )
    return bundle, comparison


def save_bundle(bundle: StockModelBundle, *, directory: Path | None = None) -> Path:
    """Persist the model bundle with joblib and return the artefact path."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "stock_model.joblib"
    save_joblib(bundle, path)
    return path


def load_bundle(path: Path | None = None) -> StockModelBundle:
    """Load a previously saved :class:`StockModelBundle`.

    Raises:
        ModelNotFoundError: When the artefact is missing or has the wrong type.
    """
    target = Path(path) if path else models_dir(PROJECT_SLUG) / "stock_model.joblib"
    loaded = load_joblib(
        target,
        description="stock price model",
        train_command="python train.py",
    )
    if not isinstance(loaded, StockModelBundle):
        raise ModelNotFoundError(
            f"'{target.name}' does not contain a StockModelBundle.",
            hint="Delete the file and re-run `python train.py`.",
        )
    return loaded


def build_prediction_table(
    ticker_model: TickerModel,
    frame: FeatureFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Return actual vs predicted values for every usable row of a ticker.

    The final ``horizon`` rows have no target yet, so they are excluded; they are
    what :func:`predict_next_day` is for.
    """
    resolved_ticker = ticker or ticker_model.ticker
    usable = ready_rows(frame.for_ticker(resolved_ticker)).sort_values("date").reset_index(drop=True)
    if usable.empty:
        return pd.DataFrame(columns=["date", "ticker", "actual", "predicted", "residual"])

    base_close = usable["close"].to_numpy(dtype=float)
    raw = ticker_model.predict_matrix(usable[FEATURE_COLUMNS].to_numpy(dtype=float))
    predicted = frame.reconstruct_price(base_close, raw)
    actual = _actual_price(frame, base_close, usable[TARGET_COLUMN].to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "date": usable["date"],
            "ticker": usable["ticker"],
            "actual": actual,
            "predicted": predicted,
            "residual": actual - predicted,
        }
    )


def test_block_predictions(
    ticker_model: TickerModel,
    frame: FeatureFrame,
    *,
    config: FeatureConfig | None = None,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> pd.DataFrame:
    """Return actual vs predicted values for the held-out test block only.

    This is the table the UI plots so that the visualised points correspond
    exactly to the reported metrics.
    """
    config = config or FeatureConfig()
    usable = ready_rows(frame.for_ticker(ticker_model.ticker)).sort_values("date").reset_index(drop=True)
    if len(usable) < MIN_TRAINING_ROWS:
        return pd.DataFrame(columns=["date", "ticker", "actual", "predicted", "residual"])

    _train_idx, _val_idx, test_idx = chronological_split(
        len(usable), train_fraction=train_fraction, val_fraction=val_fraction
    )
    test_rows = usable.iloc[test_idx]
    base_close = test_rows["close"].to_numpy(dtype=float)
    raw = ticker_model.predict_matrix(test_rows[FEATURE_COLUMNS].to_numpy(dtype=float))
    predicted = frame.reconstruct_price(base_close, raw)
    actual = _actual_price(frame, base_close, test_rows[TARGET_COLUMN].to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "date": test_rows["date"],
            "ticker": test_rows["ticker"],
            "actual": actual,
            "predicted": predicted,
            "residual": actual - predicted,
        }
    )


def predict_next_day(
    ticker_model: TickerModel,
    frame: FeatureFrame,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Produce the next-day prediction with an empirical interval.

    The most recent row with complete features is the only legitimate input for
    a next-day forecast, so it is selected explicitly rather than taking the
    last row of the raw frame (which may still be in the indicator warm-up).

    Args:
        ticker_model: A fitted model for one ticker.
        frame: Engineered features.
        confidence: Interval confidence level, one of 0.80/0.90/0.95/0.99.

    Returns:
        Dictionary describing the forecast, including the interval and the
        change versus the most recent observed close.

    Raises:
        InvalidInputError: When no complete feature row exists for the ticker.
    """
    usable = ready_rows(frame.for_ticker(ticker_model.ticker)).sort_values("date").reset_index(drop=True)

    if usable.empty:
        raise InvalidInputError(
            f"No complete feature row is available for '{ticker_model.ticker}'.",
            hint="Choose a ticker with more history so all indicators are available.",
        )

    latest = usable.iloc[[-1]]
    matrix = latest[FEATURE_COLUMNS].to_numpy(dtype=float)
    raw_prediction = float(ticker_model.predict_matrix(matrix)[0])

    last_close = float(latest["close"].iloc[0])
    # The model predicts a price change by default; add it to the latest close.
    prediction = frame.to_price(last_close, raw_prediction)
    half_width = ticker_model.interval_half_width(confidence)
    expected_change = prediction - last_close
    expected_change_pct = (expected_change / last_close * 100.0) if last_close else float("nan")

    return {
        "ticker": ticker_model.ticker,
        "as_of_date": pd.Timestamp(latest["date"].iloc[0]).strftime("%Y-%m-%d"),
        "last_close": last_close,
        "prediction": prediction,
        "lower_bound": prediction - half_width,
        "upper_bound": prediction + half_width,
        "interval_half_width": half_width,
        "confidence_level": confidence,
        "expected_change": expected_change,
        "expected_change_pct": expected_change_pct,
        "direction": "up" if expected_change > 0 else ("down" if expected_change < 0 else "flat"),
        "model_name": ticker_model.model_name,
        "model_label": MODEL_LABELS.get(ticker_model.model_name, ticker_model.model_name),
        "horizon_days": 1,
    }


def feature_importance(ticker_model: TickerModel) -> pd.DataFrame | None:
    """Return a feature-importance table, or ``None`` when unavailable.

    Tree ensembles expose impurity-based importances; linear models expose
    coefficients, which are absolute-valued here so the ranking is comparable.
    """
    estimator = ticker_model.model
    names = list(FEATURE_COLUMNS)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
        kind = "absolute coefficient"
    else:
        return None

    if len(values) != len(names):
        return None

    frame = pd.DataFrame({"feature": names, "importance": values, "kind": kind})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def build_training_report(
    bundle: StockModelBundle,
    comparison: dict[str, dict[str, dict[str, float]]],
    *,
    dataset: dict[str, Any],
    feature_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON report written to ``models/training_metrics.json``."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": bundle.trained_at,
        "split": {
            "strategy": "chronological (no shuffling)",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VAL_FRACTION,
            "test_fraction": round(1 - TRAIN_FRACTION - VAL_FRACTION, 4),
        },
        "horizon_days": bundle.horizon,
        "models_per_ticker": {ticker: model.model_name for ticker, model in bundle.models.items()},
        "model_labels": MODEL_LABELS,
        "dataset": dataset,
        "feature_summary": feature_summary,
        "comparison": comparison,
        "per_ticker_summary": bundle.summary_table().to_dict(orient="records"),
        "mean_metrics": bundle.mean_metrics(),
        "device": resolve_device(),
        "random_seed": RANDOM_SEED,
        "disclaimer": (
            "Educational model. Historical price patterns do not reliably predict future "
            "market prices. This output is not investment advice."
        ),
    }


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")