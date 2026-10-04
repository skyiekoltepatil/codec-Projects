"""Tests for project 01 - Stock Price Predictor.

These tests exercise the real modules. Data-dependent tests download the small
public dataset on first run (about 200 KB) and are marked ``network`` so they can
be skipped in a fully offline environment with ``-m "not network"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_env import use_project

# Activate this project before importing its ``src`` package. Every project uses
# the name ``src``, so this makes the target unambiguous when the whole suite
# runs in one pytest process.
PROJECT_DIR = use_project("01-stock-price-predictor")

from shared.errors import InvalidInputError, ModelNotFoundError  # noqa: E402

from src import data as stock_data  # noqa: E402
from src import features as stock_features  # noqa: E402
from src import model as stock_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_prices() -> pd.DataFrame:
    """A deterministic 300-row synthetic price series for offline tests.

    A random walk with drift is enough to exercise every code path without
    downloading anything.
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=300)
    steps = rng.normal(loc=0.15, scale=1.0, size=len(dates))
    close = 100.0 + np.cumsum(steps)
    close = np.maximum(close, 5.0)
    return pd.DataFrame({"date": dates, "ticker": "TEST", "close": close})


@pytest.fixture(scope="module")
def engineered(synthetic_prices: pd.DataFrame) -> stock_features.FeatureFrame:
    """Engineered features for the synthetic series."""
    return stock_features.make_feature_frame(synthetic_prices)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_load_price_history_returns_expected_shape() -> None:
    """The real dataset loads with the documented columns and tickers."""
    frame = stock_data.load_price_history(["AAPL"])
    assert set(frame.columns) == {"date", "ticker", "close"}
    assert frame["ticker"].unique().tolist() == ["AAPL"]
    assert len(frame) > 1000
    assert frame["date"].is_monotonic_increasing
    assert frame["close"].gt(0).all()


@pytest.mark.network
def test_load_price_history_rejects_unknown_ticker() -> None:
    """An unknown ticker produces a clear, typed error rather than a KeyError."""
    with pytest.raises(InvalidInputError) as excinfo:
        stock_data.load_price_history(["NOT_A_TICKER"])
    assert "Unknown ticker" in excinfo.value.message


@pytest.mark.network
def test_load_price_history_rejects_bad_date() -> None:
    """A malformed date is reported as invalid input."""
    with pytest.raises(InvalidInputError):
        stock_data.load_price_history(["AAPL"], start="not-a-date")


@pytest.mark.network
def test_load_price_history_empty_range_is_invalid() -> None:
    """A date range with no rows raises rather than returning an empty frame."""
    # 2017 is entirely outside the bundled dataset (which ends 2016-03-01),
    # so the filtered frame is empty and must be reported as invalid input.
    with pytest.raises(InvalidInputError):
        stock_data.load_price_history(["AAPL"], start="2017-01-01", end="2017-12-31")


def test_load_price_history_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing dataset file raises DataNotFoundError with a fix hint."""
    from shared.errors import DataNotFoundError

    monkeypatch.setattr(stock_data, "get_data_directory", lambda: tmp_path)
    with pytest.raises(DataNotFoundError) as excinfo:
        stock_data.load_price_history(["AAPL"], auto_download=False)
    assert excinfo.value.hint


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def test_features_have_expected_columns(engineered: stock_features.FeatureFrame) -> None:
    """Every declared feature is produced."""
    for column in stock_features.FEATURE_COLUMNS:
        assert column in engineered.data.columns, column


def test_features_are_free_of_future_leakage(synthetic_prices: pd.DataFrame) -> None:
    """Changing a future price must not alter any past feature row.

    This is the key correctness property for a time-series pipeline: feature
    values at day *t* may depend only on data up to day *t*.
    """
    config = stock_features.FeatureConfig()
    baseline = stock_features.engineer_ticker(synthetic_prices, config)

    mutated = synthetic_prices.copy()
    # Overwrite the final 20 closes with a wildly different value.
    mutated.loc[mutated.index[-20:], "close"] = 9999.0
    changed = stock_features.engineer_ticker(mutated, config)

    # Compare every row except the last 20, whose *features* legitimately shift
    # because a later close becomes a lag for them.
    comparable = slice(0, len(baseline) - 20)
    for column in stock_features.FEATURE_COLUMNS:
        np.testing.assert_allclose(
            baseline[column].to_numpy()[comparable],
            changed[column].to_numpy()[comparable],
            err_msg=f"feature '{column}' changed when only future prices changed",
            equal_nan=True,
        )


def test_target_is_forward_looking(synthetic_prices: pd.DataFrame) -> None:
    """The delta target equals the next close minus the current close."""
    config = stock_features.FeatureConfig(horizon=1)
    frame = stock_features.engineer_ticker(synthetic_prices, config)
    close = frame["close"].to_numpy()
    target = frame[stock_features.TARGET_COLUMN].to_numpy()

    np.testing.assert_allclose(target[:-1], close[1:] - close[:-1], rtol=1e-9)
    assert np.isnan(target[-1]), "the final row must have no future target"


def test_level_target_matches_future_close(synthetic_prices: pd.DataFrame) -> None:
    """The level target equals the raw future close."""
    config = stock_features.FeatureConfig(horizon=1, target_mode="level")
    frame = stock_features.engineer_ticker(synthetic_prices, config)
    np.testing.assert_allclose(frame[stock_features.TARGET_COLUMN].to_numpy()[:-1], frame["close"].to_numpy()[1:])


def test_invalid_target_mode_rejected() -> None:
    """An unknown target mode fails fast with a helpful message."""
    with pytest.raises(InvalidInputError):
        stock_features.FeatureConfig(target_mode="nonsense")


def test_rsi_is_bounded(synthetic_prices: pd.DataFrame) -> None:
    """RSI stays within 0-100 for a normal series."""
    rsi = stock_features.compute_rsi(synthetic_prices["close"].astype(float)).dropna()
    assert rsi.between(0.0, 100.0).all()


def test_ready_rows_drops_incomplete_rows(engineered: stock_features.FeatureFrame) -> None:
    """Usable rows are a strict subset that contains no missing features or target."""
    usable = stock_features.ready_rows(engineered)
    assert 0 < len(usable) < len(engineered.data)
    assert usable[stock_features.FEATURE_COLUMNS].notna().all().all()
    assert usable[stock_features.TARGET_COLUMN].notna().all()


def test_build_features_rejects_missing_columns() -> None:
    """Missing required columns raise a clear error."""
    with pytest.raises(InvalidInputError):
        stock_features.build_features(pd.DataFrame({"date": [], "close": []}))


def test_build_features_rejects_empty() -> None:
    """An empty frame is rejected."""
    with pytest.raises(InvalidInputError):
        stock_features.build_features(pd.DataFrame(columns=["date", "ticker", "close"]))


# --------------------------------------------------------------------------- #
# Training and prediction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", ["linear", "rf", "gb"])
def test_train_ticker_model_produces_finite_metrics(
    engineered: stock_features.FeatureFrame, model_name: str
) -> None:
    """Every estimator trains and yields finite, in-range metrics."""
    ticker_model = stock_model.train_ticker_model(engineered, "TEST", model_name)
    for key in ("mae", "rmse", "r2"):
        assert np.isfinite(ticker_model.metrics[key]), key
    assert ticker_model.metrics["mae"] >= 0
    assert ticker_model.metrics["rmse"] >= 0
    assert ticker_model.n_test > 0
    assert ticker_model.n_train > 0


def test_split_is_chronological(engineered: stock_features.FeatureFrame) -> None:
    """The test block must start after the training block ends."""
    ticker_model = stock_model.train_ticker_model(engineered, "TEST", "linear")
    assert ticker_model.train_end < ticker_model.test_start


def test_train_rejects_short_history(synthetic_prices: pd.DataFrame) -> None:
    """Too little history is reported clearly instead of producing a bad model."""
    short = synthetic_prices.head(40)
    frame = stock_features.make_feature_frame(short)
    with pytest.raises(InvalidInputError):
        stock_model.train_ticker_model(frame, "TEST", "linear")


def test_train_rejects_unknown_model(engineered: stock_features.FeatureFrame) -> None:
    """An unknown estimator key is rejected."""
    with pytest.raises(InvalidInputError):
        stock_model.train_ticker_model(engineered, "TEST", "not_a_model")


def test_train_rejects_unknown_ticker(engineered: stock_features.FeatureFrame) -> None:
    """An unknown ticker is rejected with the available list."""
    with pytest.raises(InvalidInputError):
        stock_model.train_ticker_model(engineered, "NOPE", "linear")


def test_train_bundle_selects_and_covers_all_tickers(
    engineered: stock_features.FeatureFrame,
) -> None:
    """The bundle trains one model per ticker and records a comparison."""
    bundle, comparison = stock_model.train_bundle(engineered, model_names=("linear", "gb"))
    assert bundle.tickers == ["TEST"]
    assert "TEST" in comparison
    assert "selected" in comparison["TEST"]
    assert len(bundle) == 1


def test_prediction_is_close_to_recent_price(engineered: stock_features.FeatureFrame) -> None:
    """A next-day forecast should sit near the last observed close.

    This is a sanity bound, not an accuracy claim: a daily price change is small
    relative to the price level, so a prediction an order of magnitude away
    indicates a reconstruction bug.
    """
    ticker_model = stock_model.train_ticker_model(engineered, "TEST", "linear")
    forecast = stock_model.predict_next_day(ticker_model, engineered)
    assert abs(forecast["prediction"] - forecast["last_close"]) < forecast["last_close"]
    assert forecast["lower_bound"] < forecast["prediction"] < forecast["upper_bound"]
    assert forecast["direction"] in {"up", "down", "flat"}


def test_prediction_table_shapes(engineered: stock_features.FeatureFrame) -> None:
    """Prediction tables expose the documented columns and are non-empty."""
    ticker_model = stock_model.train_ticker_model(engineered, "TEST", "linear")
    table = stock_model.build_prediction_table(ticker_model, engineered)
    assert list(table.columns) == ["date", "ticker", "actual", "predicted", "residual"]
    assert not table.empty

    test_table = stock_model.test_block_predictions(ticker_model, engineered)
    assert list(test_table.columns) == ["date", "ticker", "actual", "predicted", "residual"]
    assert not test_table.empty
    np.testing.assert_allclose(
        test_table["residual"].to_numpy(),
        test_table["actual"].to_numpy() - test_table["predicted"].to_numpy(),
        rtol=1e-9,
    )


def test_feature_importance_available(engineered: stock_features.FeatureFrame) -> None:
    """Both linear and tree models expose a ranked importance table."""
    for name in ("linear", "rf"):
        ticker_model = stock_model.train_ticker_model(engineered, "TEST", name)
        importance = stock_model.feature_importance(ticker_model)
        assert importance is not None
        assert len(importance) == len(stock_features.FEATURE_COLUMNS)
        assert importance["importance"].ge(0).all()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_and_load_roundtrip(engineered: stock_features.FeatureFrame, tmp_path: Path) -> None:
    """A saved bundle reloads and predicts identically."""
    bundle, _ = stock_model.train_bundle(engineered, model_names=("linear",))
    path = stock_model.save_bundle(bundle, directory=tmp_path)
    assert path.exists()

    reloaded = stock_model.load_bundle(path)
    assert reloaded.tickers == bundle.tickers
    assert reloaded.target_mode == bundle.target_mode

    original = stock_model.predict_next_day(bundle.get("TEST"), engineered)
    restored = stock_model.predict_next_day(reloaded.get("TEST"), engineered)
    assert original["prediction"] == pytest.approx(restored["prediction"])


def test_load_missing_model_raises_actionable_error(tmp_path: Path) -> None:
    """A missing artefact raises ModelNotFoundError naming the training command."""
    with pytest.raises(ModelNotFoundError) as excinfo:
        stock_model.load_bundle(tmp_path / "absent.joblib")
    assert "train.py" in (excinfo.value.hint or "")


# --------------------------------------------------------------------------- #
# Streamlit app smoke test
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_app_renders_without_exception() -> None:
    """The Streamlit app runs to completion and reports no unhandled exception.

    Uses Streamlit's own test harness, which executes the script the same way the
    server does and surfaces any exception raised during rendering.
    """
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_DIR / "app.py"
    if not (PROJECT_DIR / "models" / "stock_model.joblib").exists():
        pytest.skip("Run `python train.py` in 01-stock-price-predictor before this test.")

    app = AppTest.from_file(str(app_path), default_timeout=120)
    app.run()
    assert not app.exception, [str(exception.value) for exception in app.exception]
    # The page header and at least one metric must be present.
    assert app.title or app.markdown, "the app rendered no content"
