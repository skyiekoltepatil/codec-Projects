"""Tests for Project 09 - Weather Analysis & Prediction.

Every test uses a synthetic temperature series so the suite is fast and needs no
network. The tests assert real behaviour: leakage-free features, model
selection against the persistence baseline, and artefact round-tripping.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from project_env import REPO_ROOT, use_project

use_project("09-weather-analysis-prediction")

sys.path.insert(0, str(REPO_ROOT))

from shared.errors import InvalidInputError  # noqa: E402
from src import data as weather_data  # noqa: E402
from src import features as weather_features  # noqa: E402
from src import model as weather_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _synthetic_weather(days: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Return a chronological daily weather table with a seasonal signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    seasonal = 12.0 + 10.0 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    mean = seasonal + rng.normal(0.0, 1.5, days)
    return pd.DataFrame(
        {
            "date": dates,
            "temp_mean": mean,
            "temp_max": mean + rng.uniform(1.0, 4.0, days),
            "temp_min": mean - rng.uniform(1.0, 4.0, days),
            "precipitation": rng.uniform(0.0, 12.0, days),
            "wind_max": rng.uniform(0.0, 30.0, days),
        }
    )


@pytest.fixture()
def weather_frame() -> pd.DataFrame:
    """A reusable synthetic weather frame."""
    return _synthetic_weather()


@pytest.fixture()
def feature_config() -> weather_features.FeatureConfig:
    """The default feature configuration."""
    return weather_features.FeatureConfig()


# --------------------------------------------------------------------------- #
# Data module
# --------------------------------------------------------------------------- #


def test_unknown_city_is_rejected() -> None:
    """An unknown city label raises a typed error rather than a KeyError."""
    with pytest.raises(InvalidInputError):
        weather_data.cache_path("Atlantis")


def test_city_cache_paths_are_distinct() -> None:
    """Each catalogue city maps to its own cache file."""
    paths = {weather_data.cache_path(label) for label in weather_data.CITY_CATALOGUE}
    assert len(paths) == len(weather_data.CITY_CATALOGUE)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def test_features_produce_the_documented_columns(weather_frame, feature_config) -> None:
    """Every advertised feature column is present and numeric."""
    result = weather_features.build_features(weather_frame, config=feature_config)
    for column in feature_config.feature_columns:
        assert column in result.data.columns
        assert pd.api.types.is_numeric_dtype(result.data[column])


def test_features_are_free_of_future_leakage(weather_frame, feature_config) -> None:
    """Changing a future temperature must not alter any earlier feature row."""
    baseline = weather_features.build_features(weather_frame, config=feature_config).data

    mutated = weather_frame.copy()
    mutated.loc[mutated.index[-1], "temp_mean"] = 999.0
    changed = weather_features.build_features(mutated, config=feature_config).data

    columns = feature_config.feature_columns
    pd.testing.assert_frame_equal(
        baseline[columns].iloc[:-1].reset_index(drop=True),
        changed[columns].iloc[:-1].reset_index(drop=True),
    )


def test_build_features_rejects_short_history(feature_config) -> None:
    """Too few complete rows is reported clearly instead of producing a bad model."""
    short = _synthetic_weather(days=60)
    with pytest.raises(InvalidInputError):
        weather_features.build_features(short, config=feature_config)


def test_unknown_target_mode_is_rejected(weather_frame) -> None:
    """An invalid target mode is reported rather than silently accepted."""
    config = weather_features.FeatureConfig(target_mode="forecast")
    with pytest.raises(InvalidInputError):
        weather_features.build_features(weather_frame, config=config)


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #


def test_train_bundle_selects_the_lowest_rmse(weather_frame, feature_config) -> None:
    """The selected model is the one with the best test RMSE."""
    feature_frame = weather_features.make_feature_frame(weather_frame, config=feature_config)
    bundle, comparison = weather_model.train_bundle(feature_frame, city="Testville", config=feature_config)

    rmses = {
        name: payload["metrics"]["rmse"]
        for name, payload in comparison.items()
        if name != "selected"
    }
    assert bundle.model_name == min(rmses, key=rmses.get)


def test_every_model_reports_a_persistence_baseline(weather_frame, feature_config) -> None:
    """Each candidate carries a persistence-baseline metric dictionary."""
    feature_frame = weather_features.make_feature_frame(weather_frame, config=feature_config)
    _bundle, comparison = weather_model.train_bundle(feature_frame, city="Testville", config=feature_config)

    for name, payload in comparison.items():
        if name == "selected":
            continue
        assert {"mae", "rmse", "r2", "mse"} <= set(payload["baseline"])


def test_feature_importances_cover_every_column(weather_frame, feature_config) -> None:
    """Feature importances line up one-to-one with the model inputs."""
    feature_frame = weather_features.make_feature_frame(weather_frame, config=feature_config)
    bundle, _ = weather_model.train_bundle(feature_frame, city="Testville", config=feature_config)
    assert set(bundle.feature_importances) == set(bundle.feature_columns)


def test_predict_next_returns_a_future_date(weather_frame, feature_config) -> None:
    """The next-day forecast is finite and dated after the observation it uses."""
    feature_frame = weather_features.make_feature_frame(weather_frame, config=feature_config)
    bundle, _ = weather_model.train_bundle(feature_frame, city="Testville", config=feature_config)

    forecast = weather_model.predict_next(bundle, feature_frame)
    assert np.isfinite(forecast["predicted"])
    assert forecast["date"] > forecast["based_on"]


def test_bundle_round_trips_through_disk(weather_frame, feature_config, tmp_path) -> None:
    """A saved bundle and report can be reloaded unchanged."""
    feature_frame = weather_features.make_feature_frame(weather_frame, config=feature_config)
    bundle, comparison = weather_model.train_bundle(feature_frame, city="Testville", config=feature_config)

    model_path = weather_model.save_bundle(bundle, tmp_path / "bundle.joblib")
    loaded = weather_model.load_bundle(model_path)
    assert loaded.city == "Testville"
    assert loaded.model_name == bundle.model_name
    assert loaded.feature_columns == bundle.feature_columns

    report = weather_model.build_training_report(
        bundle,
        comparison,
        dataset={"city": "Testville"},
        feature_summary={"usable_rows": len(feature_frame)},
    )
    metrics_path = weather_model.save_metrics_report(report, tmp_path / "metrics.json")
    assert weather_model.load_metrics_report(metrics_path)["city"] == "Testville"

