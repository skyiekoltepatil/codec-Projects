"""Tests for project 02 - Twitter/X Sentiment Analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_env import use_project

PROJECT_DIR = use_project("02-twitter-sentiment-analysis")

from shared.errors import InvalidInputError, ModelNotFoundError  # noqa: E402

from src import data as sentiment_data  # noqa: E402
from src import model as sentiment_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: A small, hand-written three-class corpus so tests never depend on the network.
SYNTHETIC_TWEETS: list[tuple[str, str]] = [
    ("I absolutely love this airline, the crew was fantastic", "Positive"),
    ("Best flight experience I have ever had, highly recommend", "Positive"),
    ("Thank you so much, the staff were wonderful and kind", "Positive"),
    ("Great service and a very comfortable seat, will fly again", "Positive"),
    ("Amazing customer support, they resolved everything quickly", "Positive"),
    ("The flight was perfect and arrived exactly on time", "Positive"),
    ("Fantastic value for money, I am very happy with it", "Positive"),
    ("Really enjoyed the trip, everything went smoothly", "Positive"),
    ("Terrible experience, my bag was lost and nobody helped", "Negative"),
    ("Worst airline ever, the flight was delayed for hours", "Negative"),
    ("Awful service and rude staff, I am never flying again", "Negative"),
    ("They cancelled my flight and gave no explanation at all", "Negative"),
    ("Horrible, uncomfortable seats and cold food", "Negative"),
    ("I am extremely disappointed with this airline", "Negative"),
    ("Poor communication and a complete waste of money", "Negative"),
    ("The staff were unhelpful and the plane was dirty", "Negative"),
    ("The flight departed at six in the evening from gate twelve", "Neutral"),
    ("My reservation number is four five six seven", "Neutral"),
    ("I am travelling to Chicago next Tuesday morning", "Neutral"),
    ("Please confirm the baggage allowance for this route", "Neutral"),
    ("The aircraft is a Boeing seven three seven", "Neutral"),
    ("I would like to change the date of my booking", "Neutral"),
    ("What time does the check in desk open", "Neutral"),
    ("The layover is in Denver for two hours", "Neutral"),
]


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    """The synthetic three-class corpus as a DataFrame."""
    frame = pd.DataFrame(SYNTHETIC_TWEETS, columns=["text", "label"])
    # Duplicate rows to give the vectoriser enough vocabulary occurrences for
    # min_df=2 to retain meaningful features.
    return pd.concat([frame] * 6, ignore_index=True)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_cleaning_removes_urls_and_handles() -> None:
    """URLs, handles and punctuation are removed by the shared cleaner."""
    cleaned = sentiment_model.clean_for_model("Check https://example.com @airline #sale NOW!!!")
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "!" not in cleaned
    assert "sale" in cleaned


def test_cleaning_preserves_negation() -> None:
    """Negation words survive preprocessing, because they carry the signal."""
    cleaned = sentiment_model.clean_for_model("I would not say this was good")
    assert "not" in cleaned.split()
    assert "good" in cleaned.split()


def test_cleaning_is_idempotent() -> None:
    """Cleaning already-clean text does not change it."""
    once = sentiment_model.clean_for_model("the flight was not good")
    twice = sentiment_model.clean_for_model(once)
    assert once == twice


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["logreg", "nb", "svm"])
def test_every_classifier_trains(synthetic_frame: pd.DataFrame, name: str) -> None:
    """All three classifiers fit the three-class problem and report metrics."""
    model = sentiment_model.train_model(synthetic_frame, name)
    assert model.metrics["accuracy"] > 0.0
    assert 0.0 <= model.metrics["f1"] <= 1.0
    assert len(model.metrics["confusion_matrix"]) == 3
    assert len(model.per_class) == 3


def test_logreg_supports_probabilities(synthetic_frame: pd.DataFrame) -> None:
    """Logistic Regression exposes calibrated probabilities summing to one."""
    model = sentiment_model.train_model(synthetic_frame, "logreg")
    assert model.supports_probabilities
    probabilities = model.predict_proba(["I love this airline"])
    assert probabilities is not None
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


def test_svm_falls_back_to_decision_scores(synthetic_frame: pd.DataFrame) -> None:
    """The SVM has no probabilities, so the app is told to use decision scores."""
    model = sentiment_model.train_model(synthetic_frame, "svm")
    assert not model.supports_probabilities
    assert model.predict_proba(["anything"]) is None
    result = model.predict_with_confidence("I love this airline")
    assert result["confidence_kind"].startswith("normalised decision score")


def test_predicts_obvious_sentiment(synthetic_frame: pd.DataFrame) -> None:
    """The classifier gets clearly polarised examples right."""
    model = sentiment_model.train_model(synthetic_frame, "logreg")
    assert model.predict(["I absolutely love this airline"])[0] == "Positive"
    assert model.predict(["This was a terrible and awful experience"])[0] == "Negative"


def test_confidence_distribution_covers_all_classes(synthetic_frame: pd.DataFrame) -> None:
    """The returned distribution contains every class label."""
    model = sentiment_model.train_model(synthetic_frame, "logreg")
    result = model.predict_with_confidence("great service")
    assert set(result["distribution"]) == set(sentiment_data.CLASS_LABELS)
    assert 0.0 <= result["confidence"] <= 1.0


def test_train_all_models_returns_comparison(synthetic_frame: pd.DataFrame) -> None:
    """Training every candidate yields one comparison entry per classifier."""
    comparison = sentiment_model.train_all_models(synthetic_frame)
    assert set(comparison) == set(sentiment_model.SELECTABLE_MODELS)
    best = sentiment_model.select_best_model(comparison)
    assert best in sentiment_model.SELECTABLE_MODELS


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_rejects_empty_frame() -> None:
    """An empty training frame is rejected."""
    with pytest.raises(InvalidInputError):
        sentiment_model.train_model(pd.DataFrame(columns=["text", "label"]), "logreg")


def test_rejects_missing_columns(synthetic_frame: pd.DataFrame) -> None:
    """A frame without the label column is rejected."""
    with pytest.raises(InvalidInputError):
        sentiment_model.train_model(synthetic_frame[["text"]], "logreg")


def test_rejects_tiny_frame() -> None:
    """Too few rows is reported clearly."""
    tiny = pd.DataFrame({"text": ["a", "b", "c"], "label": ["Positive", "Negative", "Neutral"]})
    with pytest.raises(InvalidInputError):
        sentiment_model.train_model(tiny, "logreg")


def test_rejects_unknown_classifier(synthetic_frame: pd.DataFrame) -> None:
    """An unknown classifier key is rejected."""
    with pytest.raises(InvalidInputError):
        sentiment_model.train_model(synthetic_frame, "not_a_classifier")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_and_load_roundtrip(synthetic_frame: pd.DataFrame, tmp_path: Path) -> None:
    """A saved model reloads and predicts identically."""
    model = sentiment_model.train_model(synthetic_frame, "logreg")
    path = sentiment_model.save_model(model, directory=tmp_path)
    reloaded = sentiment_model.load_model(path)
    assert reloaded.model_name == model.model_name
    assert reloaded.predict(["great flight"]) == model.predict(["great flight"])


def test_load_missing_model_is_actionable(tmp_path: Path) -> None:
    """A missing artefact names the training command in its hint."""
    with pytest.raises(ModelNotFoundError) as excinfo:
        sentiment_model.load_model(tmp_path / "absent.joblib")
    assert "train.py" in (excinfo.value.hint or "")


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_real_dataset_loads_with_three_classes() -> None:
    """The downloaded corpus has all three classes and the documented columns."""
    frame = sentiment_data.load_dataset()
    assert set(frame.columns) == {"text", "label"}
    assert set(frame["label"].unique()) == set(sentiment_data.CLASS_LABELS)
    assert len(frame) > 5_000


@pytest.mark.network
def test_app_renders_and_predicts() -> None:
    """The Streamlit app renders, and the analyse button produces a prediction."""
    from streamlit.testing.v1 import AppTest

    if not (PROJECT_DIR / "models" / "sentiment_model.joblib").exists():
        pytest.skip("Run `python train.py` in 02-twitter-sentiment-analysis first.")

    app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception, [str(exception.value) for exception in app.exception]

    for button in list(app.button):
        if button.label.startswith("Analyse"):
            button.click().run()
            break
    assert not app.exception, [str(exception.value) for exception in app.exception]
    assert not app.error, [error.value for error in app.error]
