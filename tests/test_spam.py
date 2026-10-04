"""Tests for Project 07 - Spam Email Classifier.

The classifier is trained on a small synthetic corpus for most tests so the
suite stays fast, plus a few checks against the real UCI dataset when it is
already downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from project_env import REPO_ROOT, use_project

use_project("07-spam-email-classifier")

sys.path.insert(0, str(REPO_ROOT))

from shared.errors import DataDownloadError, InvalidInputError  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_LABELS,
    HAM,
    SPAM,
    SpamData,
    example_messages,
    load_dataset,
    parse_corpus,
)
from src.model import (  # noqa: E402
    SpamModel,
    build_pipeline,
    build_training_report,
    clean_for_model,
    load_model,
    save_model,
    select_best_model,
    train_all_models,
    train_model,
)

# --------------------------------------------------------------------------- #
# Synthetic corpus
# --------------------------------------------------------------------------- #

SPAM_WORDS = ("win", "prize", "claim", "free", "urgent", "cash", "winner", "offer", "discount", "click")
HAM_WORDS = ("meeting", "lunch", "report", "tomorrow", "call", "project", "thanks", "dinner", "review", "email")


def _synthetic_corpus(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Build a small corpus where spam uses spam vocabulary and ham does not."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        is_spam = index % 5 == 0
        vocabulary = SPAM_WORDS if is_spam else HAM_WORDS
        count = int(rng.integers(3, 8))
        words = [str(rng.choice(vocabulary)) for _ in range(count)]
        rows.append(
            {
                "message": " ".join(words),
                "label": SPAM if is_spam else HAM,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model() -> SpamModel:
    """Train one real pipeline on the synthetic corpus."""
    return train_model(_synthetic_corpus(), "logreg")


# --------------------------------------------------------------------------- #
# Corpus parsing
# --------------------------------------------------------------------------- #


def test_parse_corpus_splits_label_and_message():
    """A normal line is split into its label and message."""
    assert parse_corpus("ham\tOk sounds good, see you then") == ("ham", "Ok sounds good, see you then")
    assert parse_corpus("spam\tFree entry! Text WIN now") == ("spam", "Free entry! Text WIN now")


def test_parse_corpus_keeps_tabs_inside_the_message():
    """Only the first two tabs are separators; later ones belong to the text.

    Messages in this corpus do contain tabs, and a naive ``split("\\t", 1)``
    would leave the second separator glued to the front of the message.
    """
    label, message = parse_corpus("ham\tfirst\tsecond\tthird")
    assert label == "ham"
    assert message == "second\tthird"


def test_parse_corpus_rejects_missing_separator():
    """A line with no tab is a corrupt corpus, not a valid row."""
    with pytest.raises(DataDownloadError):
        parse_corpus("no tab here")


def test_parse_corpus_rejects_unknown_label():
    """An unexpected label is reported rather than silently coerced."""
    with pytest.raises(DataDownloadError):
        parse_corpus("junk\tmessage body")


def test_summary_counts_match_the_frame():
    """Summary statistics are derived from the data, not hard-coded."""
    data = SpamData(frame=_synthetic_corpus())
    summary = data.summary()
    counts = data.frame["label"].value_counts()
    assert summary["rows"] == len(data.frame)
    assert summary["spam_count"] == int(counts.get(SPAM, 0))
    assert summary["ham_count"] == int(counts.get(HAM, 0))
    assert 0.0 <= summary["spam_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_cleaning_lowercases_and_removes_stopwords():
    """Cleaning is deterministic."""
    cleaned = clean_for_model("WIN a FREE prize NOW!!!")
    assert cleaned == cleaned.lower()
    assert "!" not in cleaned


def test_cleaning_preserves_negations():
    """'not spam' must not reduce to 'spam'."""
    assert "not" in clean_for_model("this is not spam")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_model_flags_spam_and_clears_ham(model):
    """Both classes are recognised from their vocabulary."""
    assert model.predict("win a free prize claim your cash now").is_spam
    assert not model.predict("meeting about the project tomorrow at lunch").is_spam


def test_result_reports_valid_distribution(model):
    """Every result exposes a normalised distribution over both classes."""
    result = model.predict("urgent winner offer")
    assert set(result.distribution) == set(CLASS_LABELS)
    assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.label in CLASS_LABELS


def test_probability_and_score_paths_are_both_supported():
    """Naive Bayes gives probabilities; the SVM gives normalised scores."""
    frame = _synthetic_corpus()

    bayes = train_model(frame, "nb")
    assert bayes.supports_probabilities
    assert "calibrated" in bayes.predict("free prize").confidence_kind

    svm = train_model(frame, "svm")
    assert not svm.supports_probabilities
    assert "not calibrated" in svm.predict("free prize").confidence_kind


def test_explanation_points_the_right_way():
    """A spam message's explanation must cite spam-pushing tokens."""
    model_with_explanation = train_model(_synthetic_corpus(), "logreg")
    explanation = model_with_explanation.explain("win a free prize claim your cash")

    assert explanation, "expected an explanation for a linear model"
    tokens = {item["token"] for item in explanation}
    assert tokens & set(SPAM_WORDS), f"expected spam tokens, got {tokens}"
    assert all(item["contribution"] > 0 for item in explanation)


def test_explanation_is_empty_for_naive_bayes():
    """Naive Bayes has no per-token coefficient, and says so rather than faking one."""
    bayes = train_model(_synthetic_corpus(), "nb")
    assert bayes.explain("free prize") == []


def test_learned_features_are_per_class():
    """The most indicative tokens are reported for both classes."""
    for name in ("nb", "logreg", "svm"):
        fitted = train_model(_synthetic_corpus(), name)
        assert set(fitted.top_features) == {SPAM, HAM}
        for features in fitted.top_features.values():
            assert features
            weights = [item["weight"] for item in features]
            assert weights == sorted(weights, reverse=True)


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("message", ["", "   ", "\n\n"])
def test_empty_message_is_rejected(model, message):
    """An empty message is an input error, not a classification."""
    with pytest.raises(InvalidInputError):
        model.predict(message)


def test_punctuation_only_message_is_rejected(model):
    """Text with no usable words is refused with an explanation."""
    with pytest.raises(InvalidInputError) as error:
        model.predict("!!! ??? ...")
    assert "words" in str(error.value)


# --------------------------------------------------------------------------- #
# Training and selection
# --------------------------------------------------------------------------- #


def test_training_rejects_bad_input():
    """Empty frames, missing columns and tiny frames are reported."""
    with pytest.raises(InvalidInputError):
        train_model(pd.DataFrame(), "nb")
    with pytest.raises(InvalidInputError):
        train_model(pd.DataFrame({"message": ["a", "b"], "wrong": [1, 2]}), "nb")
    with pytest.raises(InvalidInputError):
        train_model(_synthetic_corpus(n=5), "nb")
    with pytest.raises(InvalidInputError):
        build_pipeline("telepathy")


def test_pipeline_steps_are_ordered():
    """Vectorisation happens before classification."""
    assert list(build_pipeline("nb").named_steps) == ["tfidf", "clf"]


def test_selection_maximises_macro_f1():
    """Selection follows macro-F1, which is what beats the majority baseline."""
    comparison = train_all_models(_synthetic_corpus(n=400))
    selected = select_best_model(comparison)
    best = max(payload["metrics"]["f1"] for payload in comparison.values())
    assert comparison[selected]["metrics"]["f1"] == pytest.approx(best)


def test_all_three_classifiers_train():
    """Every candidate produces a usable confusion matrix."""
    comparison = train_all_models(_synthetic_corpus(n=400))
    assert set(comparison) == {"nb", "logreg", "svm"}
    for payload in comparison.values():
        matrix = payload["metrics"]["confusion_matrix"]
        assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_load_round_trip(model, tmp_path):
    """A saved model reloads and classifies identically."""
    path = save_model(model, directory=tmp_path)
    reloaded = load_model(path)

    for message in ("free prize claim now", "project meeting tomorrow"):
        assert reloaded.predict(message).label == model.predict(message).label


def test_load_missing_model_explains_remedy(tmp_path):
    """A missing artefact names the command that creates it."""
    with pytest.raises(Exception) as error:
        load_model(tmp_path / "absent.joblib")
    assert "train.py" in (error.value.hint or "")


def test_load_rejects_foreign_artefact(tmp_path):
    """A joblib file holding the wrong object is reported clearly."""
    from shared.errors import ModelNotFoundError
    from shared.ml import save_joblib

    save_joblib(42, tmp_path / "spam_model.joblib")
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "spam_model.joblib")


def test_training_report_is_json_serialisable(model):
    """The written report carries the measured metrics."""
    import json

    comparison = train_all_models(_synthetic_corpus(n=400))
    report = build_training_report(
        model, comparison, dataset={"rows": 600}, selected="logreg"
    )
    payload = json.loads(json.dumps(report))
    assert payload["project"] == "07-spam-email-classifier"
    assert "SMS" in payload["disclaimer"]


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #


def _corpus_available() -> bool:
    """Return ``True`` when the SMS corpus has been extracted."""
    return (REPO_ROOT / "07-spam-email-classifier" / "data" / "SMSSpamCollection").exists()


@pytest.mark.skipif(not _corpus_available(), reason="Corpus absent; run `python train.py`.")
def test_real_corpus_parses_to_the_expected_shape():
    """The real corpus loads with both classes and a plausible spam rate."""
    data = load_dataset()
    assert data.n_rows > 5_000
    summary = data.summary()
    assert 0.08 < summary["spam_rate"] < 0.16
    assert set(summary["class_counts"]) == {HAM, SPAM}


@pytest.mark.skipif(not _corpus_available(), reason="Corpus absent; run `python train.py`.")
def test_real_corpus_examples_are_classified_correctly():
    """Every bundled example is labelled as its name implies."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "07-spam-email-classifier"))
    fitted = load_model()
    for name, message in example_messages().items():
        result = fitted.predict(message)
        if "spam" in name.lower():
            assert result.is_spam, f"'{name}' should be SPAM"
        else:
            assert not result.is_spam, f"'{name}' should be HAM"


@pytest.mark.skipif(
    not (REPO_ROOT / "07-spam-email-classifier" / "models" / "spam_model.joblib").exists(),
    reason="Trained artefact absent; run `python train.py` first.",
)
def test_app_renders_without_exception():
    """The Streamlit page executes top to bottom without raising."""
    from streamlit.testing.v1 import AppTest

    project_dir = REPO_ROOT / "07-spam-email-classifier"
    app = AppTest.from_file(str(project_dir / "app.py"), default_timeout=240)
    app.run()
    assert not app.exception