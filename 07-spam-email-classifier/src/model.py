"""Spam classification: TF-IDF plus three linear classifiers.

Three estimators are compared on the same stratified split:

* ``nb``     - Multinomial Naive Bayes. The required baseline. Extremely fast on
  sparse count data and a genuinely strong model here, because spam vocabulary is
  so distinctive that a simple bag-of-words model already separates it well.
* ``logreg`` - Logistic Regression. Produces calibrated probabilities, which the
  interface needs for a confidence display.
* ``svm``    - Linear Support Vector Classifier. Often the most accurate, but has
  no probability estimates without ``probability=True``, which roughly doubles
  training time on this dataset.

Selection uses **macro-F1** rather than accuracy. Only 13.4% of messages are spam,
so a classifier that answered "ham" every time would reach 86.6% accuracy while
catching no spam at all - exactly the wrong behaviour for a spam filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import classification_metrics, per_class_report
from shared.ml import RANDOM_SEED, load_joblib, save_joblib, save_json, set_global_seed
from shared.paths import models_dir
from shared.text import normalize_text

from src.data import CLASS_LABELS, HAM, POSITIVE_LABEL, SPAM

logger = logging.getLogger(__name__)

PROJECT_SLUG = "07-spam-email-classifier"

MODEL_LABELS: dict[str, str] = {
    "nb": "Multinomial Naive Bayes",
    "logreg": "Logistic Regression",
    "svm": "Linear SVM",
}

SELECTABLE_MODELS: tuple[str, ...] = ("nb", "logreg", "svm")

#: TF-IDF settings. Bigrams matter: "free entry" and "claim now" are far more
#: indicative than the individual words.
TFIDF_PARAMS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 30_000,
    "sublinear_tf": True,
}

MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "nb": {"alpha": 0.1},
    "logreg": {"C": 4.0, "max_iter": 1_000, "solver": "lbfgs", "random_state": RANDOM_SEED},
    "svm": {"C": 1.0, "max_iter": 5_000, "random_state": RANDOM_SEED},
}


def clean_for_model(text: str) -> str:
    """Preprocess a message for the vectoriser.

    Negations are preserved: "not spam" must not be reduced to "spam".
    """
    return str(normalize_text(text, keep_negations=True, strip_numbers=True))


@dataclass
class SpamResult:
    """The classification outcome for one message."""

    label: str
    confidence: float
    distribution: dict[str, float]
    confidence_kind: str
    is_spam: bool


@dataclass
class SpamModel:
    """A fitted pipeline plus the metadata the interface displays."""

    pipeline: Any
    model_name: str
    labels: list[str] = field(default_factory=lambda: list(CLASS_LABELS))
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    top_features: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    trained_at: str = ""

    @property
    def supports_probabilities(self) -> bool:
        """Whether the estimator can output class probabilities."""
        return hasattr(self.pipeline.named_steps["clf"], "predict_proba")

    def predict(self, message: str) -> SpamResult:
        """Classify one message and return its label and confidence."""
        if message is None or not str(message).strip():
            raise InvalidInputError(
                "The message is empty.",
                hint="Paste a message to classify.",
            )

        cleaned = clean_for_model(message)
        if not cleaned:
            raise InvalidInputError(
                "That message contained no usable words.",
                hint="It was only punctuation, numbers or links. Please enter some text.",
            )

        label = str(self.pipeline.predict([cleaned])[0])
        classes = [str(name) for name in self.pipeline.classes_]

        if self.supports_probabilities:
            probabilities = np.asarray(self.pipeline.predict_proba([cleaned])[0], dtype=float)
            distribution = {name: float(value) for name, value in zip(classes, probabilities)}
            confidence = float(probabilities.max())
            kind = "calibrated probability"
        else:
            scores = np.asarray(self.pipeline.decision_function([cleaned]), dtype=float).ravel()
            shifted = scores - np.max(scores)
            weights = np.exp(shifted) / np.exp(shifted).sum()
            distribution = {name: float(value) for name, value in zip(classes, weights)}
            confidence = float(weights.max())
            kind = "normalised decision score (not calibrated)"

        return SpamResult(
            label=label,
            confidence=confidence,
            distribution=distribution,
            confidence_kind=kind,
            is_spam=label == SPAM,
        )

    def explain(self, message: str, limit: int = 8) -> list[dict[str, Any]]:
        """Return the tokens in this message that most drove the decision.

        For a linear model a token's contribution is ``coefficient *
        tfidf_value``. Summing those over the whole message shows which words
        actually pushed this particular classification, which is more useful
        than a global word list.
        """
        classifier = self.pipeline.named_steps["clf"]
        vectorizer = self.pipeline.named_steps["tfidf"]
        if not hasattr(classifier, "coef_"):
            return []

        cleaned = clean_for_model(message)
        if not cleaned:
            return []

        row = vectorizer.transform([cleaned])
        if not hasattr(row, "tocsr") or row.nnz == 0:
            return []

        features = np.asarray(row.todense()).ravel()
        coefficients = np.asarray(classifier.coef_, dtype=float).ravel()
        contributions = features * coefficients
        vocabulary = vectorizer.get_feature_names_out()

        nonzero = np.argsort(np.abs(contributions))[::-1]
        results: list[dict[str, Any]] = []
        for index in nonzero:
            if features[index] == 0.0:
                continue
            results.append(
                {
                    "token": str(vocabulary[index]),
                    "contribution": float(contributions[index]),
                    "direction": (
                        "pushes towards SPAM" if contributions[index] > 0 else "pushes towards HAM"
                    ),
                }
            )
            if len(results) >= limit:
                break
        return results


def build_pipeline(name: str) -> Any:
    """Return an unfitted TF-IDF plus classifier pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    if name == "nb":
        classifier: Any = MultinomialNB(**MODEL_PARAMS["nb"])
    elif name == "logreg":
        classifier = LogisticRegression(**MODEL_PARAMS["logreg"])
    elif name == "svm":
        classifier = LinearSVC(**MODEL_PARAMS["svm"])
    else:
        raise InvalidInputError(
            f"Unknown classifier '{name}'.",
            hint=f"Choose one of: {', '.join(SELECTABLE_MODELS)}.",
        )

    return Pipeline(steps=[("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("clf", classifier)])


def _extract_top_features(pipeline: Any, top_n: int = 10) -> dict[str, list[dict[str, float]]]:
    """Return the most indicative n-grams for spam and for ham."""
    classifier = pipeline.named_steps["clf"]
    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    classes = [str(name) for name in pipeline.classes_]

    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=float)
        if coefficients.shape[0] == 1:
            # Binary: one row points towards classes_[1], which is 'spam'.
            row = coefficients[0]
            spam_top = [int(i) for i in np.argsort(row)[::-1][:top_n]]
            ham_top = [int(i) for i in np.argsort(row)[:top_n]]
            return {
                SPAM: [{"token": str(vocabulary[i]), "weight": float(row[i])} for i in spam_top],
                HAM: [{"token": str(vocabulary[i]), "weight": float(-row[i])} for i in ham_top],
            }
        rows = coefficients
    elif hasattr(classifier, "feature_log_prob_"):
        log_probabilities = np.asarray(classifier.feature_log_prob_, dtype=float)
        # Centring per feature gives "how much more likely in this class than
        # on average", which is the readable analogue of a coefficient.
        rows = log_probabilities - log_probabilities.mean(axis=0, keepdims=True)
    else:
        return {}

    result: dict[str, list[dict[str, float]]] = {}
    for index, class_name in enumerate(classes):
        if index >= rows.shape[0]:
            continue
        row = rows[index]
        order = np.argsort(row)[::-1][:top_n]
        result[class_name] = [{"token": str(vocabulary[i]), "weight": float(row[i])} for i in order]
    return result


def train_model(frame: pd.DataFrame, model_name: str = "nb", *, test_size: float = 0.2) -> SpamModel:
    """Train and evaluate one classifier on a stratified split.

    Raises:
        InvalidInputError: When the frame lacks columns or is too small.
    """
    set_global_seed()
    if frame is None or frame.empty:
        raise InvalidInputError("No training data supplied.", hint="Load the dataset first.")

    missing = [column for column in ("message", "label") if column not in frame.columns]
    if missing:
        raise InvalidInputError(
            f"Training data is missing column(s): {', '.join(missing)}.",
            hint="Reload with src.data.load_dataset().",
        )

    data = frame.dropna(subset=["message", "label"]).reset_index(drop=True)
    if len(data) < 50:
        raise InvalidInputError(
            f"Only {len(data)} labelled messages available.",
            hint="At least 50 messages are needed to train and evaluate.",
        )

    counts = data["label"].value_counts()
    if (counts < 2).any():
        raise InvalidInputError(
            "A class has fewer than two examples.",
            hint=f"Class counts: {counts.to_dict()}.",
        )

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        data["message"].tolist(),
        data["label"].tolist(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=data["label"].tolist(),
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit([clean_for_model(text) for text in x_train], y_train)

    predictions = pipeline.predict([clean_for_model(text) for text in x_test])
    probabilities = (
        pipeline.predict_proba([clean_for_model(text) for text in x_test])
        if hasattr(pipeline.named_steps["clf"], "predict_proba")
        else None
    )

    metrics = classification_metrics(y_test, predictions, labels=CLASS_LABELS, y_proba=probabilities)
    metrics["spam_recall"] = metrics["recall"]
    metrics["spam_precision"] = metrics["precision"]
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)

    return SpamModel(
        pipeline=pipeline,
        model_name=model_name,
        metrics=metrics,
        per_class=per_class_report(y_test, predictions, labels=CLASS_LABELS),
        n_train=len(x_train),
        n_test=len(x_test),
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        top_features=_extract_top_features(pipeline),
    )


def train_all_models(frame: pd.DataFrame, *, test_size: float = 0.2) -> dict[str, dict[str, Any]]:
    """Train every candidate classifier on the same split."""
    comparison: dict[str, dict[str, Any]] = {}
    for name in SELECTABLE_MODELS:
        model = train_model(frame, name, test_size=test_size)
        comparison[name] = {
            "label": MODEL_LABELS[name],
            "metrics": model.metrics,
            "per_class": model.per_class,
            "hyperparameters": MODEL_PARAMS[name],
        }
    return comparison


def select_best_model(comparison: dict[str, dict[str, Any]]) -> str:
    """Return the classifier with the highest macro-F1.

    Accuracy would be actively misleading here: with 13.4% spam, always answering
    "ham" scores 0.866 accuracy and is useless as a filter.
    """
    return max(comparison, key=lambda name: comparison[name]["metrics"]["f1"])


def save_model(model: SpamModel, *, directory: Path | None = None) -> Path:
    """Persist the fitted pipeline to ``models/spam_model.joblib``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "spam_model.joblib"
    save_joblib(model, path)
    return path


def load_model(path: Path | None = None) -> SpamModel:
    """Load a previously trained :class:`SpamModel`."""
    target = Path(path) if path else models_dir(PROJECT_SLUG) / "spam_model.joblib"
    loaded = load_joblib(target, description="spam model", train_command="python train.py")
    if not isinstance(loaded, SpamModel):
        raise ModelNotFoundError(
            f"'{target.name}' does not contain a SpamModel.",
            hint="Delete the file and re-run `python train.py`.",
        )
    return loaded


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    model: SpamModel,
    comparison: dict[str, dict[str, Any]],
    *,
    dataset: dict[str, Any],
    selected: str,
) -> dict[str, Any]:
    """Assemble the JSON report written next to the model."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": model.trained_at,
        "selected_model": selected,
        "selected_label": MODEL_LABELS.get(selected, selected),
        "selection_rule": "highest macro-F1 on the stratified test split",
        "class_labels": CLASS_LABELS,
        "positive_label": POSITIVE_LABEL,
        "tfidf": {key: str(value) for key, value in TFIDF_PARAMS.items()},
        "hyperparameters": {name: params for name, params in MODEL_PARAMS.items()},
        "dataset": dataset,
        "comparison": comparison,
        "top_features": model.top_features,
        "random_seed": RANDOM_SEED,
        "disclaimer": (
            "Trained on SMS messages, not email. Spam vocabulary and structure transfer, "
            "but the reported accuracy is an SMS figure and should not be quoted as an "
            "email result. This model is a teaching baseline, not a production filter."
        ),
    }