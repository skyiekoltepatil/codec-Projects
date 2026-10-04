"""Sentiment classification: TF-IDF vectorisation plus a linear classifier.

Three classifiers are compared on the same stratified split:

* ``logreg`` - Logistic Regression. The required baseline, and the only one of
  the three that provides calibrated class probabilities out of the box, which
  the interface needs for a confidence display.
* ``nb``     - Multinomial Naive Bayes. Very fast, strong on sparse count data,
  but its probabilities are poorly calibrated.
* ``svm``    - Linear Support Vector Classifier. Often the most accurate of the
  three, but it has no probability estimates unless ``probability=True`` is set,
  which costs a lot of training time; it is therefore reported but not used for
  the confidence display.

Text cleaning lives in ``shared.text`` so that the sentiment, chatbot and spam
projects cannot drift apart in how they preprocess input.
"""

from __future__ import annotations

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

from src.data import CLASS_LABELS

PROJECT_SLUG = "02-twitter-sentiment-analysis"

#: Human-readable estimator names.
MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic Regression",
    "nb": "Multinomial Naive Bayes",
    "svm": "Linear SVM",
}

#: Estimators that can be fitted.
SELECTABLE_MODELS: tuple[str, ...] = ("logreg", "nb", "svm")

#: TF-IDF settings. ``sublinear_tf`` dampens the effect of repeated words, and
#: the n-gram range captures negated phrases such as "not good".
TFIDF_PARAMS: dict[str, Any] = {
    "max_features": 30_000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}

#: Classifier hyperparameters, documented in the README.
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    # 'lbfgs' is used because it handles the three-class multinomial problem
    # natively. The 'liblinear' solver, used in many tutorials, raises
    # ValueError for n_classes >= 3 in current scikit-learn.
    "logreg": {"C": 4.0, "max_iter": 1_000, "solver": "lbfgs", "random_state": RANDOM_SEED},
    "nb": {"alpha": 0.3},
    "svm": {"C": 1.0, "max_iter": 5_000, "random_state": RANDOM_SEED},
}


def clean_for_model(text: str) -> str:
    """Apply the project's text preprocessing.

    Stop words are removed but negation cues are preserved, because "not good"
    and "good" must not collapse to the same feature set.
    """
    return str(normalize_text(text, keep_negations=True, strip_numbers=True))


def build_estimator(name: str) -> Any:
    """Return a fresh classifier by key.

    Raises:
        InvalidInputError: For an unknown key.
    """
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**MODEL_PARAMS["logreg"])
    if name == "nb":
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(**MODEL_PARAMS["nb"])
    if name == "svm":
        from sklearn.svm import LinearSVC

        return LinearSVC(**MODEL_PARAMS["svm"])
    raise InvalidInputError(
        f"Unknown classifier '{name}'.",
        hint=f"Choose one of: {', '.join(SELECTABLE_MODELS)}.",
    )


def build_pipeline(name: str) -> Any:
    """Return an ``(TfidfVectorizer -> classifier)`` scikit-learn pipeline.

    Wrapping both steps in a single pipeline is what prevents vectoriser leakage:
    the vocabulary and IDF weights are fitted on the training split only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", build_estimator(name)),
        ]
    )


@dataclass
class SentimentModel:
    """A fitted pipeline plus the metadata the interface needs."""

    pipeline: Any
    model_name: str
    labels: list[str] = field(default_factory=lambda: list(CLASS_LABELS))
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    trained_at: str = ""
    top_features: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    @property
    def supports_probabilities(self) -> bool:
        """Whether the wrapped classifier can produce class probabilities."""
        return hasattr(self.pipeline.named_steps["clf"], "predict_proba")

    def predict(self, texts: list[str]) -> list[str]:
        """Predict the sentiment label for each input text."""
        cleaned = [clean_for_model(text) for text in texts]
        return [str(label) for label in self.pipeline.predict(cleaned)]

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        """Return class probabilities, or ``None`` when unsupported."""
        if not self.supports_probabilities:
            return None
        cleaned = [clean_for_model(text) for text in texts]
        return np.asarray(self.pipeline.predict_proba(cleaned), dtype=float)

    def predict_with_confidence(self, text: str) -> dict[str, Any]:
        """Return the label, confidence and full class distribution for one text.

        For classifiers without probability support the decision function is
        converted to a softmax over the class scores, and the result is labelled
        as a score rather than a calibrated probability.
        """
        cleaned = clean_for_model(text)
        label = str(self.pipeline.predict([cleaned])[0])

        probabilities = self.predict_proba([text])
        if probabilities is not None:
            distribution = {
                str(name): float(value)
                for name, value in zip(self.pipeline.classes_, probabilities[0], strict=False)
            }
            return {
                "label": label,
                "confidence": float(max(probabilities[0])),
                "distribution": distribution,
                "confidence_kind": "probability",
            }

        scores = np.asarray(self.pipeline.decision_function([cleaned]), dtype=float).ravel()
        # Softmax over decision scores gives a comparable, bounded scale.
        shifted = scores - np.max(scores)
        weights = np.exp(shifted) / np.exp(shifted).sum()
        distribution = {
            str(name): float(value)
            for name, value in zip(self.pipeline.classes_, weights, strict=False)
        }
        return {
            "label": label,
            "confidence": float(max(weights)),
            "distribution": distribution,
            "confidence_kind": "normalised decision score (not calibrated)",
        }


def _extract_top_features(pipeline: Any, labels: list[str], top_n: int = 10) -> dict[str, list[dict[str, float]]]:
    """Return the most influential n-grams per class, when the model exposes them.

    For a linear model the weight of a feature for a class is its coefficient.
    Naive Bayes exposes ``feature_log_prob_``, whose per-class difference from
    the mean is the natural analogue.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    vocabulary = vectorizer.get_feature_names_out()

    result: dict[str, list[dict[str, float]]] = {}

    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=float)
        classes = list(pipeline.classes_)
        if coefficients.shape[0] == 1:
            # Binary case: a single coefficient row. The mirrored negative row is
            # used so both classes get an interpretable ranking.
            coefficients = np.vstack([-coefficients[0], coefficients[0]])
            classes = list(pipeline.classes_)
        for index, class_name in enumerate(classes):
            if index >= coefficients.shape[0]:
                continue
            row = coefficients[index]
            order = np.argsort(row)[::-1][:top_n]
            result[str(class_name)] = [
                {"feature": str(vocabulary[i]), "weight": float(row[i])} for i in order
            ]

    elif hasattr(classifier, "feature_log_prob_"):
        log_probabilities = np.asarray(classifier.feature_log_prob_, dtype=float)
        # How much more likely a feature is for this class than for the average class.
        centred = log_probabilities - log_probabilities.mean(axis=0, keepdims=True)
        for index, class_name in enumerate(pipeline.classes_):
            row = centred[index]
            order = np.argsort(row)[::-1][:top_n]
            result[str(class_name)] = [
                {"feature": str(vocabulary[i]), "weight": float(row[i])} for i in order
            ]

    return result


def train_model(
    frame: pd.DataFrame,
    model_name: str = "logreg",
    *,
    test_size: float = 0.2,
    max_features: int | None = None,
) -> SentimentModel:
    """Train and evaluate one classifier on a stratified split.

    Args:
        frame: DataFrame with ``text`` and ``label`` columns.
        model_name: One of :data:`SELECTABLE_MODELS`.
        test_size: Share of rows held out for evaluation.
        max_features: Optional override for the TF-IDF vocabulary size.

    Returns:
        A fitted :class:`SentimentModel`.

    Raises:
        InvalidInputError: When the frame lacks required columns, is too small,
            or a class has too few examples to split.
    """
    set_global_seed()
    if frame is None or frame.empty:
        raise InvalidInputError("No training data supplied.", hint="Load the dataset first.")
    missing = [column for column in ("text", "label") if column not in frame.columns]
    if missing:
        raise InvalidInputError(
            f"Training data is missing column(s): {', '.join(missing)}.",
            hint="Reload the dataset with src.data.load_dataset().",
        )

    data = frame.dropna(subset=["text", "label"]).reset_index(drop=True)
    if len(data) < 50:
        raise InvalidInputError(
            f"Only {len(data)} labelled rows available.",
            hint="At least 50 rows are required to train and evaluate a classifier.",
        )

    counts = data["label"].value_counts()
    if (counts < 2).any():
        raise InvalidInputError(
            "At least one sentiment class has fewer than two examples.",
            hint=f"Class counts: {counts.to_dict()}. Lower min_confidence when loading the data.",
        )

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=data["label"].tolist(),
    )

    pipeline = build_pipeline(model_name)
    if max_features is not None:
        pipeline.set_params(tfidf__max_features=int(max_features))

    # Clean text once, up front, so the vectoriser sees exactly what the app
    # will send it at inference time.
    pipeline.fit([clean_for_model(text) for text in x_train], y_train)

    predictions = pipeline.predict([clean_for_model(text) for text in x_test])
    probabilities = (
        pipeline.predict_proba([clean_for_model(text) for text in x_test])
        if hasattr(pipeline.named_steps["clf"], "predict_proba")
        else None
    )

    metrics = classification_metrics(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        y_proba=probabilities,
    )
    metrics["macro_f1"] = metrics["f1"]
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)

    return SentimentModel(
        pipeline=pipeline,
        model_name=model_name,
        labels=list(CLASS_LABELS),
        metrics=metrics,
        per_class=per_class_report(y_test, predictions, labels=CLASS_LABELS),
        n_train=len(x_train),
        n_test=len(x_test),
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        top_features=_extract_top_features(pipeline, list(CLASS_LABELS)),
    )


def train_all_models(frame: pd.DataFrame, *, test_size: float = 0.2) -> dict[str, dict[str, Any]]:
    """Train every candidate classifier and return their metrics keyed by name."""
    comparison: dict[str, dict[str, Any]] = {}
    for name in SELECTABLE_MODELS:
        try:
            model = train_model(frame, name, test_size=test_size)
        except InvalidInputError:
            continue
        comparison[name] = {
            "label": MODEL_LABELS[name],
            "metrics": model.metrics,
            "per_class": model.per_class,
            "hyperparameters": MODEL_PARAMS[name],
        }
    if not comparison:
        raise InvalidInputError(
            "No classifier could be trained.",
            hint="Check the dataset: it may be too small or have a class with too few rows.",
        )
    return comparison


def select_best_model(comparison: dict[str, dict[str, Any]]) -> str:
    """Return the classifier with the highest macro-F1 on the test split.

    Macro-F1 is used rather than accuracy because the airline dataset is
    imbalanced towards negative tweets; macro-F1 weights all three classes
    equally, so a model cannot win by ignoring the neutral class.
    """
    return max(comparison, key=lambda name: comparison[name]["metrics"]["f1"])


def save_model(model: SentimentModel, *, directory: Path | None = None) -> Path:
    """Persist the fitted model with joblib and return the artefact path."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "sentiment_model.joblib"
    save_joblib(model, path)
    return path


def load_model(path: Path | None = None) -> SentimentModel:
    """Load a previously saved :class:`SentimentModel`."""
    target = Path(path) if path else models_dir(PROJECT_SLUG) / "sentiment_model.joblib"
    loaded = load_joblib(
        target,
        description="sentiment model",
        train_command="python train.py",
    )
    if not isinstance(loaded, SentimentModel):
        raise ModelNotFoundError(
            f"'{target.name}' does not contain a SentimentModel.",
            hint="Delete the file and re-run `python train.py`.",
        )
    return loaded


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    model: SentimentModel,
    comparison: dict[str, dict[str, Any]],
    *,
    dataset: dict[str, Any],
    selected: str,
) -> dict[str, Any]:
    """Assemble the JSON report written to ``models/training_metrics.json``."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": model.trained_at,
        "selected_model": selected,
        "selected_label": MODEL_LABELS.get(selected, selected),
        "selection_rule": "highest macro-F1 on the stratified test split",
        "class_labels": CLASS_LABELS,
        "tfidf": {key: str(value) for key, value in TFIDF_PARAMS.items()},
        "dataset": dataset,
        "comparison": comparison,
        "top_features": model.top_features,
        "random_seed": RANDOM_SEED,
        "disclaimer": (
            "Trained on tweets about US airlines. Sentiment vocabulary is domain-specific, "
            "so accuracy on other domains will be lower than the reported figures."
        ),
    }