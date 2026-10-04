"""Churn classification: preprocessing pipeline, three models and risk banding.

Preprocessing
-------------
Categorical features are one-hot encoded and numeric features standardised,
**inside a single scikit-learn :class:`~sklearn.pipeline.Pipeline`**. That is the
whole point of the design: the imputer statistics, the one-hot category lists and
the scaler's means are all fitted on the training split only. Fitting a
``ColumnTransformer`` once on the full dataset and then cross-validating would
leak the validation rows into the encoders, and on a dataset this small that is
enough to inflate accuracy by a noticeable margin.

Why these three models
----------------------
* **Logistic Regression** - the required interpretable baseline. Its coefficients
  are readable, and its predicted probability is genuinely calibrated, which
  matters because the interface shows a probability and a risk band.
* **Random Forest** - robust to nonlinearity and interactions, and supplies an
  impurity-based importance.
* **Gradient Boosting** - usually the strongest of the three on tabular data.

Model selection uses **ROC-AUC** rather than accuracy. The dataset is 26.5%
churn, so a model that predicts "nobody churns" scores 73.5% accuracy and is
worthless. ROC-AUC measures ranking across every threshold, which is what a risk
score needs.
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

from src.data import CATEGORICAL_COLUMNS, COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

PROJECT_SLUG = "05-customer-churn-prediction"

#: The negative class, listed first so every label pair is ordered consistently.
NEGATIVE_LABEL = "No"
POSITIVE_LABEL = "Yes"

#: Fixed label order passed to every metric call, so the confusion matrix is
#: always [retained, churned] regardless of which class the data happens to start with.
CLASS_LABELS: list[str] = [NEGATIVE_LABEL, POSITIVE_LABEL]

#: Risk bands applied to the predicted churn probability.
RISK_BANDS: list[tuple[float, str]] = [
    (0.40, "HIGH"),
    (0.15, "MEDIUM"),
    (0.00, "LOW"),
]

MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

SELECTABLE_MODELS: tuple[str, ...] = ("logreg", "rf", "gb")

#: Hyperparameters, mirrored in the README.
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 2_000, "solver": "lbfgs", "random_state": RANDOM_SEED},
    "rf": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20, "n_jobs": -1,
           "random_state": RANDOM_SEED},
    "gb": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": RANDOM_SEED},
}


def risk_band(probability: float) -> str:
    """Map a churn probability onto LOW / MEDIUM / HIGH.

    The thresholds are fixed project policy rather than derived from the data, so
    the band a user sees does not silently move when the model is retrained.
    """
    for threshold, label in RISK_BANDS:
        if probability >= threshold:
            return label
    return "LOW"


def feature_columns() -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` feature lists present in the data."""
    numeric = [column for column in NUMERIC_COLUMNS]
    categorical = [column for column in CATEGORICAL_COLUMNS if column != "SeniorCitizen"]
    return numeric, categorical


def build_estimator(name: str) -> Any:
    """Return a fresh unfitted classifier by key."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**MODEL_PARAMS["logreg"])
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**MODEL_PARAMS["rf"])
    if name == "gb":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(**MODEL_PARAMS["gb"])
    raise InvalidInputError(
        f"Unknown classifier '{name}'.",
        hint=f"Choose one of: {', '.join(SELECTABLE_MODELS)}.",
    )


def build_pipeline(name: str) -> Any:
    """Return an unfitted preprocessing-plus-classifier pipeline.

    Numeric columns get median imputation (robust to the outlier-ish spend
    distribution) then standardisation, which the distance-sensitive Logistic
    Regression needs. Categorical columns get most-frequent imputation then
    one-hot encoding; unknown categories at inference time are encoded as
    all-zero rather than raising.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = feature_columns()

    numeric_transform = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_transform = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" keeps a value the model never saw during
            # training from breaking inference.
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transform, numeric),
            ("categorical", categorical_transform, categorical),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", build_estimator(name)),
        ]
    )


@dataclass
class ChurnModel:
    """A fitted pipeline plus everything the interface displays."""

    pipeline: Any
    model_name: str
    labels: list[str] = field(default_factory=lambda: list(CLASS_LABELS))
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    trained_at: str = ""

    def churn_probability(self, frame: pd.DataFrame) -> float:
        """Return ``P(churn)`` for one customer row."""
        probabilities = self.pipeline.predict_proba(frame)
        classes = list(self.pipeline.classes_)
        if POSITIVE_LABEL not in classes:
            raise InvalidInputError(
                "The trained model has no positive churn class.",
                hint="Re-run `python train.py`.",
            )
        return float(probabilities[0][classes.index(POSITIVE_LABEL)])

    def predict_customer(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Return the probability, predicted label and risk band for one customer."""
        probability = self.churn_probability(frame)
        return {
            "probability": probability,
            "label": POSITIVE_LABEL if probability >= 0.5 else NEGATIVE_LABEL,
            "risk": risk_band(probability),
            "thresholds": [
                {"band": label, "threshold": threshold} for threshold, label in RISK_BANDS
            ],
        }

    def top_factors(self, frame: pd.DataFrame, limit: int = 6) -> list[dict[str, Any]]:
        """Return the features that most moved *this* customer's churn score.

        For a linear model the per-customer contribution of a feature is
        ``coefficient * standardised_value``: that is exactly the amount it added
        to the log-odds for this row. Tree ensembles have no equivalent closed
        form, so this returns an empty list for them and the interface falls back
        to the model's global importance.

        An earlier version used ``permutation_importance`` on the single row,
        which is meaningless: permuting the columns of a one-row matrix
        reconstructs the same row, so every importance came out at zero.
        """
        classifier = self.pipeline.named_steps["clf"]
        if not hasattr(classifier, "coef_"):
            return []

        names = self._transformed_feature_names()
        if not names:
            return []

        coefficients = np.asarray(classifier.coef_, dtype=float)
        classes = list(self.pipeline.classes_)
        if coefficients.shape[0] == 1:
            # Binary layout: scikit-learn stores a single coefficient row whose
            # positive direction points at classes_[1] (the churn class).
            if len(classes) > 1 and classes[1] != POSITIVE_LABEL:
                raise InvalidInputError(
                    f"Expected '{POSITIVE_LABEL}' to be the positive class, found '{classes[1]}'.",
                    hint="Re-run `python train.py`.",
                )
        elif POSITIVE_LABEL in classes:
            # Multi-class layout: take the row belonging to the churn class.
            coefficients = coefficients[classes.index(POSITIVE_LABEL)]
        else:
            coefficients = coefficients[0]

        transformed = self.pipeline.named_steps["preprocess"].transform(frame)
        values = (
            np.asarray(transformed.todense()).ravel()
            if hasattr(transformed, "todense")
            else np.asarray(transformed).ravel()
        )
        # Keep both operands 1-D: the binary branch leaves ``coef_`` shaped
        # (1, n_features), and broadcasting that against a flat vector would
        # silently produce a 2-D contribution array.
        contributions = coefficients.ravel() * values

        # Report by magnitude: both a high risk-raise and a strong risk-reduction
        # matter when explaining a score.
        order = np.argsort(np.abs(contributions))[::-1][:limit]
        row = frame.iloc[0]
        return [
            {
                "feature": names[int(index)],
                "source_column": _raw_feature_for(names[int(index)]),
                "value": str(row.get(_raw_feature_for(names[int(index)]), "-")),
                "contribution": float(contributions[int(index)]),
                "direction": (
                    "increases churn risk"
                    if contributions[int(index)] >= 0
                    else "reduces churn risk"
                ),
            }
            for index in order
        ]

    def _transformed_feature_names(self) -> list[str]:
        """Return the post-encoding feature names from the fitted preprocessor."""
        try:
            return [str(name) for name in self.pipeline.named_steps["preprocess"].get_feature_names_out()]
        except (AttributeError, KeyError, ValueError):
            return []


def _raw_feature_for(transformed_name: str) -> str:
    """Map a one-hot column name such as ``Contract_Month-to-month`` back to its source column."""
    for column in COLUMNS:
        if transformed_name == column or transformed_name.startswith(f"{column}_"):
            return column
    return transformed_name


def _extract_global_importance(pipeline: Any, model_name: str) -> list[dict[str, Any]]:
    """Return global feature importance, whichever the estimator exposes."""
    classifier = pipeline.named_steps["clf"]
    try:
        names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    except (AttributeError, ValueError):
        return []

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=float)
        values = np.abs(coefficients[0]) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        return []

    order = np.argsort(values)[::-1][:15]
    return [{"feature": names[int(index)], "importance": float(values[int(index)])} for index in order]


def train_model(frame: pd.DataFrame, model_name: str = "logreg", *, test_size: float = 0.2) -> ChurnModel:
    """Train and evaluate one classifier on a stratified split.

    Raises:
        InvalidInputError: When the frame is missing columns or too small.
    """
    set_global_seed()
    if frame is None or frame.empty:
        raise InvalidInputError("No training data supplied.", hint="Load the dataset first.")
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing and TARGET_COLUMN not in missing:
        raise InvalidInputError(
            f"Training data is missing column(s): {', '.join(missing)}.",
            hint="Reload with src.data.load_dataset().",
        )
    if TARGET_COLUMN not in frame.columns:
        raise InvalidInputError(
            f"Training data is missing the target column '{TARGET_COLUMN}'.",
            hint="Reload with src.data.load_dataset().",
        )

    data = frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(data) < 100:
        raise InvalidInputError(
            f"Only {len(data)} labelled customers available.",
            hint="At least 100 rows are needed to train and evaluate.",
        )

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        data[list(COLUMNS)], data[TARGET_COLUMN], test_size=test_size,
        random_state=RANDOM_SEED, stratify=data[TARGET_COLUMN],
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)

    metrics = classification_metrics(
        y_test, predictions, labels=CLASS_LABELS, y_proba=probabilities,
        average="binary", pos_label=POSITIVE_LABEL,
    )
    metrics["n_train"] = int(len(x_train))
    metrics["n_test"] = int(len(x_test))
    # Recorded from the test split itself so the interface never has to assume a
    # churn rate.
    metrics["pos_rate"] = float((pd.Series(y_test) == POSITIVE_LABEL).mean())

    return ChurnModel(
        pipeline=pipeline,
        model_name=model_name,
        metrics=metrics,
        per_class=per_class_report(y_test, predictions, labels=CLASS_LABELS),
        feature_importance=_extract_global_importance(pipeline, model_name),
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
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
    """Return the classifier with the highest ROC-AUC, falling back to F1.

    ROC-AUC is primary because the dataset is imbalanced and a probability-based
    risk score needs good ranking across all thresholds.
    """
    def score(payload: dict[str, Any]) -> tuple[float, float]:
        metrics = payload["metrics"]
        roc = metrics.get("roc_auc")
        return (float(roc) if roc is not None else -1.0, float(metrics.get("f1", 0.0)))

    return max(comparison, key=lambda name: score(comparison[name]))


def save_model(model: ChurnModel, *, directory: Path | None = None) -> Path:
    """Persist the fitted pipeline to ``models/churn_model.joblib``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "churn_model.joblib"
    save_joblib(model, path)
    return path


def load_model(path: Path | None = None) -> ChurnModel:
    """Load a previously trained :class:`ChurnModel`."""
    target = Path(path) if path else models_dir(PROJECT_SLUG) / "churn_model.joblib"
    loaded = load_joblib(target, description="churn model", train_command="python train.py")
    if not isinstance(loaded, ChurnModel):
        raise ModelNotFoundError(
            f"'{target.name}' does not contain a ChurnModel.",
            hint="Delete the file and re-run `python train.py`.",
        )
    return loaded


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    model: ChurnModel,
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
        "selection_rule": "highest ROC-AUC on the stratified test split",
        "class_labels": CLASS_LABELS,
        "positive_label": POSITIVE_LABEL,
        "risk_bands": [{"band": label, "threshold": threshold} for threshold, label in RISK_BANDS],
        "dropped_columns": {
            "CustomerID": "identifier, not a feature",
            "year": "derived from TotalCharges",
        },
        "hyperparameters": {name: params for name, params in MODEL_PARAMS.items()},
        "dataset": dataset,
        "comparison": comparison,
        "metrics": model.metrics,
        "per_class": model.per_class,
        "global_feature_importance": model.feature_importance,
        "random_seed": RANDOM_SEED,
        "disclaimer": (
            "A churn probability is a statistical estimate over similar customers, not "
            "a statement that this individual will leave. It should inform retention "
            "targeting, never be the sole basis for a customer-facing decision."
        ),
    }