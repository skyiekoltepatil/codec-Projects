"""Intent classification and response selection for the support chatbot.

Pipeline: message -> clean -> TF-IDF -> Logistic Regression -> confidence gate ->
response.

Why a confidence threshold matters
----------------------------------
A Logistic Regression will always return *some* intent, even for "what is the
weather". Left ungated, the bot confidently answers a banking question with
refund-policy text. The gate turns a wrong answer into an honest "I'm not
confident I understood that", which is the behaviour a real support bot needs.

Two thresholds are applied, and they do different jobs:

* ``min_confidence`` (0.55) rejects messages the model is unsure about.
* ``margin`` (0.20) rejects messages where the top two intents are nearly tied.
  A model can be 55% confident about a class while genuinely not knowing whether
  the message was a refund or an order-status query; the margin catches that even
  when the top probability clears the first gate.

The training corpus contains an explicit ``unrecognized`` intent, which is what
makes the first threshold calibratable rather than arbitrary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import classification_metrics, per_class_report
from shared.ml import RANDOM_SEED, load_joblib, save_joblib, save_json, set_global_seed
from shared.paths import models_dir
from shared.text import normalize_text

from src.data import CLASS_LABELS, FALLBACK_INTENT, INTENT_LABELS, response_for

logger = logging.getLogger(__name__)

PROJECT_SLUG = "06-customer-service-chatbot"

#: Probability below which the bot declines to answer.
MIN_CONFIDENCE = 0.55

#: Minimum gap between the best and second-best intent probability.
MIN_MARGIN = 0.20

#: TF-IDF settings. Character-level n-grams are included so the model still
#: recognises a typo or a differently-spelled phrase ("refubd", "money bak").
TFIDF_PARAMS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 1,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

#: Logistic Regression hyperparameters.
MODEL_PARAMS: dict[str, Any] = {
    "C": 8.0,
    "max_iter": 2_000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}


def clean_for_model(text: str) -> str:
    """Apply the project's text preprocessing for intent classification.

    Negations are kept: "not a refund" must not collapse into "refund".
    """
    return str(normalize_text(text, keep_negations=True, strip_numbers=True))


@dataclass
class ChatResponse:
    """One reply from the bot, with enough context for the interface to explain it."""

    text: str
    intent: str
    intent_label: str
    confidence: float
    understood: bool
    distribution: dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    reason: str = ""


@dataclass
class ChatbotModel:
    """A fitted intent classifier plus its evaluation metadata."""

    pipeline: Any
    labels: list[str] = field(default_factory=lambda: list(CLASS_LABELS))
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    min_confidence: float = MIN_CONFIDENCE
    min_margin: float = MIN_MARGIN
    n_train: int = 0
    n_test: int = 0
    trained_at: str = ""
    top_features: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    def classify(self, message: str) -> ChatResponse:
        """Classify one message and choose a reply.

        An empty or whitespace-only message raises :class:`InvalidInputError`
        rather than returning a canned "I didn't understand", because sending an
        empty bubble is a user error worth naming.
        """
        if message is None or not str(message).strip():
            raise InvalidInputError(
                "Your message is empty.",
                hint="Type a question, for example: what are your opening hours?",
            )

        cleaned = clean_for_model(message)
        if not cleaned:
            raise InvalidInputError(
                "That message contained no usable words.",
                hint="It was only punctuation or characters. Please type a question.",
            )

        probabilities = np.asarray(self.pipeline.predict_proba([cleaned])[0], dtype=float)
        classes = [str(name) for name in self.pipeline.classes_]
        order = np.argsort(probabilities)[::-1]

        best_index = int(order[0])
        best_probability = float(probabilities[best_index])
        second_probability = float(probabilities[int(order[1])]) if len(order) > 1 else 0.0
        margin = best_probability - second_probability
        best_intent = classes[best_index]

        distribution = {
            label: float(probabilities[classes.index(label)]) for label in self.labels if label in classes
        }

        # The gate, applied in priority order so the message explains itself.
        if best_intent == FALLBACK_INTENT:
            reason = "The model matched the trained off-topic examples."
            return self._decline(best_probability, margin, distribution, reason, best_intent)

        if best_probability < self.min_confidence:
            reason = (
                f"Top intent scored {best_probability:.0%}, below the "
                f"{self.min_confidence:.0%} confidence threshold."
            )
            return self._decline(best_probability, margin, distribution, reason, best_intent)

        if margin < self.min_margin:
            reason = (
                f"Two intents scored almost equally ({best_probability:.0%} vs "
                f"{second_probability:.0%}), so the choice is not reliable."
            )
            return self._decline(best_probability, margin, distribution, reason, best_intent)

        return ChatResponse(
            text=response_for(best_intent),
            intent=best_intent,
            intent_label=INTENT_LABELS.get(best_intent, best_intent),
            confidence=best_probability,
            understood=True,
            distribution=distribution,
            margin=margin,
        )

    def _decline(
        self,
        confidence: float,
        margin: float,
        distribution: dict[str, float],
        reason: str,
        top_intent: str,
    ) -> ChatResponse:
        """Return the fallback reply that explains why the bot declined."""
        return ChatResponse(
            text=response_for(FALLBACK_INTENT),
            intent=FALLBACK_INTENT,
            intent_label=INTENT_LABELS[FALLBACK_INTENT],
            confidence=confidence,
            understood=False,
            distribution=distribution,
            margin=margin,
            reason=reason or f"The closest intent was '{top_intent}'.",
        )

    def confidence_gate_report(self, messages: Sequence[tuple[str, str]]) -> dict[str, Any]:
        """Evaluate the confidence gate against ground-truth intents.

        Args:
            messages: ``(text, true_intent)`` pairs the model did not train on.

        Four outcomes are counted, because "how often does the bot answer" on its
        own is not a useful number:

        * answered and correct - the useful case;
        * answered and **wrong** - the damaging case, and the reason the gate exists;
        * declined, and the message really was off-topic - correct caution;
        * declined, but the message had a real intent - over-caution, the cost of
          the gate being too strict.

        Comparing against the model's own prediction instead of the true label
        would always report 100% accuracy, which is why the labels are required.
        """
        answered = answered_correct = declined = declined_correct = 0

        for text, true_intent in messages:
            if true_intent not in self.labels:
                raise InvalidInputError(
                    f"'{true_intent}' is not a known intent.",
                    hint=f"Expected one of: {', '.join(self.labels)}.",
                )
            response = self.classify(text)
            if response.understood:
                answered += 1
                answered_correct += int(response.intent == true_intent)
            else:
                declined += 1
                declined_correct += int(true_intent == FALLBACK_INTENT)

        total = max(1, answered + declined)
        return {
            "n_messages": answered + declined,
            "answered": answered,
            "declined": declined,
            "answer_rate": answered / total,
            "precision_when_answering": (answered_correct / answered) if answered else 0.0,
            "wrong_answers": answered - answered_correct,
            "decline_rate": declined / total,
            "declined_were_genuinely_off_topic": declined_correct,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
        }


def build_pipeline() -> Any:
    """Return an unfitted TF-IDF plus Logistic Regression pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", LogisticRegression(**MODEL_PARAMS)),
        ]
    )


def _extract_top_features(pipeline: Any, top_n: int = 8) -> dict[str, list[dict[str, float]]]:
    """Return the most influential n-grams per intent."""
    try:
        vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
        coefficients = np.asarray(pipeline.named_steps["clf"].coef_, dtype=float)
    except (AttributeError, ValueError):
        return {}

    classes = [str(name) for name in pipeline.classes_]
    result: dict[str, list[dict[str, float]]] = {}
    for index, class_name in enumerate(classes):
        if index >= coefficients.shape[0]:
            continue
        row = coefficients[index]
        order = np.argsort(row)[::-1][:top_n]
        result[class_name] = [{"feature": str(vocabulary[i]), "weight": float(row[i])} for i in order]
    return result


def train_model(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    test_size: float = 0.25,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_MARGIN,
) -> ChatbotModel:
    """Train and evaluate the intent classifier.

    Raises:
        InvalidInputError: When inputs are misaligned, too small, or a class has
            too few examples to stratify a split.
    """
    set_global_seed()
    if len(texts) != len(labels):
        raise InvalidInputError(
            f"Got {len(texts)} messages but {len(labels)} labels.",
            hint="Each training message needs exactly one intent label.",
        )
    if len(texts) < 100:
        raise InvalidInputError(
            f"Only {len(texts)} training messages available.",
            hint="The intent corpus should produce well over a hundred examples.",
        )

    counts = {label: list(labels).count(label) for label in set(labels)}
    too_small = {label: count for label, count in counts.items() if count < 4}
    if too_small:
        raise InvalidInputError(
            f"Intent(s) with too few examples: {too_small}.",
            hint="Each intent needs at least 4 examples for a stratified split.",
        )

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        [clean_for_model(text) for text in texts],
        list(labels),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=list(labels),
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)

    metrics = classification_metrics(
        y_test, predictions, labels=CLASS_LABELS, y_proba=probabilities
    )
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)

    model = ChatbotModel(
        pipeline=pipeline,
        metrics=metrics,
        per_class=per_class_report(y_test, predictions, labels=CLASS_LABELS),
        min_confidence=float(min_confidence),
        min_margin=float(min_margin),
        n_train=len(x_train),
        n_test=len(x_test),
        top_features=_extract_top_features(pipeline),
    )
    return model


def save_model(model: ChatbotModel, *, directory: Path | None = None) -> Path:
    """Persist the classifier to ``models/chatbot_model.joblib``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "chatbot_model.joblib"
    save_joblib(model, path)
    return path


def load_model(path: Path | None = None) -> ChatbotModel:
    """Load a previously trained :class:`ChatbotModel`."""
    target = Path(path) if path else models_dir(PROJECT_SLUG) / "chatbot_model.joblib"
    loaded = load_joblib(target, description="chatbot model", train_command="python train.py")
    if not isinstance(loaded, ChatbotModel):
        raise ModelNotFoundError(
            f"'{target.name}' does not contain a ChatbotModel.",
            hint="Delete the file and re-run `python train.py`.",
        )
    return loaded


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    model: ChatbotModel,
    *,
    corpus: dict[str, Any],
    gate_report: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON report written next to the model."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": model.trained_at,
        "algorithm": "TF-IDF (word 1-2 grams) with Logistic Regression",
        "task": f"{len(CLASS_LABELS)}-class intent classification",
        "framework": "scikit-learn",
        "intents": list(CLASS_LABELS),
        "intent_labels": dict(INTENT_LABELS),
        "tfidf": {key: str(value) for key, value in TFIDF_PARAMS.items()},
        "hyperparameters": {key: str(value) for key, value in MODEL_PARAMS.items()},
        "confidence_gate": {
            "min_confidence": model.min_confidence,
            "min_margin": model.min_margin,
            "fallback_intent": FALLBACK_INTENT,
            **gate_report,
        },
        "corpus": corpus,
        "split": {"train": model.n_train, "test": model.n_test},
        "metrics": model.metrics,
        "per_class": model.per_class,
        "top_features": model.top_features,
        "random_seed": RANDOM_SEED,
        "disclaimer": (
            "This is a rule-free intent classifier trained on a small hand-written "
            "corpus. It handles the phrasings it was trained on and similar ones; it "
            "does not hold a conversation, cannot access customer accounts, and cannot "
            "action refunds or orders."
        ),
    }