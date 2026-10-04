"""Tests for Project 06 - Customer Service Chatbot.

The classifier is trained once per session on the in-source corpus, which takes
well under a second, and every test then exercises the real model rather than a
mock. That keeps the confidence-gate assertions honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from project_env import REPO_ROOT, use_project

use_project("06-customer-service-chatbot")

sys.path.insert(0, str(REPO_ROOT))

from shared.errors import InvalidInputError  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_LABELS,
    FALLBACK_INTENT,
    GATE_PROBES,
    INTENT_LABELS,
    INTENTS,
    INTENT_NAMES,
    build_corpus,
    corpus_summary,
    get_intent,
    response_for,
    suggested_messages,
)
from src.model import (  # noqa: E402
    ChatbotModel,
    build_pipeline,
    build_training_report,
    clean_for_model,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture(scope="module")
def model() -> ChatbotModel:
    """Train the real classifier once for the whole module."""
    texts, labels = build_corpus()
    return train_model(texts, labels)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def test_every_required_intent_is_present():
    """The taxonomy covers the intents the brief asks for, plus an off-topic one."""
    for name in (
        "greeting",
        "working_hours",
        "pricing",
        "refund",
        "order_status",
        "contact_support",
        "product_info",
        "goodbye",
    ):
        assert name in INTENT_NAMES

    assert FALLBACK_INTENT in INTENT_NAMES, "an explicit off-topic intent is required"


def test_every_intent_has_utterances_and_responses():
    """No intent is a placeholder without examples or a reply."""
    for intent in INTENTS:
        assert len(intent.utterances) >= 8, f"{intent.name} has too few utterances"
        assert intent.responses, f"{intent.name} has no response"
        assert intent.description


def test_utterances_are_unique_within_each_intent():
    """Duplicate examples would silently reweight a class."""
    for intent in INTENTS:
        lowered = [utterance.strip().lower() for utterance in intent.utterances]
        assert len(lowered) == len(set(lowered)), f"{intent.name} has duplicate utterances"


def test_corpus_labels_match_the_declared_intents():
    """Augmented examples are labelled with a known intent."""
    texts, labels = build_corpus()
    assert len(texts) == len(labels)
    assert len(texts) > 1_000
    assert set(labels) == set(INTENT_NAMES)


def test_corpus_augmentation_preserves_every_intent():
    """Augmentation multiplies each intent rather than dropping any of them.

    Counts differ between intents because the base utterances differ in number;
    what matters is that every intent contributes a healthy number of examples
    and that the total is a whole multiple of the base corpus.
    """
    summary = corpus_summary()
    counts = list(summary["examples_per_intent"].values())
    assert len(counts) == len(INTENT_NAMES)
    assert min(counts) > 0
    assert summary["n_training_examples"] > summary["n_base_utterances"]
    # Every intent is augmented by the same factor, prefix x suffix variants.
    factors = {
        counts[index] / summary["base_per_intent"][name]
        for index, name in enumerate(INTENT_NAMES)
    }
    assert len(factors) == 1, f"intents were augmented inconsistently: {factors}"


def test_get_intent_rejects_unknown_name():
    """An unknown intent is reported rather than silently defaulted."""
    with pytest.raises(InvalidInputError):
        get_intent("small_talk_about_weather")


def test_response_for_wraps_around():
    """Asking for more templates than exist cycles rather than raising."""
    intent = get_intent("pricing")
    for index in range(len(intent.responses) + 3):
        assert response_for("pricing", index=index) in intent.responses


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_cleaning_lowercases_strips_punctuation_and_drops_stopwords():
    """Cleaning is deterministic, lowercase and free of filler words."""
    assert clean_for_model("Hello, WHAT are your HOURS?") == "hello hours"


def test_cleaning_preserves_negations():
    """'not a refund' must not collapse into 'refund'."""
    cleaned = clean_for_model("I do not want a refund")
    assert "not" in cleaned
    assert "refund" in cleaned


def test_cleaning_keeps_typos_intact():
    """A typo stays a token so character n-grams can still match it."""
    assert "refubd" in clean_for_model("I want my refubd")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_pipeline_trains_and_reports_all_intents(model):
    """The trained model covers every intent and reports honest metrics."""
    assert model.metrics["accuracy"] > 0.95
    assert model.metrics["n_test"] > 0
    assert {row["label"] for row in model.per_class} == set(INTENT_NAMES)


def test_unseen_paraphrases_map_to_the_right_intent(model):
    """Each required intent is recognised from wording absent from the corpus."""
    expected = {
        "refund": "I would like my money back",
        "working_hours": "what time do you close today",
        "pricing": "how much does the cheapest option cost",
        "order_status": "has my parcel been delivered",
        "contact_support": "I need to talk to a real person",
        "product_info": "does it work on my phone",
        "goodbye": "that will be all, thanks",
        "greeting": "hello, good morning",
    }
    for intent, message in expected.items():
        response = model.classify(message)
        assert response.intent == intent, (
            f"'{message}' was routed to {response.intent}, expected {intent}"
        )
        assert response.understood


def test_responses_never_promise_an_action(model):
    """The bot must not claim it can issue a refund or place an order."""
    response = model.classify("I want a refund")
    text = response.text.lower()
    assert "cannot process" in text or "not issue" in text or "agent" in text
    assert SUPPORT_PHONE_LOWER in response.text or "support@" in response.text


SUPPORT_PHONE_LOWER = "+1-555-0142"


# --------------------------------------------------------------------------- #
# Confidence gate
# --------------------------------------------------------------------------- #


def test_off_topic_messages_are_declined(model):
    """Unrelated questions do not receive a confident support answer."""
    for message in ("what is the weather like", "tell me a joke", "sing me a song"):
        response = model.classify(message)
        assert not response.understood, f"'{message}' was answered instead of declined"
        assert response.intent == FALLBACK_INTENT
        assert response.reason


def test_declined_reply_matches_the_required_wording(model):
    """The fallback text tells the user what to do next."""
    text = model.classify("what is the weather like").text.lower()
    assert "not confident" in text or "outside what i can help" in text
    assert "contact" in text or "rephrase" in text


def test_gate_report_is_measured_against_ground_truth(model):
    """The gate report counts answered-wrong cases, which is the whole point."""
    report = model.confidence_gate_report(list(GATE_PROBES))
    assert report["n_messages"] == len(GATE_PROBES)
    assert report["answered"] + report["declined"] == report["n_messages"]
    assert 0.0 <= report["precision_when_answering"] <= 1.0
    assert report["wrong_answers"] == pytest.approx(
        report["answered"] * (1 - report["precision_when_answering"]), abs=1e-9
    )
    # Declining an off-topic message counts as correct caution.
    assert report["declined_were_genuinely_off_topic"] >= 0


def test_stricter_threshold_declines_more(model):
    """Raising the confidence floor can only reduce the number of answers."""
    texts, labels = build_corpus()
    loose = train_model(texts, labels, min_confidence=0.10, min_margin=0.0)
    strict = train_model(texts, labels, min_confidence=0.99, min_margin=0.0)

    loose_report = loose.confidence_gate_report(list(GATE_PROBES))
    strict_report = strict.confidence_gate_report(list(GATE_PROBES))
    assert strict_report["answered"] <= loose_report["answered"]


def test_gate_report_rejects_unknown_label(model):
    """A ground-truth label outside the taxonomy is reported."""
    with pytest.raises(InvalidInputError):
        model.confidence_gate_report([("hello", "astrology")])


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_message_is_rejected(model, message):
    """An empty bubble is a user error, not a conversational turn."""
    with pytest.raises(InvalidInputError):
        model.classify(message)


def test_punctuation_only_message_is_rejected(model):
    """A message with no usable words is rejected, not answered."""
    with pytest.raises(InvalidInputError):
        model.classify("!!! ???")


# --------------------------------------------------------------------------- #
# Training, persistence and reporting
# --------------------------------------------------------------------------- #


def test_training_rejects_misaligned_or_small_input():
    """Bad corpora are reported rather than silently training on noise."""
    with pytest.raises(InvalidInputError):
        train_model(["hello", "bye"], ["greeting", "goodbye"])

    texts, labels = build_corpus()
    with pytest.raises(InvalidInputError):
        train_model(texts[:50], labels[:50])

    # An intent reduced to two examples cannot support a stratified split.
    skewed: list[str] = []
    seen_goodbye = 0
    for text, label in zip(texts, labels):
        if label == "goodbye":
            seen_goodbye += 1
            if seen_goodbye > 2:
                skewed.append("greeting")
                continue
        skewed.append(label)
    with pytest.raises(InvalidInputError):
        train_model(texts, skewed)


def test_build_pipeline_has_expected_steps():
    """The pipeline is vectorise-then-classify, in that order."""
    pipeline = build_pipeline()
    assert list(pipeline.named_steps) == ["tfidf", "clf"]


def test_save_load_round_trip(model, tmp_path):
    """A saved chatbot reloads and answers identically."""
    path = save_model(model, directory=tmp_path)
    reloaded = load_model(path)

    for message in ("how much is the pro plan?", "what is the capital of France"):
        before = model.classify(message)
        after = reloaded.classify(message)
        assert before.intent == after.intent
        assert before.text == after.text
        assert before.understood == after.understood


def test_load_missing_model_explains_remedy(tmp_path):
    """A missing artefact names the command that creates it."""
    with pytest.raises(Exception) as error:
        load_model(tmp_path / "absent.joblib")
    assert "train.py" in (error.value.hint or "")


def test_load_rejects_foreign_artefact(tmp_path):
    """A joblib file holding the wrong object is reported clearly."""
    from shared.errors import ModelNotFoundError
    from shared.ml import save_joblib

    save_joblib("not a chatbot", tmp_path / "chatbot_model.joblib")
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "chatbot_model.joblib")


def test_training_report_is_json_serialisable(model):
    """The report contains the measured gate numbers, not placeholders."""
    import json

    report = build_training_report(
        model,
        corpus=corpus_summary(),
        gate_report=model.confidence_gate_report(list(GATE_PROBES)),
    )
    payload = json.loads(json.dumps(report))
    assert payload["project"] == "06-customer-service-chatbot"
    assert payload["confidence_gate"]["n_messages"] == len(GATE_PROBES)
    assert "TF-IDF" in payload["algorithm"]


def test_suggested_messages_cover_the_intents():
    """The interface's examples exercise several different intents."""
    messages = suggested_messages()
    assert len(messages) >= 6
    assert any("?" in message for message in messages)


def test_intent_labels_cover_every_intent():
    """No intent is missing a display name, which would show blank in the UI."""
    for name in CLASS_LABELS:
        assert INTENT_LABELS.get(name)


# --------------------------------------------------------------------------- #
# Interface smoke test
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (REPO_ROOT / "06-customer-service-chatbot" / "models" / "chatbot_model.joblib").exists(),
    reason="Trained artefact absent; run `python train.py` first.",
)
def test_app_renders_without_exception():
    """The Streamlit page executes top to bottom without raising."""
    from streamlit.testing.v1 import AppTest

    project_dir = REPO_ROOT / "06-customer-service-chatbot"
    app = AppTest.from_file(str(project_dir / "app.py"), default_timeout=240)
    app.run()
    assert not app.exception