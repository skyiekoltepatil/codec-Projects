"""Streamlit interface for the Customer Service Chatbot.

Run from this project's directory::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.paths import models_dir, portable_display  # noqa: E402
from shared.plotting import bar_chart, confusion_matrix_heatmap  # noqa: E402
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header  # noqa: E402
from shared.ui import (  # noqa: E402
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
    show_metrics_dict,
)

from src.data import (  # noqa: E402
    CLASS_LABELS,
    COMPANY_NAME,
    FALLBACK_INTENT,
    GATE_PROBES,
    INTENT_LABELS,
    build_corpus,
    corpus_summary,
    suggested_messages,
)
from src.model import (  # noqa: E402
    ChatbotModel,
    build_training_report,
    clean_for_model,
    load_model,
    save_metrics_report,
    save_model,
    train_model,
)

MODEL_PATH = models_dir("06-customer-service-chatbot") / "chatbot_model.joblib"

#: Key under which the conversation is kept in Streamlit session state.
HISTORY_KEY = "chat_history"

DISCLAIMER = (
    f"{COMPANY_NAME} is fictional. This bot classifies intent with a small "
    "hand-written corpus: it can answer the questions it was trained on, it cannot "
    "hold a conversation, and it has no access to customer accounts. It cannot issue "
    "refunds, change orders or make any change to a real system."
)


@st.cache_resource(show_spinner=False)
def _load_chatbot(path_str: str, mtime: float) -> ChatbotModel:
    """Load the trained classifier once per artefact version."""
    return load_model(Path(path_str))


def _get_model() -> ChatbotModel | None:
    """Return the trained chatbot, or ``None`` when it has not been trained."""
    if not MODEL_PATH.exists():
        return None
    return _load_chatbot(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _history() -> list[dict[str, str]]:
    """Return the conversation so far."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    return st.session_state[HISTORY_KEY]


def _append(role: str, text: str, **extra: object) -> None:
    """Append one message to the conversation history."""
    _history().append({"role": role, "text": text, **extra})


def _render_history() -> None:
    """Draw every stored message as a chat bubble."""
    for entry in _history():
        with st.chat_message(entry["role"]):
            st.markdown(entry["text"])
            if entry["role"] == "assistant" and entry.get("intent_label"):
                understood = bool(entry.get("understood", True))
                st.caption(
                    f"Intent: **{entry['intent_label']}** - "
                    f"confidence {float(entry.get('confidence', 0.0)):.0%}"
                    + ("" if understood else " - declined, below the confidence gate")
                )


def render_chat(model: ChatbotModel) -> None:
    """Render the conversation, the message box and the debugging controls."""
    section(
        "Chat",
        f"You are talking to the {COMPANY_NAME} support assistant. "
        "Ask a question and the intent classifier will route it.",
    )

    controls, examples = st.columns([1, 2])
    with controls:
        if st.button("Clear conversation", width="stretch"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()
    with examples:
        chosen = st.selectbox(
            "Example message",
            options=[""] + suggested_messages(),
            key="chat_example",
            label_visibility="collapsed",
        )
        if chosen and st.button("Send example", width="stretch"):
            _handle_message(model, chosen)
            st.rerun()

    _render_history()

    typed = st.chat_input("Type your question")
    if typed:
        _handle_message(model, typed)
        st.rerun()

    if not _history():
        st.info(
            "Try one of the examples above, or ask: "
            "*what are your opening hours?*, *how much is the Pro plan?*, "
            "*I would like a refund*."
        )


def _handle_message(model: ChatbotModel, message: str) -> None:
    """Classify a message, record both turns and re-render."""
    _append("user", message)
    try:
        response = model.classify(message)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        # Remove the user turn so an invalid message does not linger unanswered.
        _history().pop()
        return

    _append(
        "assistant",
        response.text,
        intent=response.intent,
        intent_label=response.intent_label,
        confidence=response.confidence,
        understood=response.understood,
        reason=response.reason,
        distribution=response.distribution,
    )


def render_last_turn_debug(model: ChatbotModel) -> None:
    """Show why the bot gave its most recent answer."""
    turns = [entry for entry in _history() if entry["role"] == "assistant"]
    if not turns:
        return

    last = turns[-1]
    with st.expander("Why the bot answered that", expanded=False):
        distribution = last.get("distribution") or {}
        if distribution:
            ordered = {label: distribution.get(label, 0.0) for label in CLASS_LABELS}
            show_figure(
                bar_chart(
                    [INTENT_LABELS.get(label, label) for label in ordered],
                    list(ordered.values()),
                    title="Class probabilities",
                    ylabel="Probability",
                    width=7.2,
                    height=3.2,
                )
            )
        st.markdown(f"**You asked:** {_last_user_message()}")
        st.markdown(f"**Matched intent:** {last.get('intent_label')}")
        st.markdown(f"**Confidence:** {float(last.get('confidence', 0.0)):.1%}")
        if not last.get("understood", True):
            st.warning(f"Declined. {last.get('reason', '')}")
        st.caption(
            f"Preprocessing the model actually receives: `{clean_for_model(_last_user_message())}`"
        )
        st.caption(
            f"Gate: confidence >= {model.min_confidence:.0%} and margin >= {model.min_margin:.0%}."
        )


def _last_user_message() -> str:
    """Return the most recent user message, or an empty string."""
    users = [entry for entry in _history() if entry["role"] == "user"]
    return users[-1]["text"] if users else ""


def render_evaluation(model: ChatbotModel) -> None:
    """Render held-out metrics, per-intent table and confusion matrix."""
    metrics = model.metrics
    if not metrics:
        return

    section(
        "Model evaluation",
        f"Measured on the {metrics.get('n_test', 0):,} held-out corpus messages, "
        "with augmentation variants kept in the split so a model cannot score well "
        "by memorising one string.",
    )

    show_metrics_dict(
        metrics,
        keys=["accuracy", "precision", "recall", "f1"],
        labels={
            "accuracy": "ACCURACY",
            "precision": "PRECISION (macro)",
            "recall": "RECALL (macro)",
            "f1": "F1 (macro)",
        },
    )

    left, right = st.columns([1, 1])
    with left:
        show_figure(
            confusion_matrix_heatmap(
                metrics["confusion_matrix"],
                [INTENT_LABELS.get(label, label) for label in CLASS_LABELS],
                title="Confusion matrix (held-out split)",
                width=7.4,
                height=5.6,
            )
        )
    with right:
        if model.per_class:
            frame = pd.DataFrame(model.per_class)
            frame["intent"] = frame["label"].map(lambda name: INTENT_LABELS.get(name, name))
            for column in ("precision", "recall", "f1"):
                frame[column] = frame[column].round(4)
            show_dataframe(
                frame[["intent", "precision", "recall", "f1", "support"]],
                caption="Per-intent precision, recall, F1 and support.",
            )

    section(
        "The confidence gate",
        "The bot declines to answer when the top intent is below the confidence "
        "threshold, or when the top two intents are nearly tied. Those thresholds "
        "are what stop an unrelated question from receiving a confident but wrong answer.",
    )
    metric_row(
        [
            ("Min confidence", f"{model.min_confidence:.0%}", "below this, the bot declines"),
            ("Min margin", f"{model.min_margin:.0%}", "top-two gap required"),
            ("Intents", str(len(CLASS_LABELS)), "including off-topic"),
        ]
    )
    st.caption(
        "The **Not understood** intent is trained on real off-topic messages such as "
        "'what is the weather'. Without it the classifier would confidently map every "
        "unrelated question onto whichever support intent shares the most words."
    )


def render_corpus_panel() -> None:
    """Show corpus statistics and the response catalogue."""
    summary = corpus_summary()
    with st.expander("Intent corpus"):
        metric_row(
            [
                ("Intents", str(summary["n_intents"]), "including off-topic"),
                ("Base utterances", str(summary["n_base_utterances"]), "hand written"),
                ("Training examples", f"{summary['n_training_examples']:,}", "after augmentation"),
            ]
        )
        show_dataframe(
            [
                {
                    "intent": INTENT_LABELS.get(name, name),
                    "base utterances": summary["base_per_intent"].get(name, 0),
                    "training examples": summary["examples_per_intent"].get(name, 0),
                }
                for name in CLASS_LABELS
            ],
            caption="How much of the corpus each intent contributes.",
        )
        st.markdown(
            f"**Company:** {summary['company']}  \n"
            f"**Support hours:** {summary['support_hours']}  \n"
            f"**Contact:** {summary['support_email']}"
        )


def run_training() -> bool:
    """Train and persist the classifier. Returns ``True`` on success."""
    try:
        with st.spinner("Training the intent classifier..."):
            texts, labels = build_corpus()
            model = train_model(texts, labels)
            model.trained_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
            save_model(model)
            report = build_training_report(
                model,
                corpus=corpus_summary(),
                gate_report=model.confidence_gate_report(list(GATE_PROBES)),
            )
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(
        f"Trained the intent classifier to {format_metric(model.metrics['accuracy'])} accuracy "
        f"and saved it to `{portable_display(MODEL_PATH)}`."
    )
    st.cache_resource.clear()
    return True


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Customer Service Chatbot", page_icon="06", layout="wide")
    inject_theme()

    page_header(
        "Customer Service Chatbot",
        "An intent-classification chatbot: every message is cleaned, vectorised with "
        "TF-IDF and classified by Logistic Regression, and a confidence gate decides "
        "whether the bot is allowed to answer.",
        eyebrow="Project 06",
        chips=["NLP", "Intent classification", "TF-IDF", "Confidence gate"],
    )

    if st.sidebar.button("Train / retrain", width="stretch"):
        st.sidebar.caption("Rebuilds the classifier from the intent corpus in src/data.py.")
        if run_training():
            st.rerun()

    model = _get_model()
    if model is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Train it by running:\n\n"
            "```bash\npython train.py\n```"
        )
        note(DISCLAIMER)
        return

    render_chat(model)
    render_last_turn_debug(model)
    render_evaluation(model)
    render_corpus_panel()

    model_info_panel(
        algorithm="TF-IDF (word 1-2 grams) + Logistic Regression",
        trained_on=f"{model.n_train:,} corpus messages (25% held out)",
        metrics=model.metrics,
        extra={
            "Trained at": model.trained_at,
            "Intents": str(len(CLASS_LABELS)),
            "Min confidence": f"{model.min_confidence:.0%}",
            "Min margin": f"{model.min_margin:.0%}",
            "Fallback": "declines instead of guessing",
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()