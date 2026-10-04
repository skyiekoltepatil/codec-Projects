"""Streamlit interface for the Spam Email Classifier.

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
from shared.theme import PALETTE, RISK_COLORS, inject_theme, metric_row, note, page_header, result_card  # noqa: E402
from shared.ui import (  # noqa: E402
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
    show_metrics_dict,
)

from src.data import CLASS_LABELS, HAM, SPAM, example_messages, load_dataset  # noqa: E402
from src.model import (  # noqa: E402
    MODEL_LABELS,
    SpamModel,
    build_training_report,
    clean_for_model,
    load_model,
    save_metrics_report,
    save_model,
    select_best_model,
    train_all_models,
    train_model,
)

MODEL_PATH = models_dir("07-spam-email-classifier") / "spam_model.joblib"

#: Above this spam probability the result card is styled as a warning.
SPAM_PROBABILITY_CUTOFF = 0.5

DISCLAIMER = (
    "This model was trained on **SMS messages** from the UCI SMS Spam Collection, not on "
    "email. Spam vocabulary and structure transfer between the two, but the reported "
    "accuracy is an SMS figure and should not be quoted as an email result. It is an "
    "educational baseline, not a production mail filter."
)


@st.cache_resource(show_spinner=False)
def _load_spam_model(path_str: str, mtime: float) -> SpamModel:
    """Load the trained model once per artefact version."""
    return load_model(Path(path_str))


def _get_model() -> SpamModel | None:
    """Return the trained model, or ``None`` when it has not been trained."""
    if not MODEL_PATH.exists():
        return None
    return _load_spam_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _render_sidebar() -> dict:
    """Render the sidebar and return the chosen options."""
    examples = example_messages()
    with st.sidebar:
        st.markdown("### Configuration")
        show_explanation = st.checkbox("Show word-by-word explanation", value=True)
        show_preprocessing = st.checkbox("Show preprocessing", value=False)
        st.markdown("---")
        st.markdown("### Actions")
        train_clicked = st.button("Train / retrain models", width="stretch")
        st.caption("Compares Naive Bayes, Logistic Regression and a Linear SVM.")
    return {
        "show_explanation": show_explanation,
        "show_preprocessing": show_preprocessing,
        "train_clicked": train_clicked,
        "examples": examples,
    }


def run_training() -> bool:
    """Train and persist the model. Returns ``True`` on success."""
    try:
        with st.spinner("Downloading the corpus if needed and training classifiers..."):
            data = load_dataset()
            comparison = train_all_models(data.frame)
            selected = select_best_model(comparison)
            best = train_model(data.frame, selected)
            save_model(best)
            report = build_training_report(
                best, comparison, dataset=data.summary(), selected=selected
            )
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(
        f"Trained and saved {MODEL_LABELS.get(selected, selected)} to "
        f"`{portable_display(MODEL_PATH)}`."
    )
    st.cache_resource.clear()
    return True


def render_prediction(model: SpamModel, options: dict) -> None:
    """Render the message input and the classification result."""
    section("Classify a message", "Paste an email or SMS below and the model will judge whether it is spam.")

    default_message = options["examples"]["Typical spam"]
    message = st.text_area(
        "Message",
        value=default_message,
        height=130,
        key="spam_message",
        placeholder="Paste the full message here...",
    )

    choice = st.selectbox(
        "Or load an example",
        options=[""] + list(options["examples"]),
        key="spam_example",
    )
    if choice:
        st.session_state["spam_message"] = options["examples"][choice]

    left, right = st.columns([1, 1])
    with left:
        classify = st.button("Classify message", type="primary", width="stretch")
    with right:
        if st.button("Clear", width="stretch"):
            st.session_state["spam_message"] = ""
            st.rerun()

    if not classify:
        st.caption("Press **Classify message** to run the model.")
        return

    try:
        result = model.predict(message)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return

    accent = RISK_COLORS.get("HIGH" if result.is_spam else "positive", PALETTE["accent"])
    result_card("SPAM" if result.is_spam else "NOT SPAM", f"{result.confidence:.1%} confidence", accent=accent)
    st.caption(f"Confidence is a {result.confidence_kind}.")

    ordered = {label: result.distribution.get(label, 0.0) for label in CLASS_LABELS}
    show_figure(
        bar_chart(
            CLASS_LABELS,
            list(ordered.values()),
            title="Class distribution",
            ylabel="Score",
            width=6.0,
            height=2.8,
        )
    )

    if options["show_explanation"]:
        explanation = model.explain(message)
        if explanation:
            with st.expander("Why the model decided that", expanded=True):
                show_dataframe(
                    [
                        {
                            "token": item["token"],
                            "contribution": round(item["contribution"], 4),
                            "effect": item["direction"],
                        }
                        for item in explanation
                    ],
                    caption=(
                        "Each token's contribution is its TF-IDF value multiplied by the "
                        "model's weight for that token, so this is this message's own "
                        "evidence rather than a global word list."
                    ),
                )
        else:
            st.info(
                "No per-token explanation is available for this model. A Naive Bayes "
                "classifier weights whole features rather than individual tokens."
            )

    if options["show_preprocessing"]:
        with st.expander("Preprocessing detail"):
            st.markdown("**Raw message**")
            st.code(message, language=None)
            st.markdown("**What the vectoriser receives**")
            st.code(clean_for_model(message), language=None)
            st.caption(
                "Stop words are removed but negations are kept, so 'not spam' does not "
                "collapse into 'spam'."
            )


def render_evaluation(model: SpamModel) -> None:
    """Render held-out metrics, per-class table and confusion matrix."""
    metrics = model.metrics
    if not metrics:
        return

    section(
        "Held-out evaluation",
        f"Metrics on the {metrics.get('n_test', 0):,} messages the model never saw during "
        "training, on a stratified 80/20 split.",
    )

    show_metrics_dict(
        metrics,
        keys=["accuracy", "precision", "recall", "f1"],
        labels={
            "accuracy": "ACCURACY (ALL)",
            "precision": "PRECISION (SPAM)",
            "recall": "RECALL (SPAM)",
            "f1": "F1 (SPAM)",
        },
    )
    if metrics.get("roc_auc") is not None:
        st.caption(f"ROC-AUC: **{format_metric(metrics['roc_auc'])}**")

    left, right = st.columns([1, 1])
    with left:
        show_figure(
            confusion_matrix_heatmap(
                metrics["confusion_matrix"],
                CLASS_LABELS,
                title="Confusion matrix (test split)",
                width=5.4,
                height=4.6,
            )
        )
    with right:
        if model.per_class:
            frame = pd.DataFrame(model.per_class)
            for column in ("precision", "recall", "f1"):
                frame[column] = frame[column].round(4)
            show_dataframe(frame, caption="Per-class precision, recall, F1 and support.")
        else:
            st.caption("Per-class report unavailable.")

    st.caption(
        "Macro-averaged F1 drives model selection because the classes are unbalanced: "
        "a filter tuned on accuracy alone would learn to ignore spam entirely."
    )


def render_learned_words(model: SpamModel) -> None:
    """Show the most indicative n-grams per class."""
    if not model.top_features:
        return
    section("What the model learned", "The n-grams with the strongest weight for each class.")
    columns = st.columns(len(CLASS_LABELS))
    for column, label in zip(columns, CLASS_LABELS, strict=False):
        features = model.top_features.get(label, [])
        with column:
            st.markdown(f"**{label.upper()}**")
            if not features:
                st.caption("Not available for this estimator.")
                continue
            for item in features[:8]:
                st.markdown(f"- `{item['token']}` ({item['weight']:.2f})")


def render_dataset_panel() -> None:
    """Show dataset provenance and class balance."""
    with st.expander("Dataset details"):
        try:
            with st.spinner("Loading the corpus..."):
                summary = load_dataset().summary()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        metric_row(
            [
                ("Messages", f"{summary['rows']:,}", "after duplicate removal"),
                ("Spam", f"{summary['spam_count']:,}", f"{summary['spam_rate']:.1%} of corpus"),
                ("Ham", f"{summary['ham_count']:,}", "legitimate"),
                ("Mean length", f"{summary['mean_length']:.0f}", "characters"),
            ]
        )
        counts = {SPAM: summary["spam_count"], HAM: summary["ham_count"]}
        show_figure(
            bar_chart(
                CLASS_LABELS,
                [counts[label] for label in CLASS_LABELS],
                title="Class balance",
                ylabel="Messages",
                width=6.0,
                height=3.0,
            )
        )
        st.markdown(f"**Source:** `{summary['source']}`")
        st.caption(
            "Duplicate messages are removed before splitting. They are common in this "
            "dataset and would otherwise let an identical string appear in both the "
            "training and test sets and inflate the reported accuracy."
        )


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Spam Email Classifier", page_icon="07", layout="wide")
    inject_theme()

    page_header(
        "Spam Email Classifier",
        "A TF-IDF bag-of-words classifier that flags spam messages and shows exactly "
        "which words drove the decision.",
        eyebrow="Project 07",
        chips=["NLP", "Binary classification", "TF-IDF", "Naive Bayes + SVM"],
    )

    options = _render_sidebar()

    model = _get_model()
    if options["train_clicked"]:
        if run_training():
            model = _get_model()
            st.rerun()

    if model is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Train it by running:\n\n"
            "```bash\npython train.py\n```"
        )
        render_dataset_panel()
        note(DISCLAIMER)
        return

    render_prediction(model, options)
    render_evaluation(model)
    render_learned_words(model)
    render_dataset_panel()

    model_info_panel(
        algorithm=MODEL_LABELS.get(model.model_name, model.model_name),
        trained_on=f"{model.n_train:,} messages (20% held out, stratified)",
        metrics=model.metrics,
        extra={
            "Trained at": model.trained_at,
            "Classes": ", ".join(CLASS_LABELS),
            "Confidence": (
                "calibrated probability"
                if model.supports_probabilities
                else "normalised decision score (not calibrated)"
            ),
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()