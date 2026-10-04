"""Streamlit interface for Twitter/X Sentiment Analysis.

Run from this project's directory:

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
from shared.errors import InvalidInputError  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.paths import models_dir, portable_display  # noqa: E402
from shared.plotting import bar_chart, confusion_matrix_heatmap  # noqa: E402
from shared.text import clean_text, normalize_text  # noqa: E402
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header, result_card  # noqa: E402
from shared.ui import (  # noqa: E402
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
    show_metrics_dict,
)

from src.data import CLASS_LABELS, TWEETS_SPEC, dataset_summary, load_dataset  # noqa: E402
from src.model import (  # noqa: E402
    MODEL_LABELS,
    build_training_report,
    clean_for_model,
    load_model,
    save_metrics_report,
    save_model,
    select_best_model,
    train_all_models,
    train_model,
)

MODEL_PATH = models_dir("02-twitter-sentiment-analysis") / "sentiment_model.joblib"

#: Colour used for each sentiment in the result card.
SENTIMENT_COLORS = {
    "Positive": PALETTE["positive"],
    "Negative": PALETTE["negative"],
    "Neutral": PALETTE["muted"],
}

#: Ready-made examples so a reviewer can try the app immediately.
EXAMPLES: dict[str, str] = {
    "Clearly positive": "I absolutely love this product! The service was fantastic.",
    "Clearly negative": "Terrible experience, my flight was delayed for six hours and nobody helped.",
    "Neutral / factual": "The flight departed at 6pm from gate B12 as scheduled.",
    "Negation trap": "I would not say this was a good experience.",
}

DISCLAIMER = (
    "The model was trained on tweets about US airlines. Its vocabulary is domain-specific, "
    "so accuracy on reviews, support tickets or other domains will be lower than the "
    "reported figures. It classifies the <em>text</em>, not the author's true feelings."
)


@st.cache_data(show_spinner=False)
def _dataset_summary() -> dict:
    """Cached dataset summary."""
    return dataset_summary()


@st.cache_resource(show_spinner=False)
def _load_model_cached(path_str: str, mtime: float):
    """Load the model once per artefact version."""
    return load_model(Path(path_str))


def _get_model():
    """Return the model, or ``None`` when it has not been trained yet."""
    if not MODEL_PATH.exists():
        return None
    return _load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def render_sidebar() -> dict:
    """Render the sidebar and return the selected options."""
    with st.sidebar:
        st.markdown("### Configuration")
        show_preprocessing = st.checkbox(
            "Show preprocessing steps",
            value=True,
            help="Display the cleaned text and tokens the model actually receives.",
        )
        show_distribution = st.checkbox("Show full class distribution", value=True)
        st.markdown("---")
        st.markdown("### Actions")
        train_clicked = st.button("Train / retrain models", width="stretch")
        st.caption(
            "Trains Logistic Regression, Multinomial Naive Bayes and a Linear SVM, then keeps "
            "the best by macro-F1."
        )
    return {
        "show_preprocessing": show_preprocessing,
        "show_distribution": show_distribution,
        "train_clicked": train_clicked,
    }


def run_training() -> bool:
    """Train and persist the model. Returns ``True`` on success."""
    try:
        with st.spinner("Downloading the tweet corpus if needed and training classifiers..."):
            frame = load_dataset()
            comparison = train_all_models(frame)
            selected = select_best_model(comparison)
            best = train_model(frame, selected)
            save_model(best)
            counts = frame["label"].value_counts().to_dict()
            report = build_training_report(
                best,
                comparison,
                dataset={
                    "rows": int(len(frame)),
                    "class_counts": {label: int(counts.get(label, 0)) for label in sorted(counts)},
                },
                selected=selected,
            )
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(f"Trained and saved {MODEL_LABELS.get(selected, selected)} to {portable_display(MODEL_PATH)}.")
    st.cache_resource.clear()
    return True


def render_prediction(model, options: dict) -> None:
    """Render the single-text classification panel."""
    section("Classify a message", "Enter any text and the model returns a sentiment and confidence.")

    default_text = st.session_state.get("sentiment_input", EXAMPLES["Clearly positive"])
    text = st.text_area("Message", value=default_text, height=110, key="sentiment_text_area")

    example_choice = st.selectbox("Or load an example", options=[""] + list(EXAMPLES), key="sentiment_example")
    if example_choice:
        text = EXAMPLES[example_choice]
        st.session_state["sentiment_text_area"] = text

    col_left, col_right = st.columns([1, 1])
    with col_left:
        analyse = st.button("Analyse sentiment", type="primary", width="stretch")
    with col_right:
        if st.button("Clear", width="stretch"):
            st.session_state["sentiment_text_area"] = ""
            st.rerun()

    if not analyse:
        st.caption("Press **Analyse sentiment** to run the model.")
        return

    if not text or not text.strip():
        st.error("**Empty input**\n\nEnter some text before analysing.")
        return

    try:
        result = model.predict_with_confidence(text)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return

    result_card("Predicted sentiment", result["label"], accent=SENTIMENT_COLORS.get(result["label"], PALETTE["accent"]))
    st.caption(
        f"Confidence: **{result['confidence']:.1%}** "
        f"({result['confidence_kind']})"
    )

    if options["show_distribution"]:
        distribution = result["distribution"]
        ordered = {label: distribution.get(label, 0.0) for label in CLASS_LABELS}
        show_figure(
            bar_chart(
                list(ordered),
                list(ordered.values()),
                title="Class distribution",
                ylabel="Score",
                width=6.4,
                height=3.0,
            )
        )

    if options["show_preprocessing"]:
        with st.expander("Preprocessing detail", expanded=True):
            st.markdown("**Raw text**")
            st.code(text, language=None)
            st.markdown("**After cleaning** (lowercased, URLs/handles/punctuation removed)")
            st.code(clean_text(text), language=None)
            st.markdown("**Tokens sent to the model** (stop words removed, negations kept)")
            st.code(clean_for_model(text), language=None)
            note(
                "Negation words are deliberately preserved. Removing them would collapse "
                "'not good' and 'good' into the same features and destroy the signal.",
                accent=PALETTE["accent"],
            )


def render_evaluation(model) -> None:
    """Render held-out metrics, per-class table and confusion matrix."""
    section("Held-out evaluation", "Metrics on the 20% stratified test split.")

    metrics = model.metrics
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

    roc = metrics.get("roc_auc_ovr", metrics.get("roc_auc"))
    if roc is not None:
        st.caption(f"One-vs-rest ROC-AUC: **{format_metric(roc)}**")

    left, right = st.columns([1, 1])
    with left:
        show_figure(
            confusion_matrix_heatmap(
                metrics["confusion_matrix"],
                CLASS_LABELS,
                title="Confusion matrix (test split)",
            )
        )
    with right:
        per_class = pd.DataFrame(model.per_class)
        show_dataframe(
            per_class.assign(
                precision=lambda frame: frame["precision"].round(4),
                recall=lambda frame: frame["recall"].round(4),
                f1=lambda frame: frame["f1"].round(4),
            ),
            caption="Per-class precision, recall, F1 and support.",
        )

    st.caption(
        "Macro-F1 is used for model selection because the dataset is imbalanced towards "
        "negative tweets; macro averaging prevents a model from winning by ignoring the "
        "neutral class."
    )


def render_insights(model) -> None:
    """Render the most influential n-grams per class."""
    if not model.top_features:
        return
    section("What the model learned", "Most influential n-grams per sentiment class.")
    columns = st.columns(len(CLASS_LABELS))
    for column, label in zip(columns, CLASS_LABELS, strict=False):
        features = model.top_features.get(label, [])
        with column:
            st.markdown(f"**{label}**")
            if not features:
                st.caption("Not available for this estimator.")
                continue
            for item in features[:8]:
                st.markdown(f"- `{item['feature']}` ({item['weight']:.2f})")


def render_dataset_panel() -> None:
    """Show dataset provenance and class balance."""
    with st.expander("Dataset and class balance"):
        try:
            summary = _dataset_summary()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        metric_row(
            [
                ("Tweets", f"{summary['rows']:,}", ""),
                ("Classes", str(len(summary["classes"])), ", ".join(summary["classes"])),
                (
                    "Min confidence",
                    f"{summary['min_confidence']:.1f}",
                    "annotator label confidence floor",
                ),
            ]
        )
        counts = summary["class_counts"]
        show_figure(
            bar_chart(
                list(counts),
                [counts[label] for label in counts],
                title="Class distribution in the corpus",
                ylabel="Tweets",
                width=7.0,
                height=3.2,
            )
        )
        st.markdown(f"**Source:** `{TWEETS_SPEC.url}`")
        st.caption(
            "This is a public, legally redistributable corpus. The project deliberately does "
            "not scrape Twitter/X, because that requires an unreliable unofficial method."
        )


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Twitter/X Sentiment Analysis", page_icon="02", layout="wide")
    inject_theme()

    page_header(
        "Twitter/X Sentiment Analysis",
        "Three-class sentiment classification of tweets using TF-IDF features and linear "
        "classifiers, with a proper preprocessing pipeline and honest evaluation.",
        eyebrow="Project 02",
        chips=["NLP / Classification", "scikit-learn", "TF-IDF", "Three classes"],
    )

    options = render_sidebar()

    model = _get_model()
    if options["train_clicked"]:
        if run_training():
            model = _get_model()
            st.rerun()

    if model is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Click **Train / retrain models** in the sidebar, or run:\n\n"
            "```bash\npython train.py\n```"
        )
        render_dataset_panel()
        note(DISCLAIMER)
        return

    render_prediction(model, options)
    render_evaluation(model)
    render_insights(model)
    render_dataset_panel()

    model_info_panel(
        algorithm=MODEL_LABELS.get(model.model_name, model.model_name),
        trained_on=f"{model.n_train:,} tweets (20% held out, stratified)",
        metrics=model.metrics,
        extra={
            "Trained at": model.trained_at,
            "Classes": ", ".join(model.labels),
            "Confidence": (
                "calibrated probability"
                if model.supports_probabilities
                else "normalised decision score"
            ),
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()