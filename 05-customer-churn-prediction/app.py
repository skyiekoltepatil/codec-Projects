"""Streamlit interface for Customer Churn Prediction.

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
from shared.theme import (  # noqa: E402
    PALETTE,
    RISK_COLORS,
    inject_theme,
    metric_row,
    note,
    page_header,
    result_card,
)
from shared.ui import (  # noqa: E402
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
    show_metrics_dict,
)

from src.data import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FIELD_METADATA,
    NUMERIC_COLUMNS,
    POSITIVE_LABEL,
    build_customer_input,
    load_dataset,
)
from src.model import (  # noqa: E402
    CLASS_LABELS,
    MODEL_LABELS,
    RISK_BANDS,
    ChurnModel,
    build_training_report,
    load_model,
    risk_band,
    save_metrics_report,
    save_model,
    select_best_model,
    train_all_models,
    train_model,
)

MODEL_PATH = models_dir("05-customer-churn-prediction") / "churn_model.joblib"

#: A realistic-looking default customer, so a reviewer can click once and see output.
DEFAULT_CUSTOMER: dict[str, object] = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0,
}

DISCLAIMER = (
    "A churn probability describes how similar this customer is to others who left, "
    "not whether this individual will leave. Retention offers, price changes or account "
    "closures should never be automated on the basis of this score alone."
)


@st.cache_resource(show_spinner=False)
def _load_churn_model(path_str: str, mtime: float) -> ChurnModel:
    """Load the trained model once per artefact version."""
    return load_model(Path(path_str))


def _get_model() -> ChurnModel | None:
    """Return the trained model, or ``None`` when it has not been trained."""
    if not MODEL_PATH.exists():
        return None
    return _load_churn_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _collect_customer_form() -> dict[str, object]:
    """Render the customer input form and return the collected values."""
    st.markdown(
        "Enter a customer's details. Every field matches a column of the training "
        "dataset, so the values map onto the model exactly."
    )

    values: dict[str, object] = {}

    # Four columns per row keeps the form compact without cramping the labels.
    categorical = [column for column in FIELD_METADATA if column in CATEGORICAL_COLUMNS or column == "SeniorCitizen"]
    numeric = [column for column in FIELD_METADATA if column in NUMERIC_COLUMNS]

    row = st.columns(4)
    for index, column in enumerate(categorical + numeric):
        target = row[index % 4]
        meta = FIELD_METADATA[column]
        with target:
            if column == "SeniorCitizen":
                values[column] = st.selectbox(
                    meta["label"], ["No", "Yes"], index=["No", "Yes"].index(str(DEFAULT_CUSTOMER[column])),
                    key=f"f_{column}",
                )
            elif column in NUMERIC_COLUMNS:
                bounds = meta
                values[column] = st.number_input(
                    meta["label"],
                    min_value=float(bounds["min"]),
                    max_value=float(bounds["max"]),
                    value=float(DEFAULT_CUSTOMER[column]),
                    step=float(bounds.get("step", 1.0)),
                    key=f"f_{column}",
                )
            else:
                options = list(CATEGORICAL_COLUMNS[column])
                default = str(DEFAULT_CUSTOMER[column])
                values[column] = st.selectbox(
                    meta["label"], options, index=options.index(default), key=f"f_{column}"
                )

    return values


def render_prediction(model: ChurnModel) -> None:
    """Render the customer form and the resulting risk assessment."""
    section("Predict churn risk", "Fill in the customer details, then estimate their churn risk.")

    values = _collect_customer_form()

    left, right = st.columns([1, 1])
    with left:
        predict = st.button("Estimate churn risk", type="primary", width="stretch")
    with right:
        if st.button("Reset to defaults", width="stretch"):
            for key in [f"f_{column}" for column in FIELD_METADATA]:
                st.session_state.pop(key, None)
            st.rerun()

    if not predict:
        st.caption("Press **Estimate churn risk** to run the model.")
        return

    try:
        row = build_customer_input(values)
        outcome = model.predict_customer(row)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return

    result_card(
        f"Churn risk: {outcome['risk']}",
        f"{outcome['probability']:.1%} probability of churning",
        accent=RISK_COLORS.get(outcome["risk"], PALETTE["accent"]),
    )
    st.caption(
        f"At the 0.50 decision threshold this customer is classified as "
        f"**{'likely to churn' if outcome['label'] == POSITIVE_LABEL else 'likely to stay'}**. "
        "The risk band is the number retention teams would actually act on."
    )

    show_figure(
        bar_chart(
            ["Retained", "Churn"],
            [1.0 - outcome["probability"], outcome["probability"]],
            title="Predicted outcome",
            ylabel="Probability",
            width=6.0,
            height=2.8,
        )
    )

    st.markdown("**Risk bands used**")
    show_dataframe(
        [
            {"band": label, "churn probability at or above": f"{threshold:.0%}"}
            for threshold, label in RISK_BANDS
        ],
        caption="Fixed project thresholds, so the band does not move when the model is retrained.",
    )

    render_factors(model, row)


def render_factors(model: ChurnModel, row: pd.DataFrame) -> None:
    """Show per-customer drivers when the model supports them, else global importance."""
    factors = model.top_factors(row)

    if factors:
        section(
            "What drove this prediction",
            "Each row is a feature's contribution to this customer's log-odds of churning. "
            "A positive number pushed the score up; a negative one pushed it down.",
        )
        show_dataframe(
            [
                {
                    "feature": item["feature"],
                    "value": item["value"],
                    "contribution to log-odds": round(item["contribution"], 3),
                    "effect": item["direction"],
                }
                for item in factors
            ],
            caption="Exact contributions come from the logistic model's coefficients.",
        )
        return

    st.info(
        f"{MODEL_LABELS.get(model.model_name, model.model_name)} is a tree ensemble, which has "
        "no per-customer decomposition of its score. The panel below shows the model's "
        "**average** influence of each feature across all customers; it describes the model "
        "globally, not this specific customer."
    )
    if model.feature_importance:
        with st.expander("Global feature importance", expanded=True):
            show_dataframe(
                [
                    {"feature": item["feature"], "importance": round(item["importance"], 4)}
                    for item in model.feature_importance[:12]
                ],
                caption="Mean decrease in impurity across all trees.",
            )


def render_evaluation(model: ChurnModel) -> None:
    """Render held-out metrics, per-class table and confusion matrix."""
    metrics = model.metrics
    if not metrics:
        return

    section(
        "Held-out evaluation",
        f"Measured on the {metrics.get('n_test', 0):,} customers the model never saw during "
        "training. Precision, recall and F1 refer to the churn class.",
    )

    show_metrics_dict(
        metrics,
        keys=["accuracy", "precision", "recall", "f1"],
        labels={
            "accuracy": "ACCURACY (ALL)",
            "precision": "PRECISION (CHURNED)",
            "recall": "RECALL (CHURNED)",
            "f1": "F1 (CHURNED)",
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
        f"Accuracy looks modest because only {1 - metrics.get('pos_rate', 0.265):.0%} of customers "
        "stay. A model that always predicted 'stays' would score about 73.5% accuracy while "
        "catching nobody, which is why ROC-AUC and churn-class recall are the figures to read."
    )


def render_dataset_panel() -> None:
    """Show dataset provenance and class balance."""
    with st.expander("Dataset details"):
        try:
            with st.spinner("Loading the churn dataset..."):
                summary = load_dataset().summary()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        metric_row(
            [
                ("Customers", f"{summary['rows']:,}", "after cleaning"),
                ("Churn rate", f"{summary['churn_rate']:.1%}", "positive class"),
                ("Features", str(summary["features"]), "after dropping identifiers"),
                ("Blank TotalCharges", str(summary["blank_total_charges"]), "median-imputed"),
            ]
        )
        counts = summary["class_counts"]
        show_figure(
            bar_chart(
                CLASS_LABELS,
                [counts.get(label, 0) for label in CLASS_LABELS],
                title="Class balance",
                ylabel="Customers",
                width=6.0,
                height=3.0,
            )
        )
        st.markdown(f"**Source:** `{summary['source']}`")
        st.caption(
            "`TotalCharges` arrives as text with 11 blanks (customers who never signed up). "
            "Those rows are coerced to NaN and imputed with the training-set median inside "
            "the pipeline, so no validation statistics ever leak into training."
        )


def run_training() -> bool:
    """Train and persist the model. Returns ``True`` on success."""
    try:
        with st.spinner("Loading the dataset and training three classifiers..."):
            data = load_dataset()
            comparison = train_all_models(data.frame)
            selected = select_best_model(comparison)
            best = train_model(data.frame, selected)
            save_model(best)
            report = build_training_report(best, comparison, dataset=data.summary(), selected=selected)
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(
        f"Trained {MODEL_LABELS.get(selected, selected)} with ROC-AUC "
        f"{format_metric(best.metrics.get('roc_auc'))} and saved it to "
        f"`{portable_display(MODEL_PATH)}`."
    )
    st.cache_resource.clear()
    return True


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="05", layout="wide")
    inject_theme()

    page_header(
        "Customer Churn Prediction",
        "Estimate the probability that a telecom subscriber will cancel, using a leakage-safe "
        "preprocessing pipeline and three comparable classifiers.",
        eyebrow="Project 05",
        chips=["Classification", "scikit-learn", "ROC-AUC", "7,043 customers"],
    )

    if st.sidebar.button("Train / retrain", width="stretch"):
        st.sidebar.caption("Compares Logistic Regression, Random Forest and Gradient Boosting.")
        if run_training():
            st.rerun()

    model = _get_model()
    if model is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Train it by running:\n\n"
            "```bash\npython train.py\n```"
        )
        render_dataset_panel()
        note(DISCLAIMER)
        return

    render_prediction(model)
    render_evaluation(model)
    render_dataset_panel()

    metrics = model.metrics or {}
    model_info_panel(
        algorithm=MODEL_LABELS.get(model.model_name, model.model_name),
        trained_on=f"{model.n_train:,} customers (20% held out, stratified)",
        metrics=metrics,
        extra={
            "Trained at": model.trained_at,
            "Target": f"{POSITIVE_LABEL} = churned ({metrics.get('pos_rate', 0.265):.1%} of customers)",
            "Preprocessing": "median imputation + standardisation (numeric), one-hot (categorical)",
            "Selection": "highest ROC-AUC",
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()