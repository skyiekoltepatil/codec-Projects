"""Tests for Project 05 - Customer Churn Prediction.

Most tests use a small synthetic table so the suite stays fast and needs no
network. Tests that exercise the real CSV skip when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_env import REPO_ROOT, use_project

use_project("05-customer-churn-prediction")

sys.path.insert(0, str(REPO_ROOT))

from shared.errors import InvalidInputError  # noqa: E402
from src.data import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    COLUMNS,
    NUMERIC_COLUMNS,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    build_customer_input,
    load_dataset,
)
from src.model import (  # noqa: E402
    CLASS_LABELS,
    RISK_BANDS,
    ChurnModel,
    build_pipeline,
    build_training_report,
    load_model,
    risk_band,
    save_model,
    select_best_model,
    train_all_models,
    train_model,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _valid_customer_values() -> dict[str, object]:
    """Return a complete, valid customer payload used as a base for mutation tests."""
    values: dict[str, object] = {column: None for column in COLUMNS}
    for column, options in CATEGORICAL_COLUMNS.items():
        values[column] = options[0]
    values["SeniorCitizen"] = "No"
    values["tenure"] = 10
    values["MonthlyCharges"] = 50.0
    values["TotalCharges"] = 500.0
    return values


def _synthetic_customers(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic customer table with a genuine, learnable churn signal.

    Month-to-month contracts on electronic check, short tenure and high monthly
    charges increase the churn probability. The relationship is generated as a
    logistic function of the features and then sampled, so a model can actually
    recover it and the tests below assert real behaviour rather than a fixture
    quirk.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        tenure = int(rng.integers(1, 72))
        monthly = float(np.round(rng.uniform(20, 100), 2))
        contract = str(rng.choice(["Month-to-month", "One year", "Two year"]))
        payment = str(rng.choice(["Bank transfer", "Credit card", "Electronic check", "Mailed check"]))
        internet = str(rng.choice(["DSL", "Fiber optic", "No"]))
        has_security = str(rng.choice(["No", "No internet service", "Yes"]))

        log_odds = (
            -2.4
            + 1.6 * (contract == "Month-to-month")
            + 0.9 * (payment == "Electronic check")
            + 1.0 * (internet == "Fiber optic")
            - 0.6 * (has_security == "Yes")
            - 0.02 * tenure
            + 0.012 * monthly
        )
        churned = POSITIVE_LABEL if rng.random() < 1.0 / (1.0 + np.exp(-log_odds)) else "No"

        rows.append(
            {
                "gender": str(rng.choice(["Female", "Male"])),
                "SeniorCitizen": str(rng.choice(["No", "Yes"])),
                "Partner": str(rng.choice(["No", "Yes"])),
                "Dependents": str(rng.choice(["No", "Yes"])),
                "tenure": tenure,
                "PhoneService": str(rng.choice(["No", "Yes"])),
                "MultipleLines": str(rng.choice(["No", "No phone service", "Yes"])),
                "InternetService": internet,
                "OnlineSecurity": has_security,
                "OnlineBackup": str(rng.choice(["No", "No internet service", "Yes"])),
                "TechSupport": str(rng.choice(["No", "No internet service", "Yes"])),
                "StreamingTV": str(rng.choice(["No", "No internet service", "Yes"])),
                "StreamingMovies": str(rng.choice(["No", "No internet service", "Yes"])),
                "Contract": contract,
                "PaperlessBilling": str(rng.choice(["No", "Yes"])),
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": float(np.round(monthly * tenure, 2)),
                TARGET_COLUMN: churned,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_customers() -> pd.DataFrame:
    """Return the synthetic customer table."""
    return _synthetic_customers()


def _real_csv_available() -> bool:
    """Return ``True`` when the Telco CSV has been downloaded."""
    return (
        REPO_ROOT / "05-customer-churn-prediction" / "data" / "Telco-Customer-Churn.csv"
    ).exists()


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_build_customer_input_returns_one_row_in_schema_order():
    """A complete dict becomes a single row with exactly the feature columns."""
    values = {
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
    frame = build_customer_input(values)
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == 1
    assert frame.loc[0, "Contract"] == "Month-to-month"


def test_build_customer_input_rejects_missing_field():
    """A missing field is named in the error message."""
    values = _valid_customer_values()
    del values["tenure"]
    with pytest.raises(InvalidInputError) as error:
        build_customer_input(values)
    assert "tenure" in str(error.value)


def test_build_customer_input_rejects_non_numeric_and_out_of_range():
    """Numeric fields must be finite and within documented bounds."""
    values = _valid_customer_values()

    values["tenure"] = "twelve months"
    with pytest.raises(InvalidInputError) as error:
        build_customer_input(values)
    assert "number" in str(error.value)

    values["tenure"] = 9999
    with pytest.raises(InvalidInputError) as error:
        build_customer_input(values)
    assert "at most" in str(error.value)

    values["tenure"] = -5
    with pytest.raises(InvalidInputError) as error:
        build_customer_input(values)
    assert "at least" in str(error.value)


def test_build_customer_input_rejects_unknown_category():
    """A categorical value outside the dataset's vocabulary is refused."""
    values = _valid_customer_values()
    values["Contract"] = "Lifetime deal"
    with pytest.raises(InvalidInputError) as error:
        build_customer_input(values)
    assert "Contract" in str(error.value)


def test_build_customer_input_rejects_non_dict():
    """A non-dictionary input is reported clearly."""
    with pytest.raises(InvalidInputError):
        build_customer_input(["gender", "Female"])


# --------------------------------------------------------------------------- #
# Risk banding
# --------------------------------------------------------------------------- #


def test_risk_bands_are_monotonic_and_complete():
    """Every probability maps to exactly one band, and bands never overlap."""
    assert risk_band(0.0) == "LOW"
    assert risk_band(0.149) == "LOW"
    assert risk_band(0.15) == "MEDIUM"
    assert risk_band(0.399) == "MEDIUM"
    assert risk_band(0.40) == "HIGH"
    assert risk_band(0.999) == "HIGH"

    thresholds = [threshold for threshold, _ in RISK_BANDS]
    assert thresholds == sorted(thresholds, reverse=True)


def test_risk_band_monotonic_across_range():
    """Raising the probability never lowers the band."""
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    previous = -1
    for step in range(0, 101):
        current = order[risk_band(step / 100)]
        assert current >= previous
        previous = current


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_fits_and_predicts_valid_probabilities(sample_customers):
    """A fitted pipeline returns a calibrated probability per customer."""
    model = train_model(sample_customers, "logreg")
    row = build_customer_input(
        {column: sample_customers.loc[0, column] for column in COLUMNS}
    )
    outcome = model.predict_customer(row)

    assert 0.0 <= outcome["probability"] <= 1.0
    assert outcome["label"] in CLASS_LABELS
    assert outcome["risk"] in {"LOW", "MEDIUM", "HIGH"}


def test_logreg_learns_the_synthetic_signal(sample_customers):
    """The model must beat the majority-class baseline on held-out data.

    A pipeline that silently returned the majority class would also post a high
    accuracy, so ROC-AUC is the assertion: it collapses to 0.5 for a constant
    predictor no matter how imbalanced the data is.
    """
    model = train_model(sample_customers, "logreg")
    assert model.metrics["roc_auc"] > 0.7
    assert model.metrics["n_test"] == int(len(sample_customers) * 0.2)


def test_pipeline_handles_missing_numeric_values():
    """Blank TotalCharges is imputed inside the pipeline, not at inference time."""
    frame = _synthetic_customers(n=600)
    frame.loc[0:4, "TotalCharges"] = np.nan

    model = train_model(frame, "logreg")
    row = frame.loc[[0], list(COLUMNS)].copy()
    row["TotalCharges"] = np.nan
    outcome = model.predict_customer(row)
    assert np.isfinite(outcome["probability"])


def test_pipeline_accepts_unseen_category():
    """A category never seen during training is encoded, not an exception.

    The row is built directly rather than through ``build_customer_input``,
    because that function deliberately rejects unknown categories. This test
    covers the separate layer below it: the ``OneHotEncoder`` is configured with
    ``handle_unknown="ignore"``, so an unexpected value encodes as all-zero
    instead of raising.
    """
    model = train_model(_synthetic_customers(n=600), "logreg")
    row = build_customer_input(_valid_customer_values()).assign(
        Contract="Lifetime deal", PaymentMethod="Cryptocurrency"
    )
    outcome = model.predict_customer(row)
    assert 0.0 <= outcome["probability"] <= 1.0


def test_top_factors_only_available_for_linear_models(sample_customers):
    """Logistic Regression explains a single customer; a forest reports nothing."""
    linear = train_model(sample_customers, "logreg")
    row = build_customer_input({column: sample_customers.loc[0, column] for column in COLUMNS})

    factors = linear.top_factors(row)
    assert factors, "expected contributions from a linear model"
    assert all("contribution" in item for item in factors)
    # Contributions are reported largest-magnitude first.
    magnitudes = [abs(item["contribution"]) for item in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)

    forest = train_model(sample_customers, "rf")
    assert forest.top_factors(row) == []


def test_global_importance_is_populated(sample_customers):
    """Every estimator exposes a global importance ranking."""
    for name in ("logreg", "rf", "gb"):
        model = train_model(sample_customers, name)
        assert model.feature_importance, f"no importance reported for {name}"
        values = [item["importance"] for item in model.feature_importance]
        assert values == sorted(values, reverse=True)
        assert all(value >= 0 for value in values)


def test_training_rejects_bad_input():
    """Missing target and too-few rows are reported, not silently accepted."""
    with pytest.raises(InvalidInputError):
        train_model(pd.DataFrame({column: [1] * 200 for column in COLUMNS}), "logreg")
    with pytest.raises(InvalidInputError):
        train_model(_synthetic_customers(n=5), "logreg")
    with pytest.raises(InvalidInputError):
        build_pipeline("telepathy")


# --------------------------------------------------------------------------- #
# Comparison and persistence
# --------------------------------------------------------------------------- #


def test_train_all_models_compares_every_candidate(sample_customers):
    """All three classifiers are compared on the same split."""
    comparison = train_all_models(sample_customers)
    assert set(comparison) == {"logreg", "rf", "gb"}
    for payload in comparison.values():
        assert payload["metrics"]["n_test"] == comparison["logreg"]["metrics"]["n_test"]


def test_selection_prefers_roc_auc(sample_customers):
    """Selection follows ROC-AUC, not accuracy, on an imbalanced target."""
    comparison = train_all_models(sample_customers)
    selected = select_best_model(comparison)
    best_roc = max(payload["metrics"]["roc_auc"] for payload in comparison.values())
    assert comparison[selected]["metrics"]["roc_auc"] == best_roc


def test_selection_falls_back_when_roc_missing():
    """A comparison without ROC-AUC still selects a model, by F1."""
    comparison = {
        "logreg": {"label": "Logistic Regression", "metrics": {"f1": 0.9}},
        "rf": {"label": "Random Forest", "metrics": {"f1": 0.7}},
    }
    assert select_best_model(comparison) == "logreg"


def test_save_load_round_trip(sample_customers, tmp_path):
    """A saved model reloads and predicts identically."""
    model = train_model(sample_customers, "logreg")
    path = save_model(model, directory=tmp_path)
    reloaded = load_model(path)

    row = build_customer_input({column: sample_customers.loc[0, column] for column in COLUMNS})
    assert reloaded.predict_customer(row) == model.predict_customer(row)
    assert reloaded.model_name == model.model_name
    assert reloaded.metrics["roc_auc"] == model.metrics["roc_auc"]


def test_load_missing_model_explains_remedy(tmp_path):
    """A missing artefact names the command that creates it."""
    with pytest.raises(Exception) as error:
        load_model(tmp_path / "absent.joblib")
    assert "train.py" in (error.value.hint or "")


def test_load_rejects_foreign_artefact(tmp_path):
    """A joblib file holding the wrong object is reported clearly."""
    from shared.errors import ModelNotFoundError
    from shared.ml import save_joblib

    save_joblib({"not": "a model"}, tmp_path / "churn_model.joblib")
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "churn_model.joblib")


def test_training_report_is_json_serialisable(sample_customers):
    """The written report contains the measured numbers."""
    import json

    model = train_model(sample_customers, "logreg")
    comparison = train_all_models(sample_customers)
    report = build_training_report(
        model, comparison, dataset={"rows": len(sample_customers)}, selected="logreg"
    )
    payload = json.loads(json.dumps(report))
    assert payload["project"] == "05-customer-churn-prediction"
    assert payload["selection_rule"].startswith("highest ROC-AUC")
    assert payload["metrics"]["roc_auc"] == model.metrics["roc_auc"]


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _real_csv_available(), reason="Telco CSV absent; run `python train.py`.")
def test_real_dataset_cleans_total_charges():
    """TotalCharges becomes numeric and the churn rate matches the published 26.5%."""
    data = load_dataset()
    assert data.n_rows == 7043
    assert pd.api.types.is_numeric_dtype(data.frame["TotalCharges"])
    assert data.n_blank_total_charges == 11

    summary = data.summary()
    assert 0.25 < summary["churn_rate"] < 0.28
    assert set(summary["class_counts"]) == {"No", POSITIVE_LABEL}


@pytest.mark.skipif(not _real_csv_available(), reason="Telco CSV absent; run `python train.py`.")
def test_real_dataset_has_no_leaked_identifier():
    """CustomerID is dropped before modelling."""
    data = load_dataset()
    assert "CustomerID" not in data.frame.columns
    assert "year" in data.frame.columns  # kept only as a derived helper


@pytest.mark.skipif(
    not (REPO_ROOT / "05-customer-churn-prediction" / "models" / "churn_model.joblib").exists(),
    reason="Trained artefact absent; run `python train.py` first.",
)
def test_app_renders_without_exception():
    """The Streamlit page executes top to bottom without raising."""
    from streamlit.testing.v1 import AppTest

    project_dir = REPO_ROOT / "05-customer-churn-prediction"
    app = AppTest.from_file(str(project_dir / "app.py"), default_timeout=240)
    app.run()
    assert not app.exception