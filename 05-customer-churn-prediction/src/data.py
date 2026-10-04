"""IBM Telco Customer Churn: download, cleaning and schema.

Dataset
-------
The **IBM Telco Customer Churn** dataset: 7,043 subscribers, 20 features and a
binary ``Churn`` label (26.5% positive). It is mirrored in IBM's public
``telco-customer-churn-on-icp4d`` repository, which is stable, key-free and does
not require accepting a licence click-through like some Kaggle mirrors do.

Cleaning that actually matters here
----------------------------------
``TotalCharges`` is the notorious one. It is a *string* column in the raw CSV
because the dataset contains 11 rows where the customer never signed up and the
field is blank. A naive ``pd.to_numeric`` without ``errors="coerce"`` raises, and
a naive cast without handling blanks fills those rows with the year 0, which
produces a wildly out-of-range value that then skews a scaler. This module makes
the coercion explicit, converts the failures to ``NaN`` and imputes them with the
**median** computed on the training data.

``CustomerID`` and ``TotalCharges``-derived year are dropped: an identifier is not
a predictive feature, and keeping it would let the model memorise customers
instead of learning behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.datasets import DatasetSpec
from shared.errors import DataNotFoundError, InvalidInputError
from shared.paths import data_dir

logger = logging.getLogger(__name__)

PROJECT_SLUG = "05-customer-churn-prediction"

#: Raw CSV column names, used so the app form matches the dataset exactly.
COLUMNS: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)

#: The target column in the raw CSV.
TARGET_COLUMN = "Churn"

#: Label values used in the cleaned dataset.
NEGATIVE_LABEL = "No"
POSITIVE_LABEL = "Yes"

#: Numeric features that must be coerced from strings.
NUMERIC_COLUMNS: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")

#: Columns dropped before modelling, with the reason.
DROPPED_COLUMNS: dict[str, str] = {
    "CustomerID": "A unique identifier is not a behavioural feature; keeping it invites memorisation.",
    "year": "Derived from TotalCharges; the cleaned numeric column carries the same signal.",
}

#: Categorical features, each mapped to its allowed values in the interface.
CATEGORICAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": ("No", "Yes"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("No", "Yes"),
    "MultipleLines": ("No", "No phone service", "Yes"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("No", "No internet service", "Yes"),
    "OnlineBackup": ("No", "No internet service", "Yes"),
    "TechSupport": ("No", "No internet service", "Yes"),
    "StreamingTV": ("No", "No internet service", "Yes"),
    "StreamingMovies": ("No", "No internet service", "Yes"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("No", "Yes"),
    "PaymentMethod": ("Bank transfer", "Credit card", "Electronic check", "Mailed check"),
}

#: Human-readable labels and ranges for the interface's input form.
FIELD_METADATA: dict[str, dict[str, Any]] = {
    "gender": {"label": "Gender", "type": "choice"},
    "SeniorCitizen": {"label": "Is a senior citizen?", "type": "bool"},
    "Partner": {"label": "Has a partner?", "type": "choice"},
    "Dependents": {"label": "Has dependents?", "type": "choice"},
    "tenure": {"label": "Tenure (months)", "type": "number", "min": 0, "max": 120, "step": 1},
    "PhoneService": {"label": "Has a phone line?", "type": "choice"},
    "MultipleLines": {"label": "Multiple lines?", "type": "choice"},
    "InternetService": {"label": "Internet service", "type": "choice"},
    "OnlineSecurity": {"label": "Online security?", "type": "choice"},
    "OnlineBackup": {"label": "Online backup?", "type": "choice"},
    "TechSupport": {"label": "Tech support?", "type": "choice"},
    "StreamingTV": {"label": "Streaming TV?", "type": "choice"},
    "StreamingMovies": {"label": "Streaming movies?", "type": "choice"},
    "Contract": {"label": "Contract type", "type": "choice"},
    "PaperlessBilling": {"label": "Paperless billing?", "type": "choice"},
    "PaymentMethod": {"label": "Payment method", "type": "choice"},
    "MonthlyCharges": {"label": "Monthly charges", "type": "number", "min": 0.0, "max": 120.0},
    "TotalCharges": {"label": "Total charges", "type": "number", "min": 0.0, "max": 10000.0},
}

TELCO_SPEC = DatasetSpec(
    name="IBM Telco Customer Churn",
    url=(
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    ),
    filename="Telco-Customer-Churn.csv",
    mirrors=[
        (
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv"
        )
    ],
    approx_size_mb=0.9,
    notes="Hosted in IBM's public ICP4D sample repository; no login required.",
)


@dataclass
class ChurnData:
    """A cleaned customer table ready for preprocessing."""

    frame: pd.DataFrame
    n_raw_rows: int = 0
    n_blank_total_charges: int = 0

    @property
    def n_rows(self) -> int:
        """Number of usable customers."""
        return int(len(self.frame))

    def class_counts(self) -> dict[str, int]:
        """Return the churn label distribution."""
        counts = self.frame[TARGET_COLUMN].value_counts()
        return {str(key): int(value) for key, value in counts.items()}

    def summary(self) -> dict[str, Any]:
        """Return headline statistics for the dataset panel."""
        counts = self.class_counts()
        # The label is text, so the rate is computed from the counts rather than
        # with Series.mean(), which pandas 3.x rejects for string dtype.
        positives = counts.get(POSITIVE_LABEL, 0)
        return {
            "rows": self.n_rows,
            "raw_rows": int(self.n_raw_rows or self.n_rows),
            "features": len([column for column in COLUMNS if column in self.frame.columns]),
            "churn_rate": (positives / self.n_rows) if self.n_rows else 0.0,
            "class_counts": counts,
            "blank_total_charges": int(self.n_blank_total_charges),
            "source": TELCO_SPEC.url,
        }


def download_dataset(*, force: bool = False) -> Path:
    """Download the Telco CSV into the project's ``data`` directory."""
    from shared.datasets import fetch_dataset

    return fetch_dataset(TELCO_SPEC, data_dir(PROJECT_SLUG), force=force)


def load_dataset(*, force_download: bool = False) -> ChurnData:
    """Download (if needed) and clean the Telco churn dataset.

    Returns:
        A :class:`ChurnData` whose frame has coerced numeric columns, a
        ``year`` helper column and a string ``Churn`` label.

    Raises:
        DataNotFoundError: When required columns are missing.
    """
    from shared.datasets import require_local_file

    path = require_local_file(
        download_dataset(force=force_download),
        description="Telco churn CSV",
        instructions=f"Download it manually from {TELCO_SPEC.url} and place it in data/.",
    )

    raw = pd.read_csv(path)
    missing = [column for column in (*COLUMNS, TARGET_COLUMN) if column not in raw.columns]
    if missing:
        raise DataNotFoundError(
            f"The churn CSV is missing column(s): {', '.join(missing)}.",
            hint=f"Expected columns {list(COLUMNS)} plus '{TARGET_COLUMN}'.",
        )

    n_raw = len(raw)

    # 'TotalCharges' holds blanks for customers who never signed up.
    blanks = int(raw["TotalCharges"].astype(str).str.strip().isin({"", "nan"}).sum())
    for column in NUMERIC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    # SeniorCitizen arrives as 0/1 but describes a yes/no question.
    raw["SeniorCitizen"] = raw["SeniorCitizen"].astype(int).map({0: "No", 1: "Yes"})

    for column in ("Partner", "Dependents"):
        raw[column] = raw[column].astype(str).str.strip().replace({"No": "No", "Yes": "Yes"})

    raw["year"] = pd.to_datetime(raw["TotalCharges"], errors="coerce").dt.year

    cleaned = raw.loc[:, [*COLUMNS, TARGET_COLUMN, "year"]].copy()

    invalid_target = cleaned[cleaned[TARGET_COLUMN].astype(str).str.strip().isin({"", "nan"})]
    if len(invalid_target):
        logger.warning("Dropping %d row(s) with a missing Churn label.", len(invalid_target))
        cleaned = cleaned[cleaned[TARGET_COLUMN].astype(str).str.strip().ne("")]

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(str).str.strip()
    cleaned = cleaned.reset_index(drop=True)

    logger.info(
        "Loaded Telco churn data: %d customers (%d had a blank TotalCharges)",
        len(cleaned),
        blanks,
    )
    return ChurnData(frame=cleaned, n_raw_rows=n_raw, n_blank_total_charges=blanks)


def build_customer_input(values: dict[str, Any]) -> pd.DataFrame:
    """Turn a dict of interface values into a single validated customer row.

    The resulting frame has exactly the feature columns in :data:`COLUMNS` order,
    so it can be handed straight to the fitted pipeline.

    Raises:
        InvalidInputError: When a required field is missing or out of range.
    """
    if not isinstance(values, dict):
        raise InvalidInputError(
            "Expected a dictionary of customer fields.",
            hint="Pass {column: value} pairs keyed by the dataset column names.",
        )

    row: dict[str, Any] = {}
    for column in COLUMNS:
        if column not in values:
            raise InvalidInputError(
                f"Missing customer field '{column}'.",
                hint=f"Supply all {len(COLUMNS)} features.",
            )
        value = values[column]

        if column in NUMERIC_COLUMNS:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise InvalidInputError(
                    f"'{column}' must be a number, got {value!r}.",
                    hint="Enter a numeric value, for example 24.",
                ) from error
            if not np.isfinite(number):
                raise InvalidInputError(
                    f"'{column}' must be a finite number.",
                    hint="Check for an empty or non-numeric entry.",
                )
            bounds = FIELD_METADATA.get(column, {})
            low, high = bounds.get("min"), bounds.get("max")
            if low is not None and number < low:
                raise InvalidInputError(
                    f"'{column}' must be at least {low}.",
                    hint=f"You entered {number}.",
                )
            if high is not None and number > high:
                raise InvalidInputError(
                    f"'{column}' must be at most {high}.",
                    hint=f"You entered {number}.",
                )
            row[column] = number
            continue

        text = str(value).strip()
        allowed = CATEGORICAL_COLUMNS.get(column)
        if allowed and text not in allowed:
            raise InvalidInputError(
                f"'{column}' must be one of: {', '.join(allowed)}.",
                hint=f"You entered '{text}'.",
            )
        row[column] = text

    return pd.DataFrame([row], columns=list(COLUMNS))