"""Dataset acquisition and loading for Twitter/X Sentiment Analysis.

Dataset decision (documented for the reviewer)
----------------------------------------------
Scraping Twitter/X directly requires an unreliable, unofficial method, so this
project deliberately does **not** do that. Instead it uses a public, legally
redistributable dataset of real tweets:

    "Twitter US Airline Sentiment" - 14,640 tweets about six US airlines,
    each hand-labelled as positive, negative or neutral, with a
    confidence score.

The copy used here is the widely mirrored ``Tweets.csv`` (3.4 MB). It is a real
three-class sentiment corpus, which is exactly what this project needs: the
neutral class is what makes the task non-trivial compared with the usual binary
movie-review datasets.

Only the ``text`` and ``airline_sentiment`` columns are used. Rows whose label
confidence is below the configured threshold are dropped, because a low-confidence
label is closer to noise than to ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.datasets import DatasetSpec, fetch_dataset
from shared.errors import DataNotFoundError, InvalidInputError
from shared.paths import data_dir

PROJECT_SLUG = "02-twitter-sentiment-analysis"

#: The tweet dataset. Mirrored on a public GitHub repository.
TWEETS_SPEC = DatasetSpec(
    name="Twitter US Airline Sentiment (14,640 labelled tweets)",
    url="https://raw.githubusercontent.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/master/Tweets.csv",
    filename="tweets.csv",
    mirrors=[
        "https://cdn.jsdelivr.net/gh/ruchitgandhi/Twitter-Airline-Sentiment-Analysis@master/Tweets.csv",
    ],
    approx_size_mb=3.4,
    notes="Real tweets with hand-labelled sentiment and a per-row label confidence.",
)

#: Column holding the tweet text.
TEXT_COLUMN = "text"

#: Column holding the sentiment label.
LABEL_COLUMN = "airline_sentiment"

#: Column holding the annotator confidence in the label.
CONFIDENCE_COLUMN = "airline_sentiment_confidence"

#: Mapping from the dataset's labels to the display labels used by the model.
LABEL_MAP: dict[str, str] = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}

#: Display labels in a fixed order, used for consistent metric reporting.
CLASS_LABELS: list[str] = ["Negative", "Neutral", "Positive"]

#: Minimum annotator confidence for a row to be kept.
DEFAULT_MIN_CONFIDENCE = 0.6

#: Columns required from the source CSV.
REQUIRED_COLUMNS: list[str] = [TEXT_COLUMN, LABEL_COLUMN]


def get_data_directory() -> Path:
    """Return (and create) this project's ``data/`` directory."""
    return data_dir(PROJECT_SLUG)


def download_dataset(*, force: bool = False) -> Path:
    """Download the tweet dataset into ``data/`` and return its path."""
    return fetch_dataset(TWEETS_SPEC, get_data_directory(), force=force)


def load_dataset(
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load the labelled tweet corpus.

    Args:
        min_confidence: Drop rows whose annotator confidence is below this value.
            Set to ``0.0`` to keep every row.
        auto_download: Download the CSV when it is not present locally.

    Returns:
        DataFrame with columns ``["text", "label"]``, duplicate texts removed and
        labels restricted to :data:`CLASS_LABELS`.

    Raises:
        DataNotFoundError: When the dataset is unavailable or malformed.
        InvalidInputError: When filtering leaves no usable rows.
    """
    directory = get_data_directory()
    path = directory / TWEETS_SPEC.filename
    if not path.exists() and auto_download:
        path = fetch_dataset(TWEETS_SPEC, directory)

    if not path.exists():
        raise DataNotFoundError(
            f"Dataset '{path.name}' not found.",
            hint="Run `python src/data.py` to download it, or place Tweets.csv in the data/ directory.",
        )

    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        raise DataNotFoundError(
            f"Dataset '{path.name}' could not be parsed as CSV.",
            hint=f"Delete the file and re-download it. ({type(error).__name__})",
        ) from error

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataNotFoundError(
            f"Dataset is missing column(s): {', '.join(missing)}.",
            hint="Delete the CSV and re-download it; the upstream file may have changed.",
        )

    subset = frame[REQUIRED_COLUMNS].copy()
    if CONFIDENCE_COLUMN in frame.columns:
        subset[CONFIDENCE_COLUMN] = pd.to_numeric(frame[CONFIDENCE_COLUMN], errors="coerce")

    subset[TEXT_COLUMN] = subset[TEXT_COLUMN].astype(str).str.strip()
    subset = subset[subset[TEXT_COLUMN].str.len() > 0]

    if CONFIDENCE_COLUMN in subset.columns and min_confidence > 0:
        subset = subset[subset[CONFIDENCE_COLUMN].fillna(0.0) >= min_confidence]

    subset["label"] = subset[LABEL_COLUMN].astype(str).str.lower().str.strip().map(LABEL_MAP)
    subset = subset.dropna(subset=["label"])
    subset = subset[subset["label"].isin(CLASS_LABELS)]

    subset = subset.drop_duplicates(subset=[TEXT_COLUMN])
    subset = subset[["text", "label"]].rename(columns={TEXT_COLUMN: "text"}).reset_index(drop=True)

    if subset.empty:
        raise InvalidInputError(
            "No usable rows remain after filtering.",
            hint=f"Lower min_confidence (currently {min_confidence}) or re-download the dataset.",
        )

    return subset


def dataset_summary() -> dict[str, object]:
    """Return a compact description of the dataset for the UI sidebar."""
    frame = load_dataset()
    counts = frame["label"].value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "classes": CLASS_LABELS,
        "class_counts": {label: int(counts.get(label, 0)) for label in CLASS_LABELS},
        "source": TWEETS_SPEC.url,
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
    }


if __name__ == "__main__":  # pragma: no cover - manual helper
    from shared.config import configure_logging

    configure_logging()
    path = download_dataset()
    print(f"Dataset: {path}")
    summary = dataset_summary()
    print(f"{summary['rows']} tweets; class counts: {summary['class_counts']}")
