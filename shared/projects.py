"""The catalogue of the ten projects in this repository.

This module is the single source of truth for project metadata, so the dashboard,
the README tables and the final report cannot disagree about names or
technologies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ProjectInfo:
    """Static metadata describing one project.

    Attributes:
        number: Two-digit project number, e.g. ``"01"``.
        slug: Directory name, e.g. ``"01-stock-price-predictor"``.
        title: Human-readable project name.
        short: One-line description for the dashboard card.
        ml_type: Machine-learning category shown as a chip.
        technology: Primary libraries used.
        trainable: Whether the project ships a ``train.py``.
        algorithm: The headline algorithm, used in the final report.
        dataset: Short dataset description.
        app_file: The Streamlit entry point for this project.
    """

    number: str
    slug: str
    title: str
    short: str
    ml_type: str
    technology: str
    trainable: bool
    algorithm: str
    dataset: str

    @property
    def directory(self) -> Path:
        """Absolute path to the project directory."""
        return PROJECT_ROOT / self.slug

    @property
    def app_file(self) -> Path:
        """Absolute path to the project's Streamlit app."""
        return self.directory / "app.py"

    @property
    def model_dir(self) -> Path:
        """Absolute path to the project's models directory."""
        return self.directory / "models"

    @property
    def readme(self) -> Path:
        """Absolute path to the project's README."""
        return self.directory / "README.md"

    def primary_artefact(self) -> Path | None:
        """Return the main trained artefact, or ``None`` when not applicable.

        The filename is not stored in the catalogue because projects save
        different numbers of files; the dashboard globs instead. This property is
        used only for a quick readiness check.
        """
        if not self.model_dir.is_dir():
            return None
        candidates = sorted(
            path
            for path in self.model_dir.iterdir()
            if path.suffix in {".joblib", ".pt", ".pth"} and not path.name.endswith(".json")
        )
        return candidates[0] if candidates else None


#: All ten projects, in presentation order.
PROJECTS: list[ProjectInfo] = [
    ProjectInfo(
        number="01",
        slug="01-stock-price-predictor",
        title="Stock Price Predictor",
        short="Forecast the next trading-day close from historical prices, with a persistence baseline.",
        ml_type="Regression / Time series",
        technology="scikit-learn, pandas, Plotly",
        trainable=True,
        algorithm="Linear Regression, Random Forest, Gradient Boosting",
        dataset="Daily closes for AAPL, MSFT, IBM, SBUX and the S&P 500 (2007-2016)",
    ),
    ProjectInfo(
        number="02",
        slug="02-twitter-sentiment-analysis",
        title="Twitter/X Sentiment Analysis",
        short="Classify tweets as positive, negative or neutral using TF-IDF and linear classifiers.",
        ml_type="NLP / Classification",
        technology="scikit-learn, pandas",
        trainable=True,
        algorithm="TF-IDF with Logistic Regression, Naive Bayes, Linear SVM",
        dataset="Twitter US Airline Sentiment (14,196 labelled tweets)",
    ),
    ProjectInfo(
        number="03",
        slug="03-handwritten-digit-recognizer",
        title="Handwritten Digit Recogniser",
        short="Recognise handwritten digits 0-9 with a convolutional neural network.",
        ml_type="Computer vision / Deep learning",
        technology="PyTorch, NumPy, Pillow",
        trainable=True,
        algorithm="Convolutional Neural Network (2 conv blocks)",
        dataset="MNIST (70,000 28x28 greyscale images)",
    ),
    ProjectInfo(
        number="04",
        slug="04-movie-recommendation-system",
        title="Movie Recommendation System",
        short="Recommend similar films from genres, keywords and overview text via cosine similarity.",
        ml_type="Recommender system",
        technology="scikit-learn, pandas, Plotly",
        trainable=True,
        algorithm="TF-IDF content similarity with SVD collaborative filtering",
        dataset="MovieLens ml-latest-small (100k ratings, 9,742 films)",
    ),
    ProjectInfo(
        number="05",
        slug="05-customer-churn-prediction",
        title="Customer Churn Prediction",
        short="Estimate churn probability and risk band for a telecom subscriber.",
        ml_type="Classification",
        technology="scikit-learn, pandas, Plotly",
        trainable=True,
        algorithm="Logistic Regression, Random Forest, Gradient Boosting",
        dataset="IBM Telco Customer Churn (7,043 customers)",
    ),
    ProjectInfo(
        number="06",
        slug="06-customer-service-chatbot",
        title="Customer Service Chatbot",
        short="Intent-classification chatbot that answers support questions and refuses when unsure.",
        ml_type="NLP / Intent classification",
        technology="scikit-learn, Streamlit",
        trainable=True,
        algorithm="TF-IDF with Logistic Regression and a confidence threshold",
        dataset="Curated intents corpus (8 intents, hand-written utterances)",
    ),
    ProjectInfo(
        number="07",
        slug="07-spam-email-classifier",
        title="Spam Email Classifier",
        short="Flag spam messages and explain which words drove the decision.",
        ml_type="NLP / Binary classification",
        technology="scikit-learn, pandas",
        trainable=True,
        algorithm="TF-IDF with Multinomial Naive Bayes, Logistic Regression, Linear SVM",
        dataset="SMS Spam Collection (5,574 messages)",
    ),
    ProjectInfo(
        number="08",
        slug="08-fruit-image-classifier",
        title="Fruit Image Classifier",
        short="Classify fruit photos into categories with a CNN and show confidence.",
        ml_type="Computer vision / Deep learning",
        technology="PyTorch, Pillow, NumPy",
        trainable=True,
        algorithm="Convolutional Neural Network with data augmentation",
        dataset="Synthetic-rendered fruit images generated locally (3 classes)",
    ),
    ProjectInfo(
        number="09",
        slug="09-weather-analysis-prediction",
        title="Weather Data Analysis & Prediction",
        short="Explore historical weather and predict next-day temperature with regression.",
        ml_type="Regression / Time series",
        technology="scikit-learn, pandas, Plotly",
        trainable=True,
        algorithm="Linear Regression, Random Forest, Gradient Boosting",
        dataset="Open-Meteo historical archive (daily observations)",
    ),
    ProjectInfo(
        number="10",
        slug="10-speech-to-text",
        title="Speech-to-Text Transcription",
        short="Transcribe uploaded audio offline with a Whisper model and export the text.",
        ml_type="Speech / Deep learning",
        technology="faster-whisper, Streamlit",
        trainable=False,
        algorithm="OpenAI Whisper (small English model, int8 quantised, offline)",
        dataset="No training data; pretrained model weights are downloaded on first use",
    ),
]

#: Lookup by project number.
BY_NUMBER: dict[str, ProjectInfo] = {project.number: project for project in PROJECTS}

#: Lookup by directory slug.
BY_SLUG: dict[str, ProjectInfo] = {project.slug: project for project in PROJECTS}


def get(identifier: str) -> ProjectInfo:
    """Return a project by number (``"01"``) or slug.

    Raises:
        KeyError: When the identifier does not match any project.
    """
    if identifier in BY_NUMBER:
        return BY_NUMBER[identifier]
    if identifier in BY_SLUG:
        return BY_SLUG[identifier]
    raise KeyError(f"Unknown project identifier: {identifier!r}")