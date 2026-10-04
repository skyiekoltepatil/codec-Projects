#!/usr/bin/env python
"""Train, evaluate and save the Weather Analysis & Prediction model.

Run from the ``09-weather-analysis-prediction`` directory:

    python train.py

The script downloads the Open-Meteo archive for the selected city if necessary,
engineers leakage-free features, compares Linear Regression, Random Forest and
Gradient Boosting on the same chronological split against a persistence
baseline, prints a comparison table, and saves the winner (by test RMSE) to
``models/weather_model.joblib``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``shared`` and ``src`` importable when the script is run directly.
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.ml import Timer  # noqa: E402

from src.data import (  # noqa: E402
    CITY_CATALOGUE,
    DEFAULT_CITY,
    dataset_summary,
    download_city,
    load_weather,
)
from src.features import FeatureConfig, make_feature_frame, summarise_features  # noqa: E402
from src.model import (  # noqa: E402
    MODEL_LABELS,
    build_training_report,
    load_bundle,
    save_bundle,
    save_metrics_report,
    train_bundle,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the weather prediction models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--city", default=DEFAULT_CITY, choices=list(CITY_CATALOGUE), help="City to train on.")
    parser.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=int, default=1, help="Days ahead to predict.")
    parser.add_argument(
        "--target-mode",
        choices=("delta", "level"),
        default="delta",
        help="Predict the day-to-day change or the absolute temperature.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even when a cached copy exists.",
    )
    return parser.parse_args()


def _print_comparison(comparison: dict, city: str) -> None:
    """Print the model comparison table for one city."""
    header = f"{'Model':<22}{'MAE (C)':>10}{'RMSE (C)':>10}{'R2':>8}{'Base RMSE':>12}"
    print()
    print(f"City: {city}")
    print(header)
    print("-" * len(header))

    selected = comparison.get("selected", {}).get("model", "")
    for name, payload in comparison.items():
        if name == "selected":
            continue
        metrics = payload["metrics"]
        baseline = payload["baseline"]
        marker = " *" if name == selected else "  "
        print(
            f"{MODEL_LABELS.get(name, name):<22}"
            f"{format_metric(metrics['mae']):>10}"
            f"{format_metric(metrics['rmse']):>10}"
            f"{format_metric(metrics['r2']):>8}"
            f"{format_metric(baseline['rmse']):>12}{marker}"
        )
    print()
    print("* = selected model (lowest test RMSE)")


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Weather Analysis & Prediction - training")
    logger.info("=" * 72)

    with Timer() as dataset_timer:
        download_city(args.city, force=args.force_download)
        frame = load_weather(args.city, start=args.start, end=args.end)
    logger.info("Loaded %d days for %s in %s", len(frame), args.city, dataset_timer)

    config = FeatureConfig(horizon=max(1, args.horizon), target_mode=args.target_mode)
    with Timer() as feature_timer:
        feature_frame = make_feature_frame(frame, config=config)
    summary = summarise_features(feature_frame)
    logger.info(
        "Engineered %d rows (%d usable) x %d features in %s",
        summary["total_rows"],
        summary["usable_rows"],
        summary["features"],
        feature_timer,
    )

    with Timer() as training_timer:
        bundle, comparison = train_bundle(feature_frame, city=args.city, config=config)
    logger.info(
        "Trained %d candidates in %s; selected %s",
        len(comparison) - 1,
        training_timer,
        bundle.algorithm,
    )

    _print_comparison(comparison, args.city)
    if not bundle.beats_baseline:
        logger.warning("The selected model does not beat the persistence baseline for %s.", args.city)

    report = build_training_report(
        bundle,
        comparison,
        dataset=dataset_summary(args.city),
        feature_summary=summary,
    )

    model_path = save_bundle(bundle)
    logger.info("Saved model artefact -> %s", model_path)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    # Fail loudly if the artefact cannot be reloaded, rather than discovering a
    # broken file later in the UI.
    reloaded = load_bundle(model_path)
    if reloaded.city != bundle.city:
        logger.error("Reloaded bundle is for %s but trained bundle is for %s.", reloaded.city, bundle.city)
        return 1

    print("Model saved successfully.")
    print(f"  Artefact : {model_path}")
    print(f"  Metrics  : {metrics_path}")
    print(f"  City     : {bundle.city}")
    print(f"  Model    : {bundle.algorithm}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
