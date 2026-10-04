#!/usr/bin/env python
"""Train, evaluate and save the Stock Price Predictor models.

Run from the ``01-stock-price-predictor`` directory:

    python train.py

The script downloads the dataset if necessary, engineers leakage-free features,
trains Linear Regression, Random Forest and Gradient Boosting per ticker on the
same chronological split, prints a comparison table that includes a persistence
baseline, and saves the per-ticker winners to ``models/stock_model.joblib``.
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
from shared.paths import models_dir  # noqa: E402

from src.data import download_datasets, load_price_history  # noqa: E402
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
        description="Train the stock price prediction models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=None, help="Ticker subset (default: all).")
    parser.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=int, default=1, help="Trading days ahead to predict.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the datasets even when a cached copy exists.",
    )
    return parser.parse_args()


def _print_comparison(comparison: dict) -> None:
    """Print the per-ticker model comparison table."""
    header = f"{'Ticker':<8}{'Model':<30}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'Base RMSE':>12}"
    print()
    print(header)
    print("-" * len(header))

    for ticker in sorted(comparison):
        entries = comparison[ticker]
        if not entries:
            print(f"{ticker:<8}{'(insufficient history)':<30}")
            continue
        selected = entries.get("selected", {})
        selected_name = selected.get("model", "")
        for model_name, payload in entries.items():
            if model_name == "selected":
                continue
            metrics = payload["metrics"]
            baseline = payload["baseline"]
            marker = " *" if model_name == selected_name else "  "
            print(
                f"{ticker:<8}{MODEL_LABELS.get(model_name, model_name):<30}"
                f"{format_metric(metrics['mae']):>10}"
                f"{format_metric(metrics['rmse']):>10}"
                f"{format_metric(metrics['r2']):>10}"
                f"{format_metric(baseline['rmse']):>12}{marker}"
            )
    print()
    print("* = selected model for that ticker (lowest test RMSE)")


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Stock Price Predictor - training")
    logger.info("=" * 72)

    with Timer() as dataset_timer:
        download_datasets(force=args.force_download)
        prices = load_price_history(args.tickers, start=args.start, end=args.end)
    logger.info(
        "Loaded %d price rows for %d ticker(s) in %s",
        len(prices),
        prices["ticker"].nunique(),
        dataset_timer,
    )

    config = FeatureConfig(horizon=max(1, args.horizon))
    with Timer() as feature_timer:
        feature_frame = make_feature_frame(prices, config=config)
    summary = summarise_features(feature_frame)
    logger.info(
        "Engineered %d rows (%d usable) x %d features in %s",
        summary["total_rows"],
        summary["usable_rows"],
        summary["features"],
        feature_timer,
    )

    with Timer() as training_timer:
        bundle, comparison = train_bundle(feature_frame, config=config)
    logger.info("Trained %d per-ticker models in %s", len(bundle), training_timer)

    _print_comparison(comparison)

    dataset_info = {
        "rows": int(len(prices)),
        "tickers": sorted(prices["ticker"].unique().tolist()),
        "start": prices["date"].min().strftime("%Y-%m-%d"),
        "end": prices["date"].max().strftime("%Y-%m-%d"),
    }
    report = build_training_report(
        bundle,
        comparison,
        dataset=dataset_info,
        feature_summary=summary,
    )

    model_path = save_bundle(bundle)
    logger.info("Saved model artefact -> %s", model_path)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    # Fail loudly if the artefact cannot be reloaded, rather than discovering a
    # broken file later in the UI.
    reloaded = load_bundle(model_path)
    if reloaded.tickers != bundle.tickers:
        logger.error("Reloaded bundle covers %s but trained bundle covers %s.", reloaded.tickers, bundle.tickers)
        return 1

    print("Model saved successfully.")
    print(f"  Artefact : {model_path}")
    print(f"  Metrics  : {metrics_path}")
    print(f"  Tickers  : {', '.join(bundle.tickers)}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())