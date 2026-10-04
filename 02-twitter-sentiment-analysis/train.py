#!/usr/bin/env python
"""Train, evaluate and save the Twitter/X sentiment classifiers.

Run from the ``02-twitter-sentiment-analysis`` directory:

    python train.py

Downloads the labelled tweet corpus if necessary, compares Logistic Regression,
Multinomial Naive Bayes and a Linear SVM on the same stratified split, prints a
comparison table, and saves the best model (by macro-F1) to
``models/sentiment_model.joblib``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.ml import Timer  # noqa: E402
from shared.paths import models_dir  # noqa: E402

from src.data import DEFAULT_MIN_CONFIDENCE, download_dataset, load_dataset  # noqa: E402
from src.model import (  # noqa: E402
    MODEL_LABELS,
    build_training_report,
    load_model,
    save_metrics_report,
    save_model,
    select_best_model,
    train_all_models,
    train_model,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the tweet sentiment classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Drop rows labelled with lower annotator confidence.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even when a cached copy exists.",
    )
    return parser.parse_args()


def _print_comparison(comparison: dict, selected: str) -> None:
    """Print the classifier comparison table."""
    header = f"{'Classifier':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'ROC-AUC':>10}"
    print()
    print(header)
    print("-" * len(header))
    for name, payload in comparison.items():
        metrics = payload["metrics"]
        roc = metrics.get("roc_auc_ovr", metrics.get("roc_auc"))
        roc_text = format_metric(roc) if roc is not None else "n/a"
        marker = " *" if name == selected else "  "
        print(
            f"{payload['label']:<26}"
            f"{format_metric(metrics['accuracy']):>10}"
            f"{format_metric(metrics['precision']):>11}"
            f"{format_metric(metrics['recall']):>9}"
            f"{format_metric(metrics['f1']):>9}"
            f"{roc_text:>10}{marker}"
        )
    print()
    print("* = selected model (highest macro-F1; all classes weighted equally)")


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Twitter/X Sentiment Analysis - training")
    logger.info("=" * 72)

    with Timer() as dataset_timer:
        download_dataset(force=args.force_download)
        frame = load_dataset(min_confidence=args.min_confidence)
    counts = frame["label"].value_counts().to_dict()
    logger.info("Loaded %d labelled tweets in %s", len(frame), dataset_timer)
    logger.info("Class distribution: %s", counts)

    with Timer() as training_timer:
        comparison = train_all_models(frame, test_size=args.test_size)
    logger.info("Trained %d classifiers in %s", len(comparison), training_timer)

    selected = select_best_model(comparison)
    _print_comparison(comparison, selected)

    # Refit the winner so the saved artefact is the one the metrics describe.
    best = train_model(frame, selected, test_size=args.test_size)
    model_path = save_model(best)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(
        best,
        comparison,
        dataset={
            "rows": int(len(frame)),
            "class_counts": {label: int(counts.get(label, 0)) for label in sorted(counts)},
            "min_confidence": args.min_confidence,
        },
        selected=selected,
    )
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    reloaded = load_model(model_path)
    if reloaded.model_name != best.model_name:
        logger.error("Reloaded model '%s' does not match trained '%s'.", reloaded.model_name, best.model_name)
        return 1

    print("Model saved successfully.")
    print(f"  Artefact : {model_path}")
    print(f"  Metrics  : {metrics_path}")
    print(f"  Selected : {MODEL_LABELS.get(selected, selected)}")
    print(f"  Macro F1 : {format_metric(best.metrics['f1'])}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())