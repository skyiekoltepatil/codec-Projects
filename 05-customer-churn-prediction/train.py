#!/usr/bin/env python
"""Train, evaluate and save the customer churn classifiers.

Run from the ``05-customer-churn-prediction`` directory::

    python train.py

Downloads the IBM Telco churn CSV if necessary, compares Logistic Regression,
Random Forest and Gradient Boosting on the same stratified split, prints a
comparison table, and saves the best model by ROC-AUC plus its metrics.
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
from shared.paths import portable_display  # noqa: E402

from src.data import load_dataset  # noqa: E402
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
        description="Train the customer churn classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction.")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download the CSV even if cached."
    )
    return parser.parse_args()


def _print_comparison(comparison: dict, selected: str) -> None:
    """Print the classifier comparison table."""
    header = (
        f"{'Classifier':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
        f"{'F1':>9}{'ROC-AUC':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, payload in comparison.items():
        metrics = payload["metrics"]
        roc = metrics.get("roc_auc")
        marker = " *" if name == selected else "  "
        print(
            f"{payload['label']:<24}"
            f"{format_metric(metrics['accuracy']):>10}"
            f"{format_metric(metrics['precision']):>11}"
            f"{format_metric(metrics['recall']):>9}"
            f"{format_metric(metrics['f1']):>9}"
            f"{(format_metric(roc) if roc is not None else 'n/a'):>10}{marker}"
        )
    print()
    print("* = selected model (highest ROC-AUC)")
    print(
        "Precision, recall and F1 are computed for the churn class only "
        "(average='binary'), because 'accurately predicting who stays' is not the goal."
    )


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Customer Churn Prediction - training")
    logger.info("=" * 72)

    with Timer() as data_timer:
        data = load_dataset(force_download=args.force_download)
    summary = data.summary()
    logger.info(
        "Loaded %d customers in %s (churn rate %.1f%%)",
        summary["rows"],
        data_timer,
        summary["churn_rate"] * 100,
    )
    if summary["blank_total_charges"]:
        logger.info(
            "%d customer(s) had a blank TotalCharges and were imputed with the training median.",
            summary["blank_total_charges"],
        )

    with Timer() as training_timer:
        comparison = train_all_models(data.frame, test_size=args.test_size)
    logger.info("Trained %d classifiers in %s", len(comparison), training_timer)

    selected = select_best_model(comparison)
    _print_comparison(comparison, selected)

    best = train_model(data.frame, selected, test_size=args.test_size)
    model_path = save_model(best)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(best, comparison, dataset=summary, selected=selected)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    reloaded = load_model(model_path)
    if reloaded.model_name != best.model_name:
        logger.error("Reloaded model '%s' does not match trained '%s'.", reloaded.model_name, best.model_name)
        return 1

    print()
    print("Model saved successfully.")
    print(f"  Artefact : {portable_display(model_path)}")
    print(f"  Metrics  : {portable_display(metrics_path)}")
    print(f"  Selected : {MODEL_LABELS.get(selected, selected)}")
    print(f"  ROC-AUC  : {format_metric(best.metrics.get('roc_auc'))}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())