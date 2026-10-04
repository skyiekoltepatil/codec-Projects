#!/usr/bin/env python
"""Train, evaluate and save the spam classifiers.

Run from the ``07-spam-email-classifier`` directory::

    python train.py

Downloads the UCI SMS Spam Collection if necessary, compares Multinomial Naive
Bayes, Logistic Regression and a Linear SVM on the same stratified split, prints
the comparison table, and saves the best model by macro-F1.
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

import logging
from pathlib import Path

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.ml import Timer  # noqa: E402
from shared.paths import portable_display  # noqa: E402

from src.data import example_messages, load_dataset  # noqa: E402
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
        description="Train the spam classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction.")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download the corpus even if cached."
    )
    return parser.parse_args()


def _print_comparison(comparison: dict, selected: str, spam_rate: float) -> None:
    """Print the classifier comparison table.

    Args:
        comparison: Per-classifier metrics keyed by name.
        selected: Name of the chosen model.
        spam_rate: Measured fraction of spam in the training corpus, used to show
            what a majority-class baseline would have scored. This is computed
            from the data rather than quoted from the dataset description, which
            would disagree once duplicate messages are removed.
    """
    header = (
        f"{'Classifier':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
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
            f"{payload['label']:<28}"
            f"{format_metric(metrics['accuracy']):>10}"
            f"{format_metric(metrics['precision']):>11}"
            f"{format_metric(metrics['recall']):>9}"
            f"{format_metric(metrics['f1']):>9}"
            f"{(format_metric(roc) if roc is not None else 'n/a'):>10}{marker}"
        )
    print()
    print("* = selected model (highest macro-F1)")
    print(
        f"Only {spam_rate:.1%} of messages are spam, so a model that always answered 'ham' "
        f"would score {format_metric(1 - spam_rate)} accuracy while filtering nothing."
    )


def _print_examples(model) -> None:
    """Classify the bundled examples so a reviewer sees real behaviour."""
    print()
    print("Example classifications")
    print("-" * 72)
    for name, message in example_messages().items():
        result = model.predict(message)
        print(f"[{result.label.upper():<4}] {result.confidence:>6.1%}  {name}")


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Spam Email Classifier - training")
    logger.info("=" * 72)

    with Timer() as data_timer:
        data = load_dataset(force_download=args.force_download)
    summary = data.summary()
    logger.info(
        "Loaded %d messages in %s (%.1f%% spam, %d duplicates removed)",
        summary["rows"],
        data_timer,
        summary["spam_rate"] * 100,
        summary["duplicates"],
    )

    with Timer() as training_timer:
        comparison = train_all_models(data.frame, test_size=args.test_size)
    logger.info("Trained %d classifiers in %s", len(comparison), training_timer)

    selected = select_best_model(comparison)
    _print_comparison(comparison, selected, summary["spam_rate"])

    best = train_model(data.frame, selected, test_size=args.test_size)
    model_path = save_model(best)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(best, comparison, dataset=summary, selected=selected)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    _print_examples(best)

    reloaded = load_model(model_path)
    if reloaded.model_name != best.model_name:
        logger.error("Reloaded model '%s' does not match trained '%s'.", reloaded.model_name, best.model_name)
        return 1

    print()
    print("Model saved successfully.")
    print(f"  Artefact : {portable_display(model_path)}")
    print(f"  Metrics  : {portable_display(metrics_path)}")
    print(f"  Selected : {MODEL_LABELS.get(selected, selected)}")
    print(f"  Macro F1 : {format_metric(best.metrics['f1'])}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())