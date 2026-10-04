#!/usr/bin/env python
"""Train, evaluate and save the fruit image classifier.

Run from the ``08-fruit-image-classifier`` directory::

    python train.py

Downloads Fruits-360 (~98 MB) on first run, decodes and caches the images, trains
the CNN with augmentation, evaluates on a held-out test split, and saves the
weights plus metrics.

Useful flags::

    python train.py --architecture mobilenetv2   # transfer learning
    python train.py --epochs 12                  # longer run
    python train.py --limit-train 800            # quick sanity check
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

import pandas as pd  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.ml import Timer, describe_device  # noqa: E402
from shared.paths import portable_display  # noqa: E402

from src.data import CLASS_LABELS, dataset_summary, load_dataset  # noqa: E402
from src.model import (  # noqa: E402
    ARCHITECTURE_LABELS,
    HYPERPARAMS,
    SELECTABLE_ARCHITECTURES,
    build_training_report,
    count_parameters,
    finalise,
    load_model,
    save_metrics_report,
    save_model,
    train_model,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the fruit image classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--architecture",
        choices=list(SELECTABLE_ARCHITECTURES),
        default="cnn",
        help="'cnn' trains from scratch; 'mobilenetv2' uses ImageNet transfer learning.",
    )
    parser.add_argument("--epochs", type=int, default=HYPERPARAMS["epochs"], help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=HYPERPARAMS["batch_size"], help="Mini-batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=HYPERPARAMS["learning_rate"], help="Peak learning rate."
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=0.15, help="Share held out for validation."
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="Share held out for final testing."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Compute device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--limit-train", type=int, default=0, help="Use only the first N training images (0 = all)."
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download the dataset even if cached."
    )
    return parser.parse_args()


def _log_epoch(epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
    """Progress callback forwarded to :func:`src.model.train_model`."""
    logger.info(
        "  epoch %d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    )


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    configure_logging()

    logger.info("=" * 72)
    logger.info("Fruit Image Classifier - training")
    logger.info("=" * 72)
    logger.info("Architecture : %s", ARCHITECTURE_LABELS[args.architecture])
    logger.info("Compute device: %s", describe_device())

    with Timer() as data_timer:
        data = load_dataset(force_download=args.force_download, progress=logger.info)
        train_data, val_data, test_data = data.split(
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
    logger.info("Dataset ready in %s", data_timer)
    logger.info("Split: train=%d validation=%d test=%d", len(train_data), len(val_data), len(test_data))
    logger.info("Class counts (all): %s", data.class_counts())

    if args.limit_train:
        import numpy as np

        subset = np.arange(min(args.limit_train, len(train_data)))
        train_data = train_data.subset(subset)
        logger.info("Limiting training to %d images for a quick check", len(train_data))

    device = None if args.device == "auto" else args.device

    with Timer() as training_timer:
        classifier, model = train_model(
            train_data,
            val_data,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            progress=_log_epoch,
        )
    logger.info("Training finished in %s", training_timer)

    finalise(classifier, model, test_data, device=device)

    metrics = classifier.metrics
    logger.info("-" * 72)
    logger.info("Test split (%d images)", classifier.n_test)
    logger.info("  accuracy           %s", format_metric(metrics.get("accuracy")))
    logger.info("  macro precision    %s", format_metric(metrics.get("precision")))
    logger.info("  macro recall       %s", format_metric(metrics.get("recall")))
    logger.info("  macro F1           %s", format_metric(metrics.get("f1")))
    logger.info("-" * 72)

    model_path = save_model(classifier)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(
        classifier,
        dataset=dataset_summary(),
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    )
    report["n_parameters"] = classifier.n_parameters or count_parameters(model)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    reloaded = load_model(model_path)
    if not reloaded.state_dict:
        logger.error("Reloaded model has an empty state dict.")
        return 1

    print()
    print("Model saved successfully.")
    print(f"  Artefact     : {portable_display(model_path)}")
    print(f"  Metrics      : {portable_display(metrics_path)}")
    print(f"  Architecture : {ARCHITECTURE_LABELS[classifier.architecture]}")
    print(f"  Parameters   : {report['n_parameters']:,}")
    print(f"  Test accuracy: {format_metric(metrics.get('accuracy'))}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())