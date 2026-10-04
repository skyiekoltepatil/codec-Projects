#!/usr/bin/env python
"""Train, evaluate and save the MNIST handwritten digit recogniser.

Run from the ``03-handwritten-digit-recognizer`` directory::

    python train.py

Downloads MNIST if necessary, splits the 60,000 official training images into a
training and validation set, trains the CNN, evaluates it on the 10,000 held-out
test images, and writes:

* ``models/digit_cnn.pt``            - network weights plus metadata
* ``models/training_metrics.json``   - metrics, curves and hyperparameters

Extra flags let a reviewer reproduce a longer run or force a smaller one::

    python train.py --epochs 5
    python train.py --device cpu
    python train.py --limit-train 20000
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
from shared.ml import Timer, describe_device  # noqa: E402
from shared.paths import portable_display  # noqa: E402

from src.data import CLASS_LABELS, NUM_CLASSES, dataset_summary, load_mnist  # noqa: E402
from src.model import (  # noqa: E402
    HYPERPARAMS,
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
        description="Train the MNIST handwritten digit CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=HYPERPARAMS["epochs"], help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=HYPERPARAMS["batch_size"], help="Mini-batch size."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=HYPERPARAMS["learning_rate"],
        help="Peak learning rate for the one-cycle schedule.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Compute device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Use only the first N training images (0 = all). Useful for a quick check.",
    )
    parser.add_argument(
        "--limit-test", type=int, default=0, help="Evaluate on the first N test images (0 = all)."
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download MNIST even if cached."
    )
    return parser.parse_args()


def _log_epoch(epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
    """Progress callback forwarded to :func:`src.model.train_model`."""
    logger.info(
        "  epoch %d | train loss %.4f | train acc %.4f | val loss %.4f | val acc %.4f",
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
    logger.info("Handwritten Digit Recogniser - training")
    logger.info("=" * 72)
    logger.info("Compute device: %s", describe_device())

    device = None if args.device == "auto" else args.device

    with Timer() as load_timer:
        summary = dataset_summary(download=True)
        if args.force_download:
            x_train, y_train = load_mnist("train", force=True)
            x_test, y_test = load_mnist("test", force=True)
        else:
            x_train, y_train = load_mnist("train")
            x_test, y_test = load_mnist("test")

    if args.limit_train:
        x_train, y_train = x_train[: args.limit_train], y_train[: args.limit_train]
    if args.limit_test:
        x_test, y_test = x_test[: args.limit_test], y_test[: args.limit_test]

    logger.info(
        "Loaded MNIST in %s (train=%d, test=%d, %d classes)",
        load_timer,
        len(x_train),
        len(x_test),
        NUM_CLASSES,
    )

    model_for_params = None
    with Timer() as training_timer:
        recogniser, model = train_model(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            progress=_log_epoch,
        )
        model_for_params = model

    logger.info("Training finished in %s", training_timer)
    finalise(recogniser, model_for_params, x_test, y_test)

    metrics = recogniser.metrics
    logger.info("-" * 72)
    logger.info("Test split (%d images)", recogniser.n_test)
    logger.info("  accuracy            %s", format_metric(metrics["accuracy"]))
    logger.info("  macro precision     %s", format_metric(metrics["precision"]))
    logger.info("  macro recall        %s", format_metric(metrics["recall"]))
    logger.info("  macro F1            %s", format_metric(metrics["f1"]))
    logger.info("  mean top-1 confidence %s", format_metric(metrics["mean_confidence"]))
    logger.info("-" * 72)

    model_path = save_model(recogniser)
    logger.info("Saved model artefact -> %s", model_path)

    report = build_training_report(
        recogniser,
        dataset={
            "name": "MNIST",
            "train_images_available": summary["train_images"],
            "test_images_available": summary["test_images"],
            "train_images_used": int(len(x_train)),
            "test_images_used": int(len(x_test)),
            "image_size": summary["image_size"],
            "num_classes": NUM_CLASSES,
            "class_labels": CLASS_LABELS,
            "source": summary["source"],
        },
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    )
    report["n_parameters"] = recogniser.n_parameters or count_parameters(model_for_params)
    metrics_path = save_metrics_report(report)
    logger.info("Saved metrics -> %s", metrics_path)

    reloaded = load_model(model_path)
    if not reloaded.state_dict:
        logger.error("Reloaded model has an empty state dict.")
        return 1

    print()
    print("Model saved successfully.")
    print(f"  Artefact   : {portable_display(model_path)}")
    print(f"  Metrics    : {portable_display(metrics_path)}")
    print(f"  Parameters : {report['n_parameters']:,}")
    print(f"  Test accuracy: {format_metric(metrics['accuracy'])}")
    print()
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())