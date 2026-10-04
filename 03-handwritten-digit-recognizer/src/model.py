"""A small convolutional network for MNIST, plus training and inference helpers.

Architecture
------------
Two convolution blocks followed by a classifier head::

    28x28x1  ->  [conv3x3 + BN + ReLU] x2  -> maxpool  ->  14x14x32
             ->  [conv3x3 + BN + ReLU] x2  -> maxpool  ->   7x7x64
             ->  dropout -> flatten (3136) -> dense 128 -> dropout -> dense 10

Why this shape of network
------------------------
* **Convolutions, not a dense first layer.** A fully connected layer on raw
  pixels has no notion of locality and needs roughly ten times more parameters
  for a worse result. Three-by-three kernels with padding preserve the 28x28
  spatial size so no information is downsampled before features are detected.
* **Batch normalisation** after each convolution stabilises the activations,
  which is what makes a higher learning rate safe.
* **Max pooling** halves resolution while keeping the strongest local response,
  giving the second block a wider receptive field for the same compute.
* **Dropout before the dense layers** regularises the ~400k parameters in the
  head, which is where an MNIST CNN overfits first.

The network is deliberately small: it trains to ~99% test accuracy in a couple of
minutes on a laptop CPU, which keeps the project reviewable end to end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import classification_metrics, per_class_report
from shared.ml import RANDOM_SEED, describe_device, resolve_device, save_json, set_global_seed
from shared.paths import models_dir

from src.data import CLASS_LABELS, IMAGE_SIZE, NUM_CLASSES, to_model_input

logger = logging.getLogger(__name__)

PROJECT_SLUG = "03-handwritten-digit-recognizer"

#: Filename of the saved network weights inside ``models/``.
MODEL_FILENAME = "digit_cnn.pt"

#: Hyperparameters, referenced by the README so the numbers are not duplicated.
HYPERPARAMS: dict[str, Any] = {
    "epochs": 3,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "Adam",
    "scheduler": "OneCycleLR",
    "loss": "CrossEntropyLoss",
    "dropout": 0.25,
    "head_dropout": 0.5,
}

#: Share of the 60,000 official training images held out for validation.
VALIDATION_FRACTION = 0.1


def build_model(num_classes: int = NUM_CLASSES) -> "Any":
    """Return a freshly initialised :class:`DigitCNN`."""
    import torch.nn as nn

    class DigitCNN(nn.Module):
        """Two-block convolutional network for 28x28 greyscale digits."""

        def __init__(self, classes: int = NUM_CLASSES) -> None:
            super().__init__()
            dropout = float(HYPERPARAMS["dropout"])
            head_dropout = float(HYPERPARAMS["head_dropout"])

            def block(in_channels: int, out_channels: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout),
                )

            self.features = nn.Sequential(block(1, 32), block(32, 64))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(128, classes),
            )

        def forward(self, inputs: "Any") -> "Any":
            """Return raw class logits, shape ``(batch, classes)``."""
            return self.classifier(self.features(inputs))

    return DigitCNN(num_classes)


def count_parameters(model: "Any") -> int:
    """Return the number of trainable parameters in ``model``."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


# --------------------------------------------------------------------------- #
# Inference helper
# --------------------------------------------------------------------------- #


@dataclass
class PredictionResult:
    """Outcome of classifying a batch of digit images."""

    labels: list[str]
    probabilities: np.ndarray
    images: np.ndarray

    @property
    def top1(self) -> list[str]:
        """Most likely label per image."""
        return self.labels

    def top_k(self, k: int = 3) -> list[list[dict[str, Any]]]:
        """Return the ``k`` most probable labels per image, most likely first."""
        k = max(1, min(int(k), len(CLASS_LABELS)))
        results: list[list[dict[str, Any]]] = []
        for row in self.probabilities:
            order = np.argsort(row)[::-1][:k]
            results.append(
                [{"label": CLASS_LABELS[index], "probability": float(row[index])} for index in order]
            )
        return results

    def confidence(self) -> list[float]:
        """Return the top-1 probability per image."""
        return [float(row.max()) for row in self.probabilities]


@dataclass
class DigitRecogniser:
    """A trained network plus the metadata the interface displays.

    The instance is a plain dataclass rather than a pickled ``nn.Module`` so that
    loading is safe: ``torch.load`` with ``weights_only=True`` cannot execute
    arbitrary code, and the architecture is always rebuilt from this repository's
    source rather than from whatever class was pickled.
    """

    state_dict: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    trained_at: str = ""
    device: str = "cpu"
    n_parameters: int = 0
    epochs: int = 0

    def to_model(self, device: str | None = None) -> "Any":
        """Rebuild the network and load the saved weights."""
        import torch

        target = torch.device(device or self.device)
        model = build_model()
        model.load_state_dict(self.state_dict)
        model.to(target)
        model.eval()
        return model

    def predict(self, images: np.ndarray, *, device: str | None = None) -> PredictionResult:
        """Classify one or more preprocessed ``(28, 28)`` images."""
        import torch

        array = np.asarray(images, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise InvalidInputError(
                f"Expected image(s) of shape (28, 28), got {tuple(array.shape)}.",
                hint="Run the image through src.data.preprocess_pil_image() first.",
            )

        target = torch.device(device or self.device)
        model = self.to_model(target)
        with torch.no_grad():
            logits = model(to_model_input(array).to(target))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        indices = probabilities.argmax(axis=1)
        return PredictionResult(
            labels=[CLASS_LABELS[index] for index in indices],
            probabilities=probabilities,
            images=array,
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _to_tensors(images: np.ndarray, labels: np.ndarray) -> tuple["Any", "Any"]:
    """Convert NumPy image/label arrays to PyTorch tensors."""
    import torch

    x = torch.from_numpy(np.asarray(images, dtype=np.float32)).unsqueeze(1)
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return x, y


def train_model(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = HYPERPARAMS["epochs"],
    batch_size: int = HYPERPARAMS["batch_size"],
    learning_rate: float = HYPERPARAMS["learning_rate"],
    device: str | None = None,
    validation_fraction: float = VALIDATION_FRACTION,
    progress: Any = None,
) -> tuple[DigitRecogniser, Any]:
    """Train the network and return ``(recogniser, trained_model)``.

    Args:
        images: ``(n, 28, 28)`` float array scaled to ``[0, 1]``.
        labels: ``(n,)`` integer digit labels.
        epochs: Passes over the training split.
        batch_size: Mini-batch size.
        learning_rate: Peak learning rate for the one-cycle schedule.
        device: Compute device; auto-detected when omitted.
        validation_fraction: Share of the training data held out for validation.
        progress: Optional ``callback(epoch, train_loss, train_acc, val_loss, val_acc)``.

    Returns:
        A :class:`DigitRecogniser` holding the trained weights and validation
        history, plus the live ``nn.Module``.

    Raises:
        InvalidInputError: When the inputs are inconsistent or too small.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed()
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if images.ndim != 3 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise InvalidInputError(
            f"Training images must have shape (n, {IMAGE_SIZE}, {IMAGE_SIZE}), got {tuple(images.shape)}.",
            hint="Use src.data.load_mnist('train'), which returns the correct layout.",
        )
    if len(images) != len(labels):
        raise InvalidInputError(
            f"Got {len(images)} images but {len(labels)} labels.",
            hint="The image and label arrays must be aligned.",
        )
    if len(images) < 100:
        raise InvalidInputError(
            f"Only {len(images)} training images available.",
            hint="At least 100 images are needed to train and validate the network.",
        )

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(images))
    train_idx, val_idx = train_test_split(
        indices, test_size=validation_fraction, random_state=RANDOM_SEED, stratify=labels
    )

    x_train, y_train = _to_tensors(images[train_idx], labels[train_idx])
    x_val, y_val = _to_tensors(images[val_idx], labels[val_idx])

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=max(1, min(int(batch_size), len(x_train))),
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    resolved_device = resolve_device(device)
    model = build_model().to(resolved_device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=float(HYPERPARAMS["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, epochs=max(1, epochs), steps_per_epoch=len(loader)
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, max(1, epochs) + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimiser.step()
            scheduler.step()
            running_loss += float(loss.item()) * len(batch_y)
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            seen += len(batch_y)

        val_loss, val_accuracy = _evaluate(model, x_val, y_val, criterion, resolved_device)
        history["train_loss"].append(running_loss / max(1, seen))
        history["train_accuracy"].append(correct / max(1, seen))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))

        message = (
            f"epoch {epoch}/{epochs} | train loss {history['train_loss'][-1]:.4f} "
            f"train acc {history['train_accuracy'][-1]:.4f} | val loss {val_loss:.4f} "
            f"val acc {val_accuracy:.4f} | {time.perf_counter() - started:.1f}s"
        )
        logger.info(message)
        if progress is not None:
            progress(epoch, history["train_loss"][-1], history["train_accuracy"][-1], val_loss, val_accuracy)

    recogniser = DigitRecogniser(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        history=history,
        n_train=len(x_train),
        n_val=len(x_val),
        device="cpu",
        n_parameters=count_parameters(model),
        epochs=epochs,
    )
    return recogniser, model


@staticmethod
def _evaluate(model: "Any", x: "Any", y: "Any", criterion: "Any", device: str) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` for a tensor batch, without gradients."""
    import torch

    model.eval()
    with torch.no_grad():
        tensors = x.to(device)
        targets = y.to(device)
        logits = model(tensors)
        total_loss = float(criterion(logits, targets).item())
        correct = int((logits.argmax(dim=1) == targets).sum().item())
    return total_loss, correct / max(1, len(targets))


def evaluate_on_test(
    model: "Any",
    images: np.ndarray,
    labels: np.ndarray,
    *,
    device: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a trained network on the held-out MNIST test split.

    Returns:
        ``(metrics, per_class_rows)``. ``metrics`` is the JSON-serialisable dict
        written to the training report and rendered in the interface.
    """
    resolved_device = resolve_device(device)
    x, y = _to_tensors(images, labels)
    model.to(resolved_device)
    model.eval()

    import torch

    with torch.no_grad():
        logits = model(x.to(resolved_device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, predictions, labels=CLASS_LABELS, y_proba=probabilities)
    metrics["mean_confidence"] = float(probabilities.max(axis=1).mean())
    metrics["n_test"] = int(len(labels))
    return metrics, per_class_report(labels, predictions, labels=CLASS_LABELS)


def finalise(
    recogniser: DigitRecogniser,
    model: "Any",
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> DigitRecogniser:
    """Attach test metrics, per-class report and a timestamp to ``recogniser``."""
    import pandas as pd

    metrics, per_class = evaluate_on_test(model, test_images, test_labels, device=recogniser.device)
    recogniser.metrics = metrics
    recogniser.per_class = per_class
    recogniser.n_test = int(len(test_labels))
    recogniser.trained_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    return recogniser


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_model(recogniser: DigitRecogniser, *, directory: Path | None = None) -> Path:
    """Save the network weights plus metadata to ``models/digit_cnn.pt``."""
    import torch

    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / MODEL_FILENAME
    torch.save(
        {
            "state_dict": recogniser.state_dict,
            "metrics": recogniser.metrics,
            "per_class": recogniser.per_class,
            "history": recogniser.history,
            "n_train": recogniser.n_train,
            "n_val": recogniser.n_val,
            "n_test": recogniser.n_test,
            "trained_at": recogniser.trained_at,
            "device": recogniser.device,
            "n_parameters": recogniser.n_parameters,
            "epochs": recogniser.epochs,
            "class_labels": CLASS_LABELS,
            "image_size": IMAGE_SIZE,
        },
        path,
    )
    logger.info("Saved digit CNN weights -> %s", path)
    return path


def load_model(path: Path | None = None) -> DigitRecogniser:
    """Load a previously trained :class:`DigitRecogniser`.

    Uses ``weights_only=True`` so that loading a model file can never execute
    arbitrary code.
    """
    import torch

    from shared.errors import ModelNotFoundError as _Missing

    target = Path(path) if path else models_dir(PROJECT_SLUG) / MODEL_FILENAME
    if not target.exists():
        raise _Missing(
            f"Trained digit recogniser not found at '{target.name}'.",
            hint="Run `python train.py` in this project directory to create it.",
        )

    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelNotFoundError(
            f"Trained digit recogniser at '{target.name}' could not be read.",
            hint=(
                "The file may be corrupted or was written by an incompatible PyTorch "
                f"version. Delete it and re-run training. ({type(error).__name__})"
            ),
        ) from error

    missing = [key for key in ("state_dict", "metrics") if key not in payload]
    if missing:
        raise ModelNotFoundError(
            f"'{target.name}' is missing required key(s): {', '.join(missing)}.",
            hint="Delete the file and re-run `python train.py`.",
        )

    return DigitRecogniser(
        state_dict=payload["state_dict"],
        metrics=payload.get("metrics", {}),
        per_class=payload.get("per_class", []),
        history=payload.get("history", {}),
        n_train=int(payload.get("n_train", 0)),
        n_val=int(payload.get("n_val", 0)),
        n_test=int(payload.get("n_test", 0)),
        trained_at=payload.get("trained_at", ""),
        device="cpu",
        n_parameters=int(payload.get("n_parameters", 0)),
        epochs=int(payload.get("epochs", 0)),
    )


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    recogniser: DigitRecogniser,
    *,
    dataset: dict[str, Any],
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON report written next to the weights."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": recogniser.trained_at,
        "architecture": "2-block CNN (32, 64) + dense 128",
        "algorithm": "Convolutional Neural Network",
        "task": "10-class image classification (handwritten digits 0-9)",
        "framework": "PyTorch",
        "device_used_for_training": describe_device(),
        "n_parameters": recogniser.n_parameters,
        "epochs": recogniser.epochs,
        "hyperparameters": {**HYPERPARAMS, **(hyperparams or {})},
        "dataset": dataset,
        "split": {
            "train": recogniser.n_train,
            "validation": recogniser.n_val,
            "test": recogniser.n_test,
        },
        "metrics": recogniser.metrics,
        "per_class": recogniser.per_class,
        "history": recogniser.history,
        "class_labels": CLASS_LABELS,
        "random_seed": RANDOM_SEED,
        "notes": (
            "Accuracy is measured on the official MNIST test split, which the network "
            "never sees during training. No augmentation is used for MNIST because the "
            "dataset is already large and normalised; see project 08 for an example "
            "where augmentation is required."
        ),
    }