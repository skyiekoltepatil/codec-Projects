"""Fruit image classification: a small CNN, augmentation and transfer learning.

Two architectures, chosen at training time
-----------------------------------------
``cnn`` (default)
    A three-block convolutional network trained from scratch. It needs no
    pretrained weights, so the project runs with no network access after the
    dataset is cached, and it trains to high 90s accuracy in a couple of minutes
    on a laptop CPU. This is the default because it is fully reproducible
    everywhere.

``mobilenetv2`` (optional)
    Transfer learning: the ImageNet backbone with its classification head
    replaced. It converges faster and usually edges out the scratch model, but it
    downloads ~14 MB of pretrained weights on first use. Enable it with
    ``--architecture mobilenetv2``; if the weights cannot be fetched the error
    explains how to fall back rather than silently switching architectures.

Why augmentation
----------------
Fruits-360 photographs each class in a fairly narrow setup, so a from-scratch
network will happily memorise 3,605 training images and generalise worse. Random
horizontal flips, small rotations and brightness/contrast jitter break that. The
augmentation is applied to the **training split only**; validation and test images
are used exactly as they were decoded, otherwise reported accuracy would be
measured on altered images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from shared.errors import InvalidInputError, ModelNotFoundError
from shared.metrics import classification_metrics, per_class_report
from shared.ml import RANDOM_SEED, describe_device, resolve_device, set_global_seed
from shared.paths import models_dir

from src.data import CLASS_LABELS, IMAGE_SIZE, FruitDataset

logger = logging.getLogger(__name__)

PROJECT_SLUG = "08-fruit-image-classifier"

#: Filename of the saved weights inside ``models/``.
MODEL_FILENAME = "fruit_classifier.pt"

#: Human-readable architecture names.
ARCHITECTURE_LABELS: dict[str, str] = {
    "cnn": "CNN (trained from scratch)",
    "mobilenetv2": "MobileNetV2 (transfer learning)",
}

SELECTABLE_ARCHITECTURES: tuple[str, ...] = ("cnn", "mobilenetv2")

#: Hyperparameters, mirrored in the README.
HYPERPARAMS: dict[str, Any] = {
    "epochs": 8,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "Adam",
    "scheduler": "OneCycleLR",
    "loss": "CrossEntropyLoss",
    "label_smoothing": 0.05,
}

#: Augmentation strength. Fruit photographs vary in orientation, lighting and
#: white balance, so the jitter matches the real variation in the collection.
AUGMENTATION_PARAMS: dict[str, Any] = {
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": 15,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
}


def build_model(architecture: str = "cnn", num_classes: int | None = None) -> "Any":
    """Return a fresh classifier for ``architecture``.

    Raises:
        InvalidInputError: For an unknown architecture.
        DataDownloadError: If the pretrained weights cannot be fetched.
    """
    import torch.nn as nn

    classes = int(num_classes if num_classes is not None else len(CLASS_LABELS))

    if architecture == "mobilenetv2":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        try:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception as error:  # noqa: BLE001 - network failure must be explained
            from shared.errors import DataDownloadError

            raise DataDownloadError(
                "Could not download the pretrained MobileNetV2 weights.",
                hint=(
                    "Transfer learning needs the ~14 MB ImageNet checkpoint. Connect to "
                    "the internet and retry, or train the from-scratch model instead "
                    f"with `python train.py --architecture cnn`. ({type(error).__name__})"
                ),
            ) from error

        # Replace the 1000-class head with one head per fruit.
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, classes))
        return backbone

    if architecture != "cnn":
        raise InvalidInputError(
            f"Unknown architecture '{architecture}'.",
            hint=f"Choose one of: {', '.join(SELECTABLE_ARCHITECTURES)}.",
        )

    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(128, classes),
    )


def count_parameters(model: "Any") -> int:
    """Return the number of trainable parameters in ``model``."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def training_transforms() -> "Any":
    """Return the augmentation pipeline applied to training images only.

    No ``ToTensor`` here on purpose. The decoder already produces
    ``(3, H, W)`` float arrays scaled to ``[0, 1]``, so the images are tensors
    before they reach the transform. Running ``ToTensor`` over a CHW ndarray
    would reinterpret the channels as height and fail.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=float(AUGMENTATION_PARAMS["horizontal_flip_probability"])),
            transforms.RandomRotation(degrees=float(AUGMENTATION_PARAMS["rotation_degrees"])),
            transforms.ColorJitter(
                brightness=float(AUGMENTATION_PARAMS["brightness"]),
                contrast=float(AUGMENTATION_PARAMS["contrast"]),
                saturation=float(AUGMENTATION_PARAMS["saturation"]),
            ),
        ]
    )


def eval_transforms() -> "Any":
    """Return the identity pipeline used for validation and test images.

    Those images are evaluated exactly as decoded, with no augmentation, so the
    reported accuracy describes real photographs rather than altered ones.
    """
    return None


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@dataclass
class PredictionResult:
    """Outcome of classifying one or more fruit images."""

    labels: list[str]
    probabilities: np.ndarray
    images: np.ndarray

    def confidence(self) -> list[float]:
        """Return the top-1 probability per image."""
        return [float(row.max()) for row in self.probabilities]

    def top_k(self, k: int = 3) -> list[list[dict[str, Any]]]:
        """Return the ``k`` most probable classes per image, most likely first."""
        k = max(1, min(int(k), len(CLASS_LABELS)))
        results: list[list[dict[str, Any]]] = []
        for row in self.probabilities:
            order = np.argsort(row)[::-1][:k]
            results.append(
                [{"label": CLASS_LABELS[index], "probability": float(row[index])} for index in order]
            )
        return results


@dataclass
class FruitClassifier:
    """A trained network plus the metadata the interface displays.

    Weights are stored as a ``state_dict`` rather than a pickled module, and are
    loaded with ``weights_only=True``, so loading can never execute code. The
    architecture is always rebuilt from this file.
    """

    state_dict: dict[str, Any]
    architecture: str = "cnn"
    class_labels: list[str] = field(default_factory=lambda: list(CLASS_LABELS))
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    epochs: int = 0
    n_parameters: int = 0
    trained_at: str = ""

    def to_model(self, device: str | None = None) -> "Any":
        """Rebuild the architecture and load the saved weights."""
        import torch

        model = build_model(self.architecture, num_classes=len(self.class_labels))
        model.load_state_dict(self.state_dict)
        model.to(torch.device(device or "cpu"))
        model.eval()
        return model

    def predict(self, images: np.ndarray, *, device: str | None = None) -> PredictionResult:
        """Classify one or more preprocessed ``(3, S, S)`` images."""
        import torch

        from src.data import to_tensor_batch

        batch = np.asarray(images, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None, ...]
        if batch.ndim != 4 or batch.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise InvalidInputError(
                f"Expected images of shape (3, {IMAGE_SIZE}, {IMAGE_SIZE}), "
                f"got {tuple(batch.shape)}.",
                hint="Run the image through preprocess_image() first.",
            )

        target = torch.device(device or "cpu")
        model = self.to_model(target)
        with torch.no_grad():
            logits = model(to_tensor_batch(batch).to(target))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        indices = probabilities.argmax(axis=1)
        return PredictionResult(
            labels=[self.class_labels[int(index)] for index in indices],
            probabilities=probabilities,
            images=batch,
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _make_loader(
    dataset: FruitDataset,
    *,
    shuffle: bool,
    batch_size: int,
    seed: int,
    augment: bool,
) -> "Any":
    """Build a ``DataLoader`` over in-memory arrays with the chosen transforms."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ArrayDataset(Dataset):
        def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Any) -> None:
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            import torch

            image = torch.from_numpy(np.asarray(self.images[index], dtype=np.float32))
            if self.transform is not None:
                image = self.transform(image)
            return image, int(self.labels[index])

    transform = training_transforms() if augment else eval_transforms()
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ArrayDataset(dataset.images, dataset.labels, transform),
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _evaluate(
    model: "Any",
    loader: "Any",
    criterion: "Any",
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return ``(mean_loss, accuracy, true_labels, probabilities)``."""
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    truths: list[int] = []
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            targets = batch_y.to(device)
            total_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += len(targets)
            truths.extend(int(value) for value in batch_y.tolist())
            predictions.append(torch.softmax(logits, dim=1).cpu().numpy())

    probabilities = np.vstack(predictions) if predictions else np.empty((0, 0))
    return total_loss / max(1, seen), correct / max(1, seen), np.asarray(truths), probabilities


def train_model(
    train_data: FruitDataset,
    val_data: FruitDataset,
    *,
    architecture: str = "cnn",
    epochs: int = HYPERPARAMS["epochs"],
    batch_size: int = HYPERPARAMS["batch_size"],
    learning_rate: float = HYPERPARAMS["learning_rate"],
    device: str | None = None,
    progress: Any = None,
) -> tuple[FruitClassifier, Any]:
    """Train the classifier and return ``(classifier, live_model)``.

    Raises:
        InvalidInputError: When the splits are empty or inconsistent.
    """
    import torch
    import torch.nn as nn

    set_global_seed()

    if len(train_data) == 0:
        raise InvalidInputError(
            "The training split is empty.",
            hint="Check that the dataset downloaded and decoded correctly.",
        )
    if len(train_data) != len(train_data.labels):
        raise InvalidInputError(
            f"Got {len(train_data.images)} images but {len(train_data.labels)} labels.",
            hint="The image and label arrays must be aligned.",
        )

    resolved_device = resolve_device(device)
    num_classes = len(CLASS_LABELS)

    train_loader = _make_loader(
        train_data, shuffle=True, batch_size=batch_size, seed=RANDOM_SEED, augment=True
    )
    val_loader = _make_loader(
        val_data, shuffle=False, batch_size=batch_size, seed=RANDOM_SEED, augment=False
    )

    model = build_model(architecture, num_classes=num_classes).to(resolved_device)
    # Label smoothing keeps the model from becoming over-confident on the ~3,600
    # training images, which is the usual failure mode for a small image dataset.
    criterion = nn.CrossEntropyLoss(label_smoothing=float(HYPERPARAMS["label_smoothing"]))
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=float(HYPERPARAMS["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=learning_rate,
        epochs=max(1, epochs),
        steps_per_epoch=max(1, len(train_loader)),
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    import time

    for epoch in range(1, max(1, epochs) + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for batch_x, batch_y in train_loader:
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

        val_loss, val_accuracy, _truths, _probabilities = _evaluate(
            model, val_loader, criterion, resolved_device
        )
        history["train_loss"].append(running_loss / max(1, seen))
        history["train_accuracy"].append(correct / max(1, seen))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))

        logger.info(
            "epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.1fs",
            epoch,
            epochs,
            history["train_loss"][-1],
            history["train_accuracy"][-1],
            val_loss,
            val_accuracy,
            time.perf_counter() - started,
        )
        if progress is not None:
            progress(epoch, history["train_loss"][-1], history["train_accuracy"][-1], val_loss, val_accuracy)

    classifier = FruitClassifier(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        architecture=architecture,
        history=history,
        n_train=len(train_data),
        n_val=len(val_data),
        epochs=epochs,
        n_parameters=count_parameters(model),
    )
    return classifier, model


def finalise(
    classifier: FruitClassifier,
    model: "Any",
    test_data: FruitDataset,
    *,
    device: str | None = None,
) -> FruitClassifier:
    """Attach held-out test metrics and a timestamp to ``classifier``."""
    import pandas as pd
    import torch

    if len(test_data) == 0:
        logger.warning("No test images supplied; skipping final evaluation.")
        return classifier

    resolved_device = resolve_device(device)
    loader = _make_loader(
        test_data, shuffle=False, batch_size=int(HYPERPARAMS["batch_size"]), seed=RANDOM_SEED, augment=False
    )

    labels_in_order = np.asarray(test_data.labels)
    model.to(resolved_device)
    model.eval()

    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            all_probabilities.append(torch.softmax(model(batch_x.to(resolved_device)), dim=1).cpu().numpy())
    probabilities = np.vstack(all_probabilities)

    predictions = probabilities.argmax(axis=1)
    # The model works with integer class indices, so scikit-learn is given those;
    # ``label_names`` carries the display names into the report.
    index_labels = list(range(len(CLASS_LABELS)))
    metrics = classification_metrics(
        labels_in_order,
        predictions,
        labels=index_labels,
        label_names=CLASS_LABELS,
        y_proba=probabilities,
    )
    metrics["n_test"] = int(len(labels_in_order))
    metrics["mean_confidence"] = float(probabilities.max(axis=1).mean())

    classifier.metrics = metrics
    classifier.per_class = per_class_report(
        labels_in_order, predictions, labels=index_labels, label_names=CLASS_LABELS
    )
    classifier.n_test = int(len(labels_in_order))
    classifier.trained_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    return classifier


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_model(classifier: FruitClassifier, *, directory: Path | None = None) -> Path:
    """Save weights plus metadata to ``models/fruit_classifier.pt``."""
    import torch

    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / MODEL_FILENAME
    torch.save(
        {
            "state_dict": classifier.state_dict,
            "architecture": classifier.architecture,
            "class_labels": classifier.class_labels,
            "metrics": classifier.metrics,
            "per_class": classifier.per_class,
            "history": classifier.history,
            "n_train": classifier.n_train,
            "n_val": classifier.n_val,
            "n_test": classifier.n_test,
            "epochs": classifier.epochs,
            "n_parameters": classifier.n_parameters,
            "trained_at": classifier.trained_at,
            "image_size": IMAGE_SIZE,
        },
        path,
    )
    logger.info("Saved fruit classifier -> %s", path)
    return path


def load_model(path: Path | None = None) -> FruitClassifier:
    """Load a previously trained :class:`FruitClassifier`."""
    import torch

    target = Path(path) if path else models_dir(PROJECT_SLUG) / MODEL_FILENAME
    if not target.exists():
        raise ModelNotFoundError(
            f"Trained fruit classifier not found at '{target.name}'.",
            hint="Run `python train.py` in this project directory to create it.",
        )

    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelNotFoundError(
            f"Trained fruit classifier at '{target.name}' could not be read.",
            hint=(
                "The file may be corrupt or written by an incompatible PyTorch version. "
                f"Delete it and re-run training. ({type(error).__name__})"
            ),
        ) from error

    missing = [key for key in ("state_dict", "architecture") if key not in payload]
    if missing:
        raise ModelNotFoundError(
            f"'{target.name}' is missing required key(s): {', '.join(missing)}.",
            hint="Delete the file and re-run `python train.py`.",
        )

    return FruitClassifier(
        state_dict=payload["state_dict"],
        architecture=str(payload["architecture"]),
        class_labels=[str(label) for label in payload.get("class_labels", CLASS_LABELS)],
        metrics=payload.get("metrics", {}),
        per_class=payload.get("per_class", []),
        history=payload.get("history", {}),
        n_train=int(payload.get("n_train", 0)),
        n_val=int(payload.get("n_val", 0)),
        n_test=int(payload.get("n_test", 0)),
        epochs=int(payload.get("epochs", 0)),
        n_parameters=int(payload.get("n_parameters", 0)),
        trained_at=payload.get("trained_at", ""),
    )


def save_metrics_report(report: dict[str, Any], *, directory: Path | None = None) -> Path:
    """Write the training report to ``models/training_metrics.json``."""
    from shared.ml import save_json

    target_dir = Path(directory) if directory else models_dir(PROJECT_SLUG)
    target_dir.mkdir(parents=True, exist_ok=True)
    return save_json(report, target_dir / "training_metrics.json")


def build_training_report(
    classifier: FruitClassifier,
    *,
    dataset: dict[str, Any],
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON report written next to the weights."""
    return {
        "project": PROJECT_SLUG,
        "trained_at": classifier.trained_at,
        "architecture": ARCHITECTURE_LABELS.get(classifier.architecture, classifier.architecture),
        "architecture_key": classifier.architecture,
        "algorithm": "Convolutional Neural Network",
        "task": f"{len(classifier.class_labels)}-class image classification (fruit)",
        "framework": "PyTorch",
        "device_used_for_training": describe_device(),
        "n_parameters": classifier.n_parameters,
        "epochs": classifier.epochs,
        "hyperparameters": {**HYPERPARAMS, **(hyperparams or {})},
        "augmentation": AUGMENTATION_PARAMS,
        "dataset": dataset,
        "split": {"train": classifier.n_train, "validation": classifier.n_val, "test": classifier.n_test},
        "metrics": classifier.metrics,
        "per_class": classifier.per_class,
        "history": classifier.history,
        "class_labels": classifier.class_labels,
        "random_seed": RANDOM_SEED,
        "notes": (
            "Augmentation is applied to the training split only; validation and test "
            "images are evaluated exactly as decoded. Fruits-360 labels its images by "
            "variety, so the 113 variety labels are grouped up into "
            f"{len(classifier.class_labels)} fruits before training."
        ),
    }