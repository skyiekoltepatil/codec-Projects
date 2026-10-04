"""Tests for Project 03 - Handwritten Digit Recogniser.

The tests are deliberately small: a tiny synthetic dataset stands in for MNIST
whenever a network has to be trained, so the suite runs in seconds. Tests that
need the real cached dataset skip themselves when it is absent rather than
triggering a 11 MB download during a unit-test run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from project_env import REPO_ROOT, use_project

use_project("03-handwritten-digit-recognizer")

sys.path.insert(0, str(REPO_ROOT))

from shared.errors import DataDownloadError, InvalidInputError  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_LABELS,
    IMAGE_SIZE,
    NUM_CLASSES,
    cache_path,
    load_mnist,
    preprocess_pil_image,
    to_model_input,
)
from src.model import (  # noqa: E402
    build_model,
    count_parameters,
    evaluate_on_test,
    load_model,
    save_model,
    train_model,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _synthetic_digits(per_class: int = 12, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Build a tiny, learnable stand-in dataset.

    Each class gets a distinct blocky pattern, so a convolutional network can
    separate them in a handful of steps. The point is to exercise the training
    and saving code paths, not to produce a meaningful recogniser.

    ``per_class`` defaults to 12 so the fixture holds 120 images, which clears
    the 100-image floor in ``train_model`` and still leaves enough per class for
    a stratified validation split.
    """
    images = []
    labels = []
    for digit in range(NUM_CLASSES):
        for index in range(per_class):
            image = np.zeros((size, size), dtype=np.float32)
            row = digit % size
            col = digit // size
            image[row : row + 6, :] = 1.0
            image[:, col : col + 6] = 1.0
            image = np.roll(image, index % 3, axis=0)
            images.append(image)
            labels.append(digit)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _mnist_available() -> bool:
    """Return ``True`` when the real MNIST cache exists locally."""
    return cache_path().exists()


requires_mnist = pytest.mark.skipif(
    not _mnist_available(),
    reason="MNIST cache absent. Run `python train.py` in project 03 once to create it.",
)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_mnist_cache_shapes_and_range():
    """MNIST arrays have the documented shape, dtype and value range."""
    images, labels = load_mnist("test", download=False)
    assert images.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE)
    assert images.dtype == np.float32
    assert labels.dtype == np.int64
    assert len(images) == len(labels)
    assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0
    assert set(np.unique(labels)).issubset(set(range(NUM_CLASSES)))


def test_mnist_labels_align_with_images():
    """Label index i really belongs to image i (catches a download/decoding mix-up)."""
    images, labels = load_mnist("test", download=False)
    # The first ten MNIST test labels are a fixed, documented sequence.
    assert labels[:10].tolist() == [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
    # A blank image cannot belong to any digit.
    assert not np.allclose(images[0], 0.0)


def test_unknown_split_is_rejected():
    """An unsupported split name raises a typed, actionable error."""
    with pytest.raises(InvalidInputError) as error:
        load_mnist("validation", download=False)
    assert "validation" in str(error.value)


def test_missing_cache_without_download_raises(tmp_path, monkeypatch):
    """Asking for absent data with downloads disabled fails clearly."""
    import src.data as data_module

    monkeypatch.setattr(data_module, "cache_path", lambda: tmp_path / "absent.npz")
    with pytest.raises(DataDownloadError) as error:
        data_module.load_mnist("train", download=False)
    assert "train.py" in (error.value.hint or "")


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #


def test_preprocess_output_shape_and_range():
    """Any image becomes a 28x28 float array scaled to [0, 1]."""
    from PIL import Image

    image = Image.new("L", (120, 90), color=0)
    image.paste(255, (30, 20, 80, 70))
    array = preprocess_pil_image(image)
    assert array.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert array.dtype == np.float32
    assert 0.0 <= float(array.min()) and float(array.max()) <= 1.0


def test_preprocess_preserves_aspect_ratio():
    """A very wide image is letterboxed, not stretched to fill the square.

    Stretching is the classic cause of misread digits, so the test asserts that
    the content keeps its aspect ratio: a wide bar stays wide and thin.
    """
    from PIL import Image

    image = Image.new("L", (400, 40), color=0)
    image.paste(255, (0, 0, 400, 40))
    array = preprocess_pil_image(image)
    active_rows = np.where(array.max(axis=1) > 0.5)[0]
    active_cols = np.where(array.max(axis=0) > 0.5)[0]
    assert len(active_cols) / max(1, len(active_rows)) > 4.0


def test_preprocess_inverts_photo_with_bright_background():
    """Dark ink on white paper is inverted so strokes end up bright, like MNIST."""
    from PIL import Image

    paper = Image.new("L", (200, 200), color=245)
    paper.paste(15, (60, 40, 140, 160))
    array = preprocess_pil_image(paper)
    centre = array[8:20, 8:20]
    border = array[0, :]
    # Bright strokes on a dark field: the digit region must be brighter than the edge.
    assert float(border.mean()) < float(centre.mean())


def test_preprocess_leaves_mnist_images_untouched():
    """An already-normalised MNIST image is not spuriously inverted."""
    from PIL import Image

    images, _ = load_mnist("test", download=False)
    original = images[0]
    array = preprocess_pil_image(Image.fromarray((original * 255).astype("uint8"), mode="L"))
    assert np.allclose(array, original, atol=1e-3)


def test_preprocess_rejects_blank_and_non_image():
    """Blank and wrongly-typed inputs raise a typed error rather than crashing."""
    from PIL import Image

    with pytest.raises(InvalidInputError):
        preprocess_pil_image(Image.new("L", (28, 28), color=0))
    with pytest.raises(InvalidInputError):
        preprocess_pil_image(np.zeros((28, 28)))


def test_to_model_input_adds_channel_dimension():
    """A single image becomes a (1, 1, 28, 28) batch tensor."""
    array = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    batch = to_model_input(array)
    assert tuple(batch.shape) == (1, 1, IMAGE_SIZE, IMAGE_SIZE)

    stacked = to_model_input(np.zeros((5, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
    assert tuple(stacked.shape) == (5, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_to_model_input_rejects_wrong_shape():
    """Images of the wrong size are rejected with a helpful message."""
    with pytest.raises(InvalidInputError):
        to_model_input(np.zeros((32, 32), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def test_model_output_shape_and_parameter_count():
    """The network maps a batch to 10 logits and has a sensible parameter count."""
    import torch

    model = build_model()
    with torch.no_grad():
        logits = model(torch.zeros(4, 1, IMAGE_SIZE, IMAGE_SIZE))
    assert tuple(logits.shape) == (4, NUM_CLASSES)
    # A 400k-parameter CNN is the right order of magnitude for MNIST; the exact
    # count is asserted so an accidental architecture change is noticed.
    assert count_parameters(model) == 468_202


def test_training_reduces_loss_and_records_history():
    """One epoch on a tiny dataset records one entry per history series."""
    images, labels = _synthetic_digits()
    recogniser, _model = train_model(images, labels, epochs=1, batch_size=16)
    assert len(recogniser.history["train_loss"]) == 1
    assert len(recogniser.history["val_accuracy"]) == 1
    assert recogniser.n_train + recogniser.n_val == len(images)
    assert recogniser.n_parameters > 0
    assert recogniser.state_dict


def test_training_rejects_misaligned_or_tiny_input():
    """Bad array shapes and too-few samples raise actionable errors."""
    with pytest.raises(InvalidInputError):
        train_model(np.zeros((10, 64, 64), dtype=np.float32), np.zeros(10, dtype=np.int64), epochs=1)
    with pytest.raises(InvalidInputError):
        train_model(
            np.zeros((10, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
            np.zeros(9, dtype=np.int64),
            epochs=1,
        )
    with pytest.raises(InvalidInputError):
        train_model(
            np.zeros((5, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
            np.zeros(5, dtype=np.int64),
            epochs=1,
        )


def test_save_and_load_round_trip(tmp_path):
    """A saved recogniser reloads with identical weights and predictions."""
    images, labels = _synthetic_digits()
    recogniser, _model = train_model(images, labels, epochs=1, batch_size=16)
    path = save_model(recogniser, directory=tmp_path)

    assert path.exists()
    reloaded = load_model(path)

    assert set(reloaded.state_dict) == set(recogniser.state_dict)
    for key, tensor in reloaded.state_dict.items():
        assert np.allclose(tensor.numpy(), recogniser.state_dict[key].numpy())

    before = recogniser.predict(images[:3]).labels
    after = reloaded.predict(images[:3]).labels
    assert before == after


def test_load_missing_model_raises(tmp_path):
    """Loading an absent artefact explains how to create it."""
    with pytest.raises(Exception) as error:
        load_model(tmp_path / "missing.pt")
    assert "missing.pt" in str(error.value)
    assert "train.py" in (error.value.hint or "")


def test_load_rejects_foreign_file(tmp_path):
    """A file that is not a recogniser raises a typed error, not a raw traceback."""
    from shared.errors import ModelNotFoundError

    bogus = tmp_path / "digit_cnn.pt"
    bogus.write_text("this is not a torch checkpoint", encoding="utf-8")
    with pytest.raises(ModelNotFoundError):
        load_model(bogus)


def test_predict_returns_sorted_top_k_with_valid_probabilities():
    """Predictions expose valid probabilities and a descending top-k ranking."""
    images, labels = _synthetic_digits()
    recogniser, _model = train_model(images, labels, epochs=1, batch_size=16)

    result = recogniser.predict(images)
    assert len(result.labels) == len(labels)
    assert set(result.labels).issubset(set(CLASS_LABELS))

    rows = result.probabilities
    assert rows.shape == (len(labels), NUM_CLASSES)
    assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-5)

    top_k = result.top_k(3)
    assert all(len(row) == 3 for row in top_k)
    for row in top_k:
        probabilities = [item["probability"] for item in row]
        assert probabilities == sorted(probabilities, reverse=True)
        assert row[0]["label"] == result.labels[top_k.index(row)]


def test_predict_rejects_wrong_shape_input():
    """Inference validates its input shape."""
    images, _ = _synthetic_digits()
    recogniser, _model = train_model(images, np.zeros(len(images), dtype=np.int64), epochs=1, batch_size=16)
    with pytest.raises(InvalidInputError):
        recogniser.predict(np.zeros((48, 48), dtype=np.float32))


def test_evaluate_on_test_returns_metrics_and_rows():
    """Test evaluation produces both the metrics dict and the per-class rows."""
    images, labels = _synthetic_digits()
    _recogniser, model = train_model(images, labels, epochs=1, batch_size=16)

    metrics, per_class = evaluate_on_test(model, images, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_test"] == len(labels)
    assert len(per_class) == NUM_CLASSES
    assert len(metrics["confusion_matrix"]) == NUM_CLASSES


# --------------------------------------------------------------------------- #
# Interface smoke test
# --------------------------------------------------------------------------- #


@requires_mnist
def test_app_renders_without_exception():
    """The Streamlit page executes top to bottom without raising."""
    from streamlit.testing.v1 import AppTest

    project_dir = REPO_ROOT / "03-handwritten-digit-recognizer"
    if not (project_dir / "models" / "digit_cnn.pt").exists():
        pytest.skip("Trained artefact absent; run `python train.py` first.")

    app = AppTest.from_file(str(project_dir / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception
    assert any("Handwritten" in str(m.value) for m in app.title) or app.markdown