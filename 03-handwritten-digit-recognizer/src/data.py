"""MNIST acquisition, caching and image preprocessing.

Why a cached ``.npz``?
----------------------
``torchvision`` returns tensors, which cannot be persisted to a single portable
archive as conveniently as NumPy arrays. Converting once and caching to
``data/mnist.npz`` means the Streamlit app and the test suite load the dataset in
milliseconds without re-downloading, and without keeping a second copy of the
60,000 training images on disk.

Preprocessing contract
----------------------
The network expects a ``(N, 1, 28, 28)`` float tensor scaled to ``[0, 1]`` with
bright strokes on a dark background. :func:`preprocess_pil_image` guarantees that
contract for arbitrary uploaded images, including the two common failure cases:
a photograph of paper (dark ink on white) and a photo with a light border.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any

import numpy as np

from shared.datasets import DatasetSpec
from shared.errors import DataDownloadError, InvalidInputError
from shared.paths import data_dir, ensure_parent

logger = logging.getLogger(__name__)

PROJECT_SLUG = "03-handwritten-digit-recognizer"

#: Side length of an MNIST image.
IMAGE_SIZE = 28

#: Number of classes (digits 0-9).
NUM_CLASSES = 10

#: Class labels in model output order. ``torch`` orders logits by class index,
#: so index i always corresponds to the digit i.
CLASS_LABELS: list[str] = [str(digit) for digit in range(NUM_CLASSES)]

#: Primary and mirror sources for the raw MNIST files. The official Yann LeCun
#: host has been unreliable for years, so a maintained mirror is listed first
#: and the original is kept as a fallback.
MNIST_SPECS: dict[str, DatasetSpec] = {
    "train-images-idx3-ubyte.gz": DatasetSpec(
        name="MNIST training images",
        url="https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        filename="train-images-idx3-ubyte.gz",
        mirrors=["https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"],
        approx_size_mb=9.9,
        notes="Official MNIST mirror hosted for PyTorch users.",
    ),
    "train-labels-idx1-ubyte.gz": DatasetSpec(
        name="MNIST training labels",
        url="https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        filename="train-labels-idx1-ubyte.gz",
        mirrors=["https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"],
        approx_size_mb=0.03,
    ),
    "t10k-images-idx3-ubyte.gz": DatasetSpec(
        name="MNIST test images",
        url="https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        filename="t10k-images-idx3-ubyte.gz",
        mirrors=["https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"],
        approx_size_mb=1.6,
    ),
    "t10k-labels-idx1-ubyte.gz": DatasetSpec(
        name="MNIST test labels",
        url="https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        filename="t10k-labels-idx1-ubyte.gz",
        mirrors=["https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"],
        approx_size_mb=0.03,
    ),
}

#: Name of the cached archive produced by :func:`load_mnist`.
CACHE_NAME = "mnist.npz"


def cache_path() -> Path:
    """Return the path of the cached NumPy archive."""
    return data_dir(PROJECT_SLUG) / CACHE_NAME


# --------------------------------------------------------------------------- #
# Raw IDX decoding
# --------------------------------------------------------------------------- #
#
# The IDX format is trivial, so decoding it directly (rather than going through
# torchvision) keeps the download path dependency-free and makes the byte
# offsets explicit for a reader.


def _read_gzip(path: Path) -> np.ndarray:
    """Return the uncompressed bytes of a ``.gz`` IDX file as a uint8 array.

    The four MNIST files are distributed gzip-compressed; reading the file
    without decompressing first silently produces a wrong-length array, so the
    decompression step is kept in one place.
    """
    try:
        payload = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise DataDownloadError(
            f"'{path.name}' is not readable gzip data.",
            hint="Delete it from data/ and re-run the download; the file is probably truncated.",
        ) from error
    return np.frombuffer(payload, dtype=np.uint8)


def _decode_idx_images(path: Path) -> np.ndarray:
    """Decode an IDX image file into a ``(n, 28, 28)`` uint8 array.

    The 16-byte header is the magic number, the image count and two dimensions.
    """
    raw = _read_gzip(path)
    return raw[16:].reshape(-1, IMAGE_SIZE, IMAGE_SIZE).copy()


def _decode_idx_labels(path: Path) -> np.ndarray:
    """Decode an IDX label file into a ``(n,)`` uint8 array (8-byte header)."""
    return _read_gzip(path)[8:].copy()


def download_raw_mnist(*, force: bool = False) -> dict[str, Path]:
    """Download the four MNIST IDX files into the project's ``data`` directory.

    Raises:
        DataDownloadError: When every mirror for any file fails.
    """
    from shared.datasets import fetch_dataset

    directory = data_dir(PROJECT_SLUG)
    local: dict[str, Path] = {}
    failures: list[str] = []

    for filename, spec in MNIST_SPECS.items():
        try:
            local[filename] = fetch_dataset(spec, directory, force=force)
        except DataDownloadError as error:
            failures.append(f"{filename}: {error.message}")

    if failures:
        raise DataDownloadError(
            "Could not download the MNIST dataset.",
            hint=(
                "Failed files:\n- " + "\n- ".join(failures) + "\n\n"
                "MNIST is also bundled with scikit-learn via "
                "`fetch_openml('mnist_784')`, and is mirrored at "
                "https://github.com/golbin/TensorFlow-MNIST. Place the four IDX "
                f"files in '{directory.name}/' manually if the network is restricted."
            ),
        )
    return local


# --------------------------------------------------------------------------- #
# Cached load
# --------------------------------------------------------------------------- #


def load_mnist(
    split: str = "train",
    *,
    force: bool = False,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for ``split`` as normalised float arrays.

    Args:
        split: One of ``"train"`` or ``"test"``.
        force: Rebuild the cache even when it already exists.
        download: Allow network access when the cache is absent.

    Returns:
        ``images`` has shape ``(n, 28, 28)`` and dtype float32 in ``[0, 1]``;
        ``labels`` has shape ``(n,)`` and dtype int64.

    Raises:
        InvalidInputError: For an unknown split name.
        DataDownloadError: When the data is missing and cannot be downloaded.
    """
    if split not in {"train", "test"}:
        raise InvalidInputError(
            f"Unknown MNIST split '{split}'.",
            hint="Choose either 'train' or 'test'.",
        )

    path = cache_path()
    if path.exists() and not force:
        try:
            with np.load(path) as archive:
                return archive[f"x_{split}"], archive[f"y_{split}"]
        except (OSError, ValueError, KeyError) as error:
            logger.warning("Discarding unreadable MNIST cache %s (%s)", path, error)
            path.unlink(missing_ok=True)

    if not download:
        raise DataDownloadError(
            f"MNIST cache not found at '{path.name}'.",
            hint="Run `python train.py` once, or call load_mnist(download=True).",
        )

    raw = download_raw_mnist(force=force)
    x_train = _decode_idx_images(raw["train-images-idx3-ubyte.gz"])
    y_train = _decode_idx_labels(raw["train-labels-idx1-ubyte.gz"])
    x_test = _decode_idx_images(raw["t10k-images-idx3-ubyte.gz"])
    y_test = _decode_idx_labels(raw["t10k-labels-idx1-ubyte.gz"])

    if not (len(x_train) == len(y_train) and len(x_test) == len(y_test)):
        raise DataDownloadError(
            "MNIST images and labels have different lengths.",
            hint="Delete the files in data/ and re-run; the download was probably truncated.",
        )

    ensure_parent(path)
    np.savez_compressed(
        path,
        x_train=x_train.astype(np.float32) / 255.0,
        y_train=y_train.astype(np.int64),
        x_test=x_test.astype(np.float32) / 255.0,
        y_test=y_test.astype(np.int64),
    )
    logger.info("Cached MNIST to %s (train=%d, test=%d)", path, len(x_train), len(x_test))
    return load_mnist(split)


def dataset_summary(*, download: bool = True) -> dict[str, Any]:
    """Return counts and shapes for the dataset panel in the interface."""
    x_train, y_train = load_mnist("train", download=download)
    x_test, y_test = load_mnist("test", download=download)
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES).tolist()
    test_counts = np.bincount(y_test, minlength=NUM_CLASSES).tolist()
    return {
        "train_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "pixel_value_range": [float(x_train.min()), float(x_train.max())],
        "class_labels": CLASS_LABELS,
        "train_class_counts": {label: count for label, count in zip(CLASS_LABELS, train_counts)},
        "test_class_counts": {label: count for label, count in zip(CLASS_LABELS, test_counts)},
        "source": MNIST_SPECS["train-images-idx3-ubyte.gz"].url,
    }


# --------------------------------------------------------------------------- #
# Image preprocessing for inference
# --------------------------------------------------------------------------- #


def preprocess_pil_image(image: Any, *, invert: str = "auto") -> np.ndarray:
    """Convert an arbitrary image into the ``(28, 28)`` float array the net expects.

    The steps are, in order:

    1. Convert to greyscale.
    2. Scale down to fit a 28x28 canvas **without distorting aspect ratio**, then
       centre it on a black background. Distorting a tall digit to fill the square
       is the single most common reason an uploaded digit is misread.
    3. Invert when the border is bright, because MNIST draws bright strokes on a
       dark background whereas photographs are usually dark ink on white paper.
    4. Scale to ``[0, 1]``.

    Args:
        image: A ``PIL.Image`` or any object exposing ``convert``/``size``.
        invert: ``"auto"`` to detect, ``"never"`` to keep, ``"always"`` to invert.

    Returns:
        A ``(28, 28)`` float32 array in ``[0, 1]``.

    Raises:
        InvalidInputError: When the image is degenerate (empty or zero-sized).
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise InvalidInputError(
            "Expected a PIL image.",
            hint="Load the file with PIL.Image.open(...) before calling this function.",
        )

    width, height = image.size
    if width < 1 or height < 1:
        raise InvalidInputError(
            "The uploaded image has zero width or height.",
            hint="Upload a different image file.",
        )

    greyscale = image.convert("L")

    # The bright-border test runs on the *original* image. Testing it after
    # letterboxing would be meaningless, because the padding is black and would
    # make every image look like it already had a dark border.
    should_invert = invert == "always" or (
        invert == "auto" and _has_bright_border(np.asarray(greyscale, dtype=np.float32))
    )

    # Scale to fit inside the canvas, preserving aspect ratio.
    scale = IMAGE_SIZE / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = greyscale.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    offset = ((IMAGE_SIZE - new_size[0]) // 2, (IMAGE_SIZE - new_size[1]) // 2)
    canvas.paste(resized, offset)

    array = np.asarray(canvas, dtype=np.float32)

    if should_invert:
        array = 255.0 - array

    if not np.isfinite(array).all() or array.max() <= 0.0:
        raise InvalidInputError(
            "The image is completely blank after preprocessing.",
            hint="Upload a photo with a clearly visible digit.",
        )

    return array / 255.0


def _has_bright_border(array: np.ndarray) -> bool:
    """Return ``True`` when the image border is brighter than its centre.

    A photograph of a handwritten digit is dark ink on bright paper, so its
    border is brighter than its middle. MNIST is the opposite: bright strokes on
    a black field. Comparing a thin outer frame against the central quarter is
    robust even when the digit touches the edge.
    """
    if array.ndim != 2 or array.size == 0:
        return False
    height, width = array.shape
    if height < 3 or width < 3:
        return False
    border = np.concatenate([array[0, :], array[-1, :], array[:, 0], array[:, -1]])
    inset_y, inset_x = height // 4, width // 4
    centre = array[inset_y : height - inset_y, inset_x : width - inset_x]
    if centre.size == 0:
        return False
    return float(border.mean()) > float(centre.mean())


def to_model_input(array: np.ndarray) -> "Any":
    """Convert a ``(28, 28)`` or ``(N, 28, 28)`` array to a PyTorch batch tensor."""
    import torch

    batch = np.asarray(array, dtype=np.float32)
    if batch.ndim == 2:
        batch = batch[None, ...]
    if batch.ndim != 3 or batch.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise InvalidInputError(
            f"Expected images of shape (28, 28), got {tuple(batch.shape)}.",
            hint="Use preprocess_pil_image() to normalise the image first.",
        )
    return torch.from_numpy(batch).unsqueeze(1)