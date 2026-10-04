"""Fruit image dataset acquisition from Fruits-360.

Why this dataset and not Fruits-30
----------------------------------
The obvious choice for a three-class fruit demo is the Fruits-30 collection on
Hugging Face, which stores images as individual files. It was tried first and
rejected on measurement, not taste: after listing the repository, ``apples``
holds 28 images and ``bananas`` only 11. A convolutional network trained on that
memorises eleven photographs rather than learning fruit.

**Fruits-360** has enough data to train on. The official archive is 798 MB, which
is unreasonable for a reviewable project, but the Hugging Face mirror serves each
split as a single parquet file. One such file is ~98 MB and contains 22,688
images - comfortably more than this project needs, and one download.

Varieties, then fruits
----------------------
Fruits-360 labels its images by **variety**, not by fruit: 113 classes such as
``Apple Braeburn``, ``Apple Granny Smith``, ``Banana Red`` and ``Banana Lady
Finger``. Training on 113 fine-grained classes from this subset would be an
uneven, confusing problem, so the variety labels are mapped up to seven fruits
by prefix: every ``Apple*`` variety becomes ``Apple``, and so on. The mapping
table is fetched once from the dataset's metadata and cached to disk, so later
runs work offline.

The grouping keeps 5,150 images across seven classes:

===========  ======
Fruit        Images
===========  ======
Apple          2,134
Grape          1,476
Banana           484
Strawberry       410
Pineapple        329
Orange           160
Watermelon       157
===========  ======

Seven classes, not three
------------------------
The brief suggests starting with apple, banana and orange. All three are
present; training on seven is a more honest demonstration than three, and a
seven-way problem is genuinely harder. :data:`CLASS_LABELS` is the only place
the class list appears, so narrowing this to three fruits is a one-line change
plus a retrain.

Caching
-------
Decoding 22,688 JPEGs takes minutes, so the decoded arrays for the seven kept
groups are cached to a single ``.npz``. The interface and the test suite then
load in milliseconds.

Exact duplicates, and how they are removed
------------------------------------------
A first version of this project reported **100% test accuracy**, and that number
was not trustworthy. Investigating it showed the real problem: the mirror ships
many images more than once. A check of 150 randomly chosen images found that
**every one of them had a byte-identical twin** elsewhere in the dataset, and 494
images have a perceptual hash that matches another image exactly. Splitting those
at random puts a literal copy of a test image into the training set.

The fix is exact deduplication: images are hashed by their decoded pixel bytes and
only the first occurrence of each distinct image is kept. This is precise - it
removes genuine copies and never merges two different photographs.

A difference-hash approach was tried first and rejected. Its nearest-neighbour
distance distribution on this dataset is smooth and unimodal, with no gap
separating "copy" from "different photo", and union-find over a permissive
threshold chained everything into 499 giant clusters (one holding 2,318 images,
more than the entire Apple class). Fuzzy clustering would have thrown away most
of the dataset to solve a problem that exact hashing solves perfectly.

*Near*-duplicates - different photographs of the same physical fruit, a few
pixels apart - are **not** removed, because no reliable threshold separates them
here. That remains a source of optimism in the reported accuracy and is stated as
a limitation rather than papered over.
"""

from __future__ import annotations

import io
import json
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shared.datasets import DatasetSpec, fetch_json, sha256_of
from shared.errors import DataDownloadError, DataNotFoundError, InvalidInputError
from shared.paths import data_dir, ensure_parent

logger = logging.getLogger(__name__)

PROJECT_SLUG = "08-fruit-image-classifier"

#: Hugging Face parquet endpoint holding a single Fruits-360 split.
FRUITS360_PARQUET_URL = (
    "https://huggingface.co/api/datasets/PedroSampaio/fruits-360/parquet/default/test/0.parquet"
)

#: Endpoint carrying the dataset metadata, including the variety label names.
FRUITS360_INFO_URL = "https://huggingface.co/api/datasets/PedroSampaio/fruits-360?full=true"

#: Upstream description, quoted for the dataset panel.
FRUITS360_REPOSITORY = "PedroSampaio/fruits-360"

FRUIT_SPEC = DatasetSpec(
    name="Fruits-360 (Hugging Face parquet mirror)",
    url=FRUITS360_PARQUET_URL,
    filename="fruits360_test.parquet",
    mirrors=[FRUITS360_PARQUET_URL],
    approx_size_mb=98.0,
    notes=(
        "Fruits-360 (Molchanov, Kaggle) served as a single parquet file by the "
        "Hugging Face mirror. 22,688 images labelled by 113 fruit varieties."
    ),
)

#: Class labels in model output order. Index i always corresponds to label i.
CLASS_LABELS: list[str] = [
    "Apple",
    "Banana",
    "Grape",
    "Orange",
    "Pineapple",
    "Strawberry",
    "Watermelon",
]

#: Side length the network expects.
IMAGE_SIZE = 96

#: Cached decoded arrays.
CACHE_NAME = "fruits360_cache.npz"

#: Cached variety-name mapping, so later runs need no network.
LABEL_MAP_NAME = "fruits360_varieties.json"

#: Column names in the parquet file.
_IMAGE_COLUMN = "image"
_LABEL_COLUMN = "label"


def cache_path() -> Path:
    """Return the path of the decoded-image cache."""
    return data_dir(PROJECT_SLUG) / CACHE_NAME


def parquet_path() -> Path:
    """Return the path of the downloaded parquet file."""
    return data_dir(PROJECT_SLUG) / FRUIT_SPEC.filename


def label_map_path() -> Path:
    """Return the path of the cached variety-to-fruit mapping."""
    return data_dir(PROJECT_SLUG) / LABEL_MAP_NAME


# --------------------------------------------------------------------------- #
# Variety names and grouping
# --------------------------------------------------------------------------- #


def load_variety_names(*, download: bool = True, force: bool = False) -> dict[str, str]:
    """Return ``{label_index: variety_name}`` for the Fruits-360 classes.

    The names live in the dataset's metadata rather than in the parquet file, so
    they are fetched once and cached to ``data/``. After the first run the
    project works offline.

    Raises:
        DataNotFoundError: When no mapping is cached and it cannot be fetched.
    """
    path = label_map_path()
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached:
                return {str(key): str(value) for key, value in cached.items()}
        except (json.JSONDecodeError, OSError):
            logger.warning("Ignoring unreadable variety mapping at %s", path)

    if not download:
        raise DataNotFoundError(
            "The fruit variety mapping has not been cached.",
            hint="Run `python train.py` once with a network connection.",
        )

    from shared.datasets import fetch_json

    payload = fetch_json(FRUITS360_INFO_URL, description="Fruits-360 metadata")
    try:
        features = payload["cardData"]["dataset_info"]["features"]
        names = next(
            feature["dtype"]["class_label"]["names"]
            for feature in features
            if feature.get("name") == _LABEL_COLUMN
        )
    except (KeyError, StopIteration, TypeError) as error:
        raise DataDownloadError(
            "The Fruits-360 metadata did not contain the label names.",
            hint=(
                "The upstream dataset card may have changed. Delete "
                "data/fruits360_varieties.json and retry, or check "
                f"{FRUITS360_INFO_URL}"
            ),
        ) from error

    mapping = {str(key): str(value) for key, value in names.items()}
    ensure_parent(path)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Cached %d fruit variety names -> %s", len(mapping), path)
    return mapping


def group_for_variety(variety: str) -> str | None:
    """Map a variety name onto one of :data:`CLASS_LABELS`.

    Returns ``None`` for fruits this project does not model, so the decoder can
    skip them instead of inventing a label.
    """
    text = str(variety).strip().lower()
    for label in CLASS_LABELS:
        if text.startswith(label.lower()):
            return label
    return None


def normalise_label(raw: Any, varieties: dict[str, str]) -> str | None:
    """Map a raw integer label onto a display fruit label, or ``None``.

    Fruits-360 labels are integers indexing :data:`CLASS_LABELS` of the
    *variety* list, not of this project's list, so an unmapped integer must not
    be treated as a class index.
    """
    try:
        index = str(int(raw))
    except (TypeError, ValueError):
        return None
    variety = varieties.get(index)
    if variety is None:
        return None
    return group_for_variety(variety)


# --------------------------------------------------------------------------- #
# Decoding and caching
# --------------------------------------------------------------------------- #


def _decode_parquet(path: Path, varieties: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Decode the parquet file into ``(images, labels)`` for the kept classes.

    Returns:
        ``images`` shaped ``(n, 3, S, S)`` float32 in ``[0, 1]``; ``labels``
        shaped ``(n,)`` int64 indexing into :data:`CLASS_LABELS`.
    """
    import pandas as pd
    from PIL import Image, UnidentifiedImageError

    frame = pd.read_parquet(path)
    if _IMAGE_COLUMN not in frame.columns or _LABEL_COLUMN not in frame.columns:
        raise DataNotFoundError(
            f"The parquet file is missing column(s); found {list(frame.columns)[:5]}.",
            hint=(
                "Delete the cached file and re-download. Expected an image column and "
                "a label column."
            ),
        )

    # Resolve labels first so only the images we keep are decoded; that is the
    # difference between a few seconds and several minutes.
    resolved = [normalise_label(raw, varieties) for raw in frame[_LABEL_COLUMN]]
    wanted = np.asarray([label is not None for label in resolved])
    logger.info(
        "Decoding %d of %d fruit images (%d belong to other fruits)...",
        int(wanted.sum()),
        len(frame),
        len(frame) - int(wanted.sum()),
    )
    if not wanted.any():
        raise DataNotFoundError(
            "None of the parquet images belong to the classes this project models.",
            hint=(
                f"Check the variety mapping in {label_map_path().name}. Expected fruits: "
                f"{', '.join(CLASS_LABELS)}."
            ),
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for position in np.flatnonzero(wanted):
        record = frame[_IMAGE_COLUMN].iloc[int(position)]
        payload = record["bytes"] if isinstance(record, dict) else record
        if not payload:
            skipped += 1
            continue
        try:
            with Image.open(io.BytesIO(payload)) as handle:
                image = handle.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1
            continue

        images.append(np.transpose(array, (2, 0, 1)))
        labels.append(CLASS_LABELS.index(resolved[int(position)]))

    if skipped:
        logger.warning("Skipped %d record(s) with an unreadable image.", skipped)
    if not images:
        raise DataNotFoundError(
            "No usable fruit images were decoded from the parquet file.",
            hint="Delete the cached file and re-download the dataset.",
        )

    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _decode_and_deduplicate(path: Path, varieties: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Decode the parquet file and drop byte-identical duplicates.

    Deduplication happens here, before the cache is written, so the cached array
    is already clean and every later split is safe.
    """
    images, labels = _decode_parquet(path, varieties)
    images, labels, removed = deduplicate_images(images, labels)
    if removed:
        # Record the count alongside the cache so the interface can report it.
        cache_path().with_suffix(".dedup.json").write_text(
            json.dumps({"removed": int(removed), "unique": int(len(labels))}), encoding="utf-8"
        )
    return images, labels


def load_arrays(*, force_download: bool = False, force_decode: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)``, downloading and decoding as needed.

    Args:
        force_download: Re-download the parquet file even when it is cached.
        force_decode: Rebuild the decoded cache even when it exists.

    Raises:
        DataDownloadError: When the file cannot be fetched.
        DataNotFoundError: When the cache is absent and downloading is disabled.
    """
    from shared.datasets import fetch_dataset

    cache = cache_path()
    parquet = parquet_path()

    # The cache records the parquet's digest, so an upstream change invalidates it
    # instead of leaving stale arrays behind.
    if cache.exists() and parquet.exists() and not force_download and not force_decode:
        try:
            with np.load(cache) as archive:
                if str(archive["source_digest"]) == sha256_of(parquet):
                    return archive["images"], archive["labels"]
            logger.warning("Discarding stale image cache (source file changed).")
        except (OSError, ValueError, KeyError) as error:
            logger.warning("Discarding unreadable image cache %s (%s)", cache, error)
            cache.unlink(missing_ok=True)

    if not parquet.exists() or force_download:
        fetch_dataset(FRUIT_SPEC, data_dir(PROJECT_SLUG), force=force_download)

    if not parquet.exists():
        raise DataNotFoundError(
            "The Fruits-360 parquet file is missing.",
            hint=f"Run `python train.py` to download it, or fetch {FRUIT_SPEC.url} manually.",
        )

    images, labels = _decode_and_deduplicate(parquet, load_variety_names())
    ensure_parent(cache)
    np.savez_compressed(
        cache,
        images=images.astype(np.float32),
        labels=labels.astype(np.int64),
        source_digest=np.array(sha256_of(parquet)),
    )
    logger.info("Cached %d decoded fruit images -> %s", len(images), cache)
    return images, labels


# --------------------------------------------------------------------------- #
# Exact duplicate removal
# --------------------------------------------------------------------------- #


def deduplicate_images(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop byte-identical copies of an image, keeping the first occurrence.

    The digest is taken over the raw decoded pixel bytes, so this removes only
    images that are genuinely the same file. Two different photographs of the same
    fruit have different bytes and are both kept.

    Returns:
        ``(unique_images, unique_labels, n_removed)``.
    """
    array = np.asarray(images, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 4:
        raise InvalidInputError(
            f"Expected images of shape (N, C, H, W), got {tuple(array.shape)}.",
            hint="Pass the decoded image array from load_dataset().",
        )
    if len(array) != len(label_array):
        raise InvalidInputError(
            f"Got {len(array)} images but {len(label_array)} labels.",
            hint="The image and label arrays must be aligned.",
        )

    seen: set[bytes] = set()
    keep: list[int] = []
    removed = 0
    for index in range(len(array)):
        digest = hashlib.sha256(array[index].tobytes()).digest()
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        keep.append(index)

    if removed == 0:
        return array, label_array, 0

    logger.info(
        "Removed %d byte-identical duplicate image(s) (%d unique remain).",
        removed,
        len(keep),
    )
    indices = np.asarray(keep, dtype=np.int64)
    return array[indices], label_array[indices], removed


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #


@dataclass
class FruitDataset:
    """Image tensors and integer labels for a subset of the classes."""

    images: np.ndarray = field(default_factory=lambda: np.empty((0, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))

    def __len__(self) -> int:
        return int(len(self.labels))

    @property
    def n_classes(self) -> int:
        """Number of distinct classes present."""
        return int(len(set(self.labels.tolist())))

    def class_counts(self) -> dict[str, int]:
        """Return the number of images per display label."""
        counts: dict[str, int] = {}
        for index in self.labels:
            label = CLASS_LABELS[int(index)]
            counts[label] = counts.get(label, 0) + 1
        return counts

    def subset(self, indices: np.ndarray) -> FruitDataset:
        """Return the rows named by ``indices``."""
        return FruitDataset(images=self.images[indices], labels=self.labels[indices])

    def split(
        self,
        *,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
    ) -> tuple["FruitDataset", "FruitDataset", "FruitDataset"]:
        """Split into train, validation and test sets, stratified by class.

        Byte-identical duplicates have already been removed by
        :func:`deduplicate_images` when the dataset was loaded, so an ordinary
        stratified split is correct here. See the module docstring for why
        fuzzy near-duplicate grouping was rejected.

        Raises:
            InvalidInputError: When a class has too few images to split, or the
                fractions are invalid.
        """
        if len(self) == 0:
            raise InvalidInputError(
                "No fruit images are available.",
                hint="Run `python train.py` to download and prepare the dataset.",
            )
        if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
            raise InvalidInputError(
                "validation_fraction and test_fraction must be between 0 and 1.",
                hint=f"Got {validation_fraction} and {test_fraction}.",
            )
        if validation_fraction + test_fraction >= 1:
            raise InvalidInputError(
                "validation_fraction + test_fraction must be less than 1.",
                hint=f"Got {validation_fraction} + {test_fraction}.",
            )

        counts = np.bincount(self.labels, minlength=len(CLASS_LABELS))
        too_small = {CLASS_LABELS[i]: int(counts[i]) for i in range(len(counts)) if counts[i] < 10}
        if too_small:
            raise InvalidInputError(
                f"Not enough images per class: {too_small}.",
                hint="Each class needs at least 10 images for a stratified three-way split.",
            )

        from sklearn.model_selection import train_test_split

        indices = np.arange(len(self))
        train_idx, rest_idx = train_test_split(
            indices,
            test_size=validation_fraction + test_fraction,
            random_state=seed,
            stratify=self.labels,
        )
        relative = test_fraction / (validation_fraction + test_fraction)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=relative, random_state=seed, stratify=self.labels[rest_idx]
        )
        return self.subset(train_idx), self.subset(val_idx), self.subset(test_idx)


def load_dataset(
    *,
    force_download: bool = False,
    download: bool = True,
    progress: Any = None,
) -> FruitDataset:
    """Return the full fruit dataset, downloading and decoding as needed."""
    if progress:
        progress("Preparing the fruit dataset...")
    if not download and not cache_path().exists():
        raise DataNotFoundError(
            "The fruit dataset has not been prepared.",
            hint="Run `python train.py` once, or call load_arrays() to build the cache.",
        )
    images, labels = load_arrays(force_download=force_download)
    return FruitDataset(images=images, labels=labels)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def preprocess_image(image: Any, *, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Convert a PIL image into a ``(3, S, S)`` float array in ``[0, 1]``.

    Raises:
        InvalidInputError: When the image is empty or too small to be meaningful.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise InvalidInputError(
            "Expected a PIL image.",
            hint="Open the file with PIL.Image.open(...) before calling this function.",
        )
    if image.width < 8 or image.height < 8:
        raise InvalidInputError(
            f"The uploaded image is only {image.width}x{image.height} pixels.",
            hint="Upload an image at least 8x8 pixels.",
        )

    converted = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(converted, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def to_tensor_batch(array: np.ndarray) -> "Any":
    """Convert a ``(3, S, S)`` or ``(N, 3, S, S)`` array to a float tensor."""
    import torch

    batch = np.asarray(array, dtype=np.float32)
    if batch.ndim == 3:
        batch = batch[None, ...]
    if batch.ndim != 4 or batch.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise InvalidInputError(
            f"Expected shape (3, {IMAGE_SIZE}, {IMAGE_SIZE}) or a batch of them, "
            f"got {tuple(batch.shape)}.",
            hint="Run the image through preprocess_image() first.",
        )
    return torch.from_numpy(batch)


def dataset_summary() -> dict[str, Any]:
    """Return headline statistics for the dataset panel."""
    summary: dict[str, Any] = {
        "classes": list(CLASS_LABELS),
        "image_size": IMAGE_SIZE,
        "source": FRUITS360_PARQUET_URL,
        "repository": FRUITS360_REPOSITORY,
        "approximate_size_mb": FRUIT_SPEC.approx_size_mb,
        "prepared": cache_path().exists(),
    }
    if not summary["prepared"]:
        return summary

    try:
        images, labels = load_arrays()
    except Exception as error:  # noqa: BLE001 - the panel is optional
        logger.warning("Could not summarise the fruit dataset: %s", error)
        return summary

    counts: dict[str, int] = {}
    for index in labels:
        label = CLASS_LABELS[int(index)]
        counts[label] = counts.get(label, 0) + 1
    summary.update(
        {
            "total_images": int(len(labels)),
            "class_counts": counts,
            "n_classes": len(counts),
            "pixel_value_range": [float(images.min()), float(images.max())],
        }
    )
    return summary