"""Streamlit interface for the Fruit Image Classifier.

Run from this project's directory::

    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.errors import InvalidInputError  # noqa: E402
from shared.metrics import format_metric  # noqa: E402
from shared.paths import models_dir, portable_display  # noqa: E402
from shared.plotting import bar_chart, confusion_matrix_heatmap, training_curves  # noqa: E402
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header, result_card  # noqa: E402
from shared.ui import (  # noqa: E402
    model_info_panel,
    section,
    show_dataframe,
    show_error,
    show_figure,
    show_metrics_dict,
)

from src.data import (  # noqa: E402
    CLASS_LABELS,
    IMAGE_SIZE,
    dataset_summary,
    load_dataset,
    preprocess_image,
)
from src.model import (  # noqa: E402
    ARCHITECTURE_LABELS,
    HYPERPARAMS,
    FruitClassifier,
    build_training_report,
    load_model,
    save_metrics_report,
    save_model,
    train_model,
    finalise,
)

MODEL_PATH = models_dir("08-fruit-image-classifier") / "fruit_classifier.pt"

#: Uploads larger than this are rejected before decoding.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

ALLOWED_TYPES = ("png", "jpg", "jpeg", "bmp", "webp", "gif", "tif", "tiff")

DISCLAIMER = (
    "The model is trained on Fruits-360, where every fruit was photographed against a "
    "plain background under controlled lighting. Photographs on cluttered backgrounds, "
    "partial fruit, unusual varieties or labels stacked on the fruit will be classified "
    "less reliably than the reported test accuracy suggests."
)


@st.cache_resource(show_spinner=False)
def _load_classifier(path_str: str, mtime: float) -> FruitClassifier:
    """Load the trained network once per artefact version."""
    return load_model(Path(path_str))


@st.cache_data(show_spinner=False)
def _dataset_summary() -> dict:
    """Cached dataset summary."""
    return dataset_summary()


@st.cache_data(show_spinner=False)
def _sample_gallery(per_class: int = 2) -> dict[str, list[tuple[int, str]]]:
    """Return ``{class_name: [(label_index, image_array), ...]}`` for the gallery."""
    data = load_dataset()
    gallery: dict[str, list[tuple[int, str]]] = {}
    for index, label in enumerate(data.labels):
        name = CLASS_LABELS[int(label)] if int(label) < len(CLASS_LABELS) else str(label)
        bucket = gallery.setdefault(name, [])
        if len(bucket) < per_class:
            bucket.append((int(label), data.images[int(index)]))
    return gallery


def _get_classifier() -> FruitClassifier | None:
    """Return the trained classifier, or ``None`` when it has not been built."""
    if not MODEL_PATH.exists():
        return None
    return _load_classifier(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _validate_upload(uploaded_file) -> bytes:
    """Validate an uploaded file and return its bytes."""
    if uploaded_file is None:
        raise InvalidInputError("No file was uploaded.", hint="Choose an image file first.")

    data = uploaded_file.getvalue()
    if not data:
        raise InvalidInputError(
            f"'{uploaded_file.name}' is empty (0 bytes).",
            hint="Re-save the image and upload it again.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidInputError(
            f"'{uploaded_file.name}' is {len(data) / 1e6:.1f} MB, above the "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit.",
            hint="Downscale the image before uploading.",
        )

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_TYPES:
        raise InvalidInputError(
            f"'{suffix or uploaded_file.name}' is not a supported image format.",
            hint=f"Upload one of: {', '.join(ALLOWED_TYPES)}.",
        )
    return data


def _decode_image(data: bytes, name: str = "upload") -> "object":
    """Open bytes as a PIL image, converting failures into friendly errors."""
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidInputError(
            f"'{name}' could not be read as an image.",
            hint=(
                "The file may be corrupt, or a text file with an image extension was "
                f"uploaded. ({type(error).__name__})"
            ),
        ) from error
    return image


def _image_to_rgb_array(image: "object") -> np.ndarray:
    """Render a PIL image into the CHW float array the model expects."""
    return preprocess_image(image, image_size=IMAGE_SIZE)


def _training_progress(epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
    """Streamlit progress callback for the training loop."""
    progress = st.progress(0.0)
    status = st.empty()
    progress.progress(min(epoch / max(1, HYPERPARAMS["epochs"]), 1.0))
    status.text(
        f"Epoch {epoch}/{HYPERPARAMS['epochs']} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f}"
    )


def run_training(architecture: str = "cnn") -> bool:
    """Train and persist the classifier. Returns ``True`` on success."""
    try:
        with st.spinner("Preparing the dataset and training the classifier..."):
            data = load_dataset()
            train_data, val_data, test_data = data.split()
            classifier, model = train_model(
                train_data, val_data, architecture=architecture, progress=_training_progress
            )
            finalise(classifier, model, test_data)
            save_model(classifier)
            report = build_training_report(
                classifier,
                dataset=dataset_summary(),
                hyperparams={"epochs": HYPERPARAMS["epochs"]},
            )
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(
        f"Trained {ARCHITECTURE_LABELS.get(architecture, architecture)} to "
        f"{format_metric(classifier.metrics.get('accuracy'))} test accuracy."
    )
    st.cache_resource.clear()
    return True


def render_prediction(classifier: FruitClassifier) -> None:
    """Render the upload panel, the sample gallery and the result."""
    section(
        "Classify a fruit photo",
        f"Upload a photo of a single fruit, or try one of the {len(CLASS_LABELS)} samples "
        "from the dataset.",
    )

    mode = st.radio(
        "Input source",
        options=["Upload an image", "Dataset samples"],
        horizontal=True,
        key="fruit_input_mode",
    )

    arrays: list[np.ndarray] = []
    captions: list[str] = []

    if mode == "Upload an image":
        uploaded = st.file_uploader(
            "Image file",
            type=list(ALLOWED_TYPES),
            help=(
                f"Any colour mode or size is accepted. The image is converted to RGB and "
                f"resized to {IMAGE_SIZE}x{IMAGE_SIZE}."
            ),
            key="fruit_upload",
        )
        if uploaded is None:
            st.info("Choose an image file, or switch to **Dataset samples**.")
            return
        try:
            data = _validate_upload(uploaded)
            image = _decode_image(data, uploaded.name)
            st.image(image, caption=f"Uploaded: {uploaded.name}", width=200)
            arrays.append(_image_to_rgb_array(image))
            captions.append(f"{image.width}x{image.height} {image.mode} -> {IMAGE_SIZE}x{IMAGE_SIZE} RGB")
        except Exception as error:  # noqa: BLE001 - UI boundary
            show_error(error)
            return
    else:
        try:
            gallery = _sample_gallery()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        if not gallery:
            st.warning("The dataset cache is empty. Run `python train.py` first.")
            return

        chosen = st.selectbox("Pick a fruit", list(gallery), key="fruit_sample_label")
        matches = gallery[chosen]
        columns = st.columns(len(matches))
        for column, (label_index, array) in zip(columns, matches, strict=False):
            with column:
                st.image(
                    np.transpose(array, (1, 2, 0)),
                    caption=f"True class: {chosen}",
                    width=120,
                )
        arrays = [array for _label_index, array in matches]
        captions = [f"Cached dataset sample, true class {chosen}" for _ in matches]

    # Classification runs immediately rather than behind a button: inference on a
    # single image takes milliseconds, so a button would add a click without
    # making the result any more deliberate.
    st.caption("Results update as soon as an image is selected.")
    try:
        batch = np.stack(arrays)
        result = classifier.predict(batch)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return

    for index, (label, confidence, caption) in enumerate(
        zip(result.labels, result.confidence(), captions, strict=False)
    ):
        result_card(label, f"{confidence:.1%} confidence", accent=PALETTE["accent"])
        st.caption(caption)
        st.write(
            [
                {
                    "rank": rank,
                    "fruit": item["label"],
                    "probability": round(item["probability"], 4),
                }
                for rank, item in enumerate(result.top_k(3)[index], start=1)
            ]
        )

    show_figure(
        bar_chart(
            CLASS_LABELS,
            result.probabilities[0].tolist(),
            title=f"Class probabilities for '{result.labels[0]}'",
            ylabel="Probability",
            width=7.6,
            height=3.2,
        )
    )


def render_evaluation(classifier: FruitClassifier) -> None:
    """Render held-out metrics, per-class table and confusion matrix."""
    metrics = classifier.metrics
    if not metrics:
        return

    section(
        "Held-out evaluation",
        f"Measured on {metrics.get('n_test', 0):,} images the network never saw during "
        "training. Augmentation is not applied to validation or test images.",
    )

    show_metrics_dict(
        metrics,
        keys=["accuracy", "precision", "recall", "f1"],
        labels={
            "accuracy": "ACCURACY",
            "precision": "PRECISION (macro)",
            "recall": "RECALL (macro)",
            "f1": "F1 (macro)",
        },
    )
    if metrics.get("roc_auc_ovr") is not None:
        st.caption(f"One-vs-rest ROC-AUC: **{format_metric(metrics['roc_auc_ovr'])}**")

    left, right = st.columns([1, 1])
    with left:
        show_figure(
            confusion_matrix_heatmap(
                metrics["confusion_matrix"],
                metrics.get("labels", CLASS_LABELS),
                title="Confusion matrix (test split)",
                width=7.0,
                height=5.6,
            )
        )
    with right:
        if classifier.per_class:
            frame = pd.DataFrame(classifier.per_class)
            for column in ("precision", "recall", "f1"):
                frame[column] = frame[column].round(4)
            show_dataframe(frame, caption="Per-class precision, recall, F1 and support.")
        else:
            st.caption("Per-class report unavailable.")


def render_training_curves(classifier: FruitClassifier) -> None:
    """Render per-epoch curves when history is available."""
    history = classifier.history or {}
    if not history.get("train_loss"):
        return
    section("Training curves", "Loss and accuracy per epoch on the training and validation splits.")
    show_figure(
        training_curves(
            {
                "train_loss": history.get("train_loss", []),
                "val_loss": history.get("val_loss", []),
                "train_accuracy": history.get("train_accuracy", []),
                "val_accuracy": history.get("val_accuracy", []),
            },
            title="Fruit CNN",
            width=9.0,
            height=4.6,
        )
    )
    accuracies = history.get("val_accuracy", [])
    if accuracies:
        best = int(np.argmax(accuracies)) + 1
        st.caption(
            f"Best validation accuracy {format_metric(max(accuracies))} in epoch {best} of "
            f"{classifier.epochs}. A widening gap between training and validation accuracy "
            "would indicate overfitting despite augmentation."
        )


def render_dataset_panel() -> None:
    """Show dataset provenance and class balance."""
    with st.expander("Dataset details"):
        try:
            summary = _dataset_summary()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        if not summary.get("prepared"):
            st.warning("The dataset has not been prepared yet. Run `python train.py`.")
            return

        counts = summary.get("class_counts", {})
        metric_row(
            [
                ("Images", f"{summary.get('total_images', 0):,}", "after grouping"),
                ("Classes", str(summary.get("n_classes", 0)), ", ".join(summary.get("classes", [])[:4]) + "..."),
                ("Image size", f"{summary.get('image_size')}x{summary.get('image_size')}", "RGB"),
            ]
        )
        show_figure(
            bar_chart(
                list(counts),
                list(counts.values()),
                title="Images per fruit",
                ylabel="Images",
                width=7.6,
                height=3.4,
            )
        )
        st.markdown(f"**Source:** `{summary.get('source')}`")
        st.caption(
            "Fruits-360 labels images by 113 fruit *varieties*. This project groups them up "
            "into seven fruits (every `Apple*` variety becomes `Apple`, and so on), which "
            "keeps 5,150 images and makes the problem learnable. Augmentation "
            "(flips, rotation, brightness and contrast jitter) is applied to the training "
            "split only."
        )


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Fruit Image Classifier", page_icon="08", layout="wide")
    inject_theme()

    page_header(
        "Fruit Image Classifier",
        "A convolutional network trained on Fruits-360 photographs that recognises the "
        "fruit in an uploaded image and shows how confident it is.",
        eyebrow="Project 08",
        chips=["Computer vision", "PyTorch CNN", "Fruits-360", f"{len(CLASS_LABELS)} classes"],
    )

    with st.sidebar:
        st.markdown("### Configuration")
        architecture = st.selectbox(
            "Architecture",
            options=["cnn", "mobilenetv2"],
            format_func=lambda key: ARCHITECTURE_LABELS[key],
            key="fruit_architecture",
        )
        st.markdown("---")
        st.markdown("### Actions")
        train_clicked = st.button("Train / retrain", width="stretch")
        if architecture == "mobilenetv2":
            st.caption(
                "Transfer learning downloads ~14 MB of ImageNet weights the first time it "
                "is used."
            )

    classifier = _get_classifier()
    if train_clicked:
        if run_training(architecture):
            classifier = _get_classifier()
            st.rerun()

    if classifier is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Train it by running:\n\n"
            "```bash\npython train.py\n```"
        )
        render_dataset_panel()
        note(DISCLAIMER)
        return

    render_prediction(classifier)
    render_evaluation(classifier)
    render_training_curves(classifier)
    render_dataset_panel()

    model_info_panel(
        algorithm=ARCHITECTURE_LABELS.get(classifier.architecture, classifier.architecture),
        trained_on=(
            f"{classifier.n_train:,} training images, {classifier.n_val:,} validation, "
            f"{classifier.n_test:,} test"
        ),
        metrics=classifier.metrics,
        extra={
            "Trained at": classifier.trained_at,
            "Epochs": str(classifier.epochs),
            "Parameters": f"{classifier.n_parameters:,}",
            "Framework": "PyTorch (CPU / Apple MPS / CUDA)",
            "Augmentation": "flip, rotation 15deg, brightness/contrast/saturation jitter",
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()