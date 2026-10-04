"""Streamlit interface for the Handwritten Digit Recogniser.

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
    NUM_CLASSES,
    dataset_summary,
    load_mnist,
    preprocess_pil_image,
)
from src.model import (  # noqa: E402
    HYPERPARAMS,
    build_model,
    count_parameters,
    finalise,
    load_model,
    save_metrics_report,
    save_model,
    train_model,
)

MODEL_PATH = models_dir("03-handwritten-digit-recognizer") / "digit_cnn.pt"

#: Uploads larger than this are rejected before decoding, to bound memory use.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

#: Accepted upload extensions.
ALLOWED_TYPES = ("png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp")

DISCLAIMER = (
    "The model is trained on MNIST, which contains centred, size-normalised digits "
    "written on a white background. Photographs of handwriting, screen captures and "
    "stylised fonts fall outside that distribution and accuracy on them will be "
    "noticeably lower than the reported test figure."
)


@st.cache_resource(show_spinner=False)
def _load_recogniser(path_str: str, mtime: float):
    """Load the trained network once per artefact version."""
    return load_model(Path(path_str))


@st.cache_data(show_spinner=False)
def _sample_digits(count: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a cached slice of the MNIST test split for the built-in examples."""
    images, labels = load_mnist("test")
    end = min(len(images), offset + count)
    return images[offset:end], labels[offset:end]


@st.cache_data(show_spinner=False)
def _dataset_summary() -> dict:
    """Cached dataset summary."""
    return dataset_summary()


def _get_recogniser():
    """Return the trained recogniser, or ``None`` when it has not been built yet."""
    if not MODEL_PATH.exists():
        return None
    return _load_recogniser(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _validate_upload(uploaded_file) -> bytes:
    """Check the uploaded file and return its bytes.

    Raises:
        InvalidInputError: When the file is empty, too large or the wrong type.
    """
    if uploaded_file is None:
        raise InvalidInputError(
            "No file was uploaded.",
            hint="Choose an image file, or use the MNIST examples below.",
        )

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
            hint="Downscale or crop the image before uploading.",
        )

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_TYPES:
        raise InvalidInputError(
            f"'{suffix or uploaded_file.name}' is not a supported image format.",
            hint=f"Upload one of: {', '.join(ALLOWED_TYPES)}.",
        )
    return data


def _decode_image(data: bytes, name: str = "upload") -> "object":
    """Open bytes as a PIL image, converting any failure into a friendly error."""
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidInputError(
            f"'{name}' could not be read as an image.",
            hint=(
                "The file may be corrupt, or its extension may not match its contents. "
                f"({type(error).__name__})"
            ),
        ) from error
    return image


def _run_training() -> bool:
    """Train and persist the network. Returns ``True`` on success."""
    try:
        with st.spinner("Loading MNIST and training the CNN. This can take a few minutes..."):
            x_train, y_train = load_mnist("train")
            x_test, y_test = load_mnist("test")
            recogniser, model = train_model(
                x_train, y_train, epochs=HYPERPARAMS["epochs"], progress=_training_progress
            )
            finalise(recogniser, model, x_test, y_test)
            save_model(recogniser)
            report = build_report(recogniser)
            save_metrics_report(report)
    except Exception as error:  # noqa: BLE001 - UI boundary
        show_error(error)
        return False

    st.success(
        f"Trained the digit CNN to {format_metric(recogniser.metrics['accuracy'])} test accuracy "
        f"and saved it to `{portable_display(MODEL_PATH)}`."
    )
    st.cache_resource.clear()
    return True


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


def build_report(recogniser):
    """Assemble the JSON training report (mirrors ``train.py``)."""
    from src.model import build_training_report

    summary = _dataset_summary()
    return build_training_report(
        recogniser,
        dataset={
            "name": "MNIST",
            "train_images_available": summary["train_images"],
            "test_images_available": summary["test_images"],
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "class_labels": CLASS_LABELS,
            "source": summary["source"],
        },
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_prediction(recogniser) -> None:
    """Render the upload / example panel and the prediction result."""
    section(
        "Recognise a digit",
        "Upload a photo or screenshot of a single handwritten digit, or try a sample "
        "from the MNIST test split.",
    )

    mode = st.radio(
        "Input source",
        options=["Upload an image", "Use MNIST samples"],
        horizontal=True,
        key="digit_input_mode",
    )

    array: np.ndarray | None = None
    caption = ""

    if mode == "Upload an image":
        uploaded = st.file_uploader(
            "Image file",
            type=list(ALLOWED_TYPES),
            help=(
                "Any size and colour mode is accepted. The image is converted to "
                "greyscale, scaled to fit 28x28 without distortion, and inverted if "
                "needed so the digit is bright on a dark background."
            ),
            key="digit_upload",
        )
        if uploaded is not None:
            try:
                data = _validate_upload(uploaded)
                image = _decode_image(data, uploaded.name)
                st.image(image, caption=f"Uploaded: {uploaded.name}", width=180)
                array = preprocess_pil_image(image)
                caption = (
                    f"Preprocessed to {IMAGE_SIZE}x{IMAGE_SIZE} greyscale "
                    f"(original {image.width}x{image.height}, mode {image.mode})."
                )
            except Exception as error:  # noqa: BLE001 - UI boundary
                show_error(error)
                st.info("Fix the image above, or switch to the MNIST samples tab.")
                return
    else:
        count, offset = 12, 0
        images, labels = _sample_digits(count, offset)
        options = {f"Sample {offset + i + 1} (true digit: {labels[i]})": i for i in range(len(images))}
        choice = st.selectbox("Choose a sample", options=list(options), key="digit_sample")
        index = options[choice]
        array = images[index]
        st.image(array, caption=f"True digit: {labels[index]}", width=180, clamp=True)
        caption = "Sampled directly from the MNIST test split (already 28x28, normalised)."

    st.caption(caption)

    if st.button("Recognise digit", type="primary", width="stretch"):
        try:
            result = recogniser.predict(array)
            top_k = result.top_k(3)
            confidence = result.confidence()[0]
        except Exception as error:  # noqa: BLE001 - UI boundary
            show_error(error)
            return

        result_card("Predicted digit", result.labels[0], accent=PALETTE["accent"])
        st.caption(f"Model confidence: **{confidence:.1%}**")

        show_figure(
            bar_chart(
                CLASS_LABELS,
                result.probabilities[0].tolist(),
                title="Class probabilities",
                ylabel="Probability",
                width=6.6,
                height=3.0,
            )
        )

        st.markdown("**Top predictions**")
        show_dataframe(
            [
                {"rank": rank, "digit": item["label"], "probability": round(item["probability"], 4)}
                for rank, item in enumerate(top_k[0], start=1)
            ],
            caption="The three most probable digits.",
        )


def render_evaluation(recogniser) -> None:
    """Render held-out metrics, per-class table and confusion matrix."""
    metrics = recogniser.metrics
    if not metrics:
        return

    section(
        "Held-out evaluation",
        f"Measured on the {metrics.get('n_test', 0):,} official MNIST test images, which the "
        "network never saw during training.",
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
    st.caption(f"Mean top-1 confidence: **{format_metric(metrics.get('mean_confidence', 0.0))}**")

    left, right = st.columns([1, 1])
    with left:
        show_figure(
            confusion_matrix_heatmap(
                metrics["confusion_matrix"],
                CLASS_LABELS,
                title="Confusion matrix (test split)",
                width=6.0,
                height=5.4,
            )
        )
    with right:
        if recogniser.per_class:
            show_dataframe(
                pd_per_class(recogniser.per_class),
                caption="Per-class precision, recall, F1 and support.",
            )
        else:
            st.caption("Per-class report unavailable.")


def pd_per_class(rows: list[dict]) -> "object":
    """Round the per-class rows for display."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    for column in ("precision", "recall", "f1"):
        if column in frame.columns:
            frame[column] = frame[column].round(4)
    return frame


def render_training_curves(recogniser) -> None:
    """Render the training and validation curves when history is available."""
    history = recogniser.history or {}
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
            title="MNIST CNN",
            width=9.0,
            height=4.6,
        )
    )
    best_epoch = int(np.argmax(history.get("val_accuracy", [0.0]))) + 1
    st.caption(
        f"Best validation accuracy was reached in epoch {best_epoch}. "
        f"Validation accuracy: {format_metric(max(history.get('val_accuracy', [0.0])))}."
    )


def render_dataset_panel() -> None:
    """Show dataset provenance and class balance."""
    with st.expander("Dataset details"):
        try:
            summary = _dataset_summary()
        except Exception as error:  # noqa: BLE001 - optional panel
            show_error(error)
            return

        metric_row(
            [
                ("Training images", f"{summary['train_images']:,}", "60,000 official"),
                ("Test images", f"{summary['test_images']:,}", "10,000 official"),
                ("Image size", f"{summary['image_size']}x{summary['image_size']}", "greyscale"),
                ("Classes", str(summary["num_classes"]), "digits 0-9"),
            ]
        )
        counts = summary["train_class_counts"]
        show_figure(
            bar_chart(
                CLASS_LABELS,
                [counts[label] for label in CLASS_LABELS],
                title="Training set class balance",
                ylabel="Images",
                width=7.0,
                height=3.2,
            )
        )
        st.markdown(f"**Source:** `{summary['source']}`")
        st.caption(
            "MNIST images are already centred and size-normalised, which is why the "
            "preprocessing step in the app mainly has to handle greyscale conversion, "
            "scaling and polarity."
        )


def render_sidebar() -> bool:
    """Render the sidebar. Returns ``True`` when retraining was requested."""
    with st.sidebar:
        st.markdown("### Configuration")
        st.checkbox("Show preprocessing detail", value=True, key="digit_show_detail")
        st.markdown("---")
        st.markdown("### Actions")
        train_clicked = st.button("Train / retrain", width="stretch")
        st.caption(
            f"Trains the CNN for {HYPERPARAMS['epochs']} epochs on the full MNIST training "
            "split. Takes a few minutes on CPU."
        )
    return train_clicked


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Handwritten Digit Recogniser", page_icon="03", layout="wide")
    inject_theme()

    page_header(
        "Handwritten Digit Recogniser",
        "A convolutional neural network trained on MNIST that reads handwritten digits "
        "0-9 from photographs, screenshots or the built-in sample set.",
        eyebrow="Project 03",
        chips=["Computer vision", "PyTorch CNN", "MNIST", "10 classes"],
    )

    train_clicked = render_sidebar()

    recogniser = _get_recogniser()
    if train_clicked:
        if _run_training():
            recogniser = _get_recogniser()
            st.rerun()

    if recogniser is None:
        st.warning(
            f"**No trained model found.** The app expects `{portable_display(MODEL_PATH)}`.\n\n"
            "Click **Train / retrain** in the sidebar, or run:\n\n"
            "```bash\npython train.py\n```"
        )
        render_dataset_panel()
        note(DISCLAIMER)
        return

    render_prediction(recogniser)
    render_evaluation(recogniser)
    render_training_curves(recogniser)
    render_dataset_panel()

    model_info_panel(
        algorithm="Convolutional Neural Network (2 blocks: 32, 64 filters)",
        trained_on=(
            f"{recogniser.n_train:,} training images, "
            f"{recogniser.n_val:,} validation, {recogniser.n_test:,} test"
        ),
        metrics=recogniser.metrics,
        extra={
            "Trained at": recogniser.trained_at,
            "Epochs": str(recogniser.epochs),
            "Parameters": f"{recogniser.n_parameters:,}",
            "Framework": "PyTorch (CPU / Apple MPS / CUDA)",
            "Optimizer": f"{HYPERPARAMS['optimizer']} + {HYPERPARAMS['scheduler']}",
        },
    )

    note(DISCLAIMER)


if __name__ == "__main__":
    main()