"""Streamlit interface for the Speech-to-Text Transcription app.

Run from this project's directory::

    streamlit run app.py

Upload or record a short clip and transcribe it on-device with a pretrained
Whisper model. There is no training step: the model weights are downloaded once
and then reused offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
for candidate in (str(REPO_ROOT), str(PROJECT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shared.config import configure_logging  # noqa: E402
from shared.errors import InvalidInputError  # noqa: E402
from shared.theme import PALETTE, inject_theme, metric_row, note, page_header  # noqa: E402
from shared.ui import (  # noqa: E402
    download_text_button,
    model_info_panel,
    section,
    show_dataframe,
    show_error,
)

from src.transcribe import (  # noqa: E402
    ALLOWED_EXTENSIONS,
    DEFAULT_MODEL_SIZE,
    MODEL_CATALOGUE,
    MODEL_SIZES,
    TASKS,
    TranscriptionResult,
    format_duration,
    load_model,
    segments_to_srt,
    segments_to_vtt,
    transcribe,
)

#: Uploads larger than this are rejected before decoding, to bound memory use.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

DISCLAIMER = (
    "Whisper is a general-purpose speech model. Accuracy drops on heavy accents, "
    "overlapping speakers, technical vocabulary and noisy or far-field recordings, "
    "and the model can produce fluent but incorrect text. Treat the transcript as a "
    "draft to be checked, not as a verbatim record."
)


@st.cache_resource(show_spinner=False)
def _load_whisper(model_size: str, device: str, compute_type: str):
    """Load the Whisper model once per (size, device, precision) combination."""
    return load_model(model_size, device=device, compute_type=compute_type)


def _validate_upload(uploaded) -> tuple[bytes, str]:
    """Return ``(bytes, filename)`` for an uploaded or recorded file.

    Raises:
        InvalidInputError: When nothing was provided, the file is empty, or it is
            larger than :data:`MAX_UPLOAD_BYTES`.
    """
    if uploaded is None:
        raise InvalidInputError(
            "No audio was provided.",
            hint="Upload a file or record a clip, then press Transcribe.",
        )
    name = getattr(uploaded, "name", "audio.wav")
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    if not data:
        raise InvalidInputError("The audio file is empty.", hint="Choose a file that contains audio.")
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise InvalidInputError(
            f"The file is {len(data) / 1024 / 1024:.1f} MB, larger than the {limit_mb} MB limit.",
            hint="Trim the clip or choose a smaller file.",
        )
    return data, name


def render_sidebar() -> dict:
    """Render the configuration controls and return the selected options."""
    with st.sidebar:
        st.markdown("### Configuration")
        model_size = st.selectbox(
            "Whisper model",
            options=list(MODEL_SIZES),
            index=list(MODEL_SIZES).index(DEFAULT_MODEL_SIZE),
            format_func=lambda key: f"{MODEL_CATALOGUE[key]['label']} - {key}",
            key="stt_model_size",
        )
        st.caption(str(MODEL_CATALOGUE[model_size]["note"]))

        task = st.radio("Task", options=list(TASKS), horizontal=True, key="stt_task")
        language = st.text_input(
            "Source language (optional)",
            placeholder="auto-detect, e.g. en",
            key="stt_language",
        )
        beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5, key="stt_beam")
        vad_filter = st.checkbox("Filter silence (VAD)", value=True, key="stt_vad")

        st.markdown("---")
        with st.expander("Advanced"):
            device = st.selectbox("Device", options=["auto", "cpu"], key="stt_device")
            compute_type = st.selectbox(
                "Precision",
                options=["int8", "int8_float32", "float32"],
                key="stt_compute_type",
            )
            st.caption("int8 is fastest on CPU; float32 is slowest but slightly more exact.")

    return {
        "model_size": model_size,
        "task": task,
        "language": (language or "").strip() or None,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "device": device,
        "compute_type": compute_type,
    }


def render_audio_input():
    """Render the upload and record widgets and return the chosen file, if any."""
    upload_tab, record_tab = st.tabs(["Upload a file", "Record from microphone"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Audio file",
            type=list(ALLOWED_EXTENSIONS),
            key="stt_uploader",
            help=f"Up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    with record_tab:
        st.caption("Recording uses your browser microphone; allow access when prompted.")
        recorded = st.audio_input("Record a clip", key="stt_recorder")
    return uploaded or recorded


def render_results(result: TranscriptionResult, filename: str) -> None:
    """Render the transcript, the subtitle exports and the detailed metrics."""
    section("Transcript")
    stem = Path(filename).stem or "transcript"

    text_column, export_column = st.columns([3, 1])
    with export_column:
        download_text_button(result.text + "\n", filename=f"{stem}.txt", label="Download .txt", key="stt_dl_txt")
        download_text_button(
            segments_to_srt(result.segments),
            filename=f"{stem}.srt",
            label="Download .srt",
            key="stt_dl_srt",
            mime="application/x-subrip",
        )
        download_text_button(
            segments_to_vtt(result.segments),
            filename=f"{stem}.vtt",
            label="Download .vtt",
            key="stt_dl_vtt",
            mime="text/vtt",
        )
    with text_column:
        st.text_area("Recognised text", value=result.text, height=220, key="stt_transcript_output")

    section("Details")
    metric_row(
        [
            ("Words", f"{result.word_count:,}", "whitespace-delimited"),
            ("Audio length", format_duration(result.duration), "input duration"),
            ("Language", result.language, f"{result.language_probability:.0%} confidence"),
            ("Processing", format_duration(result.elapsed_seconds), "decode time"),
            ("Real-time factor", f"{result.realtime_factor:.2f}x", "processing / audio"),
            ("Segments", f"{len(result.segments):,}", "timed spans"),
        ]
    )

    if result.segments:
        frame = pd.DataFrame(result.as_rows()).rename(
            columns={"start": "Start", "end": "End", "duration_s": "Duration (s)", "text": "Text"}
        )
        show_dataframe(frame, caption="Timed segments used to build the subtitle exports.")


def render_about() -> None:
    """Explain how the transcription works and what to expect."""
    with st.expander("How this works"):
        st.markdown(
            "1. The chosen Whisper model is downloaded once from the Hugging Face Hub "
            "and cached on this machine; later runs are fully offline.\n"
            "2. Audio is decoded with `av` and transcribed by CTranslate2 on CPU by "
            "default (set **Device** to `auto` in Advanced to allow a GPU).\n"
            "3. Every recognised segment keeps start/end timestamps, which is what the "
            "`.srt` and `.vtt` exports are built from.\n\n"
            "No audio leaves this machine: there is no third-party API call."
        )


def main() -> None:
    """Render the whole page."""
    configure_logging()
    st.set_page_config(page_title="Speech-to-Text Transcription", page_icon="10", layout="wide")
    inject_theme()

    page_header(
        "Speech-to-Text Transcription",
        "Transcribe uploaded or recorded audio entirely on-device with a pretrained "
        "Whisper model, then export the result as plain text or subtitles.",
        eyebrow="Project 10",
        chips=["Speech", "Deep learning", "faster-whisper", "Offline"],
    )

    options = render_sidebar()
    audio_file = render_audio_input()

    audio_bytes: bytes | None = None
    audio_name = "audio.wav"
    if audio_file is not None:
        try:
            audio_bytes, audio_name = _validate_upload(audio_file)
        except Exception as error:  # noqa: BLE001 - UI boundary
            show_error(error)

    if audio_bytes:
        st.audio(audio_bytes)

    clicked = st.button("Transcribe", type="primary", width="stretch", disabled=not audio_bytes)

    if clicked and audio_bytes:
        try:
            with st.spinner("Loading model and transcribing..."):
                model = _load_whisper(options["model_size"], options["device"], options["compute_type"])
                result = transcribe(
                    audio_bytes,
                    model=model,
                    model_size=options["model_size"],
                    filename=audio_name,
                    language=options["language"],
                    task=options["task"],
                    beam_size=options["beam_size"],
                    vad_filter=options["vad_filter"],
                )
            st.session_state["stt_result"] = (result, audio_name)
        except Exception as error:  # noqa: BLE001 - UI boundary
            show_error(error)

    stored = st.session_state.get("stt_result")
    if stored is not None:
        result, name = stored
        if not result.has_speech:
            st.warning("No speech was detected in this clip. Try a louder or longer recording.")
        render_results(result, name)
    else:
        st.caption("Choose or record a clip, then press **Transcribe**.")

    render_about()

    model_info_panel(
        algorithm=str(MODEL_CATALOGUE[options["model_size"]]["label"]),
        trained_on="Pretrained OpenAI Whisper weights (no local training)",
        extra={
            "Model id": options["model_size"],
            "Task": options["task"],
            "Device": options["device"],
            "Precision": options["compute_type"],
            "Framework": "faster-whisper (CTranslate2)",
        },
    )

    note(DISCLAIMER, accent=PALETTE["accent"])


if __name__ == "__main__":
    main()
