"""Offline speech recognition built on a pretrained ``faster-whisper`` model.

There is no training step for this project: Whisper weights are downloaded once
from the Hugging Face Hub and then run entirely on-device with CTranslate2. The
module keeps every model interaction in one place so ``app.py`` stays a thin
rendering layer, matching the convention used by the other nine projects.

Design notes
------------
* The model is loaded lazily and the caller (the Streamlit layer) caches it, so
  nothing is downloaded until a transcription is actually requested.
* Audio arrives as raw ``bytes`` and is written to a short-lived temporary file
  because the underlying ``av`` decoder expects a seekable input.
* Everything that leaves this module is JSON-friendly and free of NumPy types,
  which keeps the interface and the tests simple.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from shared.errors import ProjectError

logger = logging.getLogger(__name__)

#: Every model offered in the interface, smallest (fastest) first. ``.en`` models
#: are English-only; ``large-v3`` is multilingual.
MODEL_SIZES: tuple[str, ...] = ("tiny.en", "base.en", "small.en", "medium.en", "large-v3")

#: Human-readable metadata per model, surfaced in the sidebar.
MODEL_CATALOGUE: dict[str, dict[str, object]] = {
    "tiny.en": {"label": "Tiny (English)", "download_mb": 75, "multilingual": False,
                "note": "Fastest. Good for quick drafts, noticeably more errors."},
    "base.en": {"label": "Base (English)", "download_mb": 145, "multilingual": False,
                "note": "Fast and reasonable accuracy."},
    "small.en": {"label": "Small (English)", "download_mb": 480, "multilingual": False,
                 "note": "Default. Best accuracy/speed balance on CPU."},
    "medium.en": {"label": "Medium (English)", "download_mb": 1500, "multilingual": False,
                  "note": "More accurate, several times slower than small."},
    "large-v3": {"label": "Large v3 (multilingual)", "download_mb": 3100, "multilingual": True,
                 "note": "Most accurate. Needs a lot of RAM and time on CPU."},
}

#: Model used unless the user picks another one.
DEFAULT_MODEL_SIZE = "small.en"

#: File extensions accepted by the uploader / decoder.
ALLOWED_EXTENSIONS: tuple[str, ...] = (
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus",
    "webm", "mp4", "mpeg", "mpga", "wma", "aiff", "aif", "mkv",
)

#: Speech-to-text task names accepted by faster-whisper.
TASKS: tuple[str, ...] = ("transcribe", "translate")

_MULTISPACE_RE = re.compile(r"\s+")


@dataclass
class Segment:
    """One timed span of recognised speech."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Length of the span in seconds."""
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptionResult:
    """Everything the interface needs to render one transcription."""

    text: str
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model_size: str
    task: str
    elapsed_seconds: float

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words in the transcript."""
        return len(self.text.split())

    @property
    def realtime_factor(self) -> float:
        """Processing time divided by audio duration (lower is faster)."""
        if self.duration <= 0:
            return 0.0
        return self.elapsed_seconds / self.duration

    @property
    def has_speech(self) -> bool:
        """Whether any speech was recognised."""
        return bool(self.text.strip())

    def as_rows(self) -> list[dict[str, object]]:
        """Return the segments as plain dictionaries for a table."""
        return [
            {
                "start": format_timestamp(segment.start),
                "end": format_timestamp(segment.end),
                "duration_s": round(segment.duration, 2),
                "text": segment.text,
            }
            for segment in self.segments
        ]


# --------------------------------------------------------------------------- #
# Timestamp helpers
# --------------------------------------------------------------------------- #


def format_timestamp(seconds: float) -> str:
    """Format ``seconds`` as ``MM:SS`` or ``H:MM:SS`` for display."""
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _clock(seconds: float, *, millis_separator: str) -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm`` for subtitle files."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_separator}{millis:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    """Serialise segments to SubRip (``.srt``) text."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{_clock(segment.start, millis_separator=',')} --> "
            f"{_clock(segment.end, millis_separator=',')}\n"
            f"{segment.text}"
        )
    return "\n\n".join(blocks) + "\n"


def segments_to_vtt(segments: list[Segment]) -> str:
    """Serialise segments to WebVTT (``.vtt``) text."""
    blocks = []
    for segment in segments:
        blocks.append(
            f"{_clock(segment.start, millis_separator='.')} --> "
            f"{_clock(segment.end, millis_separator='.')}\n"
            f"{segment.text}"
        )
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def format_duration(seconds: float) -> str:
    """Format a duration for a metric card, e.g. ``1 m 05 s``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} m {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} m"


# --------------------------------------------------------------------------- #
# Model loading and transcription
# --------------------------------------------------------------------------- #


def load_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    *,
    device: str = "auto",
    compute_type: str = "int8",
    download_root: str | Path | None = None,
):
    """Return a ready-to-use :class:`faster_whisper.WhisperModel`.

    The first call downloads the model weights to the Hugging Face cache (or
    ``download_root`` when given); later calls are instant.

    Raises:
        ProjectError: When the model cannot be loaded, for example because the
            weights are not cached and the machine is offline.
    """
    if model_size not in MODEL_CATALOGUE:
        raise ProjectError(
            f"Unknown Whisper model '{model_size}'.",
            hint=f"Choose one of: {', '.join(MODEL_SIZES)}.",
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - dependency is present in this repo
        raise ProjectError(
            "faster-whisper is not installed.",
            hint="Install the project dependencies with `pip install -r requirements.txt`.",
        ) from error

    logger.info("Loading Whisper model %s (device=%s, compute_type=%s)", model_size, device, compute_type)
    try:
        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root is not None else None,
        )
    except Exception as error:  # noqa: BLE001 - surfaced as a friendly message
        logger.exception("Could not load Whisper model %s", model_size)
        raise ProjectError(
            f"Could not load the Whisper model '{model_size}'.",
            hint=(
                "The weights are downloaded from the Hugging Face Hub on first use. "
                "Check your internet connection, or pre-download the model while online. "
                f"(Underlying error: {error})"
            ),
        ) from error


def transcribe(
    audio: bytes,
    *,
    model,
    model_size: str = DEFAULT_MODEL_SIZE,
    filename: str = "audio.wav",
    language: str | None = None,
    task: str = "transcribe",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> TranscriptionResult:
    """Transcribe ``audio`` bytes with a preloaded ``model``.

    Args:
        audio: Raw contents of an audio file.
        model: A loaded :class:`faster_whisper.WhisperModel`.
        model_size: Model name, stored on the result for display.
        filename: Original filename; only its suffix matters, for the decoder.
        language: Force a language (``None`` lets Whisper detect it).
        task: ``"transcribe"`` keeps the spoken language, ``"translate"`` renders
            foreign speech into English.
        beam_size: Beam width for decoding; higher is slower but more accurate.
        vad_filter: Drop silence with the built-in voice-activity filter.

    Returns:
        A :class:`TranscriptionResult`.
    """
    if not audio:
        raise ProjectError("The uploaded audio file is empty.", hint="Choose a file that contains audio.")

    suffix = Path(filename).suffix or ".wav"
    temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        temporary.write(audio)
        temporary.flush()
        temporary.close()

        started = time.perf_counter()
        segment_iterator, info = model.transcribe(
            temporary.name,
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        # ``transcribe`` returns a generator; consuming it here both runs the
        # decoding and lets us time it and free the temporary file afterwards.
        raw_texts: list[str] = []
        segments: list[Segment] = []
        for raw in segment_iterator:
            raw_texts.append(raw.text)
            segments.append(Segment(float(raw.start), float(raw.end), raw.text.strip()))
        elapsed = time.perf_counter() - started
    finally:
        Path(temporary.name).unlink(missing_ok=True)

    text = _MULTISPACE_RE.sub(" ", "".join(raw_texts)).strip()
    return TranscriptionResult(
        text=text,
        segments=segments,
        language=getattr(info, "language", "unknown") or "unknown",
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        model_size=model_size,
        task=task,
        elapsed_seconds=elapsed,
    )
