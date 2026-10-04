"""Dataset acquisition with a local cache and integrity checks.

Design notes
------------
* Nothing is committed to Git except tiny text fixtures. Real datasets are
  downloaded on demand into each project's ``data/`` directory.
* Downloads are atomic (temp file + rename) so an interrupted run never leaves a
  half-written CSV that would silently corrupt training. This matters: a
  truncated CSV is one of the most confusing failure modes for a reviewer.
* Every download is recorded in a manifest with its size and SHA-256 digest.
  Subsequent runs verify the cached copy and re-download if it was corrupted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from shared.errors import DataDownloadError, DataNotFoundError
from shared.paths import ensure_parent

logger = logging.getLogger(__name__)

#: Sent with every request so dataset hosts can identify (and rate-limit) us politely.
USER_AGENT = "AI-ML-Internship-Projects/1.0 (+educational use; python-requests)"

#: Number of download attempts before giving up.
MAX_ATTEMPTS = 3

#: Human-readable form of :data:`MAX_ATTEMPTS`, used in error messages.
MAX_ATTEMPTS_LABEL = f"{MAX_ATTEMPTS} attempts"

#: Seconds to wait between retries, multiplied by the attempt number.
RETRY_BACKOFF_SECONDS = 2.0

ProgressCallback = Callable[[int, int | None], None]


@dataclass
class DatasetSpec:
    """Declarative description of one remote dataset file.

    Attributes:
        name: Human-readable dataset name used in messages and manifests.
        url: Primary download URL.
        filename: Local filename inside the destination directory.
        mirrors: Alternative URLs tried in order if the primary fails.
        expected_sha256: Optional digest; when set, a mismatch is an error.
        approx_size_mb: Used only for user-facing messages.
        notes: Free-text note surfaced in the UI when the source is unusual.
    """

    name: str
    url: str
    filename: str
    mirrors: list[str] = field(default_factory=list)
    expected_sha256: str | None = None
    approx_size_mb: float | None = None
    notes: str | None = None

    def urls(self) -> list[str]:
        """Return the primary URL followed by any mirrors."""
        return [self.url, *self.mirrors]


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #


def _manifest_path(directory: Path) -> Path:
    return directory / ".download_manifest.json"


def _read_manifest(directory: Path) -> dict[str, dict[str, object]]:
    path = _manifest_path(directory)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable download manifest at %s", path)
        return {}


def _write_manifest(directory: Path, manifest: dict[str, dict[str, object]]) -> None:
    path = _manifest_path(directory)
    try:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as error:  # pragma: no cover - disk-level failure
        logger.warning("Could not persist download manifest: %s", error)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, streamed to bound memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(
    url: str,
    *,
    description: str = "metadata",
    timeout: int = 60,
) -> Any:
    """Fetch and parse a small JSON document.

    Used for dataset metadata (label names, file listings) rather than bulk data.

    Raises:
        DataDownloadError: If the request fails or the body is not valid JSON.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            logger.warning("JSON fetch attempt %d/%d failed for %s: %s", attempt, MAX_ATTEMPTS, url, error)
        except ValueError as error:
            raise DataDownloadError(
                f"{url} did not return valid JSON.",
                hint=f"The {description} endpoint may be down or rate limiting. ({error})",
            ) from error
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise DataDownloadError(
        f"Could not fetch {description} after {MAX_ATTEMPTS} attempts.",
        hint=f"Check your internet connection. Last error: {last_error}",
    )


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    progress: ProgressCallback | None = None,
    timeout: int = 60,
) -> Path:
    """Download ``url`` to ``destination`` atomically and return the final path.

    Args:
        url: Source URL.
        destination: Final path on disk (parent directories are created).
        expected_sha256: When provided, the downloaded digest must match.
        progress: Optional ``callback(bytes_downloaded, total_bytes_or_None)``.
        timeout: Per-request socket timeout in seconds.

    Raises:
        DataDownloadError: If every attempt fails or the digest does not match.
    """
    destination = ensure_parent(Path(destination))
    temp_path = destination.with_suffix(destination.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header and total_header.isdigit() else None

                written = 0
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if progress is not None:
                            progress(written, total)

            if written == 0:
                raise DataDownloadError(
                    f"Downloaded an empty file from {url}.",
                    hint="The host may be blocking automated requests. Try the manual download instructions.",
                )

            if expected_sha256 is not None:
                actual = sha256_of(temp_path)
                if actual.lower() != expected_sha256.lower():
                    raise DataDownloadError(
                        "Downloaded file failed its integrity check.",
                        hint=(
                            f"Expected SHA-256 {expected_sha256[:12]}..., got {actual[:12]}... "
                            "The upstream file may have changed."
                        ),
                    )

            shutil.move(str(temp_path), str(destination))
            return destination

        except DataDownloadError:
            raise
        except (requests.RequestException, OSError) as error:
            last_error = error
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt, MAX_ATTEMPTS, url, error)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise DataDownloadError(
        f"Could not download '{url}' after {MAX_ATTEMPTS_LABEL}.",
        hint=(
            "Check your internet connection. If the host is unavailable, follow the manual "
            f"download instructions in the project README. Last error: {last_error}"
        ),
    )


def fetch_dataset(
    spec: DatasetSpec,
    directory: Path,
    *,
    progress: ProgressCallback | None = None,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Return a verified local copy of ``spec``, downloading it if necessary.

    A cached file is reused when its recorded size matches the manifest; if the
    manifest digest is available it is re-verified, and a corrupt cache triggers
    an automatic re-download rather than a confusing downstream error.

    Args:
        spec: The dataset description.
        directory: Directory to store the file in (usually the project's ``data/``).
        progress: Optional progress callback forwarded to :func:`download_file`.
        force: Re-download even when a valid cached copy exists.
        timeout: Per-request socket timeout in seconds.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename
    manifest = _read_manifest(directory)
    record = manifest.get(spec.filename, {})

    if target.exists() and not force:
        recorded_size = record.get("size")
        if recorded_size is None or int(recorded_size) == target.stat().st_size:
            recorded_digest = record.get("sha256")
            if isinstance(recorded_digest, str):
                if sha256_of(target) == recorded_digest:
                    logger.info("Using verified cached dataset %s", target)
                    return target
                logger.warning("Cached dataset %s failed verification; re-downloading", target.name)
            else:
                logger.info("Using cached dataset %s", target)
                return target

    errors: list[str] = []
    for url in spec.urls():
        try:
            download_file(url, target, expected_sha256=spec.expected_sha256, progress=progress, timeout=timeout)
            manifest[spec.filename] = {
                "name": spec.name,
                "source": url,
                "size": target.stat().st_size,
                "sha256": sha256_of(target),
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            _write_manifest(directory, manifest)
            logger.info("Downloaded %s from %s", spec.filename, url)
            return target
        except DataDownloadError as error:
            errors.append(f"{url} -> {error.message}")

    raise DataDownloadError(
        f"Could not obtain the dataset '{spec.name}'.",
        hint=(
            "All sources failed:\n- " + "\n- ".join(errors) + "\n\n"
            "Download the file manually and place it at "
            f"'{spec.filename}' inside '{directory.name}/'. See the README for details."
        ),
    )


# --------------------------------------------------------------------------- #
# Archive helpers
# --------------------------------------------------------------------------- #


def extract_zip(archive: Path, destination: Path, *, members: list[str] | None = None) -> Path:
    """Extract a zip archive safely, rejecting entries that escape ``destination``.

    Args:
        archive: Path to the ``.zip`` file.
        destination: Directory to extract into.
        members: Optional subset of member names to extract. Names that are not
            present in the archive are skipped rather than raising, so callers can
            list alternative spellings (``name`` and ``name.txt``) without first
            inspecting the archive.

    Raises:
        DataDownloadError: When the archive is invalid or contains a member whose
            path escapes ``destination``.
    """
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            available = set(zf.namelist())
            selected = members if members is not None else sorted(available)
            # Skip names the archive does not contain; a caller offering two
            # spellings of the same file should not have to know which one is real.
            selected = [name for name in selected if name in available]
            if members is not None and not selected:
                raise DataDownloadError(
                    f"None of the requested members are present in '{archive.name}'.",
                    hint=(
                        "Expected one of: "
                        + ", ".join(members)
                        + f". The archive contains: {', '.join(sorted(available))}"
                    ),
                )

            base = destination.resolve()
            for name in selected:
                resolved = (destination / name).resolve()
                if not str(resolved).startswith(str(base)):
                    raise DataDownloadError(
                        f"Refusing to extract unsafe archive member '{name}'.",
                        hint="The archive contains a path that escapes the destination directory.",
                    )
            zf.extractall(destination, members=selected)
    except zipfile.BadZipFile as error:
        raise DataDownloadError(
            f"'{archive.name}' is not a valid zip archive.",
            hint="Delete the file and re-run the download; it was probably truncated.",
        ) from error
    return destination


def extract_tar(archive: Path, destination: Path) -> Path:
    """Extract a tar archive, rejecting members that escape ``destination``."""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    try:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                resolved = (destination / member.name).resolve()
                if not str(resolved).startswith(str(base)):
                    raise DataDownloadError(
                        f"tar archive contains an unsafe path: '{member.name}'",
                        hint="Refusing to extract outside the destination directory.",
                    )
            tf.extractall(destination)
    except tarfile.TarError as error:
        raise DataDownloadError(
            f"'{archive.name}' is not a valid tar archive.",
            hint="Delete the file and re-run the download; it was probably truncated.",
        ) from error
    return destination


# --------------------------------------------------------------------------- #
# Local file requirements
# --------------------------------------------------------------------------- #


def require_local_file(path: Path, *, description: str, instructions: str) -> Path:
    """Return ``path`` if it exists, otherwise raise an actionable error."""
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(
            f"{description} not found at '{path.name}'.",
            hint=instructions,
        )
    if path.stat().st_size == 0:
        raise DataNotFoundError(
            f"{description} at '{path.name}' is empty.",
            hint="Delete the file and re-run the download or preparation step.",
        )
    return path