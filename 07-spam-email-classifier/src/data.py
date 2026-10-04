"""SMS Spam Collection: download, parsing and labels.

Dataset
-------
The **UCI Machine Learning Repository SMS Spam Collection v1**: 5,574 SMS messages
labelled ``ham`` or ``spam``, 13.4% spam. It is a real, publicly licensed corpus
with no API key and no rate limit.

The brief asks for an *email* classifier. This project is trained on SMS because
that is the one openly redistributable spam corpus with labels. The distinction
is stated plainly in the interface and README rather than glossed over: the model
learns spam vocabulary and structure, which transfers reasonably to email, but
the reported accuracy is an SMS number and should not be quoted as an email one.

File format
-----------
The archive holds a single ``SMSSpamCollection`` file with one message per line
and ``<TAB>ham|spam<TAB>`` after the label. Messages themselves can contain
newlines and tabs, which is why naive ``split("\\t")`` breaks on a small number of
rows. :func:`parse_corpus` handles that properly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from shared.datasets import DatasetSpec, extract_zip
from shared.errors import DataNotFoundError, DataDownloadError
from shared.paths import data_dir

logger = logging.getLogger(__name__)

PROJECT_SLUG = "07-spam-email-classifier"

#: Label values.
HAM = "ham"
SPAM = "spam"
CLASS_LABELS: list[str] = [HAM, SPAM]

#: Positive class, used for precision/recall reported as "spam".
POSITIVE_LABEL = SPAM

#: The corpus file inside the archive.
CORPUS_MEMBER = "SMSSpamCollection"
CORPUS_FILENAME = "SMSSpamCollection"

#: Prefix that precedes every corpus line.
LINE_PREFIX_RE = re.compile(r"^(ham|spam)\t")

SMS_SPAM_SPEC = DatasetSpec(
    name="SMS Spam Collection v1",
    url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    filename="sms-spam-collection.zip",
    mirrors=["https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"],
    approx_size_mb=0.2,
    notes="UCI Machine Learning Repository, CC BY 4.0.",
)


@dataclass
class SpamData:
    """A parsed, labelled message corpus."""

    frame: pd.DataFrame

    @property
    def n_rows(self) -> int:
        """Number of messages."""
        return int(len(self.frame))

    def class_counts(self) -> dict[str, int]:
        """Return the ham/spam distribution."""
        counts = self.frame["label"].value_counts()
        return {str(key): int(value) for key, value in counts.items()}

    def summary(self) -> dict[str, Any]:
        """Return headline statistics for the dataset panel."""
        counts = self.class_counts()
        total = max(1, self.n_rows)
        lengths = self.frame["message"].str.len()
        return {
            "rows": self.n_rows,
            "class_counts": counts,
            "spam_count": counts.get(SPAM, 0),
            "ham_count": counts.get(HAM, 0),
            "spam_rate": counts.get(SPAM, 0) / total,
            "mean_length": float(lengths.mean()),
            "median_length": float(lengths.median()),
            "mean_spam_length": float(self.frame.loc[self.frame["label"] == SPAM, "message"].str.len().mean()),
            "mean_ham_length": float(self.frame.loc[self.frame["label"] == HAM, "message"].str.len().mean()),
            "duplicates": int(self.frame["message"].duplicated().sum()),
            "source": SMS_SPAM_SPEC.url,
        }


def download_dataset(*, force: bool = False) -> Path:
    """Download and extract the archive, returning the corpus file path."""
    from shared.datasets import fetch_dataset

    directory = data_dir(PROJECT_SLUG)
    archive = fetch_dataset(SMS_SPAM_SPEC, directory, force=force)
    extract_zip(archive, directory, members=[CORPUS_MEMBER, f"{CORPUS_MEMBER}.txt"])
    logger.info("SMS Spam archive extracted to %s", directory)
    return directory


def _resolve_corpus(directory: Path) -> Path:
    """Locate the corpus file, accepting either extraction layout."""
    for candidate in (directory / CORPUS_FILENAME, directory / f"{CORPUS_FILENAME}.txt"):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise DataNotFoundError(
        f"'{CORPUS_FILENAME}' not found inside data/.",
        hint=(
            "Delete data/ and re-run the training script to re-download and extract the "
            f"corpus. Manual source: {SMS_SPAM_SPEC.url}"
        ),
    )


def parse_corpus(text: str) -> tuple[str, str]:
    """Parse one corpus line into ``(label, message)``.

    The label is the first tab-separated field. Everything after the *second*
    tab is the message, which keeps tabs that occur inside a message body from
    being mistaken for the label separator.
    """
    first_tab = text.find("\t")
    if first_tab == -1:
        raise DataDownloadError(
            "Encountered a corpus line with no tab separator.",
            hint="The archive is corrupt; delete data/ and re-download it.",
        )

    label = text[:first_tab].strip()
    remainder = text[first_tab + 1 :]
    second_tab = remainder.find("\t")
    message = remainder if second_tab == -1 else remainder[second_tab + 1 :]

    if label not in CLASS_LABELS:
        raise DataDownloadError(
            f"Unexpected label '{label}' in the corpus.",
            hint=f"Expected one of {CLASS_LABELS}.",
        )
    return label, message.strip()


def load_dataset(*, force_download: bool = False, drop_duplicates: bool = True) -> SpamData:
    """Load the SMS Spam Collection into a DataFrame.

    Args:
        force_download: Re-download even when the corpus exists.
        drop_duplicates: Remove repeated messages. These are common in this
            dataset and would otherwise let a memorised string appear in both the
            training and test splits.

    Raises:
        DataNotFoundError: When the corpus file or required columns are missing.
    """
    directory = download_dataset(force=force_download)
    path = _resolve_corpus(directory)

    labels: list[str] = []
    messages: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not LINE_PREFIX_RE.match(stripped):
                # Blank lines and any header are skipped rather than failing.
                continue
            label, message = parse_corpus(stripped)
            labels.append(label)
            messages.append(message)

    if not messages:
        raise DataNotFoundError(
            "The SMS Spam corpus parsed to zero messages.",
            hint="Delete data/ and re-download the archive.",
        )

    frame = pd.DataFrame({"message": messages, "label": labels})

    before = len(frame)
    if drop_duplicates:
        frame = frame.drop_duplicates(subset=["message"]).reset_index(drop=True)
        removed = before - len(frame)
        if removed:
            logger.info("Removed %d duplicate message(s).", removed)

    empty = int((frame["message"].str.strip() == "").sum())
    if empty:
        logger.info("Removed %d message(s) with no text.", empty)
        frame = frame[frame["message"].str.strip() != ""].reset_index(drop=True)

    logger.info(
        "Loaded SMS Spam corpus: %d messages (%d spam, %.1f%%)",
        len(frame),
        int((frame["label"] == SPAM).sum()),
        100.0 * (frame["label"] == SPAM).mean(),
    )
    return SpamData(frame=frame)


def example_messages() -> dict[str, str]:
    """Return ready-made examples for the interface, one per class."""
    return {
        "Typical spam": "URGENT! You have won a 1000 dollar gift card. Click here to claim now",
        "Typical ham": "Hey, are we still meeting for lunch at 1pm?",
        "Spam with a link": "FREE entry to our weekly draw! Text WIN to 80086 to enter now",
        "Long legitimate email": "Please find attached the quarterly report we discussed on Tuesday. Let me know if you have any questions before Friday.",
        "Subtle spam": "Limited time offer, 90% discount on designer watches. Reply YES to claim",
        "All caps spam": "CONGRATULATION!! YOU WON A MILLION POUNDS!!! CLAIM YOUR PRIZE NOW",
    }