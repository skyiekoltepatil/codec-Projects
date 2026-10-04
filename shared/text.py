"""Shared text cleaning, tokenisation and stop-word handling.

Three projects (sentiment analysis, chatbot, spam classifier) share this module
so that preprocessing is defined exactly once and cannot drift between them.

A note on NLTK: the repository lists NLTK as an optional dependency and prefers
``nltk.corpus.stopwords`` when it is installed *and* its data has been
downloaded. When it is not available the module falls back to a bundled English
stop-word list, so a fresh clone never fails because of a missing corpus.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Stop words
# --------------------------------------------------------------------------- #

#: Bundled English stop words. Used directly, and as a fallback when the NLTK
#: stop-words corpus is not available offline.
_BUNDLED_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can't cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how how's
    i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves out over own same
    shan't she she'd she'll she's should shouldn't so some such than that that's the their
    theirs them themselves then there there's these they they'd they'll they're they've this
    those through to too under until up very was wasn't we we'd we'll we're we've were weren't
    what what's when when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

#: Negations are meaningful for sentiment, so they are never removed when the
#: caller asks to preserve negation cues.
NEGATION_WORDS: frozenset[str] = frozenset(
    {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "cannot", "without", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't"}
)

_nltk_stopwords_cache: frozenset[str] | None = None

# --------------------------------------------------------------------------- #
# Compiled patterns
# --------------------------------------------------------------------------- #

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_MULTISPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_WORD_RE = re.compile(r"[a-z0-9']+")


def get_stopwords(prefer_nltk: bool = True) -> frozenset[str]:
    """Return the active stop-word set.

    Args:
        prefer_nltk: When ``True`` (default) and the NLTK English stop-word
            corpus is already downloaded locally.

    Returns:
        A frozenset of lower-case stop words.
    """
    global _nltk_stopwords_cache
    if _nltk_stopwords_cache is not None:
        return _nltk_stopwords_cache
    if prefer_nltk:
        try:  # pragma: no cover - depends on the local NLTK data state
            from nltk.corpus import stopwords as nltk_stopwords

            words = frozenset(nltk_stopwords.words("english"))
            if words:
                _nltk_stopwords_cache = words
                return words
        except Exception:
            # NLTK missing, or its corpus has not been downloaded. The bundled
            # list is a deliberate, documented fallback rather than a failure.
            pass
    _nltk_stopwords_cache = _BUNDLED_STOPWORDS
    return _BUNDLED_STOPWORDS


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_urls: bool = True,
    strip_mentions: bool = True,
    strip_hashtag_symbol: bool = True,
    strip_emails: bool = True,
    strip_phone_numbers: bool = True,
    strip_numbers: bool = False,
    strip_punctuation: bool = True,
    collapse_repeats: bool = True,
    strip_accents: bool = False,
) -> str:
    """Normalise free text into a clean token-friendly string.

    Args:
        text: Raw input text. Non-string input is coerced with ``str``.
        lowercase: Lower-case the result.
        strip_urls: Remove ``http(s)://`` and ``www.`` links.
        strip_mentions: Remove ``@user`` handles.
        strip_hashtag_symbol: Turn ``#tag`` into ``tag`` instead of dropping it.
        strip_emails: Remove e-mail addresses.
        strip_phone_numbers: Remove phone-like numeric sequences.
        strip_numbers: Remove all standalone numbers.
        strip_punctuation: Remove punctuation, keeping apostrophes inside words.
        collapse_repeats: Reduce ``sooooo`` to ``soo`` (three or more repeats).
        strip_accents: Transliterate accented characters to ASCII.

    Returns:
        The cleaned string. Returns ``""`` for ``None`` or non-string input that
        becomes empty after cleaning.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities and drop tags so scraped text behaves like plain text.
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)

    if strip_urls:
        text = _URL_RE.sub(" ", text)
    if strip_emails:
        text = _EMAIL_RE.sub(" ", text)
    if strip_mentions:
        text = _MENTION_RE.sub(" ", text)
    if strip_phone_numbers:
        text = _PHONE_RE.sub(" ", text)
    if strip_hashtag_symbol:
        text = _HASHTAG_RE.sub(r"\1", text)

    if strip_accents:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    if lowercase:
        text = text.lower()

    if strip_numbers:
        text = _NUMBER_RE.sub(" ", text)
    if strip_punctuation:
        # Keep intra-word apostrophes (don't, it's) but drop everything else.
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)

    if collapse_repeats:
        text = _REPEATED_CHAR_RE.sub(r"\1\1", text)

    return _MULTISPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens, keeping intra-word apostrophes."""
    return _WORD_RE.findall(text.lower())


def remove_stopwords(
    tokens: Sequence[str],
    *,
    keep_negations: bool = False,
    extra_stopwords: Iterable[str] = (),
) -> list[str]:
    """Filter stop words out of a token sequence.

    Args:
        tokens: Tokens to filter.
        keep_negations: Preserve negation cues (important for sentiment).
        extra_stopwords: Additional domain-specific words to drop.
    """
    stop = set(get_stopwords())
    stop.update(word.lower() for word in extra_stopwords)
    if keep_negations:
        stop -= NEGATION_WORDS
    return [token for token in tokens if token not in stop and len(token) > 1]


def normalize_text(
    text: str,
    *,
    keep_negations: bool = False,
    strip_numbers: bool = False,
    extra_stopwords: Iterable[str] = (),
    join: bool = True,
) -> str | list[str]:
    """Run the full clean -> tokenise -> stop-word pipeline.

    Args:
        text: Raw text.
        keep_negations: Preserve negation cues.
        strip_numbers: Remove numeric tokens during cleaning.
        extra_stopwords: Domain-specific stop words.
        join: When ``True`` return a space-joined string (ready for TF-IDF),
            otherwise return the token list.

    Returns:
        A string when ``join`` is ``True``, else a list of tokens.
    """
    cleaned = clean_text(text, strip_numbers=strip_numbers)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens, keep_negations=keep_negations, extra_stopwords=extra_stopwords)
    return " ".join(tokens) if join else tokens


def truncate(text: str, limit: int = 160) -> str | None:
    """Shorten ``text`` for display, appending an ellipsis when truncated."""
    if text is None:
        return None
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"