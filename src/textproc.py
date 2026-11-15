"""Text normalization and Devanagari-aware tokenization.

Every pipeline in this project (BM25, static embeddings, encoders, data
cleaning) passes Hindi and English text through these helpers, so that the
comparisons are not confounded by different preprocessing (``AGENTS.md`` §10.1).

The important detail is that Python's :mod:`re` word pattern does not match
Devanagari vowel signs (matras, Unicode categories Mn/Mc), which splits words
mid-token.  Token matching therefore uses the third-party :mod:`regex` package
with ``[\\p{L}\\p{M}\\p{N}]+``.
"""

from __future__ import annotations

import unicodedata

import regex

ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"
_DANDA = "\u0964\u0965"

_WORD_TOKEN = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_DEVANAGARI_LETTER = regex.compile(r"(?V1)[\p{Devanagari}&&\p{L}]")
_ANY_LETTER = regex.compile(r"\p{L}")
_LATIN_LETTER = regex.compile(r"\p{Latin}")
_WHITESPACE = regex.compile(r"\s+")

# Sentence boundaries: danda and double danda, the Latin terminators, and a
# period that is not part of a number such as ``3.5`` or an abbreviation run.
_SENTENCE_BOUNDARY = regex.compile(
    rf"[{_DANDA}?!]+"
    r"|(?<!\d)\.(?!\d)"
)

_DEVANAGARI_DIGITS = {0x0966 + offset: ord("0") + offset for offset in range(10)}


def normalize(text: str) -> str:
    """Return NFC-normalized text without zero-width characters or stray spacing.

    Steps: strip zero-width characters (ZWSP/ZWNJ/ZWJ/BOM), NFC-normalize,
    map Devanagari digits to ASCII digits, collapse whitespace runs to single
    spaces and strip the ends.  NFC already decomposes precomposed nukta
    letters, so ``U+0958`` and ``U+0915 U+093C`` normalize to the same string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    cleaned = "".join(char for char in text if char not in ZERO_WIDTH)
    cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = cleaned.translate(_DEVANAGARI_DIGITS)
    return _WHITESPACE.sub(" ", cleaned).strip()


def tokenize(text: str, lang: str, *, keep_punct: bool = False) -> list[str]:
    """Return word tokens of ``text`` without splitting Devanagari words.

    Tokens are maximal runs of letters, combining marks and digits, so matras
    and nukta marks stay attached (``किताबों``, ``हिंदी``, ``प्रधानमंत्री``).
    Latin tokens are lowercased in both languages (``AGENTS.md`` §10.1: lowercase
    English only); Devanagari has no case, so Hindi words are unchanged.
    Punctuation, including the danda, is dropped unless ``keep_punct=True``, in
    which case punctuation runs are returned as tokens.

    Args:
        text: input text, normalized with :func:`normalize` first.
        lang: ``"hi"`` or ``"en"``; both lowercase Latin tokens.
        keep_punct: return punctuation runs as separate tokens.
    """
    if lang not in {"hi", "en"}:
        raise ValueError(f"unsupported language: {lang!r}")

    normalized = normalize(text)
    if not keep_punct:
        tokens = _WORD_TOKEN.findall(normalized)
    else:
        tokens = regex.findall(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]+", normalized)
    return [token.lower() if _LATIN_LETTER.search(token) else token for token in tokens]


def sentence_split(text: str, lang: str) -> list[str]:
    """Split ``text`` into sentences on danda, ``?``, ``!`` and non-decimal ``.``.

    Danda (U+0964) and double danda (U+0965) are Hindi sentence terminators.
    A period between digits (``3.5``) is not a boundary, and empty fragments
    are dropped.  ``lang`` is accepted for symmetry with :func:`tokenize`; both
    languages share the same terminators after normalization.
    """
    if lang not in {"hi", "en"}:
        raise ValueError(f"unsupported language: {lang!r}")

    normalized = normalize(text)
    sentences: list[str] = []
    for fragment in _SENTENCE_BOUNDARY.split(normalized):
        candidate = fragment.strip()
        if candidate:
            sentences.append(candidate)
    return sentences


def is_devanagari_ratio(text: str) -> float:
    """Return the fraction of letters that are Devanagari (used to filter Wikipedia)."""
    letters = _ANY_LETTER.findall(text)
    if not letters:
        return 0.0
    devanagari = _DEVANAGARI_LETTER.findall(text)
    return len(devanagari) / len(letters)
