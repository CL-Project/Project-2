"""Tests for Devanagari-aware normalization, tokenization and sentence splitting (S05)."""

from __future__ import annotations

import pytest

from src.textproc import is_devanagari_ratio, normalize, sentence_split, tokenize


@pytest.mark.parametrize(
    "word",
    ["किताबों", "हिंदी", "प्रधानमंत्री", "बाज़ार", "ख़रीदे", "संस्कृति", "क्षेत्र"],
)
def test_matras_are_not_split(word: str) -> None:
    """Words carrying matras survive tokenization as a single token."""
    assert tokenize(word, "hi") == [word]


def test_words_with_matras_in_sentence_stay_whole() -> None:
    """A sentence yields whole Hindi words, not per-matras fragments."""
    tokens = tokenize("मैंने हिंदी की किताबें पढ़ीं।", "hi")
    assert tokens == ["मैंने", "हिंदी", "की", "किताबें", "पढ़ीं"]


def test_nfc_is_idempotent() -> None:
    """Normalizing twice equals normalizing once."""
    text = "ज़रूरत  के   लिए  हिंदी  टेक्स्ट\u200b  "
    assert normalize(normalize(text)) == normalize(text)


def test_nukta_precomposed_and_decomposed_agree() -> None:
    """U+0958 (precomposed qa) normalizes to the same string as KA + NUKTA."""
    precomposed = "\u0958"
    decomposed = "\u0915\u093c"
    assert normalize(precomposed) == normalize(decomposed)


def test_zero_width_characters_removed() -> None:
    """ZWSP, ZWNJ, ZWJ and BOM do not survive normalization."""
    text = "हिं\u200bदी\u200c प\u200dढ़ना\ufeff"
    cleaned = normalize(text)
    assert not any(char in cleaned for char in "\u200b\u200c\u200d\ufeff")
    assert cleaned == "हिंदी पढ़ना"


def test_whitespace_collapsed() -> None:
    """Tabs, newlines and repeated spaces collapse to single spaces."""
    assert normalize("  दिल्ली\t\tमें\n\nरहता   हूँ ") == "दिल्ली में रहता हूँ"


@pytest.mark.parametrize(
    ("devanagari", "ascii_digits"),
    [("२०२४", "2024"), ("०१२३४५६७८९", "0123456789")],
)
def test_devanagari_digits_become_ascii(devanagari: str, ascii_digits: str) -> None:
    """Devanagari digits map to ASCII digits, in text and in tokens."""
    assert "".join(char for char in normalize(devanagari)) == ascii_digits
    assert tokenize(f"साल {devanagari} में", "hi") == ["साल", ascii_digits, "में"]


def test_latin_words_inside_hindi_text_survive() -> None:
    """Mixed-script text keeps its Latin tokens, lowercased."""
    tokens = tokenize("मुझे Machine Learning पसंद है", "hi")
    assert "machine" in tokens
    assert "learning" in tokens
    assert "पसंद" in tokens


def test_english_lowercases_hindi_does_not() -> None:
    """English is lowercased; Hindi is left unchanged."""
    assert tokenize("Delhi IS big", "en") == ["delhi", "is", "big"]
    assert tokenize("दिल्ली बड़ी है", "hi") == ["दिल्ली", "बड़ी", "है"]


def test_punctuation_dropped_by_default() -> None:
    """Danda and other punctuation are not tokens unless requested."""
    assert tokenize("हाँ, ठीक है।", "hi") == ["हाँ", "ठीक", "है"]
    assert "।" in tokenize("हाँ, ठीक है।", "hi", keep_punct=True)


def test_danda_splits_sentences() -> None:
    """A danda terminates a sentence; the danda does not leak into output."""
    text = "मैं कल दिल्ली गया था। वह किताब पढ़ रही है।"
    assert sentence_split(text, "hi") == ["मैं कल दिल्ली गया था", "वह किताब पढ़ रही है"]


def test_double_danda_splits_sentences() -> None:
    """The double danda is a boundary too."""
    assert sentence_split("पहला वाक्य॥ दूसरा वाक्य।", "hi") == ["पहला वाक्य", "दूसरा वाक्य"]


def test_decimal_point_does_not_split() -> None:
    """``3.5`` is not a sentence boundary, but a real full stop is."""
    assert sentence_split("दर 3.5 प्रतिशत है। यह अच्छा है।", "hi") == [
        "दर 3.5 प्रतिशत है",
        "यह अच्छा है",
    ]
    assert sentence_split("It costs 3.5 dollars. Fine.", "en") == [
        "It costs 3.5 dollars",
        "Fine",
    ]


@pytest.mark.parametrize("terminator", ["?", "!"])
def test_question_and_exclamation_split(terminator: str) -> None:
    """Question and exclamation marks terminate sentences."""
    text = f"क्या हाल है{terminator} ठीक हूँ।"
    assert sentence_split(text, "hi") == ["क्या हाल है", "ठीक हूँ"]


def test_sentence_split_drops_empty_fragments() -> None:
    """Leading, trailing and repeated terminators do not produce empty sentences."""
    assert sentence_split("।। पहला वाक्य।। ", "hi") == ["पहला वाक्य"]


def test_devanagari_ratio() -> None:
    """The ratio counts letters only (matras are marks, not letters)."""
    assert is_devanagari_ratio("पूरी तरह हिंदी") == 1.0
    assert is_devanagari_ratio("only english") == 0.0
    assert is_devanagari_ratio("") == 0.0
    # ह, द are the Devanagari letters; ि and ं are combining marks.
    assert is_devanagari_ratio("हिंदी english") == pytest.approx(2 / 9)


def test_devanagari_ratio_mixed_paragraph_threshold() -> None:
    """A mostly-English paragraph falls below the 0.4 Wikipedia filter threshold."""
    paragraph = "The quick brown fox jumps over the lazy dog हिंदी"
    assert is_devanagari_ratio(paragraph) < 0.4


def test_latin_tokens_are_lowercased_in_hindi_text() -> None:
    """Mixed-script tokens are lowercased even when the language is Hindi."""
    tokens = tokenize("मुझे Machine Learning पसंद है", "hi")
    assert tokens == ["मुझे", "machine", "learning", "पसंद", "है"]


def test_unknown_language_rejected() -> None:
    """Only ``hi`` and ``en`` are accepted."""
    with pytest.raises(ValueError):
        tokenize("text", "fr")
    with pytest.raises(ValueError):
        sentence_split("text", "fr")
    with pytest.raises(TypeError):
        normalize(None)  # type: ignore[arg-type]
