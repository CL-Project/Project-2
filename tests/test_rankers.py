"""Tests for the BM25 baselines and the MT translation cache (S17).

No model is downloaded: the MT tests inject a fake translator, and BM25 runs on
a handful of toy sentences.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import mt
from src.rankers import bm25_rank, bm25_scores, bm25_settings


TOY_EN = [
    "The cat sat on the mat.",
    "A dog barked at the postman.",
    "The train to Delhi was late.",
    "Mangoes are sweet in summer.",
]


def test_bm25_settings_come_from_config():
    settings = bm25_settings()
    assert settings == {"k1": 1.5, "b": 0.75}


def test_bm25_scores_shape_and_exact_match_wins():
    """The candidate containing the query's words ranks first."""
    scores = bm25_scores(["the cat on the mat"], TOY_EN, lang="en")
    assert scores.shape == (1, 4)
    assert int(np.argmax(scores[0])) == 0
    assert scores[0, 0] > 0


def test_bm25_rank_matches_manual_ranking():
    """A query matching one document exactly has rank 1; a nonsense query ties at the bottom."""
    result = bm25_rank(["mangoes sweet summer"], TOY_EN, [[3]], lang="en")
    assert result.ranks[0] == 1

    absent = bm25_rank(["zzzz qqqq"], TOY_EN, [[1]], lang="en")
    # No term matches, every score is 0, so all candidates tie at rank 1.
    assert absent.ranks[0] == 1
    assert float(absent.best_score[0]) == 0.0


def test_bm25_rank_uses_hindi_tokenizer():
    """Devanagari words are not split, so the matching Hindi sentence ranks first."""
    docs = ["मैं कल दिल्ली गया था।", "वह किताब पढ़ रही है।", "बच्चे बगीचे में खेल रहे हैं।"]
    result = bm25_rank(["किताब"], docs, [[1]], lang="hi")
    assert result.ranks[0] == 1


def test_bm25_chunking_does_not_change_scores():
    queries = ["cat mat", "dog postman", "train delhi"]
    gold = [[0], [1], [2]]
    chunked = bm25_rank(queries, TOY_EN, gold, lang="en", chunk=1)
    whole = bm25_rank(queries, TOY_EN, gold, lang="en", chunk=100)
    np.testing.assert_array_equal(chunked.ranks, whole.ranks)
    np.testing.assert_allclose(chunked.best_score, whole.best_score, atol=1e-6)


def test_bm25_rejects_mismatched_gold():
    with pytest.raises(ValueError):
        bm25_rank(["a", "b"], TOY_EN, [[0]], lang="en")


def test_bm25_rejects_empty_input():
    with pytest.raises(ValueError):
        bm25_scores([], TOY_EN, lang="en")
    with pytest.raises(ValueError):
        bm25_scores(["query"], [], lang="en")


# --- MT cache ----------------------------------------------------------------


class _FakeTranslator:
    """Deterministic translator used instead of opus-mt."""

    def __init__(self, settings):
        self.settings = settings
        self.seen: list[str] = []

    def translate(self, texts, batch_size):
        self.seen.extend(texts)
        return [f"[en] {text}" for text in texts]

    def free(self):
        return None


@pytest.fixture
def fake_translator(monkeypatch):
    """Replace the module-level translator with a fake and expose it."""
    created: list[_FakeTranslator] = []

    def factory(settings):
        translator = _FakeTranslator(settings)
        created.append(translator)
        monkeypatch.setattr(mt, "_TRANSLATOR", translator)
        return translator

    monkeypatch.setattr(mt, "_Translator", factory)
    yield created
    monkeypatch.setattr(mt, "_TRANSLATOR", None)


def test_translate_cached_writes_and_reuses_cache(tmp_path, fake_translator):
    cache = tmp_path / "hi2en.tsv"
    texts = ["मैं कल दिल्ली गया था।", "वह किताब पढ़ रही है।"]
    first = mt.translate_cached(texts, cache)
    assert first == ["[en] मैं कल दिल्ली गया था।", "[en] वह किताब पढ़ रही है।"]
    assert cache.exists()
    header = cache.read_text(encoding="utf-8").splitlines()[0]
    assert header == "source\ttranslation"

    # A second call must not re-translate anything.
    translator = fake_translator[0]
    assert len(translator.seen) == 2
    second = mt.translate_cached(texts, cache)
    assert second == first
    assert len(translator.seen) == 2


def test_translate_cached_only_translates_new_texts(tmp_path, fake_translator):
    cache = tmp_path / "hi2en.tsv"
    mt.translate_cached(["पहला वाक्य"], cache)
    translator = fake_translator[0]
    assert translator.seen == ["पहला वाक्य"]

    mt.translate_cached(["पहला वाक्य", "दूसरा वाक्य"], cache)
    assert translator.seen == ["पहला वाक्य", "दूसरा वाक्य"]  # only the new one was translated


def test_translate_cached_force_retranslates(tmp_path, fake_translator):
    cache = tmp_path / "hi2en.tsv"
    mt.translate_cached(["वाक्य"], cache)
    translator = fake_translator[0]
    mt.translate_cached(["वाक्य"], cache, force=True)
    assert translator.seen == ["वाक्य", "वाक्य"]


def test_translate_cached_preserves_order_and_repeats(tmp_path, fake_translator):
    cache = tmp_path / "hi2en.tsv"
    texts = ["ब", "अ", "ब"]
    out = mt.translate_cached(texts, cache)
    assert out == [f"[en] {text}" for text in texts]


def test_translate_cache_bad_header_fails_loudly(tmp_path):
    cache = tmp_path / "hi2en.tsv"
    cache.write_text("wrong\theader\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected header"):
        mt.translate_cached(["वाक्य"], cache)


def test_translate_cached_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        mt.translate_cached([], tmp_path / "hi2en.tsv")
