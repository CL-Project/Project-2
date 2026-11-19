"""Tests for encoder wrappers and caching (S14).

The cache, prefix and ordering logic is tested through a fake encoder so the
suite stays offline and fast.  One integration test exercises a real cached
model; it is skipped automatically when the weights are not on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.embeddings import encoders
from src.embeddings.encoders import (
    EncodeStats,
    EncoderSettings,
    _l2_normalize,
    _sidecar_matches,
    cache_paths,
    encode,
    encoder_settings,
)


class _FakeModel:
    """Deterministic stand-in for a transformer: row order drives the vector."""

    def __init__(self, dim: int = 6) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            digest = sum(ord(char) for char in text)
            rows.append([(digest + offset) % 97 + 1.0 for offset in range(self.dim)])
        return np.asarray(rows, dtype=np.float32)


class _FakeEncoder:
    """Minimal encoder object satisfying the interface ``encode`` relies on."""

    def __init__(self, settings: EncoderSettings, dim: int = 6) -> None:
        self.settings = settings
        self.model = _FakeModel(dim)
        self._tokenizer = _FakeTokenizer()

    def tokenizer(self):
        return self._tokenizer

    def encode(self, texts, kind, batch_size, max_len):  # noqa: D401 - mirrors the real API
        return self.model.encode(list(texts), batch_size, True, False, False)

    def free(self) -> None:
        return None


class _FakeTokenizer:
    """Tokenizer that reports one token per whitespace-separated piece."""

    def __call__(self, texts, add_special_tokens=True, truncation=False, padding=False, verbose=False):
        return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}


@pytest.fixture
def fake_encoder(monkeypatch):
    """Install a fake encoder for one system and expose it."""

    def install(system: str = "labse", dim: int = 6):
        settings = encoder_settings(system)
        encoder = _FakeEncoder(settings, dim=dim)
        monkeypatch.setattr(encoders, "_get_encoder", lambda _settings: encoder)
        return encoder

    return install


def test_l2_normalize_gives_unit_rows_and_keeps_zero_rows():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = _l2_normalize(matrix)
    np.testing.assert_allclose(np.linalg.norm(normalized[0]), 1.0)
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_encoder_settings_read_config():
    labse = encoder_settings("labse")
    assert labse.max_len == 256 and labse.pooling == "sentence_transformers"
    me5 = encoder_settings("me5")
    assert me5.query_prefix == "query: " and me5.passage_prefix == "passage: "
    muril = encoder_settings("muril")
    assert muril.pooling == "mean" and muril.query_prefix == ""
    with pytest.raises(KeyError):
        encoder_settings("does-not-exist")


def test_me5_prefix_depends_on_kind():
    """mE5 is symmetric for queries, asymmetric for passages (AGENTS.md §10.2)."""
    me5 = encoder_settings("me5")
    assert me5.prefix_for("word") == "query: "
    assert me5.prefix_for("sentence") == "query: "
    assert me5.prefix_for("query") == "query: "
    assert me5.prefix_for("passage") == "passage: "
    assert encoder_settings("labse").prefix_for("passage") == ""


def test_encode_returns_unit_rows_in_input_order(fake_encoder):
    fake_encoder("labse")
    texts = ["short", "a much longer piece of text", "medium length here"]
    vectors = encode("labse", texts, "sentence", cache=False)
    assert vectors.shape == (3, 6)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    # Row i must describe text i even though encoding sorts by length.
    lengths = np.array([len(text.split()) for text in texts])
    order = np.argsort(lengths, kind="stable")
    sorted_vectors = encode("labse", [texts[i] for i in order], "sentence", cache=False)
    np.testing.assert_allclose(vectors[order], sorted_vectors, atol=1e-6)


def test_encode_is_deterministic_and_order_independent(fake_encoder):
    fake_encoder("muril")
    texts = ["alpha beta", "gamma", "delta epsilon zeta eta"]
    first = encode("muril", texts, "sentence", cache=False)
    shuffled = [texts[2], texts[0], texts[1]]
    second = encode("muril", shuffled, "sentence", cache=False)
    np.testing.assert_allclose(first[[2, 0, 1]], second, atol=1e-6)


def test_identical_strings_get_identical_vectors(fake_encoder):
    fake_encoder("labse")
    vectors = encode("labse", ["same text", "same text"], "sentence", cache=False)
    np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-6)


def test_word_kind_uses_short_max_len(fake_encoder):
    fake_encoder("labse")
    _, stats = encode("labse", ["भारत"], "word", cache=False, return_stats=True)
    assert isinstance(stats, EncodeStats)
    assert stats.max_len == 16
    assert stats.batch_size == 512


def test_truncation_is_counted(fake_encoder):
    fake_encoder("labse")
    texts = ["word " * 300, "short"]
    vectors, stats = encode("labse", texts, "sentence", cache=False, return_stats=True)
    assert vectors.shape == (2, 6)
    assert stats.n_truncated == 1
    assert 0.0 < stats.truncated_fraction < 1.0


def test_empty_input_rejected(fake_encoder):
    fake_encoder("labse")
    with pytest.raises(ValueError):
        encode("labse", [], "sentence", cache=False)
    with pytest.raises(ValueError):
        encode("labse", ["text"], "not-a-kind", cache=False)
    with pytest.raises(TypeError):
        encode("labse", [b"bytes"], "sentence", cache=False)


def test_cache_round_trip(tmp_path, fake_encoder):
    fake_encoder("labse")
    texts = ["first text", "second text"]
    array_path, meta_path = cache_paths("labse", "toy", "hi", cache_root=tmp_path)
    vectors = encode(
        "labse", texts, "sentence", dataset="toy", lang="hi", cache_root=tmp_path
    )
    assert array_path.exists() and meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["system"] == "labse" and meta["lang"] == "hi" and meta["kind"] == "sentence"
    assert meta["n_items"] == 2 and meta["normalized"] is True
    assert "n_truncated" in meta and "hf_id" in meta and "prefix" in meta

    # A second call reads the cache: the model must not be invoked again.
    model = encoders._LOADED  # noqa: F841 - documented single source of models
    reloaded = encode(
        "labse", texts, "sentence", dataset="toy", lang="hi", cache_root=tmp_path
    )
    np.testing.assert_allclose(reloaded, vectors, atol=1e-6)


def test_cache_is_skipped_when_texts_change(tmp_path, fake_encoder, monkeypatch):
    encoder = fake_encoder("labse")
    encode("labse", ["first text"], "sentence", dataset="toy", lang="hi", cache_root=tmp_path)
    assert len(encoder.model.calls) == 1
    encode("labse", ["different text"], "sentence", dataset="toy", lang="hi", cache_root=tmp_path)
    assert len(encoder.model.calls) == 2


def test_cache_use_cache_false_always_encodes(tmp_path, fake_encoder):
    encoder = fake_encoder("labse")
    for _ in range(2):
        encode(
            "labse",
            ["same text"],
            "sentence",
            dataset="toy",
            lang="hi",
            cache=False,
            use_cache=False,
            cache_root=tmp_path,
        )
    assert len(encoder.model.calls) == 2


def test_kind_tag_separates_caches(tmp_path, fake_encoder):
    fake_encoder("me5")
    query_path, _ = cache_paths("me5", "t3", "hi", "query", cache_root=tmp_path)
    passage_path, _ = cache_paths("me5", "t3", "hi", "passage", cache_root=tmp_path)
    assert query_path != passage_path
    assert query_path.name == "t3_hi_query.npy"
    assert passage_path.name == "t3_hi_passage.npy"


def test_sidecar_mismatch_is_detected():
    expected = {"system": "labse", "max_len": 256, "n_items": 2, "prefix": ""}
    assert _sidecar_matches(dict(expected), expected)
    assert not _sidecar_matches({**expected, "max_len": 128}, expected)
    assert not _sidecar_matches({**expected, "n_items": 3}, expected)
    assert not _sidecar_matches({"system": "labse"}, expected)


def test_dataset_and_lang_must_come_together(fake_encoder):
    fake_encoder("labse")
    with pytest.raises(ValueError):
        encode("labse", ["text"], "sentence", dataset="toy", cache=False)
    with pytest.raises(ValueError):
        encode("labse", ["text"], "sentence", lang="hi", cache=False)


def test_stats_fraction_is_zero_for_empty():
    stats = EncodeStats(n_items=0, n_truncated=0, max_len=16, dim=768, batch_size=512)
    assert stats.truncated_fraction == 0.0


# --- Integration with a real cached model -------------------------------------

_CACHED_MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "me5": "intfloat/multilingual-e5-base",
    "muril": "google/muril-base-cased",
}


def _model_cached(repo_id: str) -> bool:
    from pathlib import Path

    folder = "models--" + repo_id.replace("/", "--")
    return (Path.home() / ".cache" / "huggingface" / "hub" / folder / "snapshots").is_dir()


@pytest.mark.parametrize("system", sorted(_CACHED_MODELS))
def test_real_encoder_shape_identity_and_order(system):
    """A real model: right shape, cosine 1.0 for identical strings, order invariant."""
    if not _model_cached(_CACHED_MODELS[system]):
        pytest.skip(f"{system} weights are not cached")
    texts = ["मैं कल दिल्ली गया था।", "I went to Delhi yesterday.", "मैं कल दिल्ली गया था।"]
    vectors = encode(system, texts, "sentence", cache=False)
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert float(vectors[0] @ vectors[2]) > 0.999

    reordered = encode(system, [texts[1], texts[0], texts[2]], "sentence", cache=False)
    np.testing.assert_allclose(reordered[[1, 0, 2]], vectors, atol=5e-3)
