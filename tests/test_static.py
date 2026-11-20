"""Tests for static word-vector training and loading (S19), offline and tiny."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.embeddings import static


@pytest.fixture
def tiny_config():
    """A one-epoch configuration so training takes a fraction of a second."""
    config = copy.deepcopy(static.load_models_config())
    config["static"]["vocab_cap"] = 100_000
    config["static"]["word2vec"].update(epochs=1, workers=1)
    config["static"]["fasttext"].update(epochs=1, workers=1, bucket=10_000)
    return config


@pytest.fixture
def corpus(tmp_path):
    """A tiny tokenized corpus with enough repetition to pass ``min_count``."""
    sentences = [
        "india is a country in asia",
        "delhi is the capital of india",
        "water is essential for life",
        "the government passed a new law",
        "this film is very good",
        "i read a book about history",
    ]
    path = tmp_path / "wiki.en.tok.txt"
    path.write_text("\n".join(sentences * 20) + "\n", encoding="utf-8")
    return path


def test_training_params_come_from_config():
    """No hyperparameter is hard-coded: everything comes from ``models.yaml``."""
    config = static.load_models_config()
    w2v = static.training_params("w2v", config)
    ft = static.training_params("ft", config)

    assert config["static"]["word2vec"]["window"] == w2v["window"]
    assert w2v["vector_size"] == config["static"]["dim"]
    assert w2v["max_final_vocab"] == config["static"]["vocab_cap"]
    assert "min_n" not in w2v
    assert ft["min_n"] == config["static"]["fasttext"]["min_n"]
    assert ft["max_n"] == config["static"]["fasttext"]["max_n"]


def test_training_params_reject_unknown_system():
    """An unknown system id must fail loudly."""
    with pytest.raises(ValueError, match="unsupported static system"):
        static.training_params("glove")


@pytest.mark.parametrize("system", ["w2v", "ft"])
def test_train_saves_and_reloads(system, tiny_config, corpus, tmp_path):
    """Both families train, save and reload with the expected shape."""
    root = tmp_path / "vectors"
    summary = static.train(system, "en", config=tiny_config, corpus=corpus, static_root=root, force=True)

    assert summary["vocab_size"] > 0
    assert summary["vector_size"] == tiny_config["static"]["dim"]
    assert static.model_path(system, "en", root).exists()

    vectors = static.load(system, "en", static_root=root)
    assert vectors.vector_size == tiny_config["static"]["dim"]
    assert "india" in vectors


def test_train_is_idempotent(tiny_config, corpus, tmp_path):
    """A second call must reuse the existing model instead of retraining."""
    root = tmp_path / "vectors"
    first = static.train("w2v", "en", config=tiny_config, corpus=corpus, static_root=root)
    second = static.train("w2v", "en", config=tiny_config, corpus=corpus, static_root=root)

    assert first["vocab_size"] == second["vocab_size"]
    assert static.model_path("w2v", "en", root).exists()


def test_nearest_neighbours_reports_oov_words(tiny_config, corpus, tmp_path):
    """Missing words are reported with an empty list rather than raising."""
    root = tmp_path / "vectors"
    static.train("w2v", "en", config=tiny_config, corpus=corpus, static_root=root)
    vectors = static.load("w2v", "en", static_root=root)

    result = static.nearest_neighbours(vectors, ["india", "definitely_not_a_word"])

    assert isinstance(result["india"], list)
    assert result["definitely_not_a_word"] == []


def test_vector_matrix_rejects_oov():
    """A missing word must raise instead of silently shifting the rows."""

    class _Vectors:
        vector_size = 3
        key_to_index = {"a": 0}

        def __contains__(self, word):
            return word in self.key_to_index

        def __getitem__(self, word):
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(KeyError, match="not in the vocabulary"):
        static.vector_matrix(_Vectors(), ["a", "b"])
