"""Tests for Procrustes alignment (S21), using the synthetic rotated fixture.

The fixture in ``tests/conftest.py`` builds ``target = source @ rotation`` with a
random orthogonal rotation, so the correct mapping for these vectors is exactly
that rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.align import (
    apply_mapping,
    fit_mapping,
    is_orthogonal,
    l2_normalize,
    procrustes,
    read_pairs,
    seed_accuracy,
)


def test_procrustes_on_identical_vectors_returns_identity():
    """A mapping fitted against itself must be the identity."""
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.normal(size=(12, 8)))
    mapping = procrustes(vectors, vectors)
    assert np.allclose(mapping, np.eye(8), atol=1e-5)


def test_procrustes_recovers_a_known_rotation(synthetic_embeddings):
    """The fitted mapping must recover the rotation that generated the target."""
    source = synthetic_embeddings["source"]
    target = synthetic_embeddings["target"]
    rotation = synthetic_embeddings["rotation"]

    mapping = procrustes(source, target)

    assert np.allclose(mapping, rotation, atol=1e-4)
    assert np.allclose(source @ mapping, target, atol=1e-4)


def test_procrustes_returns_an_orthogonal_matrix(synthetic_embeddings):
    """Orthogonality is the defining constraint of the Procrustes solution."""
    mapping = procrustes(synthetic_embeddings["source"], synthetic_embeddings["target"])

    assert is_orthogonal(mapping)
    assert np.allclose(mapping.T @ mapping, np.eye(mapping.shape[0]), atol=1e-5)
    assert np.isclose(np.linalg.det(mapping), 1.0, atol=1e-4) or np.isclose(
        np.linalg.det(mapping), -1.0, atol=1e-4
    )


def test_apply_mapping_returns_unit_rows(synthetic_embeddings):
    """Mapped vectors are re-normalized, so cosine similarity is well defined."""
    mapping = procrustes(synthetic_embeddings["source"], synthetic_embeddings["target"])
    mapped = apply_mapping(synthetic_embeddings["source"], mapping)

    norms = np.linalg.norm(mapped, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_procrustes_rejects_mismatched_shapes():
    """A shape mismatch must fail loudly rather than align the wrong rows."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="same shape"):
        procrustes(rng.normal(size=(5, 4)), rng.normal(size=(6, 4)))


def test_procrustes_rejects_non_finite_values():
    """NaN input must not silently produce a mapping."""
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        procrustes(matrix, np.eye(4))


class _Vectors(dict):
    """Minimal stand-in for a gensim ``KeyedVectors`` (membership + lookup)."""

    def __getitem__(self, key):
        return super().__getitem__(key)


def test_fit_mapping_drops_out_of_vocabulary_pairs(synthetic_embeddings):
    """Pairs missing from either vocabulary are skipped and counted."""
    source = synthetic_embeddings["source"]
    target = synthetic_embeddings["target"]
    words = [f"w{index}" for index in range(len(source))]

    source_vectors = _Vectors({word: source[index] for index, word in enumerate(words)})
    target_vectors = _Vectors({word: target[index] for index, word in enumerate(words)})
    pairs = [(word, word) for word in words]
    pairs += [("missing_hi", words[0]), (words[0], "missing_en")]

    mapping, n_used = fit_mapping(source_vectors, target_vectors, pairs)

    assert n_used == len(words)
    assert np.allclose(mapping, synthetic_embeddings["rotation"], atol=1e-4)


def test_fit_mapping_requires_enough_pairs():
    """One surviving pair cannot determine an orthogonal mapping."""
    vectors = _Vectors({"a": np.ones(4, dtype=np.float32)})
    with pytest.raises(ValueError, match="cannot fit"):
        fit_mapping(vectors, vectors, [("a", "a"), ("b", "b")])


def test_read_pairs_parses_tab_separated_dictionary(tmp_path):
    """MUSE dictionaries are two tab-separated fields per line."""
    path = tmp_path / "hi-en.txt"
    path.write_text("के\tof\n\nमें\tin\n", encoding="utf-8")

    assert read_pairs(path) == [("के", "of"), ("में", "in")]


def test_read_pairs_rejects_malformed_lines(tmp_path):
    """A line without exactly two fields must fail with its location."""
    path = tmp_path / "bad.txt"
    path.write_text("के\tमें\tका\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected two tab-separated fields"):
        read_pairs(path)


def test_seed_accuracy_is_high_after_mapping(synthetic_embeddings):
    """Seed accuracy is near 1.0 once the mapping is fitted; a direction bug shows up as low."""
    source = synthetic_embeddings["source"]
    target = synthetic_embeddings["target"]
    words = [f"w{index}" for index in range(len(source))]
    source_vectors = _Vectors({word: source[index] for index, word in enumerate(words)})
    target_vectors = _Vectors({word: target[index] for index, word in enumerate(words)})
    pairs = [(word, word) for word in words]

    mapping, _ = fit_mapping(source_vectors, target_vectors, pairs)
    accuracy = seed_accuracy(source_vectors, target_vectors, pairs, mapping, top_k=1)

    assert accuracy >= 0.95


def test_seed_accuracy_is_low_with_the_wrong_direction(synthetic_embeddings):
    """Applying ``W^T`` instead of ``W`` must not look like a good mapping."""
    source = synthetic_embeddings["source"]
    target = synthetic_embeddings["target"]
    words = [f"w{index}" for index in range(len(source))]
    source_vectors = _Vectors({word: source[index] for index, word in enumerate(words)})
    target_vectors = _Vectors({word: target[index] for index, word in enumerate(words)})
    pairs = [(word, word) for word in words]

    mapping, _ = fit_mapping(source_vectors, target_vectors, pairs)
    accuracy = seed_accuracy(source_vectors, target_vectors, pairs, mapping.T, top_k=1)

    assert accuracy < 0.5
