"""Sanity checks for the shared synthetic fixtures (S09)."""

import numpy as np
import pytest

from src.metrics import acc_at_k, mrr


def test_synthetic_embeddings_are_unit_vectors_with_an_orthogonal_rotation(synthetic_embeddings):
    source = synthetic_embeddings["source"]
    target = synthetic_embeddings["target"]
    rotation = synthetic_embeddings["rotation"]

    assert source.shape == (20, 8)
    assert np.allclose(np.linalg.norm(source, axis=1), 1.0)
    assert np.allclose(target, source @ rotation)
    assert np.allclose(rotation.T @ rotation, np.eye(8), atol=1e-12)


def test_toy_parallel_corpus_has_ten_aligned_pairs(toy_parallel_corpus):
    assert len(toy_parallel_corpus) == 10
    for hindi, english in toy_parallel_corpus:
        assert hindi and english
        assert "।" in hindi


def test_toy_rankings_have_hand_computable_metrics(toy_rankings):
    assert acc_at_k(toy_rankings, 1) == pytest.approx(1 / 8)
    assert acc_at_k(toy_rankings, 3) == pytest.approx(4 / 8)
    expected_mrr = (1 + 1 / 2 + 1 / 2 + 1 / 3 + 1 / 5 + 1 / 8 + 1 / 10 + 1 / 20) / 8
    assert mrr(toy_rankings) == pytest.approx(expected_mrr)
