"""Tests for chunked cosine and CSLS retrieval (S07).

The reference implementations below build the full score matrix and are only
usable on toy inputs; the point is that the chunked code agrees with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import retrieval
from src.retrieval import cosine_topk, csls_topk, hubness_mean, rank_of_gold

SEED = 42
TOY_SIZE = 50
TOY_DIM = 16


def _unit(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with L2-normalized rows."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def toy_vectors() -> dict[str, np.ndarray]:
    """Fifty unit vectors in 16 dimensions, used as queries and candidates."""
    rng = np.random.default_rng(SEED)
    return {
        "matrix": _unit(rng.normal(size=(TOY_SIZE, TOY_DIM))),
        "queries": _unit(rng.normal(size=(20, TOY_DIM))),
    }


def _brute_cosine(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Full cosine score matrix (no chunking)."""
    return _unit(queries) @ _unit(candidates).T


def _brute_hubness(source: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """Brute-force ``r_S(y)``: mean cosine of each candidate to its k nearest sources."""
    scores = _brute_cosine(source, candidates)
    partition = np.sort(scores, axis=0)[::-1][:k]
    return partition.mean(axis=0)


def _brute_csls(
    queries: np.ndarray, candidates: np.ndarray, source: np.ndarray, k: int
) -> np.ndarray:
    """Brute-force CSLS score matrix, including the per-query ``r_T(x)`` term."""
    scores = _brute_cosine(queries, candidates)
    r_query = np.sort(scores, axis=1)[:, ::-1][:, :k].mean(axis=1)
    r_source = _brute_hubness(source, candidates, k)
    return 2.0 * scores - r_query[:, None] - r_source[None, :]


def test_cosine_topk_matches_brute_force(toy_vectors):
    """Chunked cosine ranking equals the full-matrix ranking."""
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    indices, scores = cosine_topk(queries, candidates, k=5, chunk=7)
    expected = _brute_cosine(queries, candidates)
    expected_order = np.argsort(-expected, axis=1, kind="stable")[:, :5]

    assert indices.shape == (queries.shape[0], 5)
    np.testing.assert_array_equal(indices, expected_order)
    np.testing.assert_allclose(scores, np.take_along_axis(expected, expected_order, axis=1), atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 7, 19, 20, 1000])
def test_chunk_size_does_not_change_results(toy_vectors, chunk):
    """Row chunking never changes indices or scores."""
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    reference_index, reference_score = cosine_topk(queries, candidates, k=4, chunk=1000)
    indices, scores = cosine_topk(queries, candidates, k=4, chunk=chunk)
    np.testing.assert_array_equal(indices, reference_index)
    np.testing.assert_allclose(scores, reference_score, atol=1e-6)


def test_column_blocking_matches_brute_force(toy_vectors, monkeypatch):
    """Streaming the candidate pool in small column blocks gives the same answer."""
    monkeypatch.setattr(retrieval, "_MAX_BLOCK_ELEMENTS", 60)
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    indices, scores = cosine_topk(queries, candidates, k=5, chunk=6)
    expected = _brute_cosine(queries, candidates)
    expected_order = np.argsort(-expected, axis=1, kind="stable")[:, :5]
    np.testing.assert_array_equal(indices, expected_order)
    np.testing.assert_allclose(scores, np.take_along_axis(expected, expected_order, axis=1), atol=1e-5)


def test_cosine_topk_is_sorted_descending(toy_vectors):
    """Returned scores decrease along each row and indices are unique."""
    indices, scores = cosine_topk(toy_vectors["queries"], toy_vectors["matrix"], k=6, chunk=5)
    assert np.all(np.diff(scores, axis=1) <= 1e-6)
    for row in indices:
        assert len(set(row.tolist())) == len(row)


def test_identical_query_and_candidate_sets_rank_first(toy_vectors):
    """When queries are the candidates themselves, every item retrieves itself first."""
    matrix = toy_vectors["matrix"]
    indices, _ = cosine_topk(matrix, matrix, k=3, chunk=8)
    np.testing.assert_array_equal(indices[:, 0], np.arange(matrix.shape[0]))


def test_csls_matches_brute_force(toy_vectors):
    """CSLS indices and scores equal the brute-force reference on toy data."""
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    source = candidates
    indices, scores = csls_topk(queries, candidates, k=5, source=source, csls_k=10, chunk=6)

    expected = _brute_csls(queries, candidates, source, k=10)
    expected_order = np.argsort(-expected, axis=1, kind="stable")[:, :5]
    np.testing.assert_array_equal(indices, expected_order)
    np.testing.assert_allclose(scores, np.take_along_axis(expected, expected_order, axis=1), atol=1e-5)


def test_csls_chunking_is_stable(toy_vectors, monkeypatch):
    """CSLS is invariant to both row and column blocking."""
    monkeypatch.setattr(retrieval, "_MAX_BLOCK_ELEMENTS", 60)
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    reference_index, reference_score = csls_topk(
        queries, candidates, k=5, source=candidates, csls_k=10, chunk=1000
    )
    indices, scores = csls_topk(queries, candidates, k=5, source=candidates, csls_k=10, chunk=6)
    np.testing.assert_array_equal(indices, reference_index)
    np.testing.assert_allclose(scores, reference_score, atol=1e-6)


def test_hubness_term_matches_brute_force(toy_vectors):
    """``r_S(y)`` equals the mean cosine to the k nearest sources."""
    matrix = toy_vectors["matrix"]
    np.testing.assert_allclose(
        hubness_mean(matrix, matrix, csls_k=10, chunk=7),
        _brute_hubness(matrix, matrix, 10),
        atol=1e-5,
    )


def test_csls_differs_from_cosine(toy_vectors):
    """CSLS is not a no-op: on toy data its top-1 differs from plain cosine for some query."""
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    cosine_index, _ = cosine_topk(queries, candidates, k=1, chunk=1000)
    csls_index, _ = csls_topk(queries, candidates, k=1, source=candidates, csls_k=10, chunk=1000)
    assert not np.array_equal(cosine_index, csls_index)


def test_shuffled_gold_labels_give_near_chance(toy_vectors):
    """Retrieving with correct queries but shuffled gold labels scores near chance."""
    rng = np.random.default_rng(SEED)
    size = 200
    matrix = _unit(rng.normal(size=(size, TOY_DIM)))
    indices, _ = cosine_topk(matrix, matrix, k=1, chunk=64)
    retrieved = indices[:, 0]

    accuracies = []
    for shuffle_seed in range(5):
        gold = np.random.default_rng(shuffle_seed).permutation(size)
        accuracies.append(float(np.mean(retrieved == gold)))
    assert np.mean(accuracies) < 0.05
    assert np.max(accuracies) < 0.10


def test_rank_of_gold_simple():
    """Ranks are 1-based, ties share the best rank and an exact top match is rank 1."""
    scores = np.array([0.9, 0.5, 0.5, 0.1], dtype=np.float32)
    assert rank_of_gold(scores, [0]) == 1
    assert rank_of_gold(scores, [1]) == 2
    assert rank_of_gold(scores, [3]) == 4
    assert rank_of_gold(scores, [1, 3]) == 2
    assert rank_of_gold(scores, [0, 1]) == 1


def test_rank_of_gold_rejects_bad_input():
    """Empty or out-of-range gold sets are an error, not a silent rank."""
    scores = np.array([0.9, 0.1], dtype=np.float32)
    with pytest.raises(ValueError):
        rank_of_gold(scores, [])
    with pytest.raises(ValueError):
        rank_of_gold(scores, [5])


def test_dimension_mismatch_fails_loudly(toy_vectors):
    """Mismatched dimensions raise instead of broadcasting silently."""
    with pytest.raises(ValueError):
        cosine_topk(np.zeros((3, 4)), np.zeros((5, 6)), k=2)


def test_invalid_k_rejected(toy_vectors):
    """``k < 1`` is rejected; ``k`` above the pool size is clipped."""
    queries, candidates = toy_vectors["queries"], toy_vectors["matrix"]
    with pytest.raises(ValueError):
        cosine_topk(queries, candidates, k=0)
    indices, _ = cosine_topk(queries, candidates, k=10_000, chunk=10)
    assert indices.shape == (queries.shape[0], candidates.shape[0])
