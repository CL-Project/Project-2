"""Tests for T1 pool construction and ranking (S22), offline and tiny."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks import bli


def test_read_dictionary_groups_multiple_translations(tmp_path):
    """A MUSE source word may have several translations; all become gold."""
    path = tmp_path / "hi-en.5000-6500.txt"
    path.write_text("के\tof\nके\tfrom\nमें\tin\n", encoding="utf-8")

    entries = bli.read_dictionary(path)

    assert entries == {"के": {"of", "from"}, "में": {"in"}}


def test_read_dictionary_rejects_malformed_lines(tmp_path):
    """Wrong column counts must fail with their location."""
    path = tmp_path / "bad.txt"
    path.write_text("के में का\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected two tab-separated fields"):
        bli.read_dictionary(path)


def test_build_pool_keeps_only_the_common_set():
    """Queries outside the shared source list or without a gold candidate are excluded."""
    dictionary = {
        "के": {"of"},
        "में": {"in"},
        "oov_source": {"in"},
        "no_gold": {"missing_target"},
    }
    pool = bli.build_pool(
        dictionary,
        setting="hi-en",
        source_candidates=["के", "में", "no_gold"],
        target_candidates=["of", "in", "other"],
    )

    assert pool.query_words == ["के", "में"]
    assert pool.n_excluded == 2
    assert pool.n_total == 4
    assert pool.gold_indices == [[0], [1]]


def test_build_pool_records_the_candidate_pool_size():
    """The candidate pool is shared and its size is reported per item."""
    pool = bli.build_pool(
        {"के": {"of"}},
        setting="hi-en",
        source_candidates=["के"],
        target_candidates=["of", "in"],
    )
    assert pool.n_candidates == 2


def test_build_pool_fails_when_nothing_survives():
    """An empty common set means the candidate lists are wrong; do not log a fake run."""
    with pytest.raises(RuntimeError, match="no query survived"):
        bli.build_pool(
            {"के": {"of"}},
            setting="hi-en",
            source_candidates=["other"],
            target_candidates=["of"],
        )


def test_rank_cosine_finds_the_gold_at_rank_one():
    """A query identical to its gold candidate must rank 1."""
    candidates = np.eye(4, dtype=np.float32)
    queries = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = bli.rank_cosine(queries, candidates, [[0]])

    assert result.ranks.tolist() == [1]
    assert result.best_index.tolist() == [0]


def test_rank_cosine_counts_better_candidates():
    """The rank is the number of candidates strictly above the best gold, plus one."""
    candidates = np.array(
        [
            [0.8, 0.6],  # gold, cosine 0.8 with the query
            [1.0, 0.0],  # strictly better than the gold
            [0.0, 1.0],  # worse
        ],
        dtype=np.float32,
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    result = bli.rank_cosine(queries, candidates, [[0]])

    assert result.ranks.tolist() == [2]


def test_rank_cosine_uses_the_best_of_several_gold_items():
    """With multiple translations, the best-ranked gold decides the rank."""
    candidates = np.array(
        [
            [0.5, 0.5],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    result = bli.rank_cosine(queries, candidates, [[0, 1]])

    assert result.ranks.tolist() == [1]


def test_rank_csls_matches_cosine_when_hubness_is_flat():
    """With a supplied constant hubness vector and k=1, CSLS ordering equals cosine ordering."""
    candidates = np.eye(3, dtype=np.float32)
    queries = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    hubness = np.zeros(3, dtype=np.float32)

    result = bli.rank_csls(queries, candidates, [[0]], source=candidates, csls_k=1, hubness=hubness)

    assert result.ranks.tolist() == [1]


def test_rank_csls_rejects_a_wrong_sized_hubness_vector():
    """A stale cached hubness vector must not be used silently."""
    candidates = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="hubness has shape"):
        bli.rank_csls(
            np.eye(3, dtype=np.float32),
            candidates,
            [[0], [1], [2]],
            source=candidates,
            hubness=np.zeros(2, dtype=np.float32),
        )


def test_per_item_frame_has_the_required_columns():
    """The per-item schema is mandatory for the later analyses."""
    pool = bli.build_pool(
        {"के": {"of"}, "में": {"in"}},
        setting="hi-en",
        source_candidates=["के", "में"],
        target_candidates=["of", "in"],
    )
    result = bli.rank_cosine(
        np.vstack([np.array([1.0, 0.0]), np.array([0.0, 1.0])]).astype(np.float32),
        np.eye(2, dtype=np.float32),
        pool.gold_indices,
    )
    frame = bli.per_item_frame(pool, result)

    assert list(frame.columns) == [
        "query_id",
        "gold_ids",
        "rank_of_best_gold",
        "top1",
        "top5",
        "rr",
        "n_candidates",
    ]
    assert frame["query_id"].tolist() == ["के", "में"]
    assert frame["top1"].tolist() == [True, True]
    assert frame["n_candidates"].tolist() == [2, 2]


def test_coverage_frame_counts_excluded_queries_as_wrong():
    """PLAN S22's secondary metric: full test dictionary, OOV queries scored as wrong."""
    dictionary = {"के": {"of"}, "में": {"in"}, "oov_source": {"in"}}
    pool = bli.build_pool(
        dictionary,
        setting="hi-en",
        source_candidates=["के", "में"],
        target_candidates=["of", "in"],
    )
    result = bli.rank_cosine(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), pool.gold_indices)

    frame = bli.coverage_per_item_frame(dictionary, ["of", "in"], pool, result)

    assert len(frame) == 3
    assert frame["query_id"].tolist() == sorted(dictionary)
    excluded = frame.loc[frame["query_id"] == "oov_source"]
    assert excluded["rank_of_best_gold"].tolist() == [pool.n_candidates + 1]
    assert excluded["top1"].tolist() == [False]
    assert float(frame["top1"].mean()) == pytest.approx(2 / 3)


def test_coverage_frame_keeps_gold_without_a_source_candidate():
    """A query outside the common set still reports its gold translations for analysis."""
    dictionary = {"के": {"of"}, "oov_source": {"in"}}
    pool = bli.build_pool(
        dictionary,
        setting="hi-en",
        source_candidates=["के"],
        target_candidates=["of", "in"],
    )
    result = bli.rank_cosine(np.array([[1.0, 0.0]], np.float32), np.eye(2, dtype=np.float32), pool.gold_indices)

    frame = bli.coverage_per_item_frame(dictionary, ["of", "in"], pool, result)

    assert frame.loc[frame["query_id"] == "oov_source", "gold_ids"].tolist() == [["in"]]


def test_pool_digest_is_stable():
    """The digest used for cache safety must be deterministic."""
    pool = bli.build_pool(
        {"के": {"of"}},
        setting="hi-en",
        source_candidates=["के"],
        target_candidates=["of", "in"],
    )
    assert bli.pool_digest(pool) == bli.pool_digest(pool)


def test_fold_assignment_is_deterministic_and_balanced():
    """S24's folds must be reproducible from the seed and roughly equal in size."""
    words = [f"w{index:03d}" for index in range(100)]
    first = bli.fold_assignment(words, folds=5, seed=42)
    second = bli.fold_assignment(words, folds=5, seed=42)

    assert first == second
    assert set(first) == set(words)
    assert set(first.values()) == set(range(5))
    sizes = [sum(1 for fold in first.values() if fold == index) for index in range(5)]
    assert max(sizes) - min(sizes) <= 1


def test_fold_assignment_changes_with_the_seed():
    """Different seeds must not silently produce the same split."""
    words = [f"w{index:03d}" for index in range(100)]
    assert bli.fold_assignment(words, folds=5, seed=42) != bli.fold_assignment(words, folds=5, seed=7)


def test_fold_assignment_rejects_too_few_words():
    """A split needs at least one word per fold."""
    with pytest.raises(ValueError, match="cannot split"):
        bli.fold_assignment(["a", "b"], folds=5)
