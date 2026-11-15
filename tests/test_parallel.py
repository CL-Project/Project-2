"""Tests for T2 parallel sentence retrieval (S15 to S17).

Everything runs on a tiny synthetic pool; no FLORES download and no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tasks import parallel
from src.tasks.parallel import (
    SentencePair,
    build_pool,
    load_flores_pairs,
    parse_setting,
    per_item_frame,
    retrieve_by_cosine,
)


def _pairs(n: int = 6) -> list[SentencePair]:
    """Synthetic pairs with distinct Hindi and English sentences."""
    return [
        SentencePair(pair_id=f"dev:{i + 1:04d}", split="dev", index=i + 1, hi=f"हिंदी वाक्य {i}", en=f"english sentence {i}")
        for i in range(n)
    ]


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_parse_setting_validation():
    assert parse_setting("hi-en") == ("hi", "en")
    assert parse_setting("en-hi") == ("en", "hi")
    for bad in ("hi", "hi-en-fr", "fr-en", ""):
        with pytest.raises(ValueError):
            parse_setting(bad)


def test_build_pool_assigns_gold_by_row():
    """Query i's gold is the candidate from pair i; ids are the pair ids."""
    pool = build_pool(_pairs(6), "hi-en")
    assert pool.query_lang == "hi" and pool.candidate_lang == "en"
    assert pool.query_ids == pool.candidate_ids
    assert pool.query_texts[2] == "हिंदी वाक्य 2"
    assert pool.candidate_texts[2] == "english sentence 2"
    assert all(gold == [index] for index, gold in enumerate(pool.gold_indices))
    assert pool.n_queries == 6 and pool.n_candidates == 6


def test_build_pool_handles_duplicate_candidate_text():
    """Identical candidate sentences all become gold ids for the matching query."""
    pairs = _pairs(3)
    pairs[2] = SentencePair(pair_id="dev:0003", split="dev", index=3, hi="हिंदी वाक्य 2", en="english sentence 0")
    pool = build_pool(pairs, "hi-en")
    assert pool.gold_indices[2] == [0, 2]


def test_build_pool_reverse_direction():
    """en-hi swaps roles without touching the pair ids."""
    pool = build_pool(_pairs(4), "en-hi")
    assert pool.query_lang == "en" and pool.candidate_lang == "hi"
    assert pool.query_texts[1] == "english sentence 1"
    assert pool.candidate_texts[1] == "हिंदी वाक्य 1"
    assert pool.gold_indices[3] == [3]


def test_build_pool_rejects_empty():
    with pytest.raises(ValueError):
        build_pool([], "hi-en")


def test_load_flores_pairs_reads_two_splits(tmp_path):
    """Lengths are asserted per split and ids are split-qualified."""
    for split, count in (("dev", 3), ("devtest", 2)):
        for code, prefix in (("hin_Deva", "hi"), ("eng_Latn", "en")):
            directory = tmp_path / split
            directory.mkdir(parents=True, exist_ok=True)
            lines = [f"{prefix} {split} {index}" for index in range(count)]
            (directory / f"{code}.{split}").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pairs = load_flores_pairs(tmp_path)
    assert len(pairs) == 5
    assert [pair.pair_id for pair in pairs] == ["dev:0001", "dev:0002", "dev:0003", "devtest:0001", "devtest:0002"]
    assert pairs[0].hi == "hi dev 0" and pairs[0].en == "en dev 0"


def test_load_flores_pairs_rejects_unequal_lengths(tmp_path):
    for split, hindi, english in (("dev", 3, 2),):
        for code, count, prefix in (("hin_Deva", hindi, "hi"), ("eng_Latn", english, "en")):
            directory = tmp_path / split
            directory.mkdir(parents=True, exist_ok=True)
            lines = [f"{prefix} {index}" for index in range(count)]
            (directory / f"{code}.{split}").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to zip"):
        load_flores_pairs(tmp_path, splits=("dev",))


def test_retrieve_identical_candidates_ranks_first():
    """When queries are the candidate rows themselves, every rank is 1."""
    rng = np.random.default_rng(42)
    matrix = _unit(rng.normal(size=(30, 8)))
    gold = [[index] for index in range(30)]
    result = retrieve_by_cosine(matrix.astype(np.float32), matrix.astype(np.float32), gold)
    np.testing.assert_array_equal(result.ranks, np.ones(30, dtype=np.int64))
    np.testing.assert_allclose(result.gold_score, 1.0, atol=1e-6)


def test_retrieve_matches_brute_force_rank():
    """Chunked retrieval equals a brute-force score matrix with rank_of_gold."""
    from src.retrieval import rank_of_gold

    rng = np.random.default_rng(7)
    queries = _unit(rng.normal(size=(25, 6))).astype(np.float32)
    candidates = _unit(rng.normal(size=(40, 6))).astype(np.float32)
    gold = [[int(index) % 40, (int(index) * 3) % 40] for index in range(25)]
    result = retrieve_by_cosine(queries, candidates, gold, chunk=7)
    full = queries @ candidates.T
    expected = np.array([rank_of_gold(full[index], gold[index]) for index in range(25)])
    np.testing.assert_array_equal(result.ranks, expected)


def test_retrieve_chunking_invariant():
    rng = np.random.default_rng(11)
    queries = _unit(rng.normal(size=(17, 5))).astype(np.float32)
    candidates = _unit(rng.normal(size=(23, 5))).astype(np.float32)
    gold = [[int(index) % 23] for index in range(17)]
    first = retrieve_by_cosine(queries, candidates, gold, chunk=1000)
    second = retrieve_by_cosine(queries, candidates, gold, chunk=1)
    np.testing.assert_array_equal(first.ranks, second.ranks)


def test_retrieve_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        retrieve_by_cosine(np.zeros((3, 4), dtype=np.float32), np.zeros((3, 5), dtype=np.float32), [[0], [1], [2]])
    with pytest.raises(ValueError):
        retrieve_by_cosine(np.zeros((3, 4), dtype=np.float32), np.zeros((3, 4), dtype=np.float32), [[0], [1]])


def test_shuffled_gold_is_near_chance():
    """Wrong gold labels on random vectors give near-chance Acc@1."""
    rng = np.random.default_rng(3)
    size = 300
    matrix = _unit(rng.normal(size=(size, 16))).astype(np.float32)
    gold = [[int(value)] for value in rng.permutation(size)]
    result = retrieve_by_cosine(matrix, matrix, gold, chunk=64)
    assert float(np.mean(result.ranks <= 1)) < 0.02


def test_per_item_frame_schema():
    """The frame is ready for ``write_per_item`` and carries the documented columns."""
    pool = build_pool(_pairs(5), "hi-en")
    rng = np.random.default_rng(1)
    matrix = _unit(rng.normal(size=(5, 4))).astype(np.float32)
    result = retrieve_by_cosine(matrix, matrix, pool.gold_indices)
    frame = per_item_frame(pool, result)
    assert list(frame.columns) == [
        "query_id",
        "gold_ids",
        "rank_of_best_gold",
        "top1",
        "top5",
        "rr",
        "n_candidates",
    ]
    assert isinstance(frame, pd.DataFrame)
    assert frame["query_id"].tolist() == pool.query_ids
    assert frame["gold_ids"].iloc[0] == ["dev:0001"]
    assert (frame["top1"]).all() and (frame["top5"]).all()
    np.testing.assert_allclose(frame["rr"].to_numpy(), 1.0)


def test_per_item_frame_rejects_wrong_length():
    pool = build_pool(_pairs(4), "hi-en")
    result = parallel.RetrievalResult(
        ranks=np.ones(3, dtype=np.int64),
        best_index=np.zeros(3, dtype=np.int64),
        best_score=np.ones(3, dtype=np.float32),
        gold_score=np.ones(3, dtype=np.float32),
        n_candidates=4,
    )
    with pytest.raises(ValueError):
        per_item_frame(pool, result)


def test_row_alignment_survives_every_setting():
    """Row i of either language matrix is pair i (AGENTS.md §10.5).

    The candidate list is built in pair order and the gold of query *i* is the
    candidate with the same pair id, so a misaligned zip fails loudly here.
    """
    pairs = _pairs(8)
    for setting in ("hi-en", "en-hi"):
        pool = build_pool(pairs, setting)
        assert pool.query_ids == pool.candidate_ids == [pair.pair_id for pair in pairs]
        for index, pair in enumerate(pairs):
            assert pool.query_ids[index] == pair.pair_id
            gold = pool.gold_indices[index]
            assert [pool.candidate_ids[position] for position in gold] == [pair.pair_id]


def test_real_flores_row_alignment():
    """On the downloaded pool, Hindi row i and English row i come from the same pair.

    Skipped when ``data/raw/flores`` is absent, which is the normal state of a
    fresh clone (data is git-ignored).
    """
    from pathlib import Path

    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "flores"
    if not (raw / "dev" / "hin_Deva.dev").exists():
        pytest.skip("FLORES is not downloaded")
    pairs = load_flores_pairs(raw)
    hindi = parallel.read_flores_split(raw / "dev" / "hin_Deva.dev")
    assert pairs[0].hi == hindi[0]
    assert len({pair.pair_id for pair in pairs}) == len(pairs)
    pool = build_pool(pairs, "hi-en")
    assert pool.gold_indices[100] == [100]
    assert pool.candidate_texts[100] == pairs[100].en
