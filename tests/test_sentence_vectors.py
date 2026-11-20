"""Tests for SIF and plain-average sentence vectors (S25), offline and tiny."""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.sentence_vectors import (
    compose_average,
    compose_sif,
    fit_pc,
    read_word_counts,
    remove_projection,
)

DIM = 6


@pytest.fixture
def vectors() -> dict[str, np.ndarray]:
    """A tiny word-vector table; ``oov_word`` is deliberately absent."""
    rng = np.random.default_rng(7)
    words = ["a", "b", "c", "d", "e", "f"]
    return {word: rng.normal(size=DIM).astype(np.float32) for word in words}


@pytest.fixture
def counts() -> dict[str, int]:
    """Plausible token counts for the fixture words."""
    return {"a": 100, "b": 50, "c": 10, "d": 5, "e": 1, "f": 1}


def test_sif_is_invariant_to_word_order(vectors, counts):
    """H3 depends on averaging being order-blind, so this must hold exactly."""
    total = sum(counts.values())
    first, _ = compose_sif([["a", "b", "c"]], vectors, counts, total, dim=DIM)
    second, _ = compose_sif([["c", "b", "a"]], vectors, counts, total, dim=DIM)

    assert np.allclose(first, second, atol=1e-6)


def test_average_is_invariant_to_word_order(vectors):
    """The plain-average ablation must also be order-blind."""
    first, _ = compose_average([["a", "b", "c"]], vectors, dim=DIM)
    second, _ = compose_average([["c", "b", "a"]], vectors, dim=DIM)

    assert np.allclose(first, second, atol=1e-6)


def test_oov_words_are_skipped_and_counted(vectors, counts):
    """Out-of-vocabulary words contribute nothing but are counted."""
    total = sum(counts.values())
    matrix, stats = compose_sif([["a", "oov_word", "b"]], vectors, counts, total, dim=DIM)

    assert matrix.shape == (1, DIM)
    assert stats.n_oov_words == 1
    assert stats.n_total_words == 3


def test_all_oov_sentence_yields_zero_vector(vectors, counts):
    """A sentence with no known word must be flagged, not crash or produce NaN."""
    total = sum(counts.values())
    matrix, stats = compose_sif([["oov_one", "oov_two"]], vectors, counts, total, dim=DIM)

    assert stats.n_empty_items == 1
    assert np.all(matrix == 0.0)


def test_output_rows_are_unit_length(vectors, counts):
    """Cosine similarity assumes unit rows."""
    total = sum(counts.values())
    matrix, _ = compose_sif([["a", "b"], ["c", "d"]], vectors, counts, total, dim=DIM)

    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_words_absent_from_counts_get_the_default_probability(vectors):
    """A word in the vectors but not in the counts uses ``p = 1/total``."""
    counts = {"a": 10}
    total = 10
    matrix, stats = compose_sif([["a", "b"]], vectors, counts, total, dim=DIM)

    assert stats.n_oov_words == 0
    assert stats.n_total_words == 2
    assert np.allclose(np.linalg.norm(matrix[0]), 1.0, atol=1e-5)


def test_fit_pc_returns_a_unit_vector(vectors):
    """The principal component is a unit vector."""
    matrix = np.vstack(list(vectors.values()))
    pc = fit_pc(matrix)

    assert pc.shape == (DIM,)
    assert np.isclose(np.linalg.norm(pc), 1.0, atol=1e-5)


def test_remove_projection_makes_rows_orthogonal_to_pc(vectors):
    """After PC removal every row is orthogonal to the component."""
    matrix = np.vstack(list(vectors.values()))
    pc = fit_pc(matrix)
    reduced = remove_projection(matrix, pc)

    assert np.allclose(reduced @ pc, 0.0, atol=1e-4)


def test_read_word_counts(tmp_path):
    """``wordcounts`` files are ``word<TAB>count``."""
    path = tmp_path / "wordcounts_en.tsv"
    path.write_text("the\t100\ncat\t5\n", encoding="utf-8")

    counts, total = read_word_counts(path)

    assert counts == {"the": 100, "cat": 5}
    assert total == 105


def test_fit_pc_rejects_degenerate_input():
    """An all-zero matrix has no meaningful principal component."""
    with pytest.raises(ValueError, match="non-zero rows"):
        fit_pc(np.zeros((5, DIM), dtype=np.float32))


def test_sample_fit_paragraphs_takes_whole_paragraphs_from_the_fit_shard(tmp_path):
    """T3 fits its own component on paragraphs, never on the training shard."""
    from src.embeddings.sentence_vectors import sample_fit_paragraphs

    path = tmp_path / "wiki.en.para.tsv"
    rows = [
        f"train\t1\tt\t0\tfirst train paragraph that is long enough",
        f"fit\t2\tt\t0\tfirst fit paragraph with a danda। second fit sentence here",
        f"distractor\t3\tt\t0\tdistractor paragraph",
        f"fit\t4\tt\t0\tsecond fit paragraph text",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    paragraphs = sample_fit_paragraphs(path, shard="fit", limit=10, seed=42)

    assert sorted(paragraphs) == sorted(
        ["first fit paragraph with a danda। second fit sentence here", "second fit paragraph text"]
    )
    assert all("train" not in paragraph and "distractor" not in paragraph for paragraph in paragraphs)


def test_sample_fit_paragraphs_is_deterministic(tmp_path):
    from src.embeddings.sentence_vectors import sample_fit_paragraphs

    path = tmp_path / "wiki.en.para.tsv"
    rows = [f"fit\t{index}\tt\t0\tparagraph number {index} with words" for index in range(50)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    first = sample_fit_paragraphs(path, shard="fit", limit=10, seed=42)
    second = sample_fit_paragraphs(path, shard="fit", limit=10, seed=42)

    assert first == second
    assert len(first) == 10


def test_sample_fit_paragraphs_rejects_an_empty_shard(tmp_path):
    from src.embeddings.sentence_vectors import sample_fit_paragraphs

    path = tmp_path / "wiki.en.para.tsv"
    path.write_text("train\t1\tt\t0\tonly training text here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no fit-shard paragraphs"):
        sample_fit_paragraphs(path, shard="fit")
