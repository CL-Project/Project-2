"""Sentence and paragraph vectors from static word vectors (S25).

Two compositions are implemented:

* **SIF** (Arora et al., 2017): a word of probability ``p(w)`` is weighted by
  ``a / (a + p(w))`` with ``a = 1e-3``, the weighted mean is taken over the
  in-vocabulary words, and the projection on the first principal component is
  removed.  The component is fitted on held-out Wikipedia text from the ``fit``
  shard, never on evaluation data.
* **plain average**: the unweighted mean, used as the ablation.

Out-of-vocabulary words are skipped and counted (``AGENTS.md`` §10.8).  A
sentence with no in-vocabulary word yields a zero vector and is flagged instead
of crashing the run.

Word-order invariance is the property H3 depends on, so it is tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import re

import numpy as np

from src.textproc import normalize, sentence_split, tokenize

LOGGER = logging.getLogger("sentence_vectors")

DEFAULT_A = 1e-3


@dataclass
class CompositionStats:
    """Counters for one composition run."""

    n_items: int
    n_oov_words: int
    n_empty_items: int
    n_total_words: int

    @property
    def oov_rate(self) -> float:
        """Fraction of word tokens that were out of vocabulary."""
        return self.n_oov_words / self.n_total_words if self.n_total_words else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready representation."""
        return {
            "n_items": self.n_items,
            "n_oov_words": self.n_oov_words,
            "n_empty_items": self.n_empty_items,
            "n_total_words": self.n_total_words,
            "oov_rate": round(self.oov_rate, 6),
        }


def read_word_counts(path: Path | str) -> tuple[dict[str, int], int]:
    """Read ``wordcounts_<lang>.tsv`` into ``(counts, total)``.

    ``total`` is the number of tokens counted, so ``p(w) = count/total``.
    """
    counts: dict[str, int] = {}
    total = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected two columns, got {len(parts)}")
            word, count = parts
            value = int(count)
            counts[word] = value
            total += value
    if total == 0:
        raise ValueError(f"{path} has no counts")
    return counts, total


def _lookup(vectors, word: str) -> np.ndarray | None:
    """Return the vector of ``word`` or ``None`` when it is out of vocabulary."""
    try:
        return np.asarray(vectors[word], dtype=np.float32)
    except KeyError:
        return None


def _dimension(vectors, dim: int | None) -> int:
    """Return the embedding dimension, from ``dim`` or the first stored vector."""
    if dim is not None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        return int(dim)
    if getattr(vectors, "vector_size", None):
        return int(vectors.vector_size)
    if isinstance(vectors, dict) and vectors:
        return int(len(next(iter(vectors.values()))))
    raise ValueError("cannot infer the embedding dimension; pass dim explicitly")


def compose_sif(
    token_lists: Sequence[Sequence[str]],
    vectors,
    counts: dict[str, int],
    total: int,
    *,
    dim: int | None = None,
    a: float = DEFAULT_A,
    pc: np.ndarray | None = None,
) -> tuple[np.ndarray, CompositionStats]:
    """Return SIF vectors for ``token_lists`` plus composition statistics.

    ``vectors`` is any mapping from word to vector (a gensim ``KeyedVectors`` or
    a plain ``dict``); ``pc`` is the first principal component to remove, and
    when ``None`` no projection is removed (used while fitting the component).
    """
    if a <= 0:
        raise ValueError("a must be > 0")
    if total <= 0:
        raise ValueError("total must be > 0")

    dim = _dimension(vectors, dim)
    matrix = np.zeros((len(token_lists), dim), dtype=np.float32)
    n_oov = 0
    n_words = 0
    n_empty = 0

    # Weight cache: only the words actually seen are looked up, so a 800k-entry
    # count table is never turned into a weight dict in full.
    weights: dict[str, float] = {}

    for row, tokens in enumerate(token_lists):
        accumulator = np.zeros(dim, dtype=np.float32)
        weight_sum = 0.0
        for token in tokens:
            n_words += 1
            vector = _lookup(vectors, token)
            if vector is None:
                n_oov += 1
                continue
            weight = weights.get(token)
            if weight is None:
                weight = a / (a + counts.get(token, 0) / total) if token in counts else a / (a + 1.0 / total)
                weights[token] = weight
            accumulator += weight * vector
            weight_sum += weight
        if weight_sum == 0.0:
            n_empty += 1
            continue
        matrix[row] = accumulator / weight_sum

    if pc is not None:
        matrix = remove_projection(matrix, pc)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.zeros_like(matrix)
    np.divide(matrix, norms, out=normalized, where=norms > 0)
    return normalized, CompositionStats(len(token_lists), n_oov, n_empty, n_words)


def compose_average(
    token_lists: Sequence[Sequence[str]],
    vectors,
    *,
    dim: int | None = None,
) -> tuple[np.ndarray, CompositionStats]:
    """Return plain unweighted average vectors (the ablation) plus statistics."""
    dim = _dimension(vectors, dim)
    matrix = np.zeros((len(token_lists), dim), dtype=np.float32)
    n_oov = 0
    n_words = 0
    n_empty = 0

    for row, tokens in enumerate(token_lists):
        accumulator = np.zeros(dim, dtype=np.float32)
        seen = 0
        for token in tokens:
            n_words += 1
            vector = _lookup(vectors, token)
            if vector is None:
                n_oov += 1
                continue
            accumulator += vector
            seen += 1
        if seen == 0:
            n_empty += 1
            continue
        matrix[row] = accumulator / seen

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.zeros_like(matrix)
    np.divide(matrix, norms, out=normalized, where=norms > 0)
    return normalized, CompositionStats(len(token_lists), n_oov, n_empty, n_words)


def fit_pc(matrix: np.ndarray) -> np.ndarray:
    """Return the first principal component (unit vector) of an uncentered matrix.

    Computed from the eigen-decomposition of ``X^T X``, which is 300x300, instead
    of an SVD of the full matrix: same answer, far cheaper.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("matrix must be 2-D with at least two rows")
    # Drop all-zero rows so empty sentences do not bias the component.
    nonzero = np.any(values != 0.0, axis=1)
    values = values[nonzero]
    if values.shape[0] < 2:
        raise ValueError("matrix has fewer than two non-zero rows")
    eigenvalues, eigenvectors = np.linalg.eigh(values.T @ values)
    component = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = np.linalg.norm(component)
    if norm == 0:
        raise ValueError("degenerate principal component")
    return (component / norm).astype(np.float32)


def remove_projection(matrix: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """Remove the projection of every row on ``pc``."""
    values = np.asarray(matrix, dtype=np.float32)
    component = np.asarray(pc, dtype=np.float32)
    if component.ndim != 1 or component.shape[0] != values.shape[1]:
        raise ValueError("pc must be a 1-D vector matching the matrix width")
    return values - np.outer(values @ component, component)


_SENTENCE_SPLIT = re.compile(r"[\n]+")


def sample_fit_paragraphs(
    para_path: Path | str,
    *,
    shard: str = "fit",
    limit: int = 20_000,
    seed: int = 42,
) -> list[str]:
    """Sample up to ``limit`` whole paragraphs from ``shard`` of ``wiki.<lang>.para.tsv``.

    Paragraph-level composition (T3) needs its own principal component: a
    paragraph is not a sentence, so the component fitted on sentences would
    remove the wrong direction.  The sample is a deterministic seeded shuffle of
    the eligible paragraphs, never evaluation data.
    """
    rng = np.random.default_rng(seed)
    paragraphs: list[str] = []
    with Path(para_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t", 4)
            if len(fields) != 5 or fields[0] != shard:
                continue
            text = fields[4].strip()
            if text:
                paragraphs.append(normalize(text))
            if len(paragraphs) >= limit * 4:
                break
    if not paragraphs:
        raise ValueError(f"no {shard}-shard paragraphs found in {para_path}")
    order = rng.permutation(len(paragraphs))[:limit]
    return [paragraphs[index] for index in order]


def sample_fit_sentences(
    para_path: Path | str,
    lang: str,
    *,
    shard: str = "fit",
    limit: int = 100_000,
    seed: int = 42,
) -> list[str]:
    """Sample up to ``limit`` sentences from the ``shard`` of ``wiki.<lang>.para.tsv``.

    The paragraph table has columns ``shard, article_id, title, para_idx, text``.
    Sentences are produced with :func:`src.textproc.sentence_split`, and the
    selection is a deterministic seeded shuffle of the candidate sentences so
    the principal component does not depend on article order.
    """
    rng = np.random.default_rng(seed)
    sentences: list[str] = []
    with Path(para_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] != shard:
                continue
            sentences.extend(sentence_split(parts[4], lang))
            if len(sentences) >= limit * 4:
                break
    if not sentences:
        raise ValueError(f"no {shard}-shard sentences found in {para_path}")
    order = rng.permutation(len(sentences))[:limit]
    return [sentences[index] for index in order]


def tokenize_all(texts: Iterable[str], lang: str) -> list[list[str]]:
    """Tokenize every text with the shared tokenizer."""
    return [tokenize(text, lang) for text in texts]
