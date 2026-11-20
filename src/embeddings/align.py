"""Procrustes alignment of two embedding spaces from a seed dictionary (S21).

The mapping is orthogonal and learned from translation pairs: ``X`` holds source
(Hindi) embeddings of the seed words, ``Y`` the target (English) embeddings of
their translations, and the solution of

    min_W || X W - Y ||_F  subject to  W^T W = I

is ``W = U V^T`` where ``X^T Y = U S V^T`` (Schönemann, 1966).  Vectors are
L2-normalized before fitting and again after mapping (``AGENTS.md`` §10.6).

One mapping is fitted **per direction** (``W_hi_en`` from ``hi-en`` seed pairs,
``W_en_hi`` from ``en-hi``), because the two dictionaries are different and
using one for both would leak the test direction into its own mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger("align")


def l2_normalize(matrix: ArrayLike) -> NDArray[np.float32]:
    """Return ``matrix`` as float32 with every row scaled to unit length.

    Zero rows stay zero rather than producing NaNs.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {values.ndim} dimensions")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    np.divide(values, norms, out=normalized, where=norms > 0)
    return normalized


def procrustes(source: ArrayLike, target: ArrayLike) -> NDArray[np.float32]:
    """Return the orthogonal ``W`` mapping ``source`` into the ``target`` space.

    Args:
        source: ``(n, d)`` embeddings of the seed words in the source language.
        target: ``(n, d)`` embeddings of the same seed words in the target language.

    Returns:
        ``(d, d)`` orthogonal matrix with ``source @ W`` closest to ``target``.
    """
    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("source and target must be 2-D matrices")
    if left.shape != right.shape:
        raise ValueError(f"source {left.shape} and target {right.shape} must have the same shape")
    if left.shape[0] < 1:
        raise ValueError("at least one translation pair is required")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("embeddings must be finite")

    left = l2_normalize(left)
    right = l2_normalize(right)
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    return (u @ vt).astype(np.float32)


def apply_mapping(vectors: ArrayLike, mapping: ArrayLike) -> NDArray[np.float32]:
    """Map ``vectors`` with ``mapping`` and re-normalize the result.

    The mapping is applied first and normalization second, so the output can be
    compared with cosine similarity directly (``AGENTS.md`` §10.6).
    """
    values = np.asarray(vectors, dtype=np.float32)
    matrix = np.asarray(mapping, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("vectors must be a 2-D matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mapping must be a square matrix")
    if values.shape[1] != matrix.shape[0]:
        raise ValueError(
            f"vectors have dimension {values.shape[1]} but the mapping expects {matrix.shape[0]}"
        )
    return l2_normalize(values @ matrix)


def is_orthogonal(mapping: ArrayLike, tol: float = 1e-4) -> bool:
    """Return whether ``mapping`` is orthogonal within ``tol``."""
    matrix = np.asarray(mapping, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity = matrix.T @ matrix
    return bool(np.allclose(identity, np.eye(matrix.shape[0]), atol=tol))


def read_pairs(path: Path | str) -> list[tuple[str, str]]:
    """Read a MUSE dictionary: one ``source<TAB>target`` pair per line."""
    pairs: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected two tab-separated fields, got {len(parts)}")
            source, target = (part.strip() for part in parts)
            if source and target:
                pairs.append((source, target))
    if not pairs:
        raise ValueError(f"{path} contains no translation pairs")
    return pairs


def fit_mapping(
    source_vectors: "MappingLike",
    target_vectors: "MappingLike",
    pairs: Sequence[tuple[str, str]],
) -> tuple[NDArray[np.float32], int]:
    """Fit the mapping from ``pairs``, keeping only pairs whose words are in both vocabularies.

    Args:
        source_vectors: object exposing ``__contains__`` and ``__getitem__``
            (a gensim ``KeyedVectors`` in practice).
        target_vectors: same, for the target language.
        pairs: ``(source_word, target_word)`` pairs.

    Returns:
        ``(mapping, n_used)``; ``n_used`` is the number of pairs actually used.

    Raises:
        ValueError: when fewer than two pairs survive, because an orthogonal
            mapping is under-determined from a single pair.
    """
    kept: list[tuple[str, str]] = []
    missing_source = 0
    missing_target = 0
    for source_word, target_word in pairs:
        in_source = source_word in source_vectors
        in_target = target_word in target_vectors
        if in_source and in_target:
            kept.append((source_word, target_word))
        else:
            missing_source += int(not in_source)
            missing_target += int(not in_target)

    LOGGER.info(
        "seed pairs: %d total, %d used, %d source OOV, %d target OOV",
        len(pairs),
        len(kept),
        missing_source,
        missing_target,
    )
    if len(kept) < 2:
        raise ValueError(f"only {len(kept)} usable seed pairs; cannot fit an orthogonal mapping")

    source_matrix = np.vstack([source_vectors[word] for word, _ in kept])
    target_matrix = np.vstack([target_vectors[word] for _, word in kept])
    mapping = procrustes(source_matrix, target_matrix)
    return mapping, len(kept)


class MappingLike:
    """Structural type for the vector lookups :func:`fit_mapping` needs."""

    def __contains__(self, key: str) -> bool:  # pragma: no cover - typing aid
        raise NotImplementedError

    def __getitem__(self, key: str) -> ArrayLike:  # pragma: no cover - typing aid
        raise NotImplementedError


def seed_accuracy(
    source_vectors: "MappingLike",
    target_vectors: "MappingLike",
    pairs: Iterable[tuple[str, str]],
    mapping: ArrayLike,
    *,
    top_k: int = 1,
) -> float:
    """Return the top-*k* accuracy of ``pairs`` after mapping.

    Accuracy on the seed pairs themselves should be far above test accuracy; a
    near-zero value means the mapping is applied in the wrong direction
    (``AGENTS.md`` §10.6).
    """
    used = [(source, target) for source, target in pairs if source in source_vectors and target in target_vectors]
    if not used:
        raise ValueError("no usable pairs")
    sources = l2_normalize(np.vstack([source_vectors[word] for word, _ in used]) @ np.asarray(mapping))
    targets = l2_normalize(np.vstack([target_vectors[word] for _, word in used]))

    scores = sources @ targets.T
    if top_k == 1:
        predicted = np.argmax(scores, axis=1)
        correct = predicted == np.arange(len(used))
    else:
        top = np.argsort(-scores, axis=1)[:, :top_k]
        correct = np.any(top == np.arange(len(used))[:, None], axis=1)
    return float(np.mean(correct))
