"""Chunked cosine and CSLS similarity search.

The T1 candidate pool is about 200,000 words per language, so a full
query-by-candidate score matrix would be hundreds of gigabytes.  Every search
here works on blocks: at most ``chunk x col_block`` float32 scores are held in
memory at once, and the running top-*k* is merged block by block.

CSLS (Conneau et al., 2018) is

    CSLS(x, y) = 2 cos(x, y) - r_T(x) - r_S(y),

where ``r_T(x)`` is the mean cosine of query ``x`` to its ``k`` nearest
neighbours in the candidate set and ``r_S(y)`` the mean cosine of candidate
``y`` to its ``k`` nearest neighbours in the source matrix ``S``.  ``r_T(x)`` is
constant within a row, so it never changes that query's ranking; it is applied
at the end so the returned scores are true CSLS scores.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_CHUNK = 1024
DEFAULT_CSLS_K = 10

_MAX_BLOCK_ELEMENTS = 16_000_000
_TORCH_MIN_ELEMENTS = 50_000_000


def _as_matrix(values: ArrayLike, name: str) -> NDArray[np.float32]:
    """Validate a 2-D numeric input and return it as a float32 matrix."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got {matrix.ndim} dimensions")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ``matrix`` with every row scaled to unit length (zero rows kept)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scaled = np.zeros_like(matrix)
    np.divide(matrix, norms, out=scaled, where=norms > 0)
    return scaled


def _check_k(k: int, n_candidates: int) -> int:
    """Validate ``k`` and clip it to the number of candidates."""
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be >= 1")
    return int(min(k, n_candidates))


def _resolve_backend(backend: str, n_q: int, n_d: int) -> str:
    """Return ``"torch"`` or ``"numpy"`` for the requested backend.

    ``"auto"`` uses torch only when CUDA is available and the score matrix is
    large enough for the transfer to pay off.
    """
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError(f"unsupported backend: {backend!r}")
    if backend == "numpy":
        return "numpy"
    try:
        import torch
    except ImportError:
        if backend == "torch":
            raise RuntimeError("torch backend requested but torch is not installed") from None
        return "numpy"
    if not torch.cuda.is_available():
        if backend == "torch":
            raise RuntimeError("torch backend requested but CUDA is not available")
        return "numpy"
    if backend == "torch" or n_q * n_d >= _TORCH_MIN_ELEMENTS:
        return "torch"
    return "numpy"


def _block_scores(left: NDArray, right: NDArray, backend: str) -> NDArray[np.float32]:
    """Return cosine scores ``left @ right.T`` for two row-normalized blocks."""
    if backend == "torch":
        import torch

        with torch.no_grad():
            device = torch.device("cuda")
            left_t = torch.from_numpy(left).to(device)
            right_t = torch.from_numpy(right).to(device)
            return (left_t @ right_t.T).cpu().numpy()
    return (left @ right.T).astype(np.float32, copy=False)


def _topk_along_rows(block: NDArray[np.float32], k: int) -> tuple[NDArray, NDArray]:
    """Return the top-*k* column indices and scores per row, sorted deterministically.

    Ties are broken by the smaller column index so repeated runs are identical.
    """
    width = min(k, block.shape[1])
    part = np.argpartition(-block, width - 1, axis=1)[:, :width]
    scores = np.take_along_axis(block, part, axis=1)
    order = np.lexsort((part, -scores), axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(scores, order, axis=1)


def _topk_along_columns(block: NDArray[np.float32], k: int) -> tuple[NDArray, NDArray]:
    """Return the top-*k* row indices and scores per column, sorted deterministically."""
    width = min(k, block.shape[0])
    part = np.argpartition(-block, width - 1, axis=0)[:width, :]
    scores = np.take_along_axis(block, part, axis=0)
    order = np.lexsort((part, -scores), axis=0)
    return np.take_along_axis(part, order, axis=0), np.take_along_axis(scores, order, axis=0)


def _merge_topk(
    current_index: NDArray,
    current_score: NDArray,
    new_index: NDArray,
    new_score: NDArray,
    k: int,
) -> tuple[NDArray, NDArray]:
    """Merge a running top-*k* with a new candidate block (both row-wise)."""
    if current_index.size == 0:
        return new_index[:, :k], new_score[:, :k]
    scores = np.concatenate([current_score, new_score], axis=1)
    index = np.concatenate([current_index, new_index], axis=1)
    order = np.lexsort((index, -scores), axis=1)[:, :k]
    return np.take_along_axis(index, order, axis=1), np.take_along_axis(scores, order, axis=1)


def _merge_scores(current: NDArray, new: NDArray, k: int) -> NDArray:
    """Merge a running top-*k* score block with a new one (indices not tracked)."""
    if current.size == 0:
        return new[:, :k]
    merged = np.concatenate([current, new], axis=1)
    return -np.partition(-merged, k - 1, axis=1)[:, :k]


def _row_blocks(size: int, block: int):
    """Yield ``(start, stop)`` row-block slices covering ``range(size)``."""
    for start in range(0, size, block):
        yield start, min(start + block, size)


def _column_block_size(rows: int, block_limit: int = _MAX_BLOCK_ELEMENTS) -> int:
    """Return the largest column block keeping ``rows x columns`` under the budget."""
    return max(1, block_limit // max(1, rows))


def cosine_topk(
    queries: ArrayLike,
    candidates: ArrayLike,
    k: int,
    *,
    chunk: int = DEFAULT_CHUNK,
    backend: str = "auto",
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Return the top-*k* cosine neighbours of each query.

    Args:
        queries: ``(n_queries, dim)`` query vectors.
        candidates: ``(n_candidates, dim)`` candidate vectors.
        k: number of neighbours per query, clipped to ``n_candidates``.
        chunk: maximum number of queries scored at once.
        backend: ``"auto"``, ``"numpy"`` or ``"torch"``.

    Returns:
        ``(indices, scores)``, both ``(n_queries, k)`` and sorted by descending
        score with ties broken by the smaller candidate index.
    """
    query = _as_matrix(queries, "queries")
    candidate = _as_matrix(candidates, "candidates")
    if query.shape[1] != candidate.shape[1]:
        raise ValueError(
            f"dimension mismatch: queries {query.shape[1]} vs candidates {candidate.shape[1]}"
        )
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    k = _check_k(k, candidate.shape[0])
    query = _l2_normalize(query)
    candidate = _l2_normalize(candidate)
    n_rows = query.shape[0]
    resolved = _resolve_backend(backend, n_rows, candidate.shape[0])

    indices = np.empty((n_rows, k), dtype=np.int64)
    scores = np.empty((n_rows, k), dtype=np.float32)
    column_block = _column_block_size(min(chunk, n_rows))

    for row_start, row_stop in _row_blocks(n_rows, chunk):
        query_block = query[row_start:row_stop]
        best_index = np.empty((row_stop - row_start, 0), dtype=np.int64)
        best_score = np.empty((row_stop - row_start, 0), dtype=np.float32)
        for col_start, col_stop in _row_blocks(candidate.shape[0], column_block):
            block = _block_scores(query_block, candidate[col_start:col_stop], resolved)
            block_index, block_score = _topk_along_rows(block, k)
            best_index, best_score = _merge_topk(
                best_index, best_score, block_index + col_start, block_score, k
            )
        indices[row_start:row_stop] = best_index
        scores[row_start:row_stop] = best_score
    return indices, scores


def hubness_mean(
    source: ArrayLike,
    candidates: ArrayLike,
    csls_k: int = DEFAULT_CSLS_K,
    *,
    chunk: int = DEFAULT_CHUNK,
    backend: str = "auto",
) -> NDArray[np.float32]:
    """Return ``r_S(y)``: mean cosine of each candidate to its ``csls_k`` nearest sources.

    Computed once per candidate pool, block by block: the source matrix is
    streamed and only a running top-``csls_k`` per candidate is kept, so no
    ``n_source x n_candidates`` matrix is ever materialized.
    """
    source_matrix = _l2_normalize(_as_matrix(source, "source"))
    candidate_matrix = _l2_normalize(_as_matrix(candidates, "candidates"))
    if source_matrix.shape[1] != candidate_matrix.shape[1]:
        raise ValueError("source and candidates must have the same dimension")

    k = _check_k(csls_k, source_matrix.shape[0])
    n_candidates = candidate_matrix.shape[0]
    resolved = _resolve_backend(backend, source_matrix.shape[0], n_candidates)

    result = np.empty(n_candidates, dtype=np.float32)
    column_block = _column_block_size(min(chunk, n_candidates))
    row_block = max(1, _MAX_BLOCK_ELEMENTS // max(1, column_block))

    for col_start, col_stop in _row_blocks(n_candidates, column_block):
        candidate_block = candidate_matrix[col_start:col_stop]
        best_index = np.empty((0, col_stop - col_start), dtype=np.int64)
        best_score = np.empty((0, col_stop - col_start), dtype=np.float32)
        for row_start, row_stop in _row_blocks(source_matrix.shape[0], row_block):
            block = _block_scores(source_matrix[row_start:row_stop], candidate_block, resolved)
            block_index, block_score = _topk_along_columns(block, k)
            if best_index.size == 0:
                best_index, best_score = block_index, block_score
            else:
                merged_score = np.concatenate([best_score, block_score], axis=0)
                merged_index = np.concatenate([best_index, block_index], axis=0)
                order = np.lexsort((merged_index, -merged_score), axis=0)[:k, :]
                best_index = np.take_along_axis(merged_index, order, axis=0)
                best_score = np.take_along_axis(merged_score, order, axis=0)
        result[col_start:col_stop] = best_score.mean(axis=0)
    return result


def csls_topk(
    queries: ArrayLike,
    candidates: ArrayLike,
    k: int,
    source: ArrayLike,
    *,
    csls_k: int = DEFAULT_CSLS_K,
    chunk: int = DEFAULT_CHUNK,
    backend: str = "auto",
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Return the top-*k* CSLS neighbours of each query.

    ``source`` is the matrix used for the candidate-side hubness term ``r_S(y)``
    (for T1 it is the mapped source candidate list).  Query vectors and
    candidates must already be in the same space.

    Returns ``(indices, scores)`` exactly like :func:`cosine_topk`, with scores
    being ``CSLS(x, y)`` including the per-query ``r_T(x)`` term.
    """
    query = _l2_normalize(_as_matrix(queries, "queries"))
    candidate = _l2_normalize(_as_matrix(candidates, "candidates"))
    if query.shape[1] != candidate.shape[1]:
        raise ValueError("queries and candidates must have the same dimension")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    k = _check_k(k, candidate.shape[0])
    neighbour_k = _check_k(csls_k, candidate.shape[0])
    r_source = hubness_mean(source, candidate, neighbour_k, chunk=chunk, backend=backend)

    n_rows = query.shape[0]
    resolved = _resolve_backend(backend, n_rows, candidate.shape[0])
    raw_width = max(k, neighbour_k)

    indices = np.empty((n_rows, k), dtype=np.int64)
    scores = np.empty((n_rows, k), dtype=np.float32)
    column_block = _column_block_size(min(chunk, n_rows))

    for row_start, row_stop in _row_blocks(n_rows, chunk):
        query_block = query[row_start:row_stop]
        rows = row_stop - row_start
        adj_index = np.empty((rows, 0), dtype=np.int64)
        adj_score = np.empty((rows, 0), dtype=np.float32)
        raw_score = np.empty((rows, 0), dtype=np.float32)

        for col_start, col_stop in _row_blocks(candidate.shape[0], column_block):
            block = _block_scores(query_block, candidate[col_start:col_stop], resolved)

            block_index, block_score = _topk_along_rows(2.0 * block - r_source[col_start:col_stop], k)
            adj_index, adj_score = _merge_topk(
                adj_index, adj_score, block_index + col_start, block_score, k
            )

            block_score = _topk_along_rows(block, raw_width)[1]
            raw_score = _merge_scores(raw_score, block_score, raw_width)

        r_query = raw_score[:, :neighbour_k].mean(axis=1)
        indices[row_start:row_stop] = adj_index
        scores[row_start:row_stop] = adj_score - r_query[:, None]
    return indices, scores


def rank_of_gold(scores: ArrayLike, gold_indices: ArrayLike) -> int:
    """Return the 1-based rank of the best gold candidate in a full score vector.

    ``scores`` holds one score per candidate (1-D).  Candidates with a score
    strictly greater than the best gold score rank above it; ties share the
    rank, so an exact match ranks 1.  Raises ``ValueError`` when no gold index
    is a valid candidate index.
    """
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    gold = np.atleast_1d(np.asarray(gold_indices))
    if gold.size == 0:
        raise ValueError("gold_indices must not be empty")
    if gold.dtype.kind not in "iu":
        raise TypeError("gold_indices must contain integers")
    if np.any(gold < 0) or np.any(gold >= values.size):
        raise ValueError(f"gold index out of range for {values.size} candidates")
    best_gold = float(values[gold].max())
    return int(np.count_nonzero(values > best_gold)) + 1
