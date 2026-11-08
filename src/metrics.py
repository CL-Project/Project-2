"""Ranking metrics and bootstrap confidence intervals.

Per-item inputs are 1-based ranks of the best gold item, as stored in
``results/per_item/*.csv`` (column ``rank_of_best_gold``).  A rank of 1 means the
best gold item was retrieved first.  Bootstrap defaults (1,000 resamples, 95%
percentile interval, seed 42) come from ``configs/tasks.yaml``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 42


def _as_ranks(ranks: ArrayLike) -> NDArray[np.int64]:
    """Validate and convert 1-based ranks to a 1-D int64 array."""
    values = np.asarray(ranks)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranks must be a non-empty 1-D array")
    if values.dtype.kind not in "iuf":
        raise TypeError("ranks must be numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError("ranks must be finite")
    if np.any(values < 1):
        raise ValueError("ranks are 1-based and must be >= 1")
    if np.any(values != np.floor(values)):
        raise ValueError("ranks must be integers")
    return values.astype(np.int64)


def _check_k(k: int) -> int:
    """Validate a cut-off *k* and return it as a Python int."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return int(k)


def _check_bootstrap(n: int, alpha: float) -> None:
    """Validate bootstrap sample count and significance level."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _bootstrap_means(values: NDArray[np.float64], n: int, rng: np.random.Generator) -> NDArray:
    """Return ``n`` bootstrap means of ``values`` (resampling item indices)."""
    indices = rng.integers(0, values.size, size=(n, values.size))
    return values[indices].mean(axis=1)


def _percentile_interval(samples: NDArray, alpha: float) -> tuple[float, float]:
    """Return the percentile bootstrap interval of ``samples``."""
    low, high = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def acc_at_k(ranks: ArrayLike, k: int) -> float:
    """Fraction of queries whose best gold item is ranked at or above *k*.

    With one gold item per query this is also hit@k, which is the MUSE-style
    ``P@k`` used for T1.
    """
    values = _as_ranks(ranks)
    return float(np.mean(values <= _check_k(k)))


def recall_at_k(ranks: ArrayLike, k: int) -> float:
    """Recall@k for the best-gold rank input (equals :func:`acc_at_k`).

    The per-item files store only the rank of the best gold item, so the number
    of retrieved gold items is unknown; with a single gold item recall@k is
    exactly whether that item appears in the top *k*.
    """
    return acc_at_k(ranks, k)


def precision_at_k(gold_hits: ArrayLike, k: int) -> float:
    """Mean fraction of the top *k* results that are gold, for multi-gold tasks.

    ``gold_hits`` holds, per query, the number of gold items retrieved in the
    top *k* (0 to *k*).  For T1, where the plan defines ``P@k`` as *any* gold in
    the top *k* (MUSE convention), use :func:`acc_at_k` instead.
    """
    values = np.asarray(gold_hits, dtype=float)
    k = _check_k(k)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("gold_hits must be a non-empty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("gold_hits must be finite")
    if np.any(values < 0) or np.any(values > k):
        raise ValueError("gold_hits must be between 0 and k")
    return float(np.mean(values) / k)


def mrr(ranks: ArrayLike) -> float:
    """Mean reciprocal rank of the best gold item."""
    values = _as_ranks(ranks)
    return float(np.mean(1.0 / values))


def bootstrap_ci(
    per_item: ArrayLike,
    n: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, low, high)`` for a per-item score with a percentile CI.

    Resamples item indices with replacement ``n`` times using ``seed``; the
    interval is deterministic for a fixed seed.
    """
    values = np.asarray(per_item, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("per_item must be a non-empty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("per_item must be finite")
    _check_bootstrap(n, alpha)
    rng = np.random.default_rng(seed)
    samples = _bootstrap_means(values, n, rng)
    low, high = _percentile_interval(samples, alpha)
    return float(values.mean()), low, high


def paired_bootstrap_diff(
    a: ArrayLike,
    b: ArrayLike,
    n: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float, float]:
    """Return ``(mean_diff, low, high, p_two_sided_approx)`` for paired scores.

    ``mean_diff`` is ``mean(a) - mean(b)``.  ``a`` and ``b`` must be aligned per
    item (same query order); the bootstrap resamples item indices jointly so
    the pairing is preserved.  The p-value is the usual bootstrap two-sided
    approximation ``2 * min(P(diff <= 0), P(diff >= 0))`` and is approximate, not
    a substitute for a named test.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays")
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape (paired items)")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("a and b must be finite")
    _check_bootstrap(n, alpha)

    diff = left - right
    rng = np.random.default_rng(seed)
    samples = _bootstrap_means(diff, n, rng)
    low, high = _percentile_interval(samples, alpha)
    p_value = min(1.0, 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0))))
    return float(diff.mean()), low, high, p_value
