"""Tests for ranking metrics and bootstrap intervals (S06).

Toy rankings have hand-computed answers; bootstrap tests check determinism,
that the interval brackets the mean, and that paired resampling preserves the
pairing.
"""

import numpy as np
import pytest

from src.metrics import (
    acc_at_k,
    bootstrap_ci,
    mrr,
    paired_bootstrap_diff,
    precision_at_k,
    recall_at_k,
)

TOY_RANKS = np.array([1, 1, 2, 3, 4, 10])


def test_acc_at_k_hand_computed():
    assert acc_at_k(TOY_RANKS, 1) == pytest.approx(2 / 6)
    assert acc_at_k(TOY_RANKS, 3) == pytest.approx(4 / 6)
    assert acc_at_k(TOY_RANKS, 5) == pytest.approx(5 / 6)
    assert acc_at_k(TOY_RANKS, 10) == pytest.approx(1.0)
    assert acc_at_k([1, 2], 1) == pytest.approx(0.5)


def test_recall_at_k_equals_hit_at_k_for_best_gold_ranks():
    for k in (1, 3, 5, 10):
        assert recall_at_k(TOY_RANKS, k) == pytest.approx(acc_at_k(TOY_RANKS, k))


def test_precision_at_k_hand_computed():
    # Three queries with 1, 0 and 2 gold items in their top-3.
    assert precision_at_k([1, 0, 2], 3) == pytest.approx(1 / 3)
    assert precision_at_k([2, 2], 2) == pytest.approx(1.0)
    assert precision_at_k([0, 0], 5) == pytest.approx(0.0)


def test_mrr_hand_computed():
    expected = (1 + 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 10) / 6
    assert mrr(TOY_RANKS) == pytest.approx(expected)


@pytest.mark.parametrize("bad_ranks", [np.array([]), [0, 1], [1.5, 2], [np.nan, 1], [[1, 2]]])
def test_rank_validation(bad_ranks):
    with pytest.raises((ValueError, TypeError)):
        acc_at_k(bad_ranks, 5)


def test_k_validation():
    with pytest.raises(ValueError):
        acc_at_k(TOY_RANKS, 0)


def test_bootstrap_ci_deterministic_and_brackets_mean():
    scores = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    first = bootstrap_ci(scores, n=500, seed=42)
    second = bootstrap_ci(scores, n=500, seed=42)
    assert first == second

    mean, low, high = first
    assert mean == pytest.approx(0.6)
    assert low <= mean <= high
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_diff_of_identical_arrays_is_exactly_zero():
    a = np.array([0.0, 1.0, 0.5, 1.0, 0.0])
    mean_diff, low, high, p_value = paired_bootstrap_diff(a, a.copy(), n=200, seed=42)
    assert mean_diff == 0.0
    assert low == 0.0 and high == 0.0
    assert p_value == 1.0


def test_paired_bootstrap_diff_uses_joint_resampling():
    a = np.array([2.0, 1.0, 3.0, 2.0])
    b = np.array([1.0, 0.0, 2.0, 1.0])
    mean_diff, low, high, p_value = paired_bootstrap_diff(a, b, n=200, seed=42)
    assert mean_diff == pytest.approx(1.0)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)
    assert p_value == pytest.approx(0.0)


def test_paired_bootstrap_diff_brackets_mean_and_requires_equal_lengths():
    a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    mean_diff, low, high, p_value = paired_bootstrap_diff(a, b, n=500, seed=42)
    assert low <= mean_diff <= high
    assert 0.0 <= p_value <= 1.0

    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 2.0], [1.0])
