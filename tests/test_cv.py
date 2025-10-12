from __future__ import annotations

import numpy as np

from src.fftboost.cv import blocked_cv


def _collect_folds(n: int, k: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(tr, va) for tr, va in blocked_cv(n, k, gap)]


def test_gate1_leak_proof_min_distance() -> None:
    n, k, gap = 50, 5, 2
    folds = _collect_folds(n, k, gap)
    for train_idx, val_idx in folds:
        if val_idx.size == 0:
            continue
        a = int(val_idx.min())
        b = int(val_idx.max())
        if train_idx.size > 0:
            # No train indices within the forbidden interval [a-gap, b+gap]
            forbidden_lo = a - gap
            forbidden_hi = b + gap
            assert not np.any((train_idx >= forbidden_lo) & (train_idx <= forbidden_hi))

            # Boundary form per side (if that side exists)
            left = train_idx[train_idx < a]
            right = train_idx[train_idx > b]
            if left.size > 0:
                assert int(left.max()) <= a - gap - 1
            if right.size > 0:
                assert int(right.min()) >= b + gap + 1


def test_gate2_full_coverage_exact_once() -> None:
    n, k, gap = 37, 6, 3
    seen = np.zeros(n, dtype=np.int64)
    for _, val_idx in blocked_cv(n, k, gap):
        seen[val_idx] += 1
    # Every index appears exactly once in validation
    assert np.all(seen == 1)


def test_gate3_not_divisible_sizes() -> None:
    n, k = 23, 5
    gap = 1
    vals = [va.size for _, va in blocked_cv(n, k, gap)]
    # Remainder 23 % 5 = 3 -> first 3 folds get +1
    assert vals == [5, 5, 5, 4, 4]


def test_gate3_gap_zero_behavior() -> None:
    n, k, gap = 20, 4, 0
    for train_idx, val_idx in blocked_cv(n, k, gap):
        # Train is the complement of val
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        expected_train = np.nonzero(mask)[0]
        np.testing.assert_array_equal(train_idx, expected_train)


def test_gate3_large_gap_allows_empty_train() -> None:
    n, k, gap = 10, 2, 10
    folds = _collect_folds(n, k, gap)
    # Each val size is 5; with gap 10, training should be empty
    for train_idx, val_idx in folds:
        assert val_idx.size == 5
        assert train_idx.size == 0


def test_gate3_min_splits_two() -> None:
    n, k, gap = 9, 2, 1
    folds = _collect_folds(n, k, gap)
    # Fold sizes (val) should be [5,4]
    val_sizes = [va.size for _, va in folds]
    assert val_sizes == [5, 4]
    # No leakage across the gap
    for train_idx, val_idx in folds:
        if val_idx.size == 0:
            continue
        a = int(val_idx.min())
        b = int(val_idx.max())
        assert not np.any((train_idx >= a - gap) & (train_idx <= b + gap))
