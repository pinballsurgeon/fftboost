from __future__ import annotations

from collections.abc import Generator

import numpy as np


def blocked_cv(
    n_windows: int, n_splits: int, gap: int = 0
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield (train_indices, validation_indices) for leak-safe blocked CV.

    - Contiguous validation blocks per fold.
    - Deterministic remainder handling: the first (n_windows % n_splits) folds
      receive one extra window.
    - gap excludes indices within `gap` of the validation block on both sides.
    - Pure NumPy implementation.
    """
    base = n_windows // n_splits
    rem = n_windows % n_splits

    start = 0
    for k in range(n_splits):
        val_size = base + (1 if k < rem else 0)
        val_start = start
        val_end = val_start + val_size - 1

        if val_size == 0:
            val_idx = np.empty(0, dtype=np.int64)
        else:
            val_idx = np.arange(val_start, val_end + 1, dtype=np.int64)

        # Train indices: everything outside [val_start-gap, val_end+gap]
        left_end = max(-1, val_start - gap - 1)
        right_start = min(n_windows, val_end + gap + 1)

        left = (
            np.arange(0, left_end + 1, dtype=np.int64)
            if left_end >= 0
            else np.empty(0, dtype=np.int64)
        )
        right = (
            np.arange(right_start, n_windows, dtype=np.int64)
            if right_start < n_windows
            else np.empty(0, dtype=np.int64)
        )

        if left.size == 0:
            train_idx = right
        elif right.size == 0:
            train_idx = left
        else:
            train_idx = np.concatenate((left, right))

        yield train_idx, val_idx

        start = val_end + 1
