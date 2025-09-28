from typing import Optional

import numpy as np


def zscale(
    a: np.ndarray, mu: Optional[np.ndarray] = None, sd: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mu is None:
        mu = a.mean(axis=0)
    if sd is None:
        sd = a.std(axis=0) + 1e-9
    return (a - mu) / sd, mu, sd


def robust_scale(
    y: np.ndarray, ref: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float, float]:
    if ref is None:
        ref = y
    median = np.median(ref)
    iqr = np.percentile(ref, 75) - np.percentile(ref, 25) + 1e-9
    return (y - median) / iqr, median, iqr
