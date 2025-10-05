from __future__ import annotations

from typing import Any

import numpy as np


def zscale(
    x: np.ndarray[Any, Any], mu: float | None = None, sd: float | None = None
) -> tuple[np.ndarray[Any, Any], float, float]:
    if mu is None:
        mu = float(np.nanmean(x))
    if sd is None:
        sd = float(np.nanstd(x))
    if sd == 0:
        return x - mu, mu, sd
    return (x - mu) / sd, mu, sd


def robust_scale(
    x: np.ndarray[Any, Any], ref: np.ndarray[Any, Any] | None = None
) -> tuple[np.ndarray[Any, Any], float, float]:
    source = ref if ref is not None else x
    med = float(np.nanmedian(source))
    iqr = float(np.nanquantile(source, 0.75) - np.nanquantile(source, 0.25))
    if iqr == 0:
        return x - med, med, iqr
    return (x - med) / iqr, med, iqr
