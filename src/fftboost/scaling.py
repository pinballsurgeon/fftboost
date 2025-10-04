from __future__ import annotations

import numpy as np
import numpy.typing as npt


def zscale(
    x: npt.NDArray, mu: float | None = None, sd: float | None = None
) -> tuple[npt.NDArray, float, float]:
    if mu is None:
        mu = float(np.nanmean(x))
    if sd is None:
        sd = float(np.nanstd(x))
    if sd == 0:
        return x - mu, mu, sd
    return (x - mu) / sd, mu, sd


def robust_scale(
    x: npt.NDArray, ref: npt.NDArray | None = None
) -> tuple[npt.NDArray, float, float]:
    source = ref if ref is not None else x
    med = float(np.nanmedian(source))
    iqr = float(np.nanquantile(source, 0.75) - np.nanquantile(source, 0.25))
    if iqr == 0:
        return x - med, med, iqr
    return (x - med) / iqr, med, iqr
