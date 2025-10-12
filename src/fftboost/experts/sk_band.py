from __future__ import annotations

from typing import cast

import numpy as np

from .types import ExpertContext
from .types import Proposal


def _excess_kurtosis(x: np.ndarray) -> float:
    # Population-style excess kurtosis (standardized 4th moment - 3.0)
    mu = float(np.mean(x))
    xc = x - mu
    v2 = float(np.mean(xc * xc))
    if v2 == 0.0:
        return 0.0
    v4 = float(np.mean(xc**4))
    return v4 / (v2 * v2) - 3.0


def propose(
    residual: np.ndarray,
    ctx: ExpertContext,
    *,
    bands: list[tuple[float, float]],
    top_m: int = 3,
    boost_kurtosis: bool = True,
    kurtosis_weight: float = 0.1,
) -> Proposal:
    psd = ctx.psd  # (n_windows, n_bins)
    freqs = ctx.freqs  # (n_bins,)
    n_windows, _ = psd.shape

    # Build band energy matrix H_band: (n_windows, n_bands)
    band_cols: list[np.ndarray] = []
    descriptors: list[dict[str, object]] = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            col = np.zeros(n_windows, dtype=np.float64)
        else:
            col = cast(np.ndarray, np.sum(psd[:, mask], axis=1))
        band_cols.append(col)
        descriptors.append(
            {"type": "sk_band", "low_hz": float(lo), "high_hz": float(hi)}
        )

    if len(band_cols) == 0:
        H = np.empty((n_windows, 0), dtype=np.float64)
        mu = np.empty((0,), dtype=np.float64)
        sigma = np.empty((0,), dtype=np.float64)
        return Proposal(H=H, descriptors=[], mu=mu, sigma=sigma)

    H = np.stack(band_cols, axis=1)

    # Base score via z-scored dot product
    res_mu = residual.mean()
    res_std = residual.std()
    H_mu = H.mean(axis=0)
    H_std = H.std(axis=0)
    res_z = (residual - res_mu) / (res_std + 1e-12)
    H_z = (H - H_mu) / (H_std + 1e-12)
    corrs = np.abs(res_z @ H_z) / float(n_windows)

    # Optional spectral-kurtosis-based boost
    if boost_kurtosis and kurtosis_weight != 0.0:
        boosts = np.array(
            [max(_excess_kurtosis(H[:, j]), 0.0) for j in range(H.shape[1])],
            dtype=np.float64,
        )
        scores = corrs + kurtosis_weight * boosts
    else:
        scores = corrs

    m = min(top_m, H.shape[1])
    order = np.argpartition(scores, -m)[-m:]
    order = order[np.argsort(scores[order])[::-1]]
    H_sel = H[:, order]
    mu = H_mu[order]
    sigma = H_std[order]

    desc_sel = [descriptors[i] for i in order]
    return Proposal(H=H_sel, descriptors=desc_sel, mu=mu, sigma=sigma)
