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
    bands: list[tuple[float, float]] | None = None,
    top_m: int | None = None,
    boost_kurtosis: bool | None = None,
    kurtosis_weight: float | None = None,
    n_select: int | None = None,
    kurtosis_boost: float | None = None,
) -> Proposal:
    psd = ctx.psd  # (n_windows, n_bins)
    freqs = ctx.freqs  # (n_bins,)
    n_windows, _ = psd.shape

    # Build band energy matrix H_band: (n_windows, n_bands)
    band_cols: list[np.ndarray] = []
    descriptors: list[dict[str, object]] = []
    # Determine band definitions
    band_list: list[tuple[float, float]]
    if bands is not None:
        band_list = list(bands)
    elif ctx.band_edges_hz is not None and ctx.band_edges_hz.size >= 2:
        edges: np.ndarray = ctx.band_edges_hz.astype(np.float64)
        band_list = [
            (float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)
        ]
    else:
        band_list = []

    for lo, hi in band_list:
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            col = np.zeros(n_windows, dtype=np.float64)
        else:
            col = cast(np.ndarray, np.sum(psd[:, mask], axis=1))
        band_cols.append(col)
        descriptors.append(
            {
                "type": "sk_band",
                "low_hz": float(lo),
                "high_hz": float(hi),
                "band_hz": (float(lo), float(hi)),
            }
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
    use_boost = (boost_kurtosis if boost_kurtosis is not None else True) and (
        (kurtosis_weight if kurtosis_weight is not None else (kurtosis_boost or 0.0))
        != 0.0
    )
    weight = kurtosis_weight if kurtosis_weight is not None else (kurtosis_boost or 0.0)
    if use_boost:
        boosts = np.array(
            [max(_excess_kurtosis(H[:, j]), 0.0) for j in range(H.shape[1])],
            dtype=np.float64,
        )
        scores = corrs + float(weight) * boosts
    else:
        scores = corrs

    m_param = top_m if top_m is not None else (n_select if n_select is not None else 3)
    m = min(int(m_param), H.shape[1])
    order = np.argpartition(scores, -m)[-m:]
    order = order[np.argsort(scores[order])[::-1]]
    H_sel = H[:, order]
    mu = H_mu[order]
    sigma = H_std[order]

    desc_sel = [descriptors[i] for i in order]
    return Proposal(H=H_sel, descriptors=desc_sel, mu=mu, sigma=sigma)
