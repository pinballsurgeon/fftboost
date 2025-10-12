from __future__ import annotations

from typing import Any
from typing import cast

import numpy as np

from .types import ExpertContext
from .types import Proposal


def _vectorized_excess_kurtosis(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=0, keepdims=True)
    m4 = (xc**4).mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = np.nan_to_num(m4 / (var**2 + 1e-12)) - 3.0
    return cast(np.ndarray[Any, Any], kurt.squeeze())


def propose(
    residual: np.ndarray[Any, Any],
    ctx: ExpertContext,
    *,
    n_select: int = 1,
    kurtosis_boost: float = 0.0,
) -> Proposal:
    psd = ctx.psd
    freqs = ctx.freqs
    n_windows, _ = psd.shape

    if ctx.band_edges_hz is None or ctx.band_edges_hz.size < 2:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.array([]),
            sigma=np.array([]),
        )

    edges = ctx.band_edges_hz
    band_cols: list[np.ndarray[Any, Any]] = []
    descriptors: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        band_cols.append(
            psd[:, mask].sum(axis=1) if mask.any() else np.zeros(n_windows)
        )
        descriptors.append({"type": "sk_band", "band_hz": (float(lo), float(hi))})

    H = np.stack(band_cols, axis=1)
    if H.shape[1] == 0:
        return Proposal(H=H, descriptors=[], mu=np.array([]), sigma=np.array([]))

    res_z = (residual - residual.mean()) / (residual.std() + 1e-12)
    H_mu, H_std = H.mean(axis=0), H.std(axis=0)
    H_z = (H - H_mu) / (H_std + 1e-12)

    scores = np.abs(res_z @ H_z) / n_windows
    if kurtosis_boost > 0.0:
        boosts = np.maximum(0.0, _vectorized_excess_kurtosis(H))
        scores += kurtosis_boost * boosts

    m = min(n_select, H.shape[1])
    top_indices = np.argpartition(scores, -m)[-m:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return Proposal(
        H=H[:, top_indices],
        descriptors=[descriptors[i] for i in top_indices],
        mu=H_mu[top_indices],
        sigma=H_std[top_indices],
    )
