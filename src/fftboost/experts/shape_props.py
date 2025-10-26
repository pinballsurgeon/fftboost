from __future__ import annotations

from typing import Any

import numpy as np

from .types import ExpertContext
from .types import Proposal


def propose(
    residual: np.ndarray[Any, Any], ctx: ExpertContext, *, top_k: int = 1
) -> Proposal:
    """
    Proposes features based on the overall shape of the power spectrum.
    """
    psd = ctx.psd
    freqs = ctx.freqs
    n_windows = psd.shape[0]

    # --- Calculate Shape Features ---
    # Each feature is a time series (one value per window)

    # 1. Spectral Centroid: The "center of mass" of the spectrum.
    # Indicates where the majority of the energy is concentrated.
    spec_sum = psd.sum(axis=1) + 1e-20
    centroid = (psd * freqs).sum(axis=1) / spec_sum

    # 2. Spectral Spread: The standard deviation of the spectrum around the centroid.
    # Indicates the bandwidth of the spectrum.
    spread = np.sqrt(
        np.sum(psd * (freqs - centroid[:, np.newaxis]) ** 2, axis=1) / spec_sum
    )

    # 3. Spectral Skewness: The third moment, measures asymmetry.
    skewness = np.sum(psd * (freqs - centroid[:, np.newaxis]) ** 3, axis=1) / (
        spec_sum * spread**3
    )

    # 4. Spectral Kurtosis: The fourth moment, measures "peakiness".
    kurtosis = np.sum(psd * (freqs - centroid[:, np.newaxis]) ** 4, axis=1) / (
        spec_sum * spread**4
    )

    # --- Assemble into a feature matrix H ---
    H = np.column_stack([centroid, spread, skewness, kurtosis])
    H = np.nan_to_num(H)  # Handle potential divisions by zero

    descriptors = [
        {"type": "shape_prop", "name": "centroid"},
        {"type": "shape_prop", "name": "spread"},
        {"type": "shape_prop", "name": "skewness"},
        {"type": "shape_prop", "name": "kurtosis"},
    ]

    # --- Calculate correlation scores and select top-k ---
    res_z = (residual - residual.mean()) / (residual.std() + 1e-12)
    H_mu = H.mean(axis=0)
    H_std = H.std(axis=0)
    H_centered = H - H_mu
    scores = np.abs(res_z @ H_centered) / len(residual)

    k = min(top_k, H.shape[1])
    if k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
        )

    top_indices = np.argsort(scores)[-k:][::-1]

    return Proposal(
        H=H[:, top_indices],
        descriptors=[descriptors[i] for i in top_indices],
        mu=H_mu[top_indices],
        sigma=H_std[top_indices],
    )
