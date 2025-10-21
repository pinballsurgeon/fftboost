from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np

from .types import ExpertContext
from .types import Proposal


def _fscore(
    psd: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    # labels are binary {0,1}
    mask1 = labels > 0.5
    mask0 = ~mask1
    # Protect against empty classes
    if not mask0.any() or not mask1.any():
        return np.zeros(psd.shape[1], dtype=np.float64)
    x1 = psd[mask1]
    x0 = psd[mask0]
    mu1 = x1.mean(axis=0)
    mu0 = x0.mean(axis=0)
    var1 = x1.var(axis=0)
    var0 = x0.var(axis=0)
    num = (mu1 - mu0) ** 2
    den = var1 + var0
    # Add a small epsilon to prevent division by zero
    scores = num / (den + 1e-12)
    return scores


def _mutual_info(
    psd: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        return np.zeros(psd.shape[1], dtype=np.float64)
    # mutual_info_classif expects (n_samples, n_features)
    return mutual_info_classif(psd, labels.astype(int), discrete_features=False)


def propose(
    residual: np.ndarray[Any, Any],
    ctx: ExpertContext,
    *,
    top_k: int = 5,
    method: Literal["fscore", "mi"] = "fscore",
) -> Proposal:
    psd = ctx.psd
    freqs = ctx.freqs
    n_windows, n_bins = psd.shape

    # No labels -> no proposal
    if ctx.y_labels is None:
        return Proposal(
            H=np.empty((n_windows, 0), dtype=np.float64),
            descriptors=[],
            mu=np.empty((0,), dtype=np.float64),
            sigma=np.empty((0,), dtype=np.float64),
        )

    labels = ctx.y_labels.astype(np.float64, copy=False)

    if method == "mi":
        scores = _mutual_info(psd, labels)
    else:
        scores = _fscore(psd, labels)

    # Penalties
    if freqs.size > 0 and ctx.lambda_hf > 0.0:
        hf_penalty = ctx.lambda_hf * (freqs / (ctx.fs / 2.0))
        scores = scores - hf_penalty

    # Enforce min separation around selected bins
    if ctx.selected_bins is not None and ctx.selected_bins.size > 0:
        for b in ctx.selected_bins:
            lo = max(0, int(b) - ctx.min_sep_bins)
            hi = min(n_bins, int(b) + ctx.min_sep_bins + 1)
            scores[lo:hi] = -np.inf

    k = min(top_k, n_bins)
    if k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0), dtype=np.float64),
            descriptors=[],
            mu=np.empty(0, dtype=np.float64),
            sigma=np.empty(0, dtype=np.float64),
        )

    # Use argpartition for performance; it finds the k-largest elements
    # without a full sort. We then sort the top k to get them in order.
    top_indices_unsorted = np.argpartition(scores, -k)[-k:]
    top_scores = scores[top_indices_unsorted]
    top_indices = top_indices_unsorted[np.argsort(top_scores)[::-1]]

    H = psd[:, top_indices]
    mu = psd.mean(axis=0)[top_indices]
    sigma = psd.std(axis=0)[top_indices]
    descriptors = [
        {"type": "clf_bin", "freq_hz": float(freqs[i]), "bin": int(i)}
        for i in top_indices
    ]
    return Proposal(H=H, descriptors=descriptors, mu=mu, sigma=sigma)
