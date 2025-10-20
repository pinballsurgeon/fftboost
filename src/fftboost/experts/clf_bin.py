from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np

from .types import ExpertContext
from .types import Proposal


def _fscore(
    psd: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    sample_weight: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    # labels are binary {0,1}
    mask1 = labels > 0.5
    mask0 = ~mask1
    # Protect against empty classes
    if not mask0.any() or not mask1.any():
        return np.zeros(psd.shape[1], dtype=np.float64)

    if sample_weight is None:
        sample_weight = np.ones_like(labels)

    w1 = sample_weight[mask1]
    w0 = sample_weight[mask0]
    w1_sum = w1.sum()
    w0_sum = w0.sum()

    if w1_sum == 0 or w0_sum == 0:
        return np.zeros(psd.shape[1], dtype=np.float64)

    x1 = psd[mask1]
    x0 = psd[mask0]

    mu1 = np.sum(x1 * w1[:, np.newaxis], axis=0) / w1_sum
    mu0 = np.sum(x0 * w0[:, np.newaxis], axis=0) / w0_sum

    var1 = np.sum(w1[:, np.newaxis] * (x1 - mu1) ** 2, axis=0) / w1_sum
    var0 = np.sum(w0[:, np.newaxis] * (x0 - mu0) ** 2, axis=0) / w0_sum
    num = (mu1 - mu0) ** 2
    den = var1 + var0 + 1e-12
    return num / den


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

    # Weight samples by absolute residual to focus on errors
    sample_weight = np.abs(residual)
    scores = _fscore(psd, labels, sample_weight=sample_weight)

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

    k = min(top_k * 2, n_bins)
    if k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0), dtype=np.float64),
            descriptors=[],
            mu=np.empty(0, dtype=np.float64),
            sigma=np.empty(0, dtype=np.float64),
        )

    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    H = psd[:, top_indices]
    mu = psd.mean(axis=0)[top_indices]
    sigma = psd.std(axis=0)[top_indices]
    descriptors = [
        {"type": "clf_bin", "freq_hz": float(freqs[i]), "bin": int(i)}
        for i in top_indices
    ]
    return Proposal(H=H, descriptors=descriptors, mu=mu, sigma=sigma)
