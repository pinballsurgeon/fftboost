from __future__ import annotations

import numpy as np

from .types import ExpertContext
from .types import Proposal


def propose(residual: np.ndarray, ctx: ExpertContext, *, top_k: int = 5) -> Proposal:
    """
    Proposes top-K FFT bins by correlation with the residual, applying priors.
    """
    psd = ctx.psd  # (n_windows, n_bins)
    freqs = ctx.freqs  # (n_bins,)
    n_windows, n_bins = psd.shape

    # 1. Z-score inputs for correlation calculation
    res_mu = residual.mean()
    res_std = residual.std()
    psd_mu = psd.mean(axis=0)
    psd_std = psd.std(axis=0)
    res_z = (residual - res_mu) / (res_std + 1e-12)
    psd_z = (psd - psd_mu) / (psd_std + 1e-12)

    # 2. Compute correlation scores (absolute mean dot-product)
    corrs = np.abs(res_z @ psd_z) / float(n_windows)

    # 3. Apply physics-aware penalties to scores
    scores = corrs.copy()
    if freqs.size > 0 and ctx.lambda_hf > 0.0:
        hf_penalty = ctx.lambda_hf * (freqs / (ctx.fs / 2.0))
        scores -= hf_penalty

    # Enforce min separation around already selected bins
    if ctx.selected_bins is not None and ctx.selected_bins.size > 0:
        for b in ctx.selected_bins:
            lo = max(0, int(b) - ctx.min_sep_bins)
            hi = min(n_bins, int(b) + ctx.min_sep_bins + 1)
            scores[lo:hi] = -np.inf

    # 4. Select top-K candidates
    k = min(top_k, n_bins)
    if k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0), dtype=np.float64),
            descriptors=[],
            mu=np.empty(0, dtype=np.float64),
            sigma=np.empty(0, dtype=np.float64),
        )

    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    # 5. Construct the proposal
    H = psd[:, top_indices]
    mu = psd_mu[top_indices]
    sigma = psd_std[top_indices]
    descriptors = [{"type": "fft_bin", "freq_hz": float(freqs[i])} for i in top_indices]

    return Proposal(H=H, descriptors=descriptors, mu=mu, sigma=sigma)
