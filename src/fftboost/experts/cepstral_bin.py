from __future__ import annotations

from typing import Any

import numpy as np

from .types import ExpertContext
from .types import Proposal


def propose(
    residual: np.ndarray[Any, Any], ctx: ExpertContext, *, top_k: int = 5
) -> Proposal:
    """
    Proposes features based on the real cepstrum of the signal.

    The cepstrum can reveal periodic structures in the spectrum, such as
    harmonic series (pitch) or echoes.
    """
    # 1. Calculate the real cepstrum for each window's power spectrum.
    # The cepstrum is the inverse FFT of the log-power spectrum.
    log_psd = np.log(ctx.psd + 1e-20)
    cepstrum = np.fft.irfft(log_psd, axis=1)

    # The feature matrix H is the time series of the cepstral coefficients.
    # We skip the first coefficient (c[0]) as it represents the average log power.
    H = cepstrum[:, 1:]

    # 2. Calculate scores based on cepstral peak magnitude.
    # For this expert, the best feature is the strongest peak in the cepstrum,
    # as this corresponds to the most prominent harmonic series.
    # We use the mean cepstral value across all windows as the score.
    scores = H.mean(axis=0)
    H_mu = H.mean(axis=0)
    H_std = H.std(axis=0)

    # 3. Apply a prior: search for pitch only in a reasonable quefrency range.
    # This avoids selecting low-quefrency components corresponding to spectral shape.
    # We'll search for fundamentals roughly between 70 Hz and 400 Hz.
    min_period = 1.0 / 400.0  # Corresponds to 400 Hz
    max_period = 1.0 / 70.0  # Corresponds to 70 Hz

    quefrencies = np.arange(1, H.shape[1] + 1) / ctx.fs

    valid_indices = np.where((quefrencies >= min_period) & (quefrencies <= max_period))[
        0
    ]

    # Create a mask to invalidate scores outside the valid range
    mask = np.full(scores.shape, -np.inf)
    if valid_indices.size > 0:
        mask[valid_indices] = 0

    scores += mask

    # 4. Select top-K candidates
    k = min(top_k, H.shape[1])
    if k <= 0:
        return Proposal(
            H=np.empty((residual.shape[0], 0)),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
        )

    top_indices = np.argsort(scores)[-k:][::-1]

    # 5. Construct the proposal
    # Descriptors for cepstral features are in "quefrency" (units of time).
    # A peak at a certain quefrency implies a periodic pattern in the spectrum
    # with a spacing of 1/quefrency Hz.
    quefrencies = np.arange(1, H.shape[1] + 1) / ctx.fs

    descriptors = [
        {
            "type": "cepstral_bin",
            "quefrency_s": float(quefrencies[i]),
            "index": int(i + 1),
        }
        for i in top_indices
    ]

    return Proposal(
        H=H[:, top_indices],
        descriptors=descriptors,
        mu=H_mu[top_indices],
        sigma=H_std[top_indices],
    )
