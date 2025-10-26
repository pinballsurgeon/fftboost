from __future__ import annotations

from typing import Any

import numpy as np

from .types import ExpertContext
from .types import Proposal


def propose(
    residual: np.ndarray[Any, Any], ctx: ExpertContext, *, top_k: int = 5
) -> Proposal:
    """
    Proposes features based on the instantaneous frequency of FFT bins.

    Instantaneous frequency is the time-derivative of the unwrapped phase,
    capturing frequency modulation within a signal.
    """
    if ctx.complex_rfft is None:
        return Proposal(
            H=np.empty((residual.shape[0], 0)),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
        )

    # 1. Calculate unwrapped phase along the time (window) axis
    unwrapped_phase = np.unwrap(np.angle(ctx.complex_rfft), axis=0)

    # 2. Calculate instantaneous frequency (IF) as the discrete time-derivative
    # of phase. The result is the deviation from the center frequency of each bin.
    # We prepend the first value to maintain the original shape.
    if_deviation = np.diff(
        unwrapped_phase, n=1, axis=0, prepend=unwrapped_phase[0:1, :]
    )

    # The feature matrix H is the time series of these frequency deviations.
    # It's crucial to center the IF deviation to remove any DC offset from phase drift.
    H = if_deviation - if_deviation.mean(axis=0, keepdims=True)

    # 3. Calculate correlation scores, weighted by the power in each bin. This
    # ensures that we only consider phase information from bins with significant
    # energy.
    res_z = (residual - residual.mean()) / (residual.std() + 1e-12)
    H_mu = H.mean(axis=0)
    H_std = H.std(axis=0)
    H_centered = H - H_mu

    # Raw correlation of instantaneous frequency with the residual
    if_scores = np.abs(res_z @ H_centered) / len(residual)

    # Weight by the mean power in each bin to suppress noise from low-power bins.
    # We use normalized power as a weighting factor between 0 and 1.
    mean_power = ctx.psd.mean(axis=0)
    power_weights = mean_power / (mean_power.max() + 1e-12)

    scores = if_scores * power_weights

    # 4. Apply penalties (re-using logic from fft_bin expert)
    if ctx.freqs.size > 0 and ctx.lambda_hf > 0.0:
        hf_penalty = ctx.lambda_hf * (ctx.freqs / (ctx.fs / 2.0))
        scores -= hf_penalty

    if ctx.selected_bins is not None and ctx.selected_bins.size > 0:
        for b in ctx.selected_bins:
            lo = max(0, int(b) - ctx.min_sep_bins)
            hi = min(scores.shape[0], int(b) + ctx.min_sep_bins + 1)
            scores[lo:hi] = -np.inf

    # 5. Select top-K candidates
    k = min(top_k, H.shape[1])
    if k <= 0:
        return Proposal(
            H=np.empty((residual.shape[0], 0)),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
        )

    top_indices = np.argsort(scores)[-k:][::-1]

    # 6. Construct the proposal
    descriptors = [
        {"type": "phase_bin", "freq_hz": float(ctx.freqs[i]), "bin": int(i)}
        for i in top_indices
    ]

    return Proposal(
        H=H[:, top_indices],
        descriptors=descriptors,
        mu=H_mu[top_indices],
        sigma=H_std[top_indices],
    )
