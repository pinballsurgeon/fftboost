from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .types import ExpertContext
from .types import Proposal


def _standardize_cols(
    H: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    mu = H.mean(axis=0)
    sigma = H.std(axis=0)
    Z = (H - mu) / (sigma + 1e-12)
    return Z, mu, sigma


def propose_flux(
    residual: np.ndarray[Any, Any],
    ctx: ExpertContext,
    *,
    top_k: int = 5,
) -> Proposal:
    psd = ctx.psd  # (n_windows, n_bins)
    freqs = ctx.freqs
    n_windows, n_bins = psd.shape
    if n_windows < 2 or top_k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.array([]),
            sigma=np.array([]),
        )

    # Temporal difference along windows (flux)
    d = np.zeros_like(psd)
    d[1:, :] = psd[1:, :] - psd[:-1, :]

    # Flux-energy score per bin (robust to monotonic trends)
    scores = np.sqrt(np.mean(d * d, axis=0))
    # Moments for downstream normalization
    mu = d.mean(axis=0)
    sigma = d.std(axis=0)

    # Apply penalties and min-sep around selected bins
    if freqs.size > 0 and ctx.lambda_hf > 0.0:
        hf_penalty = ctx.lambda_hf * (freqs / (ctx.fs / 2.0))
        scores = scores - hf_penalty
    if ctx.selected_bins is not None and ctx.selected_bins.size > 0:
        for b in ctx.selected_bins:
            lo = max(0, int(b) - ctx.min_sep_bins)
            hi = min(n_bins, int(b) + ctx.min_sep_bins + 1)
            scores[lo:hi] = -np.inf

    k = min(top_k, n_bins)
    idxs = np.argpartition(scores, -k)[-k:]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    H = d[:, idxs]
    desc = [
        {"type": "flux_bin", "freq_hz": float(freqs[i]), "bin": int(i)} for i in idxs
    ]
    return Proposal(H=H, descriptors=desc, mu=mu[idxs], sigma=sigma[idxs])


def propose_lagstack(
    residual: np.ndarray[Any, Any],
    ctx: ExpertContext,
    *,
    bins: np.ndarray[Any, Any],
    lags: Iterable[int] = (1, 2, 3),
    top_k: int = 5,
) -> Proposal:
    psd = ctx.psd
    freqs = ctx.freqs
    n_windows, _ = psd.shape
    bins = np.array(bins, dtype=np.int64)
    lags = [int(lag) for lag in lags if int(lag) > 0]
    if bins.size == 0 or len(lags) == 0 or n_windows < 2 or top_k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.array([]),
            sigma=np.array([]),
        )

    cols: list[np.ndarray[Any, Any]] = []
    desc: list[dict[str, object]] = []
    for b in bins:
        if not (0 <= b < freqs.size):
            continue
        x = psd[:, int(b)]
        for L in lags:
            v = np.zeros_like(x)
            v[L:] = x[:-L]
            cols.append(v)
            desc.append(
                {
                    "type": "lag_bin",
                    "freq_hz": float(freqs[int(b)]),
                    "bin": int(b),
                    "lag": int(L),
                }
            )

    if not cols:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.array([]),
            sigma=np.array([]),
        )

    H = np.column_stack(cols)
    Z, mu, sigma = _standardize_cols(H)
    rz = (residual - residual.mean()) / (residual.std() + 1e-12)
    scores = np.abs(rz @ Z) / float(n_windows)

    k = min(top_k, H.shape[1])
    idxs = np.argpartition(scores, -k)[-k:]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    return Proposal(
        H=H[:, idxs],
        descriptors=[desc[i] for i in idxs],
        mu=mu[idxs],
        sigma=sigma[idxs],
    )


def propose_burstpool(
    residual: np.ndarray[Any, Any],
    ctx: ExpertContext,
    *,
    widths: Iterable[int] = (3, 5),
    top_k: int = 5,
) -> Proposal:
    psd = ctx.psd
    freqs = ctx.freqs
    n_windows, n_bins = psd.shape
    widths = [int(w) for w in widths if int(w) >= 2]
    if not widths or top_k <= 0:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.array([]),
            sigma=np.array([]),
        )

    cols: list[np.ndarray[Any, Any]] = []
    desc: list[dict[str, object]] = []
    for b in range(n_bins):
        x = psd[:, b]
        for w in widths:
            k = np.ones(w, dtype=np.float64) / float(w)
            y = np.convolve(x, k, mode="same")
            cols.append(y)
            desc.append(
                {
                    "type": "pool_bin",
                    "freq_hz": float(freqs[b]),
                    "bin": int(b),
                    "width": int(w),
                }
            )

    if not cols:
        return Proposal(
            H=np.empty((n_windows, 0)),
            descriptors=[],
            mu=np.array([]),
            sigma=np.array([]),
        )

    H = np.column_stack(cols)
    Z, mu, sigma = _standardize_cols(H)
    rz = (residual - residual.mean()) / (residual.std() + 1e-12)
    scores = np.abs(rz @ Z) / float(n_windows)

    k = min(top_k, H.shape[1])
    idxs = np.argpartition(scores, -k)[-k:]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    return Proposal(
        H=H[:, idxs],
        descriptors=[desc[i] for i in idxs],
        mu=mu[idxs],
        sigma=sigma[idxs],
    )
