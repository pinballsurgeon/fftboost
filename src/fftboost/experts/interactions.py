from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .types import ExpertContext
from .types import Proposal


def propose(
    residual: np.ndarray[Any, Any], ctx: ExpertContext, *, top_k: int = 1
) -> Proposal:
    """
    Proposes features based on the product of already-selected FFT bins.
    """
    if ctx.selected_bins is None or len(ctx.selected_bins) < 2:
        return Proposal(
            H=np.empty((residual.shape[0], 0)),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
        )

    # --- Generate all pairwise product combinations of selected features ---
    selected_psd = ctx.psd[:, ctx.selected_bins]

    pair_indices = list(combinations(range(len(ctx.selected_bins)), 2))

    if not pair_indices:
        return Proposal(
            H=np.empty((residual.shape[0], 0)),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
        )

    H_list = []
    descriptors = []
    for i, j in pair_indices:
        H_list.append(selected_psd[:, i] * selected_psd[:, j])

        bin1 = ctx.selected_bins[i]
        bin2 = ctx.selected_bins[j]

        descriptors.append(
            {
                "type": "interaction",
                "bins": (int(bin1), int(bin2)),
                "freqs_hz": (float(ctx.freqs[bin1]), float(ctx.freqs[bin2])),
            }
        )

    H = np.column_stack(H_list)

    # --- Calculate correlation scores and select top-k ---
    res_z = (residual - residual.mean()) / (residual.std() + 1e-12)
    H_mu = H.mean(axis=0)
    H_std = H.std(axis=0)
    H_centered = H - H_mu
    scores = np.abs(res_z @ H_centered) / len(residual)

    k = min(top_k, H.shape[1])
    if k <= 0:
        return Proposal(
            H=np.empty((residual.shape[0], 0)),
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
