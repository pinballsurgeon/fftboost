from __future__ import annotations

import numpy as np

from fftboost.experts.interactions import propose
from fftboost.experts.types import ExpertContext


def test_interaction_expert_identifies_product() -> None:
    """
    Verify that the interaction expert can identify a product of two frequencies.
    """
    fs = 2000.0
    n_windows = 100
    n_bins = 50

    # Create a synthetic PSD where two bins are meaningful
    psd = np.random.rand(n_windows, n_bins) * 0.1
    bin1, bin2 = 10, 30
    psd[:, bin1] = 1.0 + np.random.rand(n_windows)
    psd[:, bin2] = 1.0 + np.random.rand(n_windows)

    # The target is the product of the power in these two bins
    y = psd[:, bin1] * psd[:, bin2]
    residual = y - y.mean()

    freqs = np.linspace(0, fs / 2, n_bins)

    # --- Run the proposal ---
    # The context simulates that bins 10 and 30 have already been selected
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=0,
        lambda_hf=0.0,
        selected_bins=np.array([bin1, bin2]),
    )

    proposal = propose(residual, ctx, top_k=1)

    # --- Assertions ---
    assert len(proposal.descriptors) == 1
    descriptor = proposal.descriptors[0]

    assert descriptor["type"] == "interaction"

    # The expert should have picked the interaction between bin1 and bin2
    selected_bins = descriptor["bins"]
    assert set(selected_bins) == {bin1, bin2}

    # The feature matrix H should be the product of the two columns
    expected_H = (psd[:, bin1] * psd[:, bin2]).reshape(-1, 1)
    np.testing.assert_allclose(proposal.H, expected_H)
