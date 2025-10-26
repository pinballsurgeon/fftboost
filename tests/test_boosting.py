from __future__ import annotations

import numpy as np

from src.fftboost.boosting import StageRecord
from src.fftboost.boosting import fit_stage
from src.fftboost.experts.types import Proposal


def _make_proposals_from_matrix(H_list: list[np.ndarray]) -> list[Proposal]:
    proposals: list[Proposal] = []
    for idx, H in enumerate(H_list):
        mu = H.mean(axis=0)
        sigma = H.std(axis=0)
        desc = [
            {"type": "synthetic", "idx": int(idx), "col": int(j)}
            for j in range(H.shape[1])
        ]
        proposals.append(Proposal(H=H, descriptors=desc, mu=mu, sigma=sigma))
    return proposals


def test_fit_stage_converges_toward_residual() -> None:
    # Synthetic residual: smooth envelope with slight noise
    rng = np.random.default_rng(123)
    n = 2048
    t = np.linspace(0.0, 10.0, n, dtype=np.float64)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t)
    residual = envelope + 0.05 * rng.standard_normal(n)

    # Create proposals: include residual itself and some distractor columns
    H1 = residual.reshape(-1, 1)
    H2 = rng.standard_normal((n, 4))  # distractors
    proposals = _make_proposals_from_matrix([H1, H2])

    applied, rec = fit_stage(residual, proposals, ridge_alpha=1e-3, nu=0.8)

    # Correlation between applied step and residual should be strong
    corr = np.corrcoef(applied, residual)[0, 1]
    assert corr > 0.85

    # Sanity on record content
    assert isinstance(rec, StageRecord)
    assert rec.weights.shape[0] == 1
    assert len(rec.descriptors) == 1
    assert rec.mu.shape == rec.sigma.shape == (1,)


def test_fit_stage_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    n = 1024
    residual = rng.standard_normal(n)

    # Two proposals with fixed content
    H1 = np.vstack(
        [
            residual,
            residual**2,
        ]
    ).T
    H2 = np.ones((n, 1))
    proposals = _make_proposals_from_matrix([H1, H2])

    applied1, rec1 = fit_stage(residual, proposals, ridge_alpha=1e-2, nu=0.5)
    applied2, rec2 = fit_stage(residual, proposals, ridge_alpha=1e-2, nu=0.5)

    np.testing.assert_array_equal(applied1, applied2)
    np.testing.assert_array_equal(rec1.weights, rec2.weights)
    assert rec1.descriptors == rec2.descriptors
    np.testing.assert_array_equal(rec1.mu, rec2.mu)
    np.testing.assert_array_equal(rec1.sigma, rec2.sigma)
    assert rec1.gamma == rec2.gamma
    assert rec1.ridge_alpha == rec2.ridge_alpha
    assert rec1.nu == rec2.nu
