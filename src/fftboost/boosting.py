from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from .experts.types import Proposal


@dataclass(frozen=True)
class StageRecord:
    """
    Minimal, reproducible record of a single boosting stage.

    Stores everything needed to regenerate the stage contribution given
    the original proposals and inputs.
    """

    weights: np.ndarray  # shape: (n_features,)
    descriptors: list[dict[str, object]]
    mu: np.ndarray  # per-feature means used for z-scoring
    sigma: np.ndarray  # per-feature stds used for z-scoring
    gamma: float  # 1D line-search scalar
    ridge_alpha: float
    nu: float


def _concat_proposals(
    proposals: list[Proposal],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """
    Stack proposal matrices column-wise, and concatenate mu/sigma and descriptors.
    Returns (H, mu, sigma, descriptors).
    """
    if len(proposals) == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
        )

    H_list = [p.H for p in proposals]
    mu_list = [p.mu for p in proposals]
    sigma_list = [p.sigma for p in proposals]
    desc_list: list[dict[str, object]] = []
    for p in proposals:
        desc_list.extend(p.descriptors)

    H = np.hstack(H_list) if len(H_list) > 1 else H_list[0]
    mu = np.hstack(mu_list) if len(mu_list) > 1 else mu_list[0]
    sigma = np.hstack(sigma_list) if len(sigma_list) > 1 else sigma_list[0]
    return cast(np.ndarray, H), cast(np.ndarray, mu), cast(np.ndarray, sigma), desc_list


def _ridge_fit_via_cholesky(Z: np.ndarray, r: np.ndarray, alpha: float) -> np.ndarray:
    """
    Solve (Z^T Z + alpha I) w = Z^T r via Cholesky.
    """
    n_features = Z.shape[1]
    if n_features == 0:
        return cast(np.ndarray, np.empty((0,), dtype=np.float64))

    G = Z.T @ Z
    # Add ridge on the diagonal
    G.flat[:: n_features + 1] += alpha
    y = Z.T @ r

    L = np.linalg.cholesky(G)
    # Solve L z = y
    z = np.linalg.solve(L, y)
    # Solve L^T w = z
    w = np.linalg.solve(L.T, z)
    return cast(np.ndarray, w)


def fit_stage(
    residual: np.ndarray,
    proposals: list[Proposal],
    ridge_alpha: float,
    nu: float,
) -> tuple[np.ndarray, StageRecord]:
    """
    Fit a single boosting stage against the residual using expert proposals.

    Steps:
      1) Concatenate features; z-score per proposal-provided mu/sigma
      2) Solve ridge for weights via Cholesky
      3) raw_step = Z @ w
      4) gamma from 1D least-squares line search
      5) applied_step = nu * gamma * raw_step
    """
    # Concatenate proposals
    H, mu, sigma, descriptors = _concat_proposals(proposals)

    n_windows = residual.shape[0]
    if H.size == 0:
        # No features: zero step
        applied = np.zeros(n_windows, dtype=np.float64)
        rec = StageRecord(
            weights=np.empty((0,), dtype=np.float64),
            descriptors=descriptors,
            mu=mu,
            sigma=sigma,
            gamma=0.0,
            ridge_alpha=ridge_alpha,
            nu=nu,
        )
        return applied, rec

    # Z-score with provided moments (protect against zero variance)
    Z = (H - mu) / (sigma + 1e-12)

    # Fit ridge weights
    w = _ridge_fit_via_cholesky(Z, residual, ridge_alpha)

    # Compute raw step and line-search gamma
    raw_step = Z @ w
    denom = float(raw_step @ raw_step)
    if denom == 0.0:
        gamma = 0.0
    else:
        gamma = float((residual @ raw_step) / denom)

    applied_step = (nu * gamma) * raw_step

    record = StageRecord(
        weights=w,
        descriptors=descriptors,
        mu=mu,
        sigma=sigma,
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        nu=nu,
    )
    return applied_step, record
