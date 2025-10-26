from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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

    weights: np.ndarray[Any, Any]  # shape: (n_features,)
    descriptors: list[dict[str, object]]
    mu: np.ndarray[Any, Any]  # per-feature means used for z-scoring
    sigma: np.ndarray[Any, Any]  # per-feature stds used for z-scoring
    gamma: float  # 1D line-search scalar
    ridge_alpha: float
    nu: float


def _concat_proposals(
    proposals: list[Proposal],
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[dict[str, object]],
]:
    """
    Stack proposal matrices column-wise, and concatenate mu/sigma and descriptors.
    Returns (H, mu, sigma, descriptors).
    """
    if not proposals:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
        )

    # Ensure all proposals have a consistent number of windows (first dimension)
    n_windows = proposals[0].H.shape[0]
    for p in proposals:
        if p.H.shape[0] != n_windows:
            # This case should ideally not happen if experts are implemented correctly
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                [],
            )

    H_list = [p.H for p in proposals if p.H.size > 0]
    mu_list = [p.mu for p in proposals if p.H.size > 0]
    sigma_list = [p.sigma for p in proposals if p.H.size > 0]
    desc_list: list[dict[str, object]] = []
    for p in proposals:
        if p.H.size > 0:
            desc_list.extend(p.descriptors)

    if not H_list:
        return (
            np.empty((n_windows, 0), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
        )

    H = np.hstack(H_list)
    mu = np.concatenate(mu_list)
    sigma = np.concatenate(sigma_list)
    return H, mu, sigma, desc_list


def _ridge_fit_via_cholesky(
    Z: np.ndarray[Any, Any], r: np.ndarray[Any, Any], alpha: float
) -> np.ndarray[Any, Any]:
    """
    Solve (Z^T Z + alpha I) w = Z^T r via Cholesky.
    """
    n_features = Z.shape[1]
    if n_features == 0:
        return np.empty((0,), dtype=np.float64)

    G = Z.T @ Z
    # Add ridge on the diagonal
    G.flat[:: n_features + 1] += alpha
    y = Z.T @ r

    try:
        L = np.linalg.cholesky(G)
        # Solve L z = y
        z = np.linalg.solve(L, y)
        # Solve L^T w = z
        w = np.linalg.solve(L.T, z)
        return cast(np.ndarray[Any, Any], w)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix
        return np.linalg.pinv(G) @ y


def fit_stage(
    residual: np.ndarray[Any, Any],
    proposals: list[Proposal],
    ridge_alpha: float,
    nu: float,
    ensemble_k: int = 1,
) -> tuple[np.ndarray[Any, Any], StageRecord]:
    """
    Fit a single boosting stage against the residual using expert proposals.
    This version implements a robust greedy selection of the single best feature.
    """
    # Concatenate all proposals into one large feature matrix
    H_all, mu_all, sigma_all, descriptors_all = _concat_proposals(proposals)

    n_windows = residual.shape[0]
    if H_all.size == 0:
        return np.zeros(n_windows), StageRecord(
            weights=np.empty(0),
            descriptors=[],
            mu=np.empty(0),
            sigma=np.empty(0),
            gamma=0.0,
            ridge_alpha=ridge_alpha,
            nu=nu,
        )

    # Z-score all features for scoring
    Z_all = (H_all - mu_all) / (sigma_all + 1e-12)

    # Score every proposed feature by its correlation with the residual
    res_z = (residual - residual.mean()) / (residual.std() + 1e-12)
    scores = np.abs(res_z @ Z_all)

    # Select only the single best feature from all proposals
    best_idx = np.argmax(scores)

    # Create the final feature matrix for this stage with only the best feature
    H_best = H_all[:, [best_idx]]
    mu_best = mu_all[[best_idx]]
    sigma_best = sigma_all[[best_idx]]
    descriptors_best = [descriptors_all[best_idx]]
    Z_best = (H_best - mu_best) / (sigma_best + 1e-12)

    # Fit ridge weights (will be a 1D regression)
    w = _ridge_fit_via_cholesky(Z_best, residual, ridge_alpha)

    # Compute raw step and line-search gamma
    raw_step = Z_best @ w
    denom = float(raw_step @ raw_step)
    gamma = 0.0 if denom == 0.0 else float((residual @ raw_step) / denom)

    applied_step = (nu * gamma) * raw_step

    record = StageRecord(
        weights=w,
        descriptors=descriptors_best,
        mu=mu_best,
        sigma=sigma_best,
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        nu=nu,
    )
    return applied_step, record
