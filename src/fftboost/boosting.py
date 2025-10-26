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

    L = np.linalg.cholesky(G)
    # Solve L z = y
    z = np.linalg.solve(L, y)
    # Solve L^T w = z
    w = np.linalg.solve(L.T, z)
    return cast(np.ndarray[Any, Any], w)


def fit_stage(
    residual: np.ndarray[Any, Any],
    proposals: list[Proposal],
    ridge_alpha: float,
    nu: float,
    ensemble_k: int = 1,
) -> tuple[np.ndarray[Any, Any], StageRecord]:
    """
    Fit a single boosting stage against the residual using expert proposals.

    Steps:
      1) Concatenate all proposed features from all experts.
      2) Score each feature by its correlation with the residual.
      3) Select the top `ensemble_k` features.
      4) Solve a single multi-variate Ridge regression on this ensemble.
      5) Perform a 1D line search (gamma) on the combined step.
      6) The final step is nu * gamma * (Z_ensemble @ w).
    """
    # Concatenate all proposals into one large feature matrix
    H_all, mu_all, sigma_all, descriptors_all = _concat_proposals(proposals)

    n_windows = residual.shape[0]
    if H_all.size == 0:
        # No features proposed, return a zero step
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

    # Select the top `ensemble_k` features
    k = min(ensemble_k, Z_all.shape[1])
    top_indices = np.argsort(scores)[-k:][::-1]

    # Create the final ensemble feature matrix for this stage
    H_ensemble = H_all[:, top_indices]
    mu_ensemble = mu_all[top_indices]
    sigma_ensemble = sigma_all[top_indices]
    descriptors_ensemble = [descriptors_all[i] for i in top_indices]
    Z_ensemble = (H_ensemble - mu_ensemble) / (sigma_ensemble + 1e-12)

    # Fit ridge weights on the selected ensemble
    w = _ridge_fit_via_cholesky(Z_ensemble, residual, ridge_alpha)

    # Compute raw step and line-search gamma
    raw_step = Z_ensemble @ w
    denom = float(raw_step @ raw_step)
    gamma = 0.0 if denom == 0.0 else float((residual @ raw_step) / denom)

    applied_step = (nu * gamma) * raw_step

    record = StageRecord(
        weights=w,
        descriptors=descriptors_ensemble,
        mu=mu_ensemble,
        sigma=sigma_ensemble,
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        nu=nu,
    )
    return applied_step, record
