from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.fft import rfftfreq
from sklearn.linear_model import Ridge

from .config import FFTBoostConfig


def _calculate_penalties(
    config: FFTBoostConfig, n_fft_features: int, fs: int
) -> npt.NDArray:
    freqs = rfftfreq(n_fft_features * 2, 1 / fs)[1 : n_fft_features + 1]
    hf_penalty = config.lambda_hf * (freqs / (fs / 2))
    coh_penalty = config.lambda_coh * (1 - np.ones(n_fft_features))
    return hf_penalty + coh_penalty


def _fit_model(
    x_fft: npt.NDArray,
    x_aux: npt.NDArray,
    y: npt.NDArray,
    config: FFTBoostConfig,
    fs: int,
) -> dict[str, Any]:
    _, n_fft_features = x_fft.shape
    residuals = y.copy()
    active_atoms: list[int] = []
    penalties = _calculate_penalties(config, n_fft_features, fs)

    for _ in range(config.atoms):
        corrs = np.abs(np.corrcoef(residuals, x_fft, rowvar=False)[0, 1:])
        scores = corrs - penalties

        for atom_idx in active_atoms:
            start = max(0, atom_idx - config.min_sep_bins)
            end = min(n_fft_features, atom_idx + config.min_sep_bins + 1)
            scores[start:end] = -np.inf

        best_atom = int(np.argmax(scores))
        active_atoms.append(best_atom)

        selected_features = x_fft[:, active_atoms]
        model = Ridge(alpha=1.0)
        model.fit(selected_features, y)
        residuals = y - model.predict(selected_features)

    final_features = np.hstack([x_fft[:, active_atoms], x_aux])
    final_model = Ridge(alpha=config.ridge_alpha)
    final_model.fit(final_features, y)

    return {"active_atoms": active_atoms, "final_model": final_model}
