from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.fft import rfftfreq
from sklearn.linear_model import Ridge


def fit_fftboost_model(
    x_fft_tr: np.ndarray,
    x_aux_tr: np.ndarray,
    y_tr: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    n_samples, n_fft_features = x_fft_tr.shape
    fftboost_cfg = config["fftboost"]
    atoms = fftboost_cfg["atoms"]
    win_len = config["fft_win_len"]
    fs = config["fs"]

    freqs = rfftfreq(win_len, 1 / fs)
    cand_bins = np.arange(1, len(freqs))
    freqs_cand = freqs[cand_bins]

    lam_hf = fftboost_cfg["lambda_hf"]
    lam_coh = fftboost_cfg["lambda_coh"]
    coh_vec = config["coherence_vec"]

    freq_penalty = lam_hf * (np.arange(n_fft_features) / max(1, n_fft_features - 1)) ** 2
    coh_penalty = lam_coh * (1.0 - coh_vec) ** 2
    penalty = freq_penalty + coh_penalty

    residuals = y_tr.copy()
    active_atoms = []
    forbidden_indices: set[int] = set()

    for _ in range(atoms):
        corr = x_fft_tr.T @ residuals
        score = np.abs(corr) - penalty
        score[list(forbidden_indices)] = -np.inf

        best_idx = int(np.argmax(score))
        if not np.isfinite(score[best_idx]) or score[best_idx] <= 0:
            break

        active_atoms.append(best_idx)
        f_k = max(freqs_cand[best_idx], 1.0)
        adaptive_sep = max(1, int(round(fftboost_cfg["min_sep_bins"] * (100.0 / f_k) ** 0.3)))

        for d in range(-adaptive_sep, adaptive_sep + 1):
            k_plus_d = best_idx + d
            if 0 <= k_plus_d < n_fft_features:
                forbidden_indices.add(k_plus_d)

        if len(active_atoms) > 0:
            current_features = np.hstack([x_fft_tr[:, active_atoms], x_aux_tr])
            ridge = Ridge(alpha=fftboost_cfg["ridge_alpha"], fit_intercept=False)
            ridge.fit(current_features, y_tr)
            predictions = ridge.predict(current_features)
            residuals = y_tr - predictions

    final_model = Ridge(alpha=fftboost_cfg["ridge_alpha"], fit_intercept=True)
    if not active_atoms:
        final_model.fit(x_aux_tr, y_tr)
    else:
        final_features = np.hstack([x_fft_tr[:, active_atoms], x_aux_tr])
        final_model.fit(final_features, y_tr)

    fitted_state = {"active_atoms": active_atoms, "model": final_model}
    return fitted_state


def predict_with_fftboost_model(
    x_fft_va: np.ndarray, x_aux_va: np.ndarray, fitted_state: dict[str, Any]
) -> np.ndarray:
    active_atoms = fitted_state["active_atoms"]
    model = fitted_state["model"]

    if not active_atoms:
        return model.predict(x_aux_va)
    else:
        final_features = np.hstack([x_fft_va[:, active_atoms], x_aux_va])
        return model.predict(final_features)
