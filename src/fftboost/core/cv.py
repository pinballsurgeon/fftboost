from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from data.synthetic_generator import generate_synthetic_signals
from fftboost.core.features import (
    compute_fft,
    compute_labels,
    create_windows,
    extract_features_from_signals,
)
from fftboost.core.fftboost import fit_fftboost_model, predict_with_fftboost_model
from fftboost.utils.scaling import robust_scale, zscale


def run_full_cv_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    seed = config["seed"]
    i_signal, v_signal, _ = generate_synthetic_signals(config, seed)
    x_fft_all, x_aux_all, _ = extract_features_from_signals(i_signal, v_signal, config)

    iwins = create_windows(i_signal, config)
    _, freqs_full = compute_fft(iwins, config["fs"], iwins.shape[1])
    y_all = compute_labels(iwins, freqs_full, config)

    outer_cv = TimeSeriesSplit(n_splits=config["cv_folds"])
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(x_fft_all)):
        x_fft_tr, x_fft_va = x_fft_all[tr_idx], x_fft_all[va_idx]
        x_aux_tr, x_aux_va = x_aux_all[tr_idx], x_aux_all[va_idx]
        y_tr_raw, y_va_raw = y_all[tr_idx], y_all[va_idx]

        x_fft_tr_s, mu_fft, sd_fft = zscale(x_fft_tr)
        x_fft_va_s, _, _ = zscale(x_fft_va, mu_fft, sd_fft)
        x_aux_tr_s, mu_aux, sd_aux = zscale(x_aux_tr)
        x_aux_va_s, _, _ = zscale(x_aux_va, mu_aux, sd_aux)
        y_tr, _, _ = robust_scale(y_tr_raw)
        y_va, _, _ = robust_scale(y_va_raw, ref=y_tr_raw)

        base_model = Ridge(alpha=1.0)
        base_model.fit(np.hstack([x_fft_tr_s, x_aux_tr_s]), y_tr)
        y_gbdt_va = base_model.predict(np.hstack([x_fft_va_s, x_aux_va_s]))

        fit_config = config.copy()
        coh_placeholder = np.ones(x_fft_tr.shape[1]) * 0.5
        fit_config["coherence_vec"] = np.mean(np.tile(coh_placeholder, (len(tr_idx), 1)), axis=0)
        fit_config["fft_win_len"] = int(config["window_s"] * config["fs"])

        fitted_state = fit_fftboost_model(x_fft_tr_s, x_aux_tr_s, y_tr, fit_config)
        y_fft_va = predict_with_fftboost_model(x_fft_va_s, x_aux_va_s, fitted_state)

        r2_g = r2_score(y_va, y_gbdt_va)
        r2_f = r2_score(y_va, y_fft_va)
        fold_results.append({"R2_GBDT": r2_g, "R2_FFT": r2_f, "dFFT": r2_f - r2_g})

    return {"fold_results": fold_results}
