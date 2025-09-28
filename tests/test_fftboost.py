from typing import Any

import numpy as np
from fftboost.core.fftboost import fit_fftboost_model, predict_with_fftboost_model


def test_atom_selection_finds_perfect_feature() -> None:
    n_samples = 100
    n_fft_features = 20
    n_aux_features = 5
    perfect_feature_idx = 15

    rng = np.random.default_rng(1337)
    x_fft = rng.standard_normal((n_samples, n_fft_features))
    x_aux = rng.standard_normal((n_samples, n_aux_features))
    y = x_fft[:, perfect_feature_idx] * 2.0 + 0.1 * rng.standard_normal(n_samples)

    config: dict[str, Any] = {
        "fs": 1000,
        "fft_win_len": 400,
        "coherence_vec": np.ones(n_fft_features) * 0.5,
        "fftboost": {
            "atoms": 1,
            "lambda_hf": 0.0,
            "lambda_coh": 0.0,
            "min_sep_bins": 3,
            "ridge_alpha": 0.1,
        },
    }

    fitted_state = fit_fftboost_model(x_fft, x_aux, y, config)
    active_atoms = fitted_state["active_atoms"]

    assert len(active_atoms) == 1
    assert active_atoms[0] == perfect_feature_idx


def test_prediction_output_shape() -> None:
    n_samples_tr = 100
    n_samples_va = 50
    n_fft_features = 20
    n_aux_features = 5
    active_atoms = [3, 8, 12]

    rng = np.random.default_rng(1337)
    x_fft_tr = rng.standard_normal((n_samples_tr, n_fft_features))
    x_aux_tr = rng.standard_normal((n_samples_tr, n_aux_features))
    y_tr = rng.standard_normal(n_samples_tr)

    x_fft_va = rng.standard_normal((n_samples_va, n_fft_features))
    x_aux_va = rng.standard_normal((n_samples_va, n_aux_features))

    config: dict[str, Any] = {
        "fs": 1000,
        "fft_win_len": 400,
        "coherence_vec": np.ones(n_fft_features) * 0.5,
        "fftboost": {
            "atoms": len(active_atoms),
            "lambda_hf": 0.0,
            "lambda_coh": 0.0,
            "min_sep_bins": 1,
            "ridge_alpha": 0.1,
        },
    }

    fitted_state = fit_fftboost_model(x_fft_tr, x_aux_tr, y_tr, config)
    predictions = predict_with_fftboost_model(x_fft_va, x_aux_va, fitted_state)

    assert predictions.shape == (n_samples_va,)
