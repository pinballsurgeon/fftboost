from __future__ import annotations

import numpy as np

from fftboost import BoosterConfig
from fftboost import FFTBoost


def test_model_solves_known_problem() -> None:
    """
    This is our new, consolidated acceptance test.
    It proves that the model can be imported and can solve a simple, known
    signal processing task with a high degree of accuracy.
    This replaces the old CLI and notebook-based CI steps.
    """
    # 1. ARRANGE: Create a simple, solvable dataset
    fs = 2000
    duration_s = 10
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    amplitude_modulation = 1.0 + np.sin(2 * np.pi * 0.2 * t)
    y_true_signal = amplitude_modulation * np.sin(2 * np.pi * 50 * t)
    i_signal = y_true_signal + 0.5 * np.random.randn(n_samples)

    # 2. ACT: Configure and run the model using the public API
    config = BoosterConfig(n_stages=5, nu=0.5, ridge_alpha=1e-3)

    # Create the target variable
    win_len = int(0.25 * fs)
    hop_len = int(0.1 * fs)
    n_windows = (y_true_signal.shape[0] - win_len) // hop_len + 1
    shape = (n_windows, win_len)
    strides = (y_true_signal.strides[0] * hop_len, y_true_signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        y_true_signal, shape=shape, strides=strides
    )
    y_target = np.sqrt(np.mean(windows**2, axis=1))

    model = FFTBoost(config)
    model.fit(i_signal, y_target, fs=float(fs), window_s=0.25, hop_s=0.1, val_size=0.2)
    predictions = model.predict(i_signal, fs=float(fs), window_s=0.25, hop_s=0.1)

    # 3. ASSERT: Check if the R^2 score meets acceptance criteria
    y_trimmed = y_target[: len(predictions)]

    corr_matrix = np.corrcoef(y_trimmed, predictions)
    correlation = corr_matrix[0, 1]
    r_squared = correlation**2

    print(f"\nAcceptance Test R^2 score: {r_squared:.4f}")
    assert (
        r_squared > 0.8
    ), "ACCEPTANCE GATE FAILED: Model R^2 score is below the 0.8 threshold."
