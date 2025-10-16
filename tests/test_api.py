from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.fftboost import BoosterConfig
from src.fftboost import FFTBoost


@pytest.fixture  # type: ignore[misc]
def synthetic_data() -> Any:
    fs = 4000.0
    duration_s = 5.0
    n_total = int(fs * duration_s)
    tvec = np.arange(n_total) / fs
    amplitude_modulation = 0.5 + 0.5 * np.cos(2 * np.pi * 0.2 * tvec)
    harmonics = np.sin(2 * np.pi * 50 * tvec) + 0.5 * np.sin(2 * np.pi * 100 * tvec)
    y_true_signal = amplitude_modulation * harmonics
    noise = 0.8 * np.random.randn(n_total)
    i_signal = y_true_signal + noise
    window_s = 0.5
    hop_s = 0.25
    win_len = int(window_s * fs)
    hop_len = int(hop_s * fs)
    n_windows = (n_total - win_len) // hop_len + 1
    windows = np.lib.stride_tricks.as_strided(
        y_true_signal,
        shape=(n_windows, win_len),
        strides=(y_true_signal.strides[0] * hop_len, y_true_signal.strides[0]),
    )
    y = np.sqrt(np.mean(windows**2, axis=1))
    return i_signal.astype(np.float64), y.astype(np.float64), fs, window_s, hop_s


def test_model_initialization() -> None:
    config = BoosterConfig(n_stages=10)
    model = FFTBoost(config)
    assert not model.is_fitted


def test_fit_predict_flow(synthetic_data: Any) -> None:
    i_signal, y, fs, window_s, hop_s = synthetic_data
    config = BoosterConfig(n_stages=5, min_sep_bins=2)
    model = FFTBoost(config)
    model.fit(i_signal, y, fs=fs, window_s=window_s, hop_s=hop_s, val_size=0.2)
    assert model.is_fitted
    predictions = model.predict(i_signal, fs=fs, window_s=window_s, hop_s=hop_s)
    expected_windows = (len(i_signal) - int(window_s * fs)) // int(hop_s * fs) + 1
    assert predictions.shape == (expected_windows,)


def test_unfitted_model_raises_error(synthetic_data: Any) -> None:
    i_signal, _, fs, window_s, hop_s = synthetic_data
    model = FFTBoost(BoosterConfig())
    with pytest.raises(RuntimeError):
        model.predict(i_signal, fs=fs, window_s=window_s, hop_s=hop_s)
