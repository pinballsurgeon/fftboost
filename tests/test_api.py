from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.fftboost import FeatureConfig
from src.fftboost import FFTBoost
from src.fftboost import FFTBoostConfig


@pytest.fixture  # type: ignore[misc]
def synthetic_data() -> Any:
    fs = 4000
    duration_s = 5
    n_total = fs * duration_s
    tvec = np.arange(n_total) / fs
    amplitude_modulation = 0.5 + 0.5 * np.cos(2 * np.pi * 0.2 * tvec)
    harmonics = np.sin(2 * np.pi * 50 * tvec) + 0.5 * np.sin(2 * np.pi * 100 * tvec)
    y_true_signal = amplitude_modulation * harmonics
    noise = 0.8 * np.random.randn(n_total)
    i_signal = y_true_signal + noise
    feature_config = FeatureConfig(fs=fs, window_s=0.5, hop_s=0.25)
    win_len = int(feature_config.window_s * fs)
    hop_len = int(feature_config.hop_s * fs)
    n_windows = (n_total - win_len) // hop_len + 1
    windows = np.lib.stride_tricks.as_strided(
        y_true_signal,
        shape=(n_windows, win_len),
        strides=(y_true_signal.strides[0] * hop_len, y_true_signal.strides[0]),
    )
    y = np.sqrt(np.mean(windows**2, axis=1))
    return i_signal, y, feature_config


def test_model_initialization() -> None:
    fftboost_config = FFTBoostConfig(atoms=10)
    feature_config = FeatureConfig(fs=1000)
    model = FFTBoost(fftboost_config, feature_config)
    assert model.fftboost_config.atoms == 10
    assert model.feature_config.fs == 1000
    assert not model.is_fitted


def test_fit_predict_flow(synthetic_data: Any) -> None:
    i_signal, y, feature_config = synthetic_data
    fftboost_config = FFTBoostConfig(atoms=5, min_sep_bins=2)
    model = FFTBoost(fftboost_config, feature_config)
    model.fit(i_signal, y)
    assert model.is_fitted
    assert "active_atoms" in model._fitted_state
    assert "final_model" in model._fitted_state
    predictions = model.predict(i_signal)
    expected_windows = (
        len(i_signal) - int(feature_config.window_s * feature_config.fs)
    ) // int(feature_config.hop_s * feature_config.fs) + 1
    assert predictions.shape == (expected_windows,)


def test_unfitted_model_raises_error(synthetic_data: Any) -> None:
    i_signal, _, feature_config = synthetic_data
    fftboost_config = FFTBoostConfig()
    model = FFTBoost(fftboost_config, feature_config)
    with pytest.raises(RuntimeError):
        model.predict(i_signal)
