from __future__ import annotations

import numpy as np

from src.fftboost.config import FeatureConfig
from src.fftboost.features import _create_windows
from src.fftboost.features import _extract_features


def test_create_windows() -> None:
    signal = np.arange(100)
    config = FeatureConfig(fs=10, window_s=2.0, hop_s=1.0)
    windows = _create_windows(signal, config)
    assert windows.shape == (9, 20)
    np.testing.assert_array_equal(windows[0], np.arange(20))
    np.testing.assert_array_equal(windows[1], np.arange(10, 30))


def test_extract_features_shapes() -> None:
    fs = 1000
    signal = np.random.randn(10 * fs)
    config = FeatureConfig(
        fs=fs, window_s=0.5, hop_s=0.25, use_wavelets=True, use_hilbert_phase=True
    )
    x_fft, x_aux = _extract_features(signal, np.zeros_like(signal), config)
    win_len = int(config.window_s * fs)
    hop_len = int(config.hop_s * fs)
    expected_windows = (len(signal) - win_len) // hop_len + 1
    assert x_fft.shape[0] == expected_windows
    assert x_aux.shape[0] == expected_windows
    assert x_fft.shape[1] == win_len // 2
    expected_aux_features = config.wavelet_level + 2
    assert x_aux.shape[1] == expected_aux_features
