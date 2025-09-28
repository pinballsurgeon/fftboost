from typing import Any, cast

import numpy as np
from fftboost.core import features


def test_windowing_output_shape_and_count() -> None:
    config: dict[str, Any] = {"fs": 1000, "duration_s": 10, "window_s": 0.4, "hop_s": 0.2}
    signal = np.random.randn(cast(int, config["duration_s"]) * cast(int, config["fs"]))
    windows = features.create_windows(signal, config)

    win_len = int(cast(float, config["window_s"]) * cast(int, config["fs"]))
    hop_len = int(cast(float, config["hop_s"]) * cast(int, config["fs"]))
    expected_n_windows = (len(signal) - win_len) // hop_len + 1

    assert windows.shape == (expected_n_windows, win_len)


def test_fft_peak_location() -> None:
    fs = 1000
    win_len = 1000
    test_freq = 150.0
    t = np.arange(win_len) / fs
    signal = np.sin(2 * np.pi * test_freq * t)
    windows = np.array([signal])

    fft_features, freqs = features.compute_fft(windows, fs, win_len)
    peak_index = np.argmax(fft_features[0, :])
    detected_freq = freqs[peak_index]

    assert np.isclose(detected_freq, test_freq, atol=1.0)


def test_feature_shapes() -> None:
    n_windows = 10
    win_len = 400
    fs = 1000
    config: dict[str, Any] = {"wavelet_family": "db4", "wavelet_level": 4}
    windows = np.random.randn(n_windows, win_len)
    fft_features = np.random.rand(n_windows, win_len // 2)
    freqs = np.linspace(0, fs / 2, win_len // 2)

    wavelet_energies = features.compute_wavelet_energies(windows, config)
    wavelet_level = cast(int, config["wavelet_level"])
    assert wavelet_energies.shape == (n_windows, wavelet_level + 1)

    hilbert_stats = features.compute_hilbert_phase_stats(windows, fs)
    assert hilbert_stats.shape == (n_windows, 3)

    spectral_moments = features.compute_spectral_moments(fft_features, freqs)
    assert spectral_moments.shape == (n_windows, 2)


def test_orchestrator_output_shapes() -> None:
    config: dict[str, Any] = {
        "fs": 1000,
        "duration_s": 10,
        "window_s": 0.4,
        "hop_s": 0.2,
        "use_wavelets": True,
        "wavelet_family": "db4",
        "wavelet_level": 4,
        "use_hilbert_phase": True,
        "coherence_subbands": [[1, 40], [40, 100]],
    }
    i_signal = np.random.randn(cast(int, config["duration_s"]) * cast(int, config["fs"]))
    v_signal = np.random.randn(cast(int, config["duration_s"]) * cast(int, config["fs"]))

    x_fft, x_aux, freqs = features.extract_features_from_signals(i_signal, v_signal, config)

    win_len = int(cast(float, config["window_s"]) * cast(int, config["fs"]))
    hop_len = int(cast(float, config["hop_s"]) * cast(int, config["fs"]))
    expected_n_windows = (len(i_signal) - win_len) // hop_len + 1
    expected_n_fft_bins = win_len // 2

    assert x_fft.shape == (expected_n_windows, expected_n_fft_bins)
    assert freqs.shape == (expected_n_fft_bins,)

    wavelet_level = cast(int, config["wavelet_level"])
    coherence_subbands = cast(list[list[int]], config["coherence_subbands"])
    expected_aux_cols = (wavelet_level + 1) + 3 + 2 + len(coherence_subbands)
    assert x_aux.shape == (expected_n_windows, expected_aux_cols)
