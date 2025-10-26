from __future__ import annotations

from typing import Any

import numpy as np

from fftboost.experts.shape_props import propose
from fftboost.experts.types import ExpertContext


def generate_band_limited_noise(
    fs: float, duration: float, f_min: float, f_max: float
) -> np.ndarray[Any, Any]:
    """Generates noise signal with energy concentrated in a specific band."""
    n_samples = int(fs * duration)
    noise = np.random.randn(n_samples)
    fft_noise = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Create a band-pass filter
    fft_noise[(freqs < f_min) | (freqs > f_max)] = 0

    return np.fft.irfft(fft_noise)


def test_shape_props_identifies_centroid_shift() -> None:
    """
    Verify that the shape_props expert can detect a shift in spectral centroid.
    """
    fs = 2000.0
    window_s = 0.256
    win_len = int(fs * window_s)

    # Create two sets of windows: one low-frequency, one high-frequency
    low_freq_signal = generate_band_limited_noise(
        fs, duration=5.0, f_min=100, f_max=200
    )
    high_freq_signal = generate_band_limited_noise(
        fs, duration=5.0, f_min=800, f_max=900
    )

    # Create windows from signals
    n_win_low = (len(low_freq_signal) - win_len) // win_len + 1
    windows_low = np.lib.stride_tricks.as_strided(
        low_freq_signal,
        shape=(n_win_low, win_len),
        strides=(low_freq_signal.strides[0] * win_len, low_freq_signal.strides[0]),
    )
    n_win_high = (len(high_freq_signal) - win_len) // win_len + 1
    windows_high = np.lib.stride_tricks.as_strided(
        high_freq_signal,
        shape=(n_win_high, win_len),
        strides=(high_freq_signal.strides[0] * win_len, high_freq_signal.strides[0]),
    )

    all_windows = np.vstack([windows_low, windows_high])

    # Create a binary target that distinguishes the two signal types
    y = np.array([0] * n_win_low + [1] * n_win_high)
    residual = y - y.mean()

    # Prepare context
    psd = np.abs(np.fft.rfft(all_windows, axis=1))[:, 1:]
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)[1:]

    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=0,
        lambda_hf=0.0,
    )

    # Run the proposal
    proposal = propose(residual, ctx, top_k=1)

    # --- Assertions ---
    assert len(proposal.descriptors) == 1
    descriptor = proposal.descriptors[0]

    # The most discriminative feature should be the spectral centroid
    assert descriptor["type"] == "shape_prop"
    assert descriptor["name"] == "centroid"

    # The feature values should be significantly different for the two classes
    centroid_values = proposal.H.flatten()
    mean_low = centroid_values[:n_win_low].mean()
    mean_high = centroid_values[n_win_low:].mean()

    assert mean_high > mean_low
    assert (mean_high - mean_low) > 500  # Expect a large difference in centroid
