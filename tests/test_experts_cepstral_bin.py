from __future__ import annotations

from typing import Any

import numpy as np

from fftboost.experts.cepstral_bin import propose
from fftboost.experts.types import ExpertContext


def generate_harmonic_signal(
    fs: float, duration: float, f0: float, n_harmonics: int, amp: float = 1.0
) -> np.ndarray[Any, Any]:
    """Generates a signal with a fundamental frequency f0 and its harmonics."""
    t = np.arange(int(fs * duration)) / fs
    signal = np.zeros_like(t)
    for i in range(1, n_harmonics + 1):
        signal += (amp / i) * np.sin(2 * np.pi * (f0 * i) * t)
    return signal


def test_cepstral_bin_identifies_harmonic_series() -> None:
    """
    Verify that the cepstral_bin expert can identify the fundamental
    frequency of a harmonic series.
    """
    fs = 4000.0
    window_s = 0.256
    hop_s = 0.128
    win_len = int(fs * window_s)
    hop_len = int(fs * hop_s)
    f0 = 100.0  # Fundamental frequency

    signal = generate_harmonic_signal(fs, duration=5.0, f0=f0, n_harmonics=10)

    # Create windows
    n_win = (len(signal) - win_len) // hop_len + 1
    windows = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_win, win_len),
        strides=(signal.strides[0] * hop_len, signal.strides[0]),
    )

    # Create a simple ramp target
    residual = np.linspace(0, 1, n_win)

    # Prepare context
    psd = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]
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
    assert descriptor["type"] == "cepstral_bin"

    # The expert should pick the quefrency corresponding to the fundamental
    # frequency's period.
    selected_quefrency_s = float(descriptor["quefrency_s"])
    expected_quefrency_s = 1.0 / f0  # Period of the fundamental frequency

    # We allow for some tolerance due to discretization
    # The resolution of the cepstrum is related to the sampling frequency
    assert np.isclose(selected_quefrency_s, expected_quefrency_s, atol=1.0 / fs)
