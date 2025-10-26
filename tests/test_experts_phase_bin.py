from __future__ import annotations

from typing import Any

import numpy as np

from fftboost.experts.phase_bin import propose
from fftboost.experts.types import ExpertContext


def generate_fm_signal(
    fs: float, duration: float, f_start: float, f_end: float, amp: float = 1.0
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generates a frequency-modulated (chirp) signal and its target modulation."""
    t = np.arange(int(fs * duration)) / fs
    # Instantaneous frequency: linear sweep from f_start to f_end
    inst_freq = np.linspace(f_start, f_end, len(t))
    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    signal = amp * np.sin(phase)
    return signal, inst_freq


def test_phase_bin_identifies_fm_signal() -> None:
    """
    Verify that the phase_bin expert can identify a frequency-modulated signal.
    """
    fs = 2000.0
    window_s = 0.256
    hop_s = 0.128
    win_len = int(fs * window_s)
    hop_len = int(fs * hop_s)

    # Signal: A chirp from 90 Hz to 110 Hz, centered at 100 Hz
    # Target: The instantaneous frequency, which is what the expert should detect
    signal, inst_freq_target = generate_fm_signal(
        fs, duration=10.0, f_start=90.0, f_end=110.0
    )

    # Create windows
    n_win = (len(signal) - win_len) // hop_len + 1
    windows = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_win, win_len),
        strides=(signal.strides[0] * hop_len, signal.strides[0]),
    )

    # Create a target `y` that is correlated with the modulation
    y = np.linspace(0, 1, n_win)  # Simple ramp target

    # The residual is the negative gradient, for a simple squared loss it's
    # y_true - y_pred. Here, we assume y_pred is zero, so residual is just y.
    residual = y

    # Prepare context for the expert
    complex_rfft = np.fft.rfft(windows, axis=1)[:, 1:]
    psd = np.abs(complex_rfft)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)[1:]

    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=3,
        lambda_hf=0.0,
        complex_rfft=complex_rfft,
        raw_windows=windows,
    )

    # Run the proposal
    proposal = propose(residual, ctx, top_k=1)

    # --- Assertions ---
    assert len(proposal.descriptors) == 1
    descriptor = proposal.descriptors[0]
    assert descriptor["type"] == "phase_bin"

    # The expert should pick the frequency bin closest to the center of the chirp
    # (100 Hz).
    selected_freq = float(descriptor["freq_hz"])
    center_freq = 100.0
    bin_resolution = fs / win_len
    assert abs(selected_freq - center_freq) <= bin_resolution

    # The feature matrix H should contain the instantaneous frequency deviation
    # which should be non-zero for a chirp signal.
    assert proposal.H.shape == (n_win, 1)
    assert np.std(proposal.H) > 1e-3  # Check that it's not a constant value
