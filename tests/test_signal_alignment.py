from __future__ import annotations

import numpy as np


def _make_signal(
    fs: float, duration: float, freq: float, noise: float, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float64) / fs
    x = np.sin(2 * np.pi * float(freq) * t)
    return x + float(noise) * rng.standard_normal(n)


def test_correlation_top_bin_matches_expected_frequency() -> None:
    fs = 200.0
    window_s, hop_s = 0.5, 0.25
    f0 = 10.0
    x = _make_signal(fs, 5.0, f0, noise=0.1, seed=7)

    win_len = int(window_s * fs)
    hop = int(hop_s * fs)
    n = x.shape[0]
    n_win = (n - win_len) // hop + 1
    shape = (n_win, win_len)
    strides = (x.strides[0] * hop, x.strides[0])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    psd = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)[1:]

    # RMS target aligned to the same windows
    y = np.sqrt(np.mean(windows**2, axis=1))

    # Compute per-bin correlation with the target
    rz = (y - y.mean()) / (y.std() + 1e-12)
    Z = (psd - psd.mean(axis=0)) / (psd.std(axis=0) + 1e-12)
    scores = np.abs(rz @ Z) / float(n_win)
    top_idx = int(np.argmax(scores))
    top_freq = float(freqs[top_idx])

    # Assert top frequency within one bin of f0
    bin_res = fs / win_len
    assert abs(top_freq - f0) <= bin_res
