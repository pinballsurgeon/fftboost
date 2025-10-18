from __future__ import annotations

from typing import cast

import numpy as np

from src.fftboost.experts.clf_bin import propose as clfbin_propose
from src.fftboost.experts.types import ExpertContext


def _make_psd_with_class_tone(
    n_windows: int, win_len: int, fs: int, f0: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(win_len) / fs
    freqs_full = np.fft.rfftfreq(win_len, d=1.0 / fs)
    freqs = freqs_full[1:]
    psd = np.zeros((n_windows, freqs.shape[0]), dtype=np.float64)
    labels = np.zeros(n_windows, dtype=np.float64)
    # Class 1 windows contain the tone; class 0 do not
    ones = np.arange(0, n_windows, 3)
    labels[ones] = 1.0
    for w in range(n_windows):
        amp = 1.2 if labels[w] > 0.5 else 0.0
        x = amp * np.sin(2.0 * np.pi * f0 * t) + 0.1 * rng.standard_normal(win_len)
        X = np.fft.rfft(x)
        mag = np.abs(X)[1:]
        psd[w] = mag
    return psd, freqs, labels


def test_clf_bin_identifies_discriminative_bin() -> None:
    rng = np.random.default_rng(123)
    fs = 1000
    win_len = 512
    n_windows = 256
    f0 = 75.0
    psd, freqs, labels = _make_psd_with_class_tone(n_windows, win_len, fs, f0, rng)
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=2,
        lambda_hf=0.0,
        selected_bins=None,
        y_labels=labels,
    )
    prop = clfbin_propose(
        np.zeros(n_windows, dtype=np.float64), ctx, top_k=1, method="fscore"
    )
    assert prop.H.shape == (n_windows, 1)
    picked = cast(float, prop.descriptors[0]["freq_hz"])
    bin_res = fs / win_len
    assert abs(picked - f0) <= bin_res


def test_clf_bin_determinism() -> None:
    rng = np.random.default_rng(7)
    fs = 1000
    win_len = 256
    n_windows = 300
    f0 = 60.0
    psd, freqs, labels = _make_psd_with_class_tone(n_windows, win_len, fs, f0, rng)
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=1,
        lambda_hf=0.0,
        selected_bins=None,
        y_labels=labels,
    )
    p1 = clfbin_propose(np.zeros(n_windows), ctx, top_k=5, method="fscore")
    p2 = clfbin_propose(np.zeros(n_windows), ctx, top_k=5, method="fscore")
    np.testing.assert_array_equal(p1.H, p2.H)
    assert p1.descriptors == p2.descriptors
    np.testing.assert_allclose(p1.mu, p2.mu)
    np.testing.assert_allclose(p1.sigma, p2.sigma)
