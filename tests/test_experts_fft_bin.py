from __future__ import annotations

import time
from typing import cast

import numpy as np

from src.fftboost.experts.fft_bin import propose as fftbin_propose
from src.fftboost.experts.types import ExpertContext


def _make_psd_with_modulated_tone(
    n_windows: int, win_len: int, fs: int, f0: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(win_len) / fs
    freqs_full = np.fft.rfftfreq(win_len, d=1.0 / fs)
    freqs = freqs_full[1:]  # drop DC to match typical features

    psd = np.zeros((n_windows, freqs.shape[0]), dtype=np.float64)
    amps = rng.uniform(0.5, 1.5, size=n_windows).astype(np.float64)
    for w in range(n_windows):
        x = amps[w] * np.sin(2.0 * np.pi * f0 * t) + 0.1 * rng.standard_normal(win_len)
        X = np.fft.rfft(x)
        mag = np.abs(X)[1:]
        psd[w] = mag
    residual = amps - float(np.mean(amps))
    return psd, freqs, residual


def test_signal_recovery_fft_bin_identifies_tone() -> None:
    rng = np.random.default_rng(123)
    fs = 1000
    win_len = 512
    n_windows = 256
    f0 = 50.0
    psd, freqs, residual = _make_psd_with_modulated_tone(
        n_windows, win_len, fs, f0, rng
    )
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=2,
        lambda_hf=0.0,
        selected_bins=None,
    )
    proposal = fftbin_propose(residual, ctx, top_k=1)
    assert proposal.H.shape == (n_windows, 1)
    picked_freq = cast(float, proposal.descriptors[0]["freq_hz"])
    bin_res = fs / win_len
    assert abs(picked_freq - f0) <= bin_res


def test_performance_fft_bin_throughput() -> None:
    rng = np.random.default_rng(42)
    fs = 1000
    win_len = 256
    n_windows = 20000
    f0 = 60.0
    psd, freqs, residual = _make_psd_with_modulated_tone(
        n_windows, win_len, fs, f0, rng
    )
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=2,
        lambda_hf=0.1,
        selected_bins=None,
    )

    t0 = time.perf_counter()
    _ = fftbin_propose(residual, ctx, top_k=8)
    elapsed = time.perf_counter() - t0
    throughput = n_windows / max(elapsed, 1e-9)
    assert throughput > 20000.0


def test_determinism_fft_bin() -> None:
    rng = np.random.default_rng(7)
    fs = 1000
    win_len = 256
    n_windows = 512
    f0 = 75.0
    psd, freqs, residual = _make_psd_with_modulated_tone(
        n_windows, win_len, fs, f0, rng
    )
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=fs,
        min_sep_bins=1,
        lambda_hf=0.05,
        selected_bins=np.array([10], dtype=np.int64),
    )
    p1 = fftbin_propose(residual, ctx, top_k=5)
    p2 = fftbin_propose(residual, ctx, top_k=5)
    np.testing.assert_array_equal(p1.H, p2.H)
    assert p1.descriptors == p2.descriptors
    np.testing.assert_allclose(p1.mu, p2.mu)
    np.testing.assert_allclose(p1.sigma, p2.sigma)
