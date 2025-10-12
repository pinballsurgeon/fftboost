from __future__ import annotations

import time
from typing import cast

import numpy as np

from src.fftboost.experts.sk_band import propose as skband_propose
from src.fftboost.experts.types import ExpertContext


def _make_psd_with_band_impulse(
    n_windows: int,
    n_bins: int,
    band_idx: tuple[int, int],
    impulse_windows: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    # Build freqs as 1..n_bins Hz for simplicity (bin width = 1 Hz)
    freqs = np.arange(1, n_bins + 1, dtype=np.float64)
    psd = rng.random((n_windows, n_bins), dtype=np.float64) * 0.1
    # Inject impulse energy across the target band for selected windows
    lo_i, hi_i = band_idx
    for w in impulse_windows:
        psd[w, lo_i : hi_i + 1] += 5.0
    # Residual: 1 for impulse windows, 0 else
    residual = np.zeros(n_windows, dtype=np.float64)
    residual[np.array(impulse_windows, dtype=np.int64)] = 1.0
    # Define bands in Hz to match indices (bin i corresponds to freq i+1)
    bands = [(float(lo_i + 1), float(hi_i + 1))]
    # Add some distractor bands
    bands += [(10.0, 20.0), (40.0, 50.0), (80.0, 90.0)]
    return psd, freqs, residual, bands


def test_signal_recovery_sk_band_identifies_impulse_band() -> None:
    rng = np.random.default_rng(123)
    n_windows = 512
    n_bins = 200
    target_band = (60, 80)  # indices inclusive
    impulse_windows = [100, 140, 300]
    psd, freqs, residual, bands = _make_psd_with_band_impulse(
        n_windows, n_bins, target_band, impulse_windows, rng
    )
    # fs not used by sk_band, but ExpertContext requires it
    fs = 2.0 * float(freqs.max())  # Nyquist maps to max(freqs)
    # Use only the target band edges to avoid unintended sub-band splits
    edges = np.array([bands[0][0], bands[0][1]], dtype=np.float64)
    ctx = ExpertContext(
        psd=psd, freqs=freqs, fs=fs, min_sep_bins=0, lambda_hf=0.0, band_edges_hz=edges
    )
    proposal = skband_propose(residual, ctx, n_select=1, kurtosis_boost=0.5)
    assert proposal.H.shape == (n_windows, 1)
    top_desc = proposal.descriptors[0]
    assert top_desc["type"] == "sk_band"
    # Should pick the target band (lo,hi) in Hz
    picked = cast(tuple[float, float], top_desc["band_hz"])
    true_band_hz = (float(target_band[0] + 1), float(target_band[1] + 1))
    assert picked == true_band_hz


def test_performance_sk_band_throughput() -> None:
    rng = np.random.default_rng(7)
    n_windows = 20000
    n_bins = 128
    target_band = (20, 40)
    impulse_windows = list(range(0, n_windows, 500))
    psd, freqs, residual, bands = _make_psd_with_band_impulse(
        n_windows, n_bins, target_band, impulse_windows, rng
    )
    fs = 2.0 * float(freqs.max())
    edges = np.unique(
        np.array(bands + [(1.0, 5.0), (90.0, 100.0)], dtype=np.float64).reshape(-1)
    )
    ctx = ExpertContext(
        psd=psd, freqs=freqs, fs=fs, min_sep_bins=0, lambda_hf=0.0, band_edges_hz=edges
    )

    t0 = time.perf_counter()
    _ = skband_propose(residual, ctx, n_select=2, kurtosis_boost=0.5)
    elapsed = time.perf_counter() - t0
    throughput = n_windows / max(elapsed, 1e-9)
    assert throughput > 20000.0


def test_determinism_sk_band() -> None:
    rng = np.random.default_rng(11)
    n_windows = 300
    n_bins = 150
    target_band = (30, 60)
    impulse_windows = [10, 100, 200]
    psd, freqs, residual, bands = _make_psd_with_band_impulse(
        n_windows, n_bins, target_band, impulse_windows, rng
    )
    fs = 2.0 * float(freqs.max())
    edges = np.unique(np.array(bands, dtype=np.float64).reshape(-1))
    ctx = ExpertContext(
        psd=psd, freqs=freqs, fs=fs, min_sep_bins=0, lambda_hf=0.0, band_edges_hz=edges
    )
    p1 = skband_propose(residual, ctx, n_select=3, kurtosis_boost=0.2)
    p2 = skband_propose(residual, ctx, n_select=3, kurtosis_boost=0.2)
    np.testing.assert_array_equal(p1.H, p2.H)
    assert p1.descriptors == p2.descriptors
    np.testing.assert_allclose(p1.mu, p2.mu)
    np.testing.assert_allclose(p1.sigma, p2.sigma)
