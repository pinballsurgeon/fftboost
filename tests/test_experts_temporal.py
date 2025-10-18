from __future__ import annotations

from typing import cast

import numpy as np

from src.fftboost.experts.temporal import propose_flux
from src.fftboost.experts.temporal import propose_lagstack
from src.fftboost.experts.types import ExpertContext


def _make_psd_changing_bin(
    n_windows: int, n_bins: int, target_bin: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.linspace(1.0, float(n_bins), n_bins, dtype=np.float64)
    psd = rng.random((n_windows, n_bins), dtype=np.float64) * 0.1
    # Inject a ramp in the target bin so flux/lag correlates with residual
    ramp = np.linspace(0.0, 1.0, n_windows)
    psd[:, target_bin] += ramp
    residual = ramp + 0.01 * rng.standard_normal(n_windows)
    return psd, freqs, residual


def test_temporal_flux_picks_changing_bin() -> None:
    rng = np.random.default_rng(42)
    n_windows, n_bins, tb = 300, 64, 10
    psd, freqs, residual = _make_psd_changing_bin(n_windows, n_bins, tb, rng)
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=2.0 * float(freqs.max()),
        min_sep_bins=1,
        lambda_hf=0.0,
    )
    prop = propose_flux(residual, ctx, top_k=1)
    picked_bin = cast(int, prop.descriptors[0]["bin"])
    # Allow a small tolerance: pick must be within ±1 bin of target
    assert abs(picked_bin - tb) <= 1


def test_temporal_lagstack_uses_selected_bins() -> None:
    rng = np.random.default_rng(7)
    n_windows, n_bins, tb = 256, 50, 12
    psd, freqs, residual = _make_psd_changing_bin(n_windows, n_bins, tb, rng)
    ctx = ExpertContext(
        psd=psd,
        freqs=freqs,
        fs=2.0 * float(freqs.max()),
        min_sep_bins=1,
        lambda_hf=0.0,
        selected_bins=np.array([tb], dtype=np.int64),
    )
    prop = propose_lagstack(
        residual, ctx, bins=np.array([tb], dtype=np.int64), lags=(1, 2), top_k=1
    )
    assert prop.H.shape == (n_windows, 1)
    d = prop.descriptors[0]
    assert d["type"] == "lag_bin"
    assert cast(int, d["bin"]) == tb
