from __future__ import annotations

import os

import numpy as np
import pytest

from src.fftboost.api import FFTBoost
from src.fftboost.booster import BoosterConfig


STRICT = os.getenv("FFTBOOST_STRICT_GAUNTLET", "0") == "1"


def test_fft_only_regression_baseline_easy() -> None:
    if not STRICT:
        pytest.xfail(
            "FFT-only baseline exploratory threshold; "
            "set FFTBOOST_STRICT_GAUNTLET=1 to enforce"
        )
    fs = 200.0
    t = np.arange(0.0, 5.0, 1.0 / fs, dtype=np.float64)
    x = np.sin(2 * np.pi * 10.0 * t) + 0.2 * np.random.default_rng(0).standard_normal(
        t.shape[0]
    )
    window_s, hop_s = 0.5, 0.25
    win_len = int(window_s * fs)
    hop = int(hop_s * fs)
    n = x.shape[0]
    n_win = (n - win_len) // hop + 1
    W = np.lib.stride_tricks.as_strided(
        x, shape=(n_win, win_len), strides=(x.strides[0] * hop, x.strides[0])
    )
    y = np.sqrt(np.mean(W**2, axis=1))

    cfg = BoosterConfig(
        loss="huber",
        n_stages=64,
        early_stopping_rounds=10,
        nu=0.3,
        k_fft=8,
        min_sep_bins=2,
        temporal_use=False,
        lambda_hf=0.05,
    )
    model = FFTBoost(cfg).fit(
        x,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        val_gap_windows=1,
        center_target=True,
    )
    y_hat = model.predict(x, fs=fs, window_s=window_s, hop_s=hop_s)
    m = min(y.shape[0], y_hat.shape[0])
    y0, p0 = y[:m], y_hat[:m]
    c = float(np.corrcoef(y0, p0)[0, 1]) if m > 1 else 0.0
    r2 = float(c * c)
    assert r2 >= 0.6
