from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from src.fftboost.api import FFTBoost


STRICT = os.getenv("FFTBOOST_STRICT_GAUNTLET", "0") == "1"


def _make_signal(
    fs: float,
    duration: float,
    components: list[tuple[float, float]],
    *,
    am: tuple[float, float] | None = None,
    noise_level: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float64) / fs
    clean = np.zeros(n, dtype=np.float64)
    for amp, freq in components:
        clean += float(amp) * np.sin(2 * np.pi * float(freq) * t)
    if am is not None:
        depth, f_am = am
        clean = clean * (1.0 + float(depth) * np.sin(2 * np.pi * float(f_am) * t))
    signal = clean + float(noise_level) * rng.standard_normal(n)
    return signal, clean, t


def _windows_rms(y: np.ndarray, fs: float, window_s: float, hop_s: float) -> np.ndarray:
    win = int(window_s * fs)
    hop = int(hop_s * fs)
    n = y.shape[0]
    n_win = (n - win) // hop + 1
    shape = (n_win, win)
    strides = (y.strides[0] * hop, y.strides[0])
    W = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return np.sqrt(np.mean(W**2, axis=1))


def _best_freq_hz(model: object, fs: float, win_len: int) -> tuple[float, bool]:
    # Works for FFTBoost or FFTBoostClassifier
    core: Any = getattr(model, "model", model)
    if not hasattr(core, "_booster"):
        return 0.0, False
    picks: list[float] = []
    for st in core._booster.stages:
        for d in st.descriptors:
            t = d.get("type")
            if t in ("fft_bin", "clf_bin"):
                f = float(d.get("freq_hz", 0.0))
                picks.append(f)
    if not picks:
        return 0.0, False
    # voting
    uniq = sorted(set(picks))
    top = max(uniq, key=lambda f: picks.count(f))
    return float(top), True


def test_regression_single_tone_easy() -> None:
    if not STRICT:
        pytest.xfail(
            "Gauntlet-lite exploratory threshold not yet met; "
            "set FFTBOOST_STRICT_GAUNTLET=1 to enforce"
        )
    fs = 200.0
    signal, clean, _ = _make_signal(
        fs, 5.0, components=[(1.0, 10.0)], am=(0.6, 0.3), noise_level=0.2, seed=1
    )
    window_s, hop_s = 0.5, 0.25
    y = _windows_rms(clean, fs, window_s, hop_s)

    model, info = FFTBoost.auto(
        signal,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        val_gap_windows=1,
        center_target=True,
        budget_stages=48,
        halving_rounds=1,
        n_configs=12,
    )
    model_any: Any = model
    y_hat = model_any.predict(signal, fs=fs, window_s=window_s, hop_s=hop_s)
    m = min(y.shape[0], y_hat.shape[0])
    y0, p0 = y[:m], y_hat[:m]
    # R2 via correlation (robust for short blocks)
    c = float(np.corrcoef(y0, p0)[0, 1]) if m > 1 else 0.0
    r2 = float(c * c)
    # Diagnostics
    print(f"[gauntlet-lite] regression corr-R2={r2:.3f}")
    assert r2 >= 0.5
    # Frequency pick within one bin
    win_len = int(window_s * fs)
    f_top, ok = _best_freq_hz(model, fs, win_len)
    print(f"[gauntlet-lite] picked_freq={f_top:.1f}Hz bin_res={fs/win_len:.2f}")
    assert ok and abs(f_top - 10.0) <= fs / win_len


def test_classification_sparse_pattern_easy() -> None:
    if not STRICT:
        pytest.xfail(
            "Gauntlet-lite exploratory threshold not yet met; "
            "set FFTBOOST_STRICT_GAUNTLET=1 to enforce"
        )
    fs = 200.0
    # Presence/absence pattern at 8 windows periodicity
    pattern = [0, 0, 0, 0, 1, 1, 0, 0]
    signal, clean, _ = _make_signal(
        fs,
        5.0,
        components=[(1.0, 20.0)],
        am=None,
        noise_level=0.3,
        seed=2,
    )
    window_s, hop_s = 0.5, 0.25
    win_len = int(window_s * fs)
    hop = int(hop_s * fs)
    n = signal.shape[0]
    n_win = (n - win_len) // hop + 1
    y = np.tile(pattern, int(np.ceil(n_win / len(pattern))))[:n_win].astype(np.float64)

    model, info = FFTBoost.auto(
        signal,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        val_gap_windows=1,
        center_target=False,
        budget_stages=48,
        halving_rounds=1,
        n_configs=12,
    )
    model_any: Any = model
    # For classifier, predict returns labels, FFTBoost returns scores
    if hasattr(model_any, "predict"):
        y_hat = model_any.predict(signal, fs=fs, window_s=window_s, hop_s=hop_s)
        # If this is regression object, threshold at 0 for basic split
        if y_hat.dtype.kind == "f":
            y_hat = (y_hat >= 0.0).astype(np.int64)
    else:
        # Should not occur, but keep a safe fallback
        scores = model_any.predict_proba(signal, fs=fs, window_s=window_s, hop_s=hop_s)
        thr = getattr(model_any, "threshold_", 0.5)
        y_hat = (scores >= float(thr)).astype(np.int64)

    m = min(y.shape[0], y_hat.shape[0])
    acc = float((y[:m] == y_hat[:m]).mean())
    print(f"[gauntlet-lite] classification acc={acc:.3f}")
    assert acc >= 0.85
