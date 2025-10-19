from __future__ import annotations

import numpy as np

from src.fftboost.api import FFTBoost


def _toy_signal_regression(
    fs: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    t = np.arange(0.0, 5.0, 1.0 / fs, dtype=np.float64)
    x = np.sin(2 * np.pi * 7.0 * t) + 0.1 * np.random.default_rng(0).standard_normal(
        t.shape[0]
    )
    y = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return x, y, fs, 0.5, 0.25


def _toy_signal_binary(
    fs: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    t = np.arange(0.0, 5.0, 1.0 / fs, dtype=np.float64)
    x = np.sin(2 * np.pi * 10.0 * t)
    y = (np.sin(2 * np.pi * 0.3 * t) > 0.0).astype(np.float64)
    return x, y, fs, 0.5, 0.25


def test_automl_returns_model_and_info_regression() -> None:
    x, y, fs, window_s, hop_s = _toy_signal_regression()
    model, info = FFTBoost.auto(
        x,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        val_gap_windows=1,
        center_target=True,
        n_configs=6,
        budget_stages=32,
        halving_rounds=1,
    )
    assert model is not None
    assert isinstance(info, dict)
    assert info.get("task") == "regression"
    assert "config" in info and isinstance(info["config"], dict)
    assert "score" in info


def test_automl_returns_model_and_info_binary() -> None:
    x, y, fs, window_s, hop_s = _toy_signal_binary()
    model, info = FFTBoost.auto(
        x,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        val_gap_windows=1,
        center_target=False,
        n_configs=6,
        budget_stages=32,
        halving_rounds=1,
    )
    assert model is not None
    assert isinstance(info, dict)
    assert info.get("task") == "binary"
    assert "config" in info and isinstance(info["config"], dict)
    assert "score" in info
