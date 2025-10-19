from __future__ import annotations

from typing import Any
from typing import cast

import numpy as np

from src.fftboost.booster import Booster
from src.fftboost.booster import BoosterConfig


def _make_windows(
    signal: np.ndarray, fs: float, window_s: float, hop_s: float
) -> tuple[np.ndarray, np.ndarray]:
    win_len = int(window_s * fs)
    hop = int(hop_s * fs)
    n = signal.shape[0]
    n_win = (n - win_len) // hop + 1
    shape = (n_win, win_len)
    strides = (signal.strides[0] * hop, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    psd = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)[1:]
    return psd, freqs


def test_predict_descriptor_shapes_consistent() -> None:
    # Synthetic signal and binary labels
    fs = 200.0
    t = np.arange(0.0, 5.0, 1.0 / fs, dtype=np.float64)
    # Single tone with amplitude modulation to make classification separable
    f0 = 10.0
    x = np.sin(2 * np.pi * f0 * t)
    y = (np.sin(2 * np.pi * 0.2 * t) > 0.0).astype(np.float64)

    cfg = BoosterConfig(
        n_stages=6,
        nu=0.3,
        ridge_alpha=1e-3,
        early_stopping_rounds=3,
        loss="logistic",
        k_fft=3,
        min_sep_bins=2,
        clf_use=True,
        clf_k=2,
        clf_method="fscore",
        temporal_use=True,
        temporal_k=2,
        temporal_lags=(1, 2),
        temporal_pool_widths=(3,),
    )

    booster = Booster(cfg).fit(
        x, y, fs=fs, window_s=0.5, hop_s=0.25, val_size=0.2, val_gap_windows=1
    )

    # Build the validation PSD/freq grid the same way Booster.fit does
    psd, freqs = _make_windows(x, fs, window_s=0.5, hop_s=0.25)
    total = psd.shape[0]
    val_n = max(1, int(total * 0.2))
    val_psd = psd[total - val_n : total]

    # For each stage, ensure descriptor/weight/moment shapes align and that
    # reconstructed column count matches
    same_grid = (
        booster.freqs is not None
        and booster.freqs.shape == freqs.shape
        and np.allclose(booster.freqs, freqs)
    )

    def nearest_index(fgrid: np.ndarray, f: float) -> int:
        i = int(np.searchsorted(fgrid, f, side="left"))
        if i <= 0:
            return 0
        if i >= fgrid.size:
            return int(fgrid.size - 1)
        return int(i - 1 if abs(f - fgrid[i - 1]) <= abs(fgrid[i] - f) else i)

    # Precompute temporal difference for flux features
    d_psd = np.zeros_like(val_psd)
    if val_psd.shape[0] >= 2:
        d_psd[1:, :] = val_psd[1:, :] - val_psd[:-1, :]

    for rec in booster.stages:
        # Basic alignment between metadata arrays
        assert rec.weights.shape[0] == rec.mu.shape[0] == rec.sigma.shape[0]
        # Rebuild columns per descriptor the same way as Booster._predict
        cols: list[np.ndarray] = []
        for d in rec.descriptors:
            dobj = cast(dict[str, Any], d)
            dt = cast(str, dobj.get("type", ""))
            if dt == "fft_bin" or dt == "clf_bin":
                if same_grid:
                    cols.append(val_psd[:, int(cast(int, dobj["bin"]))])
                else:
                    idx = nearest_index(freqs, float(cast(Any, dobj["freq_hz"])))
                    cols.append(val_psd[:, idx])
            elif dt == "sk_band":
                band_obj = dobj.get("band_hz")
                if isinstance(band_obj, (tuple, list)) and len(band_obj) == 2:
                    lo: float = float(cast(Any, band_obj[0]))
                    hi: float = float(cast(Any, band_obj[1]))
                else:
                    lo, hi = 0.0, 0.0
                mask = (freqs >= lo) & (freqs < hi)
                cols.append(
                    val_psd[:, mask].sum(axis=1)
                    if mask.any()
                    else np.zeros(val_psd.shape[0])
                )
            elif dt == "flux_bin":
                if same_grid:
                    cols.append(d_psd[:, int(cast(int, dobj["bin"]))])
                else:
                    idx = nearest_index(freqs, float(cast(Any, dobj["freq_hz"])))
                    cols.append(d_psd[:, idx])
            elif dt == "lag_bin":
                lag = int(cast(Any, dobj.get("lag", 1)))
                if same_grid:
                    series = val_psd[:, int(cast(int, dobj["bin"]))].copy()
                else:
                    idx = nearest_index(freqs, float(cast(Any, dobj["freq_hz"])))
                    series = val_psd[:, idx].copy()
                v = np.zeros_like(series)
                if lag > 0:
                    v[lag:] = series[:-lag]
                cols.append(v)
            elif dt == "pool_bin":
                width = max(2, int(cast(Any, dobj.get("width", 3))))
                if same_grid:
                    series = val_psd[:, int(cast(int, dobj["bin"]))].copy()
                else:
                    idx = nearest_index(freqs, float(cast(Any, dobj["freq_hz"])))
                    series = val_psd[:, idx].copy()
                k = np.ones(width, dtype=np.float64) / float(width)
                y = np.convolve(series, k, mode="same")
                if y.shape[0] != series.shape[0]:
                    extra = y.shape[0] - series.shape[0]
                    start = max(0, extra // 2)
                    y = y[start : start + series.shape[0]]
                cols.append(y)

        H = np.column_stack(cols) if cols else np.zeros((val_psd.shape[0], 0))
        # Validate alignment with recorded shapes
        assert (
            H.shape[1] == rec.weights.shape[0] == rec.mu.shape[0] == rec.sigma.shape[0]
        )

    # Also ensure Booster._predict runs without shape errors and returns correct length
    preds = booster._predict(val_psd, freqs, booster.stages)
    assert preds.shape[0] == val_psd.shape[0]
