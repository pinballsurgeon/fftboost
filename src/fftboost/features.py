from __future__ import annotations

from typing import Any
from typing import cast

import numpy as np
import pywt
from scipy.fft import rfft
from scipy.signal import hilbert

from .config import FeatureConfig


def _create_windows(
    signal: np.ndarray[Any, Any], config: FeatureConfig
) -> np.ndarray[Any, Any]:
    win_len = int(config.window_s * config.fs)
    hop_len = int(config.hop_s * config.fs)
    n_windows = (signal.shape[0] - win_len) // hop_len + 1
    shape = (n_windows, win_len)
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    return cast(
        np.ndarray[Any, Any],
        np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides),
    )


def _compute_fft_features(windows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    fft_result = rfft(windows, axis=1)
    return cast(np.ndarray[Any, Any], np.abs(fft_result[:, 1:]))


def _compute_wavelet_features(
    windows: np.ndarray[Any, Any], config: FeatureConfig
) -> np.ndarray[Any, Any]:
    coeffs = pywt.wavedec(
        windows, config.wavelet_family, level=config.wavelet_level, axis=1
    )
    energies: list[np.ndarray[Any, Any]] = [
        np.sqrt(np.sum(detail_coeffs**2, axis=1, keepdims=True))
        for detail_coeffs in coeffs[1:]
    ]
    return cast(np.ndarray[Any, Any], np.hstack(energies))


def _compute_hilbert_features(windows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    analytic_signal = hilbert(windows, axis=1)
    instant_phase = np.unwrap(np.angle(analytic_signal), axis=1)
    instant_freq = np.diff(instant_phase, axis=1)
    mean_freq = np.mean(instant_freq, axis=1, keepdims=True)
    std_freq = np.std(instant_freq, axis=1, keepdims=True)
    return cast(np.ndarray[Any, Any], np.hstack([mean_freq, std_freq]))


def _extract_features(
    i_signal: np.ndarray[Any, Any],
    v_signal: np.ndarray[Any, Any],
    config: FeatureConfig,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    i_windows = _create_windows(i_signal, config)
    x_fft = _compute_fft_features(i_windows)
    aux_features_list: list[np.ndarray[Any, Any]] = []
    if config.use_wavelets:
        aux_features_list.append(_compute_wavelet_features(i_windows, config))
    if config.use_hilbert_phase:
        aux_features_list.append(_compute_hilbert_features(i_windows))
    if not aux_features_list:
        x_aux = np.empty((x_fft.shape[0], 0))
    else:
        x_aux = np.hstack(aux_features_list)
    return x_fft, x_aux
