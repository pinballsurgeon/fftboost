from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pywt
from scipy.fft import rfft
from scipy.signal import hilbert

from .config import FeatureConfig


def _create_windows(signal: npt.NDArray, config: FeatureConfig) -> npt.NDArray:
    win_len = int(config.window_s * config.fs)
    hop_len = int(config.hop_s * config.fs)
    n_windows = (signal.shape[0] - win_len) // hop_len + 1
    shape = (n_windows, win_len)
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


def _compute_fft_features(windows: npt.NDArray) -> npt.NDArray:
    fft_result = rfft(windows, axis=1)
    return np.abs(fft_result[:, 1:])


def _compute_wavelet_features(
    windows: npt.NDArray, config: FeatureConfig
) -> npt.NDArray:
    coeffs = pywt.wavedec(
        windows, config.wavelet_family, level=config.wavelet_level, axis=1
    )
    energies: list[npt.NDArray] = [
        np.sqrt(np.sum(detail_coeffs**2, axis=1, keepdims=True))
        for detail_coeffs in coeffs[1:]
    ]
    return np.hstack(energies)


def _compute_hilbert_features(windows: npt.NDArray) -> npt.NDArray:
    analytic_signal = hilbert(windows, axis=1)
    instant_phase = np.unwrap(np.angle(analytic_signal), axis=1)
    instant_freq = np.diff(instant_phase, axis=1)
    mean_freq = np.mean(instant_freq, axis=1, keepdims=True)
    std_freq = np.std(instant_freq, axis=1, keepdims=True)
    return np.hstack([mean_freq, std_freq])


def _extract_features(
    i_signal: npt.NDArray, v_signal: npt.NDArray, config: FeatureConfig
) -> tuple[npt.NDArray, npt.NDArray]:
    i_windows = _create_windows(i_signal, config)
    x_fft = _compute_fft_features(i_windows)
    aux_features_list: list[npt.NDArray] = []
    if config.use_wavelets:
        aux_features_list.append(_compute_wavelet_features(i_windows, config))
    if config.use_hilbert_phase:
        aux_features_list.append(_compute_hilbert_features(i_windows))
    if not aux_features_list:
        x_aux = np.empty((x_fft.shape[0], 0))
    else:
        x_aux = np.hstack(aux_features_list)
    return x_fft, x_aux
