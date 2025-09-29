from collections.abc import Mapping
from typing import Any

import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.signal import coherence, get_window, hilbert


def create_windows(signal: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    fs = config["fs"]
    window_s = config["window_s"]
    hop_s = config["hop_s"]

    win_len = int(window_s * fs)
    hop_len = int(hop_s * fs)
    n_total = len(signal)

    hann_win = get_window("hann", win_len, fftbins=True)
    indices = np.arange(0, n_total - win_len + 1, hop_len)
    windows = np.stack([signal[i : i + win_len] * hann_win for i in indices], axis=0)
    return windows


def compute_fft(windows: np.ndarray, fs: int, n_fft_bins: int) -> tuple[np.ndarray, np.ndarray]:
    freqs = rfftfreq(n_fft_bins, 1 / fs)
    fft_features = np.abs(rfft(windows, n=n_fft_bins, axis=1))
    return fft_features, freqs


def compute_wavelet_energies(windows: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    wavelet_family = config["wavelet_family"]
    wavelet_level = config["wavelet_level"]
    all_energies = []
    for window in windows:
        coeffs = pywt.wavedec(window, wavelet_family, level=wavelet_level)
        energies = [np.sum(c**2) for c in coeffs if c.size > 0]
        all_energies.append(energies)
    return np.array(all_energies)


def compute_hilbert_phase_stats(windows: np.ndarray, fs: int) -> np.ndarray:
    all_stats = []
    for window in windows:
        analytic_signal = hilbert(window)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) * fs) / (2.0 * np.pi)
        stats = [
            np.mean(instantaneous_frequency),
            np.std(instantaneous_frequency),
            np.median(instantaneous_frequency),
        ]
        all_stats.append(stats)
    return np.array(all_stats)


def compute_spectral_moments(fft_features: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    power = fft_features**2
    total_power = np.sum(power, axis=1, keepdims=True) + 1e-12
    freq_grid = freqs[None, :]

    centroid = np.sum(power * freq_grid, axis=1, keepdims=True) / total_power
    spread = np.sqrt(
        np.sum(power * (freq_grid - centroid) ** 2, axis=1, keepdims=True) / total_power
    )
    return np.hstack([centroid, spread])


def compute_coherence_subbands(
    iwins: np.ndarray, vwins: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    fs = config["fs"]
    win_len = iwins.shape[1]
    subband_edges = config["coherence_subbands"]
    freqs_full = rfftfreq(win_len, 1 / fs)

    f_c, cxy = coherence(iwins, vwins, fs=fs, nperseg=min(128, win_len // 4), axis=1)
    coh_full = np.array(
        [np.interp(freqs_full, f_c, coh_row, left=0.0, right=0.0) for coh_row in cxy]
    )

    coh_subbands = []
    for start_freq, end_freq in subband_edges:
        mask = (freqs_full >= start_freq) & (freqs_full < end_freq)
        if np.any(mask):
            subband_coh = np.mean(coh_full[:, mask], axis=1, keepdims=True)
            coh_subbands.append(subband_coh)

    return np.hstack(coh_subbands)


def extract_features_from_signals(
    i_signal: np.ndarray, v_signal: np.ndarray, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iwins = create_windows(i_signal, config)
    vwins = create_windows(v_signal, config)
    n_windows, n_fft_bins = iwins.shape

    cand_bins = np.arange(1, n_fft_bins // 2 + 1)

    fft_features_full, freqs_full = compute_fft(iwins, config["fs"], n_fft_bins)
    x_fft = fft_features_full[:, cand_bins]
    freqs = freqs_full[cand_bins]

    aux_features_list = []
    if config.get("use_wavelets", False):
        wavelet_features = compute_wavelet_energies(iwins, config)
        aux_features_list.append(wavelet_features)

    if config.get("use_hilbert_phase", False):
        hilbert_features = compute_hilbert_phase_stats(iwins, config["fs"])
        aux_features_list.append(hilbert_features)

    spectral_moments = compute_spectral_moments(x_fft, freqs)
    aux_features_list.append(spectral_moments)

    if config.get("coherence_subbands"):
        coherence_features = compute_coherence_subbands(iwins, vwins, config)
        aux_features_list.append(coherence_features)

    x_aux = np.hstack(aux_features_list)
    return x_fft, x_aux, freqs


def compute_labels(windows: np.ndarray, freqs: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    ehi_weights = config["target"]["ehi_weights"]
    fft_features = np.abs(rfft(windows, axis=1))

    nyq_bin = len(freqs) - 1
    labels = []
    for row in fft_features:
        fb = np.where((freqs >= 45) & (freqs <= 55))[0]
        if fb.size == 0:
            thd, ipr = 0.0, 0.0
        else:
            k1 = fb[np.argmax(row[fb])]
            v1_sq = row[k1] ** 2 + 1e-12
            h_sq = sum(
                np.max(
                    row[max(1, int(round(m * k1)) - 1) : min(nyq_bin - 1, int(round(m * k1)) + 2)]
                )
                ** 2
                for m in range(2, 9)
                if 1 <= int(round(m * k1)) < nyq_bin
            )
            thd = np.sqrt(h_sq / v1_sq)

            ib = np.where((freqs >= 90) & (freqs <= 130))[0]
            ip_sq = float(np.sum(row[ib] ** 2))
            tot_sq = float(np.sum(row[1:nyq_bin] ** 2) + 1e-12)
            ipr = ip_sq / tot_sq

        labels.append(ehi_weights["thd"] * thd + ehi_weights["ipr"] * ipr)

    return np.array(labels)
