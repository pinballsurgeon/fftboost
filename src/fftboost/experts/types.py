from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExpertContext:
    psd: np.ndarray[Any, Any]  # shape: (n_windows, n_bins), float64
    freqs: np.ndarray[Any, Any]  # shape: (n_bins,), float64
    fs: float  # sampling rate (Hz)
    min_sep_bins: int
    lambda_hf: float
    selected_bins: np.ndarray[Any, Any] | None = None  # shape: (k,), int64 or None
    band_edges_hz: np.ndarray[Any, Any] | None = None  # optional band edges for sk_band


@dataclass(frozen=True)
class Proposal:
    H: np.ndarray[Any, Any]  # shape: (n_windows, n_features)
    descriptors: list[dict[str, object]]
    mu: np.ndarray[Any, Any]  # shape: (n_features,)
    sigma: np.ndarray[Any, Any]  # shape: (n_features,)

    def with_features(
        self,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return self.H, self.mu, self.sigma
