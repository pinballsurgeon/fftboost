from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FFTBoostConfig:
    atoms: int = 20
    lambda_hf: float = 2.0
    lambda_coh: float = 0.0
    min_sep_bins: int = 15
    ridge_alpha: float = 0.01


@dataclass
class FeatureConfig:
    fs: int = 4000
    window_s: float = 1.024
    hop_s: float = 0.512
    use_wavelets: bool = True
    wavelet_family: str = "db4"
    wavelet_level: int = 4
    use_hilbert_phase: bool = False
    coherence_subbands: list[list[int]] | None = None
