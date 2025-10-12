from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from .boosting import StageRecord
from .boosting import fit_stage
from .config import FFTBoostConfig
from .experts.fft_bin import propose as fftbin_propose
from .experts.types import ExpertContext
from .io import BoosterArtifact


@dataclass(frozen=True)
class BoosterConfig:
    n_stages: int
    ridge_alpha: float
    nu: float
    top_k_fft: int


class Booster:
    def __init__(self, cfg: BoosterConfig):
        self.cfg = cfg
        self.stages: list[StageRecord] = []
        self.freqs: np.ndarray | None = None

    def fit(
        self,
        psd: np.ndarray,
        freqs: np.ndarray,
        y: np.ndarray,
        fft_cfg: FFTBoostConfig,
        fs: float,
    ) -> Booster:
        n_windows = psd.shape[0]
        y = y[:n_windows]
        y_pred = np.zeros_like(y, dtype=np.float64)

        ctx_base = {
            "fs": float(fs),
            "min_sep_bins": int(fft_cfg.min_sep_bins),
            "lambda_hf": float(fft_cfg.lambda_hf),
        }

        self.stages = []
        for _ in range(self.cfg.n_stages):
            residual = (y - y_pred) / float(max(1, y.shape[0]))
            ctx = ExpertContext(
                psd=psd,
                freqs=freqs,
                fs=int(fs),
                min_sep_bins=int(ctx_base["min_sep_bins"]),
                lambda_hf=float(ctx_base["lambda_hf"]),
                selected_bins=None,
            )
            prop = fftbin_propose(residual, ctx, top_k=self.cfg.top_k_fft)
            applied, rec = fit_stage(
                residual, [prop], ridge_alpha=self.cfg.ridge_alpha, nu=self.cfg.nu
            )
            y_pred = y_pred + applied
            self.stages.append(rec)

        self.freqs = freqs.astype(np.float64, copy=True)
        return self

    def predict(self, psd: np.ndarray) -> np.ndarray:
        if self.freqs is None:
            raise RuntimeError("Booster is not fitted")
        yhat = np.zeros(psd.shape[0], dtype=np.float64)
        for s in self.stages:
            # reconstruct columns from descriptors (fft_bin)
            idxs = []
            for d in s.descriptors:
                f = cast(float, d.get("freq_hz"))
                idxs.append(int(np.argmin(np.abs(self.freqs - f))))
            H = psd[:, idxs]
            Z = (H - s.mu) / (s.sigma + 1e-12)
            yhat = yhat + (s.nu * s.gamma) * (Z @ s.weights)
        return cast(np.ndarray, np.asarray(yhat, dtype=np.float64))

    @property
    def artifact(self) -> BoosterArtifact:
        if self.freqs is None:
            raise RuntimeError("Booster is not fitted")
        return BoosterArtifact(
            schema_version="1",
            fftboost_version="1.0",
            freqs=self.freqs,
            stages=self.stages,
            config={
                "cfg": {
                    "n_stages": self.cfg.n_stages,
                    "ridge_alpha": self.cfg.ridge_alpha,
                    "nu": self.cfg.nu,
                    "top_k_fft": self.cfg.top_k_fft,
                }
            },
        )
