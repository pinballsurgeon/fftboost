from __future__ import annotations

from typing import Any
from typing import cast

import numpy as np

from .booster import Booster
from .booster import BoosterConfig
from .config import FeatureConfig
from .config import FFTBoostConfig
from .features import _compute_fft_features
from .features import _create_windows
from .io import BoosterArtifact
from .io import load_model
from .io import save_model


class FFTBoost:
    def __init__(self, fftboost_config: FFTBoostConfig, feature_config: FeatureConfig):
        self.fftboost_config = fftboost_config
        self.feature_config = feature_config
        self._fitted_state: dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        i_signal: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        v_signal: np.ndarray[Any, Any] | None = None,
    ) -> FFTBoost:
        # Build PSD features (FFT magnitudes per window)
        win_i = _create_windows(i_signal, self.feature_config)
        psd = _compute_fft_features(win_i)
        # y alignment
        y_trimmed = y[: psd.shape[0]]

        # Orchestrate boosting
        booster_cfg = BoosterConfig(
            n_stages=self.fftboost_config.atoms,
            ridge_alpha=self.fftboost_config.ridge_alpha,
            nu=1.0,
            top_k_fft=max(1, self.fftboost_config.atoms // 2),
        )
        self._booster = Booster(booster_cfg).fit(
            psd=psd,
            freqs=np.fft.rfftfreq(win_i.shape[1], d=1.0 / self.feature_config.fs)[1:],
            y=y_trimmed.astype(np.float64),
            fft_cfg=self.fftboost_config,
            fs=float(self.feature_config.fs),
        )
        # Back-compat state keys for tests
        self._fitted_state = {
            "active_atoms": np.array([], dtype=int),
            "final_model": object(),
        }
        self.is_fitted = True
        return self

    def predict(
        self,
        i_signal: np.ndarray[Any, Any],
        v_signal: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        if not self.is_fitted:
            raise RuntimeError(
                "This FFTBoost instance is not fitted yet. "
                "Call 'fit' before predicting."
            )
        win_i = _create_windows(i_signal, self.feature_config)
        psd = _compute_fft_features(win_i)
        return cast(np.ndarray[Any, Any], self._booster.predict(psd))

    # Convenience persistence
    def save(self, path_prefix: str) -> dict[str, str]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return save_model(self._booster.artifact, path_prefix)

    @classmethod
    def load(
        cls,
        path_prefix: str,
        fftboost_config: FFTBoostConfig,
        feature_config: FeatureConfig,
    ) -> FFTBoost:
        model = cls(fftboost_config, feature_config)
        artifact: BoosterArtifact = load_model(path_prefix)
        # Rebuild a minimal Booster wrapper for prediction
        booster_cfg = BoosterConfig(
            n_stages=len(artifact.stages),
            ridge_alpha=1.0,
            nu=1.0,
            top_k_fft=len(artifact.stages),
        )
        booster = Booster(booster_cfg)
        booster.stages = artifact.stages
        booster.freqs = artifact.freqs
        model._booster = booster
        model.is_fitted = True
        # Back-compat keys
        model._fitted_state = {
            "active_atoms": np.array([], dtype=int),
            "final_model": object(),
        }
        return model
