from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .config import FeatureConfig, FFTBoostConfig
from .features import _extract_features
from .models import _fit_model


class FFTBoost:
    def __init__(self, fftboost_config: FFTBoostConfig, feature_config: FeatureConfig):
        self.fftboost_config = fftboost_config
        self.feature_config = feature_config
        self._fitted_state: dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        i_signal: npt.NDArray,
        y: npt.NDArray,
        v_signal: npt.NDArray | None = None,
    ) -> FFTBoost:
        v_signal_proc = v_signal if v_signal is not None else np.zeros_like(i_signal)
        x_fft, x_aux = _extract_features(i_signal, v_signal_proc, self.feature_config)
        y_trimmed = y[: x_fft.shape[0]]
        self._fitted_state = _fit_model(
            x_fft, x_aux, y_trimmed, self.fftboost_config, self.feature_config.fs
        )
        self.is_fitted = True
        return self

    def predict(
        self, i_signal: npt.NDArray, v_signal: npt.NDArray | None = None
    ) -> npt.NDArray:
        if not self.is_fitted:
            raise RuntimeError(
                "This FFTBoost instance is not fitted yet. "
                "Call 'fit' before predicting."
            )
        v_signal_proc = v_signal if v_signal is not None else np.zeros_like(i_signal)
        x_fft, x_aux = _extract_features(i_signal, v_signal_proc, self.feature_config)
        final_model = self._fitted_state["final_model"]
        active_atoms = self._fitted_state["active_atoms"]
        features_to_predict = np.hstack([x_fft[:, active_atoms], x_aux])
        return final_model.predict(features_to_predict)
