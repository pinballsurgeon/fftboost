from __future__ import annotations

from typing import Any
from typing import cast

import numpy as np

from .booster import Booster
from .booster import BoosterConfig
from .io import BoosterArtifact
from .io import load_model
from .io import save_model


class FFTBoost:
    def __init__(self, config: BoosterConfig):
        self.config = config
        self._booster: Booster | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        signal: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        *,
        fs: float,
        window_s: float,
        hop_s: float,
        val_size: float = 0.2,
        val_gap_windows: int = 1,
    ) -> FFTBoost:
        self._booster = Booster(self.config).fit(
            signal.astype(np.float64),
            y.astype(np.float64),
            fs=fs,
            window_s=window_s,
            hop_s=hop_s,
            val_size=val_size,
            val_gap_windows=val_gap_windows,
        )
        self.is_fitted = True
        return self

    def predict(
        self,
        signal: np.ndarray[Any, Any],
        *,
        fs: float,
        window_s: float,
        hop_s: float,
    ) -> np.ndarray[Any, Any]:
        if not self.is_fitted or self._booster is None:
            msg = (
                "This FFTBoost instance is not fitted yet. "
                "Call 'fit' before predicting."
            )
            raise RuntimeError(msg)
        return cast(
            np.ndarray[Any, Any],
            self._booster.predict(
                signal.astype(np.float64),
                fs=fs,
                window_s=window_s,
                hop_s=hop_s,
            ),
        )

    # Convenience persistence
    def save(self, path_prefix: str) -> dict[str, str]:
        if not self.is_fitted or self._booster is None:
            raise RuntimeError("Model not fitted")
        return save_model(self._booster.artifact, path_prefix)

    @classmethod
    def load(cls, path_prefix: str) -> FFTBoost:
        artifact: BoosterArtifact = load_model(path_prefix)
        config = BoosterConfig(**(artifact.config or {}).get("cfg", {}))
        model = cls(config)
        booster = Booster(config)
        booster.stages = artifact.stages
        booster.freqs = artifact.freqs
        model._booster = booster
        model.is_fitted = True
        return model
