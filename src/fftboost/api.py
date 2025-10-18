from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any
from typing import Union
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
        self._y_offset: float | None = None

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
        center_target: bool = False,
    ) -> FFTBoost:
        y_arr = y.astype(np.float64)
        if center_target:
            # Center the provided target prior to internal window truncation.
            # Booster.fit will truncate y to the derived window count.
            self._y_offset = float(np.mean(y_arr))
            y_arr = y_arr - self._y_offset
        else:
            self._y_offset = None

        booster = Booster(self.config).fit(
            signal.astype(np.float64),
            y_arr,
            fs=fs,
            window_s=window_s,
            hop_s=hop_s,
            val_size=val_size,
            val_gap_windows=val_gap_windows,
        )
        self._booster = booster
        self.is_fitted = True

        # Guardrail: warn if training made no real progress
        # (likely centering/tuning issue)
        try:
            if getattr(booster, "best_iteration_", -1) <= 1:
                warnings.warn(
                    (
                        "Training stopped after <=1 stage. "
                        "If target is positive-only (e.g., RMS), "
                        "try center_target=True and/or increase nu/n_stages."
                    ),
                    UserWarning,
                )
        except Exception:
            # Be resilient to unexpected attribute/state issues
            pass
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
        yhat = self._booster.predict(
            signal.astype(np.float64), fs=fs, window_s=window_s, hop_s=hop_s
        )
        if self._y_offset is not None:
            yhat = yhat + float(self._y_offset)
        return yhat

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
        model._y_offset = None
        return model

    def get_feature_importances(self, kind: str = "all") -> list[dict[str, object]]:
        """
        Aggregate simple per-feature contribution scores from fitted stages.

        Scoring heuristic: sum over stages of (nu * |gamma| * |w_j|) per descriptor.
        This provides an interpretable, stage-accumulated magnitude for each
        selected feature (fft_bin or sk_band).

        kind: 'fft_bin' | 'sk_band' | 'all'
        """
        if not self.is_fitted or self._booster is None:
            raise RuntimeError("Model not fitted")
        if kind not in ("fft_bin", "sk_band", "all"):
            raise ValueError("kind must be 'fft_bin', 'sk_band', or 'all'")

        Key = tuple[str, Union[float, tuple[float, float]]]
        scores: dict[Key, float] = {}
        for rec in self._booster.stages:
            if rec.weights.size == 0:
                continue
            # Descriptor list aligns with weights order in StageRecord
            for d, w in zip(rec.descriptors, rec.weights):
                d_type = d.get("type")
                if d_type not in ("fft_bin", "sk_band"):
                    continue
                if kind != "all" and d_type != kind:
                    continue
                if d_type == "fft_bin":
                    fval = float(cast(Any, d.get("freq_hz", 0.0)))
                    key: Key = ("fft_bin", fval)
                else:
                    band_obj = d.get("band_hz", (0.0, 0.0))
                    if isinstance(band_obj, tuple) and len(band_obj) == 2:
                        lo_raw = band_obj[0]
                        hi_raw = band_obj[1]
                        band_t = (
                            float(cast(Any, lo_raw)),
                            float(cast(Any, hi_raw)),
                        )
                    else:
                        band_t = (0.0, 0.0)
                    key = ("sk_band", band_t)
                contrib = float(rec.nu) * abs(float(rec.gamma)) * abs(float(w))
                scores[key] = scores.get(key, 0.0) + contrib

        # Format as list of descriptors with score
        out: list[dict[str, object]] = []
        for key, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            if key[0] == "fft_bin":
                out.append({"type": "fft_bin", "freq_hz": key[1], "score": sc})
            else:
                out.append({"type": "sk_band", "band_hz": key[1], "score": sc})
        return out


class FFTBoostClassifier:
    """
    Simple binary classifier wrapper around FFTBoost using logistic loss.

    - Fits an internal FFTBoost configured with loss='logistic'.
    - Exposes predict_proba (sigmoid of logits) and predict (thresholded).
    - Default threshold selection uses the validation block means midpoint.
    """

    def __init__(self, config: BoosterConfig, threshold: float | str = "auto"):
        self.base_config = config
        self.threshold_spec = threshold
        self.model: FFTBoost | None = None
        self.threshold_: float | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return 1.0 / (1.0 + np.exp(-z))

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
    ) -> FFTBoostClassifier:
        # Ensure binary 0/1 float labels
        y_bin = (y > 0.5).astype(np.float64, copy=False)
        cfg = replace(self.base_config, loss="logistic")
        clf = FFTBoost(cfg)
        clf.fit(
            signal,
            y_bin,
            fs=fs,
            window_s=window_s,
            hop_s=hop_s,
            val_size=val_size,
            val_gap_windows=val_gap_windows,
            center_target=False,
        )
        # Store
        self.model = clf

        # Select threshold
        scores = clf.predict(signal, fs=fs, window_s=window_s, hop_s=hop_s)
        # scores are logits under logistic loss; default decision boundary at 0.0
        if isinstance(self.threshold_spec, (int, float)):
            thr = float(self.threshold_spec)
        else:
            total = scores.shape[0]
            val_n = max(1, int(total * val_size))
            val_idx = slice(total - val_n, total)
            ys = y_bin[:total]
            s_val = scores[val_idx]
            y_val = ys[val_idx]
            pos = s_val[y_val > 0.5]
            neg = s_val[y_val <= 0.5]
            if pos.size == 0 or neg.size == 0:
                thr = 0.0
            else:
                thr = 0.5 * (float(pos.mean()) + float(neg.mean()))
        self.threshold_ = thr
        return self

    def predict_proba(
        self,
        signal: np.ndarray[Any, Any],
        *,
        fs: float,
        window_s: float,
        hop_s: float,
    ) -> np.ndarray[Any, Any]:
        if self.model is None:
            raise RuntimeError("Classifier not fitted")
        logits = self.model.predict(
            signal.astype(np.float64), fs=fs, window_s=window_s, hop_s=hop_s
        )
        p = self._sigmoid(logits)
        # Two-column style not required; return positive class probability
        return p

    def predict(
        self,
        signal: np.ndarray[Any, Any],
        *,
        fs: float,
        window_s: float,
        hop_s: float,
    ) -> np.ndarray[Any, Any]:
        if self.model is None or self.threshold_ is None:
            raise RuntimeError("Classifier not fitted")
        logits = self.model.predict(
            signal.astype(np.float64), fs=fs, window_s=window_s, hop_s=hop_s
        )
        return (logits >= self.threshold_).astype(np.int64, copy=False)
