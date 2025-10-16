# src\fftboost\losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import final

import numpy as np


@final
class SquaredLoss:
    """
    Mean squared loss with a 0.5 factor per-sample:
      L = 0.5 * mean((y_pred - y_true)**2)

    Gradient is the mean gradient w.r.t. y_pred:
      dL/dy_pred = (y_pred - y_true) / n
    """

    def __call__(
        self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
    ) -> float:
        r = y_pred - y_true
        return float(0.5 * np.mean(r * r))

    def gradient(
        self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        n = float(y_true.shape[0])
        r = y_pred - y_true
        return cast(np.ndarray[Any, Any], (r / n).astype(np.float64, copy=False))


@final
@dataclass(frozen=True)
class HuberLoss:
    """
    Mean Huber loss with parameter delta > 0.

    Per-sample loss for residual r = y_pred - y_true:
      if |r| <= delta: 0.5 * r^2
      else:            delta * (|r| - 0.5 * delta)

    Gradient (mean) w.r.t. y_pred:
      if |r| <  delta: r / n
      if |r| >  delta: delta * sign(r) / n
      if |r| == delta: uses the outer branch (delta * sign(r) / n)
    """

    delta: float

    def __post_init__(self) -> None:
        if not (self.delta > 0.0):
            raise ValueError("delta must be > 0")

    def __call__(
        self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
    ) -> float:
        r = y_pred - y_true
        abs_r = np.abs(r)
        quad = 0.5 * (r * r)
        lin = self.delta * (abs_r - 0.5 * self.delta)
        loss = np.where(abs_r <= self.delta, quad, lin)
        return float(np.mean(loss))

    def gradient(
        self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        n = float(y_true.shape[0])
        r = y_pred - y_true
        abs_r = np.abs(r)
        grad = np.where(abs_r <= self.delta, r, self.delta * np.sign(r))
        return cast(np.ndarray[Any, Any], (grad / n).astype(np.float64, copy=False))


@final
@dataclass(frozen=True)
class QuantileLoss:
    """
    Mean pinball (quantile) loss with parameter alpha in (0, 1).

    Using residual r = y_pred - y_true, per-sample loss:
      if r >= 0: (1 - alpha) * r
      if r <  0: -alpha * r

    Mean gradient w.r.t. y_pred:
      if r > 0:  (1 - alpha) / n
      if r < 0:  (-alpha)     / n
      if r = 0:  uses r <= 0 branch (-alpha / n)
    """

    alpha: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

    def __call__(
        self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
    ) -> float:
        r = y_pred - y_true
        # Proper pinball loss with residual r = y_pred - y_true
        loss = np.maximum((1.0 - self.alpha) * r, -self.alpha * r)
        return float(np.mean(loss))

    def gradient(
        self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        n = float(y_true.shape[0])
        # Deterministic subgradient at r == 0 via y_true comparison
        grad = -np.where(y_true > y_pred, self.alpha, self.alpha - 1.0)
        return cast(np.ndarray[Any, Any], (grad / n).astype(np.float64, copy=False))
