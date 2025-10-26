from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np


class ModelHead(ABC):
    @abstractmethod
    def fit(self, H: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None: ...

    @abstractmethod
    def predict(self, H: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...


class RidgeHead(ModelHead):
    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        self.weights: np.ndarray[Any, Any] | None = None

    def fit(self, H: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        # Using a standard Cholesky decomposition solver for Ridge
        Z = H  # Assume H is already z-scored
        n_features = Z.shape[1]
        if n_features == 0:
            self.weights = np.empty((0,), dtype=np.float64)
            return

        G = Z.T @ Z
        G.flat[:: n_features + 1] += self.alpha
        y_corr = Z.T @ y

        try:
            L = np.linalg.cholesky(G)
            z = np.linalg.solve(L, y_corr)
            w = np.linalg.solve(L.T, z)
            self.weights = w
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            G_inv = np.linalg.pinv(G)
            self.weights = G_inv @ y_corr

    def predict(self, H: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self.weights is None:
            raise RuntimeError("RidgeHead is not fitted.")
        return H @ self.weights
