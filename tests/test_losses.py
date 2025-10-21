from __future__ import annotations

from typing import Callable
from typing import cast

import numpy as np

from src.fftboost.losses import HuberLoss
from src.fftboost.losses import QuantileLoss
from src.fftboost.losses import SquaredLoss


def _finite_difference_grad(
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    n = y_true.shape[0]
    approx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        e = np.zeros(n, dtype=np.float64)
        e[i] = eps
        lp = loss_fn(y_true, y_pred + e)
        lm = loss_fn(y_true, y_pred - e)
        approx[i] = (lp - lm) / (2.0 * eps)
    return cast(np.ndarray, approx)


def _assert_grad_close(analytical: np.ndarray, numerical: np.ndarray) -> None:
    # Relative L2 error with absolute fallback near zero
    num = float(np.linalg.norm(analytical - numerical))
    den = max(float(np.linalg.norm(analytical)), 1e-12)
    rel = num / den
    assert rel < 1e-6, f"relative error too large: {rel}"


def test_squared_loss_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(123)
    n = 32
    y_true = rng.standard_normal(n).astype(np.float64)
    y_pred = rng.standard_normal(n).astype(np.float64)

    loss = SquaredLoss()
    analytical = loss.gradient(y_true, y_pred)
    numerical = _finite_difference_grad(loss, y_true, y_pred)
    _assert_grad_close(analytical, numerical * n)


def test_huber_loss_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(456)
    n = 32
    y_true = rng.standard_normal(n).astype(np.float64)
    y_pred = rng.standard_normal(n).astype(np.float64)

    loss = HuberLoss(delta=0.75)
    analytical = loss.gradient(y_true, y_pred)
    numerical = _finite_difference_grad(loss, y_true, y_pred)
    _assert_grad_close(analytical, numerical * n)


def test_quantile_loss_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(789)
    n = 32
    y_true = rng.standard_normal(n).astype(np.float64)
    y_pred = rng.standard_normal(n).astype(np.float64)

    loss = QuantileLoss(alpha=0.2)
    analytical = loss.gradient(y_true, y_pred)
    numerical = _finite_difference_grad(loss, y_true, y_pred)
    _assert_grad_close(analytical, numerical * n)


def test_huber_kink_boundary_stability() -> None:
    # Construct residuals exactly at the kink then jitter deterministically
    delta = 0.5
    loss = HuberLoss(delta=delta)

    n = 10
    y_true = np.zeros(n, dtype=np.float64)
    # r = y_pred - y_true exactly at +/- delta
    base = np.array([delta, -delta] * (n // 2), dtype=np.float64)
    y_pred = base.copy()

    # Deterministic jitter around the boundary
    rng = np.random.default_rng(42)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
    eps = 1e-8
    y_pred_jitter = y_pred + signs * eps

    # Expected gradient depends on which side of the kink we are on
    r = y_pred_jitter - y_true
    expected = np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    analytical = loss.gradient(y_true, y_pred_jitter)
    # Allow tiny numerical noise around the kink
    np.testing.assert_allclose(analytical, expected, rtol=0.0, atol=2e-8)


def test_quantile_kink_boundary_stability() -> None:
    # r = 0 kink; jitter determines the side
    alpha = 0.3
    loss = QuantileLoss(alpha=alpha)

    n = 10
    y_true = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y_pred = y_true.copy()  # r == 0 exactly

    rng = np.random.default_rng(1234)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
    eps = 1e-8
    y_pred_jitter = y_pred + signs * eps

    # If r > 0: grad = (1 - alpha); if r < 0: grad = -alpha
    r = y_pred_jitter - y_true
    expected = np.where(r > 0.0, (1.0 - alpha), -alpha)

    analytical = loss.gradient(y_true, y_pred_jitter)
    np.testing.assert_allclose(analytical, expected, rtol=0.0, atol=1e-12)
