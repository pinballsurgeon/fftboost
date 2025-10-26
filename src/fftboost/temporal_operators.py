from __future__ import annotations

from typing import Any

import numpy as np


def apply_lag(series: np.ndarray[Any, Any], lag: int) -> np.ndarray[Any, Any]:
    """
    Applies a lag to a time series.

    Positive lags shift the series into the future (past values).
    """
    if lag == 0:
        return series
    # Use NaN for values that are shifted out of existence
    result = np.full_like(series, np.nan, dtype=np.float64)
    if lag > 0:
        if lag < len(series):
            result[lag:] = series[:-lag]
    else:  # Negative lag (lead) is not typically used but supported
        if abs(lag) < len(series):
            result[:lag] = series[-lag:]
    return np.nan_to_num(result)


def apply_diff(series: np.ndarray[Any, Any], order: int = 1) -> np.ndarray[Any, Any]:
    """
    Computes the difference of a time series.
    """
    if order < 1:
        return series
    # Prepend to maintain original shape
    return np.diff(series, n=order, axis=0, prepend=series[0:order])


def apply_moving_average(
    series: np.ndarray[Any, Any], width: int
) -> np.ndarray[Any, Any]:
    """
    Computes the moving average of a time series using convolution.
    """
    if width <= 1:
        return series
    # Use a 1D convolution with a flat kernel for the moving average
    kernel = np.ones(width, dtype=np.float64) / float(width)
    # 'same' mode ensures the output has the same length as the input
    return np.convolve(series, kernel, mode="same")
