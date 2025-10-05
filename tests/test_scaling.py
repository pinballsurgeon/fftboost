from __future__ import annotations

import numpy as np

from src.fftboost.scaling import robust_scale  # CORRECTED IMPORT
from src.fftboost.scaling import zscale  # CORRECTED IMPORT


def test_zscale_properties() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaled_data, mu, sd = zscale(data)
    assert np.isclose(scaled_data.mean(), 0.0)
    assert np.isclose(scaled_data.std(), 1.0)


def test_robust_scale_calculation() -> None:
    data = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
    scaled_data, median, iqr = robust_scale(data)

    expected_median = 10.0
    expected_iqr = 11.0 - 2.0

    assert np.isclose(median, expected_median)
    assert np.isclose(iqr, expected_iqr)
