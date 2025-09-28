import numpy as np
from fftboost.utils.scaling import robust_scale, zscale


def test_zscale_properties() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaled_data, mu, sd = zscale(data)

    assert np.isclose(scaled_data.mean(), 0.0)
    assert np.isclose(scaled_data.std(), 1.0)
    assert np.isclose(mu, 3.0)
    assert sd > 0


def test_robust_scale_calculation() -> None:
    data = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
    scaled_data, median, iqr = robust_scale(data)

    expected_median = 10.0
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    expected_iqr = q75 - q25

    assert median == expected_median
    assert np.isclose(iqr, expected_iqr + 1e-9)

    expected_scaled_data = (data - expected_median) / (expected_iqr + 1e-9)
    assert np.allclose(scaled_data, expected_scaled_data)
