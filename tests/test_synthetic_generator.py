import numpy as np

from data.synthetic_generator import generate_synthetic_signals


def test_signal_properties() -> None:
    config = {"fs": 1000, "duration_s": 10}
    i_signal, v_signal, tvec = generate_synthetic_signals(config, seed=1337)

    expected_length = config["fs"] * config["duration_s"]
    assert i_signal.shape == (expected_length,)
    assert v_signal.shape == (expected_length,)
    assert tvec.shape == (expected_length,)
    assert isinstance(i_signal, np.ndarray)
    assert isinstance(v_signal, np.ndarray)


def test_determinism() -> None:
    config = {"fs": 100, "duration_s": 5}
    i1, v1, _ = generate_synthetic_signals(config, seed=42)
    i2, v2, _ = generate_synthetic_signals(config, seed=42)
    i3, v3, _ = generate_synthetic_signals(config, seed=99)

    assert np.array_equal(i1, i2)
    assert np.array_equal(v1, v2)
    assert not np.array_equal(i1, i3)
