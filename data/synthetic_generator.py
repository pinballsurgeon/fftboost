from collections.abc import Mapping

import numpy as np


def generate_synthetic_signals(
    config: Mapping[str, float | int], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fs = config["fs"]
    duration_s = config["duration_s"]

    n_total = int(duration_s * fs)
    tvec = np.arange(n_total) / fs

    i_signal = 100 * np.sin(2 * np.pi * 50 * tvec)
    i_signal += 15 * np.sin(2 * np.pi * 150 * tvec + 0.3)
    i_signal += 8 * np.sin(2 * np.pi * 117 * tvec + 1.1)
    i_signal += 30 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * tvec)) * np.sin(2 * np.pi * 123 * tvec)
    i_signal += 12 * np.sin(2 * np.pi * 250 * tvec + 0.7)
    i_signal += 7 * rng.standard_normal(n_total)

    v_signal = np.roll(i_signal, int(0.004 * fs)) + 5 * rng.standard_normal(n_total)

    return i_signal, v_signal, tvec
