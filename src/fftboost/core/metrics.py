from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import t as tdist


def paired_ci(d: Sequence[float], conf: float = 0.95) -> tuple[float, float]:
    d_arr = np.asarray(d, dtype=float)
    if d_arr.size < 2:
        mean_val = float(np.mean(d_arr)) if d_arr.size else 0.0
        return mean_val, float("inf")

    mean_val = float(np.mean(d_arr))
    std_val = float(np.std(d_arr, ddof=1))
    se_val = std_val / np.sqrt(d_arr.size)
    tc = tdist.ppf((1 + conf) / 2.0, d_arr.size - 1)
    return mean_val, tc * se_val


def calculate_j_beats_gate(
    fold_results: list[dict[str, float]], conf: float = 0.95
) -> dict[str, Any]:
    deltas = [r["dFFT"] for r in fold_results]
    mean_delta, ci_width = paired_ci(deltas, conf)
    ci_low = mean_delta - ci_width
    fft_wins = sum(1 for d in deltas if d > 0)
    n_folds = len(deltas)

    return {
        "j_beats_pass": ci_low > 0,
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": mean_delta + ci_width,
        "fft_wins": fft_wins,
        "n_folds": n_folds,
    }


def calculate_m_latency_gate(inference_times_ms: list[float], budget_ms: float) -> dict[str, Any]:
    mean_latency = np.mean(inference_times_ms) if inference_times_ms else 0.0
    return {
        "latency_pass": mean_latency <= budget_ms,
        "mean_latency_ms": mean_latency,
        "budget_ms": budget_ms,
    }
