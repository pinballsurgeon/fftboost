from typing import Any, Optional, cast

import numpy as np
import yaml

from fftboost.core.cv import run_full_cv_evaluation
from fftboost.core.metrics import (
    calculate_j_beats_gate,
    calculate_m_latency_gate,
)
from fftboost.core.types import CvResults, JBeatsResult, LatencyResult


# A placeholder generator for the API, in case a user wants to run a demo
def _placeholder_generator(
    config: dict[str, Any], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fs = config.get("fs", 1000)
    duration_s = config.get("duration_s", 1)
    n_total = int(duration_s * fs)
    tvec = np.arange(n_total) / fs
    i_signal = np.random.randn(n_total)
    v_signal = np.random.randn(n_total)
    return i_signal, v_signal, tvec


class FFTBoost:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.results: Optional[CvResults] = None

    def run_evaluation_with_generator(self) -> None:
        from fftboost.cli import generate_synthetic_signals

        self.results = run_full_cv_evaluation(self.config, generate_synthetic_signals)

    def get_j_beats_telemetry(self) -> JBeatsResult:
        if self.results is None or self.results["fold_results"] is None:
            return {
                "j_beats_pass": False,
                "mean_delta": -1.0,
                "ci_low": -1.0,
                "ci_high": -1.0,
                "fft_wins": 0,
                "n_folds": 0,
            }
        return calculate_j_beats_gate(self.results["fold_results"])

    def get_m_latency_telemetry(self) -> LatencyResult:
        budget = self.config.get("latency_budget_ms", 2.0)
        times: list[float] = []
        if self.results and self.results["inference_times_ms"]:
            times = self.results["inference_times_ms"]
        return calculate_m_latency_gate(times, budget)


def load_config_from_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        untyped_config = yaml.safe_load(f)
    config = cast(dict[str, Any], untyped_config)
    return config
