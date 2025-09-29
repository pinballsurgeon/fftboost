import argparse
import json
from typing import Any

import numpy as np

from fftboost.api import FFTBoost, load_config_from_yaml


# Move the generator to the top level of the module
def generate_synthetic_signals(
    config: dict[str, Any], seed: int
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Modify the API to accept the generator
def run_cli_evaluation(model: FFTBoost) -> None:
    from fftboost.core.cv import run_full_cv_evaluation

    model.results = run_full_cv_evaluation(model.config, generate_synthetic_signals)


def main() -> None:
    # ... (main function is identical)
    parser = argparse.ArgumentParser(description="Run the FFTBoost evaluation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--save-telemetry",
        type=str,
        help="Path to save the final telemetry JSON file.",
    )
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    model = FFTBoost(config)
    run_cli_evaluation(model)

    j_beats_telemetry = model.get_j_beats_telemetry()
    m_latency_telemetry = model.get_m_latency_telemetry()

    final_telemetry = {
        "acceptance": {**j_beats_telemetry, **m_latency_telemetry},
        "config": config,
    }

    telemetry_json = json.dumps(final_telemetry, indent=4, cls=NumpyEncoder)
    print(telemetry_json)

    if args.save_telemetry:
        with open(args.save_telemetry, "w") as f:
            f.write(telemetry_json)


if __name__ == "__main__":
    main()
