import argparse
import json
from typing import Any

import numpy as np

from fftboost.api import FFTBoost, load_config_from_yaml


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


def main() -> None:
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
    model.run_evaluation()

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
