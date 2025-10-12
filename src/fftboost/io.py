from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from typing import cast

import numpy as np

from .boosting import StageRecord


@dataclass(frozen=True)
class BoosterArtifact:
    schema_version: str
    fftboost_version: str
    freqs: np.ndarray
    stages: list[StageRecord]
    config: dict[str, Any] | None = None


def save_model(artifact: BoosterArtifact, path_prefix: str) -> None:
    """
    Save artifact to path_prefix/model.json and path_prefix/model.npz.
    Arrays are stored in NPZ; JSON stores metadata and NPZ keys.
    """
    os.makedirs(path_prefix, exist_ok=True)
    json_path = os.path.join(path_prefix, "model.json")
    npz_path = os.path.join(path_prefix, "model.npz")

    arrays: dict[str, np.ndarray] = {"freqs": artifact.freqs}
    stages_json: list[dict[str, Any]] = []
    for i, s in enumerate(artifact.stages):
        w_key = f"s{i}_weights"
        mu_key = f"s{i}_mu"
        sd_key = f"s{i}_sigma"
        arrays[w_key] = s.weights
        arrays[mu_key] = s.mu
        arrays[sd_key] = s.sigma
        stages_json.append(
            {
                "weights_key": w_key,
                "mu_key": mu_key,
                "sigma_key": sd_key,
                "gamma": float(s.gamma),
                "ridge_alpha": float(s.ridge_alpha),
                "nu": float(s.nu),
                "descriptors": s.descriptors,
            }
        )

    meta = {
        "schema_version": artifact.schema_version,
        "fftboost_version": artifact.fftboost_version,
        "config": artifact.config or {},
        "n_stages": len(artifact.stages),
        "stages": stages_json,
    }

    # Stable JSON formatting
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, sort_keys=True, separators=(",", ":")))

    # Save arrays; order of fields is deterministic due to Python 3.7+ dict order
    _save_func: Any = np.savez
    _save_func(npz_path, **arrays)


def load_model(path_prefix: str) -> BoosterArtifact:
    """
    Load artifact from path_prefix/model.json and path_prefix/model.npz.
    """
    json_path = os.path.join(path_prefix, "model.json")
    npz_path = os.path.join(path_prefix, "model.npz")

    with open(json_path, encoding="utf-8") as f:
        meta = json.loads(f.read())

    with np.load(npz_path) as z:
        freqs = cast(np.ndarray, z["freqs"])  # freq vector
        stages: list[StageRecord] = []
        for i, s_meta in enumerate(cast(list[dict[str, Any]], meta["stages"])):
            w = cast(np.ndarray, z[s_meta["weights_key"]])
            mu = cast(np.ndarray, z[s_meta["mu_key"]])
            sigma = cast(np.ndarray, z[s_meta["sigma_key"]])
            stages.append(
                StageRecord(
                    weights=w,
                    descriptors=cast(list[dict[str, object]], s_meta["descriptors"]),
                    mu=mu,
                    sigma=sigma,
                    gamma=float(s_meta["gamma"]),
                    ridge_alpha=float(s_meta["ridge_alpha"]),
                    nu=float(s_meta["nu"]),
                )
            )

    return BoosterArtifact(
        schema_version=cast(str, meta["schema_version"]),
        fftboost_version=cast(str, meta["fftboost_version"]),
        freqs=freqs,
        stages=stages,
        config=cast(dict[str, Any], meta.get("config", {})),
    )
