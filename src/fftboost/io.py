from __future__ import annotations

import hashlib
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
    freqs: np.ndarray[Any, Any]
    stages: list[StageRecord]
    config: dict[str, Any] | None = None


def _stable_json_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_meta_and_arrays(
    meta: dict[str, Any], arrays: dict[str, np.ndarray[Any, Any]]
) -> str:
    h = hashlib.sha256()
    h.update(_stable_json_dumps(meta).encode("utf-8"))
    for name in sorted(arrays.keys()):
        arr = arrays[name]
        h.update(name.encode("utf-8"))
        h.update(hashlib.sha256(arr.tobytes(order="C")).digest())
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
    return h.hexdigest()


def save_model(artifact: BoosterArtifact, path_prefix: str) -> dict[str, str]:
    """
    Save artifact to path_prefix.json and path_prefix.npz.
    Returns dict with paths and deterministic sha256 over meta+arrays.
    """
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    json_path = f"{path_prefix}.json"
    npz_path = f"{path_prefix}.npz"

    arrays: dict[str, np.ndarray[Any, Any]] = {"freqs": artifact.freqs}
    stages_json: list[dict[str, Any]] = []
    for i, s in enumerate(artifact.stages):
        keys = {"weights": f"s{i}_weights", "mu": f"s{i}_mu", "sigma": f"s{i}_sigma"}
        arrays[keys["weights"]] = s.weights
        arrays[keys["mu"]] = s.mu
        arrays[keys["sigma"]] = s.sigma
        stages_json.append(
            {
                "keys": keys,
                "gamma": float(s.gamma),
                "ridge_alpha": float(s.ridge_alpha),
                "nu": float(s.nu),
                "descriptors": s.descriptors,
            }
        )

    meta: dict[str, Any] = {
        "schema_version": artifact.schema_version,
        "fftboost_version": artifact.fftboost_version,
        "config": artifact.config or {},
        "n_stages": len(artifact.stages),
        "stages": stages_json,
    }

    sha256 = _hash_meta_and_arrays(meta, arrays)

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(_stable_json_dumps(meta))

    _savez: Any = np.savez_compressed
    _savez(npz_path, **arrays)
    return {"json_path": json_path, "npz_path": npz_path, "sha256": sha256}


def load_model(path_prefix: str) -> BoosterArtifact:
    """
    Load artifact from path_prefix.json and path_prefix.npz.
    """
    json_path = f"{path_prefix}.json"
    npz_path = f"{path_prefix}.npz"

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)

    with np.load(npz_path) as z:
        freqs = cast(np.ndarray[Any, Any], z["freqs"])  # freq vector
        stages: list[StageRecord] = []
        for s_meta in cast(list[dict[str, Any]], meta["stages"]):
            keys = cast(dict[str, str], s_meta["keys"])
            w = cast(np.ndarray[Any, Any], z[keys["weights"]])
            mu = cast(np.ndarray[Any, Any], z[keys["mu"]])
            sigma = cast(np.ndarray[Any, Any], z[keys["sigma"]])
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
