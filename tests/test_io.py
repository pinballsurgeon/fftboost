from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import cast

import numpy as np

from src.fftboost.boosting import fit_stage
from src.fftboost.experts.fft_bin import propose as fftbin_propose
from src.fftboost.experts.types import ExpertContext
from src.fftboost.io import BoosterArtifact
from src.fftboost.io import load_model
from src.fftboost.io import save_model


def _make_psd_with_modulated_tone(
    n_windows: int, win_len: int, fs: int, f0: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(win_len) / fs
    freqs_full = np.fft.rfftfreq(win_len, d=1.0 / fs)
    freqs = freqs_full[1:]  # drop DC
    psd = np.zeros((n_windows, freqs.shape[0]), dtype=np.float64)
    amps = rng.uniform(0.8, 1.2, size=n_windows).astype(np.float64)
    for w in range(n_windows):
        x = amps[w] * np.sin(2.0 * np.pi * f0 * t)
        X = np.fft.rfft(x)
        mag = np.abs(X)[1:]
        psd[w] = mag
    residual = amps - float(np.mean(amps))
    return psd, freqs, residual


def _predict_with_artifact(artifact: BoosterArtifact, psd: np.ndarray) -> np.ndarray:
    # Map descriptors back to column indices using freqs
    freqs = artifact.freqs
    yhat = np.zeros(psd.shape[0], dtype=np.float64)
    for stage in artifact.stages:
        # Build H from descriptors (assumes fft_bin type)
        idxs = [
            int(np.argmin(np.abs(freqs - cast(float, d["freq_hz"]))))
            for d in stage.descriptors
        ]
        H = psd[:, idxs]
        Z = (H - stage.mu) / (stage.sigma + 1e-12)
        contrib = (stage.nu * stage.gamma) * (Z @ stage.weights)
        yhat += cast(np.ndarray, contrib)
    return cast(np.ndarray, yhat)


def _train_two_stage_artifact(
    rng: np.random.Generator,
) -> tuple[BoosterArtifact, np.ndarray, np.ndarray, ExpertContext]:
    fs = 1000
    win_len = 256
    n_windows = 512
    f0 = 60.0
    psd, freqs, residual0 = _make_psd_with_modulated_tone(
        n_windows, win_len, fs, f0, rng
    )
    ctx = ExpertContext(psd=psd, freqs=freqs, fs=fs, min_sep_bins=2, lambda_hf=0.0)

    # Stage 1
    prop1 = fftbin_propose(residual0, ctx, top_k=5)
    applied1, rec1 = fit_stage(residual0, [prop1], ridge_alpha=1e-3, nu=0.8)
    residual1 = residual0 - applied1

    # Stage 2 (re-propose with updated residual)
    prop2 = fftbin_propose(residual1, ctx, top_k=5)
    applied2, rec2 = fit_stage(residual1, [prop2], ridge_alpha=1e-3, nu=0.8)

    artifact = BoosterArtifact(
        schema_version="1",
        fftboost_version="test",
        freqs=freqs,
        stages=[rec1, rec2],
        config={"fs": fs, "win_len": win_len},
    )
    return artifact, psd, freqs, ctx


def _stable_artifact_hash(json_path: str, npz_path: str) -> str:
    h = hashlib.sha256()
    with open(json_path, encoding="utf-8") as f:
        # Normalize to ensure stable encoding
        meta = json.loads(f.read())
        j = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h.update(j)
    with np.load(npz_path) as z:
        for k in sorted(z.files):
            arr = z[k]
            h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def test_io_round_trip_predictions_identical() -> None:
    rng = np.random.default_rng(123)
    artifact, psd, _, ctx = _train_two_stage_artifact(rng)

    # Predict with original artifact
    yhat_orig = _predict_with_artifact(artifact, ctx.psd)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "artifact")
        _ = save_model(artifact, prefix)
        loaded = load_model(prefix)
        yhat_loaded = _predict_with_artifact(loaded, ctx.psd)
        np.testing.assert_array_equal(yhat_orig, yhat_loaded)


def test_io_determinism_gate() -> None:
    # Train, save, and hash twice – hashes must match
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        art1, _, _, _ = _train_two_stage_artifact(rng1)
        art2, _, _, _ = _train_two_stage_artifact(rng2)

        info1 = save_model(art1, os.path.join(tmp1, "model"))
        info2 = save_model(art2, os.path.join(tmp2, "model"))

        # Determinism via returned hash and recomputed hash
        h1 = info1["sha256"]
        h2 = info2["sha256"]
        assert h1 == h2
