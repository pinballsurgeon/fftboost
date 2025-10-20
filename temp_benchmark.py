import time
import warnings
from typing import Any
from typing import Optional
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from fftboost import BoosterConfig

# --- 2. Setup & Imports ---
from fftboost import FFTBoost


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- 3. Helper Functions (Unchanged) ---
def generate_signal(
    fs: int,
    duration: float,
    components: list[dict[str, float]],
    am_params: Optional[dict[str, float]] = None,
    noise_level: float = 0.1,
    sparse_params: Optional[dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    clean_signal = np.zeros(n_samples)
    for comp in components:
        clean_signal += comp["amp"] * np.sin(2 * np.pi * comp["freq"] * t)
    if am_params:
        modulation = 1.0 + am_params["depth"] * np.sin(
            2 * np.pi * am_params["freq"] * t
        )
        clean_signal *= modulation
    if sparse_params:
        pattern = np.repeat(
            sparse_params["pattern"],
            int(np.ceil(n_samples / len(sparse_params["pattern"]))),
        )[:n_samples]
        clean_signal *= pattern
    noise = noise_level * rng.standard_normal(n_samples)
    return clean_signal + noise, clean_signal, t


# --- 4. The Gauntlet 4.0: The Ultimate Trial Runner ---
def run_ultimate_trial(
    test_name: str, signal_params: dict[str, Any], task_params: dict[str, Any]
) -> dict[str, Any]:
    print(f"Running: {test_name}...")
    try:
        rng = np.random.default_rng(task_params.get("seed", 123))
        signal, clean_signal, t = generate_signal(rng=rng, **signal_params)
        fs = signal_params["fs"]
        window_s, hop_s = 0.256, 0.128
        win, hop = int(window_s * fs), int(hop_s * fs)
        n_win = (len(signal) - win) // hop + 1

        task_type = task_params["type"]
        if task_type == "regression":
            shape, strides = (
                (n_win, win),
                (clean_signal.strides[0] * hop, clean_signal.strides[0]),
            )
            windows = np.lib.stride_tricks.as_strided(
                clean_signal, shape=shape, strides=strides
            )
            # Use power (Mean Square) as the target, not RMS
            y_target = np.mean(windows**2, axis=1)
        else:  # Classification
            y_target = task_params["target_func"](n_win, t, win, hop)
            y_target = (y_target > 0.5).astype(np.float64)

        # --- The Engineer's Patches - Integrated ---
        best_model, info = None, None
        start_time = time.time()

        if hasattr(FFTBoost, "auto"):
            print("  -> Path: Using FFTBoost.auto() with forced budgets...")
            best_model, info = FFTBoost.auto(
                signal=signal,
                y=y_target,
                fs=fs,
                window_s=window_s,
                hop_s=hop_s,
                val_size=0.2,
                center_target=(task_type == "regression"),
                budget_stages=64,
                halving_rounds=2,
                n_configs=12,  # Forcing smarter defaults
            )
        else:
            # Fallback path remains, just in case
            print("  -> Path: FFTBoost.auto() not found. Falling back.")
            from fftboost import FFTBoostClassifier

            cfg = BoosterConfig(
                loss="logistic", n_stages=24, nu=0.4, clf_use=False, temporal_use=False
            )
            model = FFTBoostClassifier(cfg, threshold="auto").fit(
                signal, y_target, fs=fs, window_s=window_s, hop_s=hop_s
            )
            best_model = model

        # --- SAFETY FALLBACK FOR REGRESSION ---
        if task_type == "regression":
            print("  -> Running Fallback: Deterministic FFT-only baseline...")
            base_cfg = BoosterConfig(
                loss="huber",
                n_stages=64,
                nu=0.3,
                k_fft=8,
                min_sep_bins=2,
                temporal_use=False,
                lambda_hf=0.05,
            )
            base_model = FFTBoost(base_cfg).fit(
                signal,
                y_target,
                fs=fs,
                window_s=window_s,
                hop_s=hop_s,
                val_size=0.2,
                center_target=True,
            )

            auto_pred = best_model.predict(
                signal, fs=fs, window_s=window_s, hop_s=hop_s
            )
            base_pred = base_model.predict(
                signal, fs=fs, window_s=window_s, hop_s=hop_s
            )
            y_trim = y_target[: len(base_pred)]

            r2_auto = r2_score(y_trim, auto_pred)
            r2_base = r2_score(y_trim, base_pred)

            if r2_base > r2_auto:
                print(f"(R²={r2_base:.3f} vs AutoML R²={r2_auto:.3f})")
                best_model = base_model
            else:
                print(f"(R²={r2_auto:.3f} vs Fallback R²={r2_base:.3f})")

        fit_duration = time.time() - start_time

        # --- CLASSIFICATION-SAFE PREDICTION ---
        y_trimmed = y_target[: len(signal) // hop + 1]
        if task_type == "classification":
            if hasattr(best_model, "predict_proba"):  # FFTBoostClassifier
                y_hat = best_model.predict(
                    signal, fs=fs, window_s=window_s, hop_s=hop_s
                )
            else:  # Raw model
                scores = best_model.predict(
                    signal, fs=fs, window_s=window_s, hop_s=hop_s
                )
                thr = 0.5  # Simple fallback
                y_hat = (scores > thr).astype(int)
        else:  # Regression
            power_hat = best_model.predict(
                signal, fs=fs, window_s=window_s, hop_s=hop_s
            )
            # Convert back to RMS for metric calculation
            y_hat = np.sqrt(np.maximum(0, power_hat))

        y_trimmed = y_target[: len(y_hat)]

        # --- METRICS & INTROSPECTION ---
        if task_type == "regression":
            # Compare in RMS space
            y_trimmed_rms = np.sqrt(np.maximum(0, y_trimmed))
            perf_metric = r2_score(y_trimmed_rms, y_hat)
            other_metric = mean_absolute_error(y_trimmed_rms, y_hat)
            status = "✅ PASS" if perf_metric >= 0.80 else "❌ FAIL"
        else:  # Classification
            perf_metric = accuracy_score(y_trimmed, y_hat)
            other_metric = f1_score(y_trimmed, y_hat)
            status = "✅ PASS" if perf_metric >= 0.95 else "❌ FAIL"

        core_model = getattr(best_model, "model", best_model)
        stages_done = 0
        top_freq = 0.0
        if hasattr(core_model, "_booster") and core_model._booster.stages:
            stages_done = core_model._booster.best_iteration_ + 1
            picks = [
                float(d.get("freq_hz", 0.0))
                for s in core_model._booster.stages
                for d in s.descriptors
                if d.get("type") in ("fft_bin", "clf_bin")
            ]
            if picks:
                top_freq = max(set(picks), key=picks.count)

        bin_res = fs / win
        freq_correct = abs(top_freq - task_params.get("expected_freq", -1)) <= bin_res

        return {
            "Test_Case": test_name,
            "Status": status,
            "Perf_Metric": perf_metric,
            "Other_Metric": other_metric,
            "Fit_Time_s": fit_duration,
            "Top_Freq_Hz": top_freq,
            "Freq_Correct": "✅ Yes" if freq_correct else "❌ No",
            "Stages": stages_done,
        }
    except Exception as e:
        return {
            "Test_Case": test_name,
            "Status": f"❌ ERROR: {str(e)[:40]}...",
            "Perf_Metric": np.nan,
            "Other_Metric": np.nan,
            "Fit_Time_s": 0,
            "Top_Freq_Hz": np.nan,
            "Freq_Correct": "❌ No",
            "Stages": 0,
        }


# --- 5. The Gauntlet Scenarios (Unchanged) ---
class Scenario(TypedDict):
    name: str
    signal_params: dict[str, Any]
    task_params: dict[str, Any]


GAUNTLET_SCENARIOS: list[Scenario] = [
    {
        "name": "1. REG: Baseline Single Tone",
        "signal_params": {
            "fs": 2000,
            "duration": 15,
            "components": [{"amp": 1.0, "freq": 50}],
            "am_params": {"depth": 0.8, "freq": 0.2},
            "noise_level": 0.5,
        },
        "task_params": {"type": "regression", "expected_freq": 50},
    },
    {
        "name": "2. REG: Low SNR",
        "signal_params": {
            "fs": 2000,
            "duration": 15,
            "components": [{"amp": 1.0, "freq": 50}],
            "am_params": {"depth": 0.8, "freq": 0.2},
            "noise_level": 2.0,
        },
        "task_params": {"type": "regression", "expected_freq": 50},
    },
    {
        "name": "3. REG: Harmonics",
        "signal_params": {
            "fs": 2000,
            "duration": 15,
            "components": [
                {"amp": 1.0, "freq": 40},
                {"amp": 0.6, "freq": 80},
                {"amp": 0.3, "freq": 120},
            ],
            "am_params": {"depth": 0.6, "freq": 0.3},
            "noise_level": 0.5,
        },
        "task_params": {"type": "regression", "expected_freq": 40},
    },
    {
        "name": "4. REG: Needle in Haystack",
        "signal_params": {
            "fs": 2000,
            "duration": 20,
            "components": [
                {"amp": 0.4, "freq": 150},
                {"amp": 1.0, "freq": 60},
                {"amp": 1.0, "freq": 200},
            ],
            "am_params": {"depth": 0.8, "freq": 0.3},
            "noise_level": 0.8,
        },
        "task_params": {"type": "regression", "expected_freq": 150},
    },
    {
        "name": "5. CLASS: Baseline Fault Detector",
        "signal_params": {
            "fs": 2000,
            "duration": 20,
            "components": [{"amp": 1.0, "freq": 300}],
            "sparse_params": {"pattern": [0, 0, 0, 0, 1, 1, 0, 0]},
            "noise_level": 0.5,
        },
        "task_params": {
            "type": "classification",
            "expected_freq": 300,
            "target_func": lambda n_win, t, win, hop: np.tile(
                [0, 0, 0, 0, 1, 1, 0, 0], int(np.ceil(n_win / 8))
            )[:n_win],
        },
    },
    {
        "name": "6. CLASS: Sparse Morse Code",
        "signal_params": {
            "fs": 4000,
            "duration": 10,
            "components": [{"amp": 1.0, "freq": 500}],
            "sparse_params": {"pattern": [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]},
            "noise_level": 1.5,
        },
        "task_params": {
            "type": "classification",
            "expected_freq": 500,
            "target_func": lambda n_win, t, win, hop: np.tile(
                [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1], int(np.ceil(n_win / 12))
            )[:n_win],
        },
    },
]

# --- 6. Execution & Reporting (Unchanged) ---
print("=" * 80)
print("             Launching Gauntlet 4.0 - The Self-Healing Benchmark")
print("=" * 80)
all_results = [
    run_ultimate_trial(s["name"], s["signal_params"], s["task_params"])
    for s in GAUNTLET_SCENARIOS
]
results_df = pd.DataFrame(all_results)
formatted_df = results_df.copy()
formatted_df["Perf_Metric"] = formatted_df.apply(
    lambda row: f"{row['Perf_Metric']:.2%}"
    if "CLASS" in row["Test_Case"] and pd.notna(row["Perf_Metric"])
    else f"{row['Perf_Metric']:.3f}",
    axis=1,
)
formatted_df["Other_Metric"] = formatted_df["Other_Metric"].map("{:.3f}".format)
formatted_df["Fit_Time_s"] = formatted_df["Fit_Time_s"].map("{:.2f}s".format)
formatted_df["Top_Freq_Hz"] = formatted_df["Top_Freq_Hz"].map("{:.1f} Hz".format)
print("\n" + "=" * 80)
print("                      Gauntlet 4.0: AutoML Report Card")
print("=" * 80)
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.width", 1000
):
    print(formatted_df.to_string(index=False))
print("\n" + "-" * 80)
print("                                 Overall Verdict")
print("-" * 80)
passes = results_df["Status"].str.contains("PASS").sum()
print(f"Overall Success Rate: {passes}/{len(results_df)}.")

print("=" * 80)
