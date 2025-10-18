from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from fftboost import AutoMLConfig
from fftboost import AutoMLController
from fftboost import BoosterConfig
from fftboost import FFTBoost
from fftboost import FFTBoostClassifier


def _window_indices(n: int, win: int, hop: int) -> tuple[int, int, int]:
    n_win = (n - win) // hop + 1
    return n_win, win, hop


def _window_rms(x: np.ndarray, fs: float, window_s: float, hop_s: float) -> np.ndarray:
    win = int(window_s * fs)
    hop = int(hop_s * fs)
    n = x.shape[0]
    n_win, win, hop = _window_indices(n, win, hop)
    shape = (n_win, win)
    strides = (x.strides[0] * hop, x.strides[0])
    W = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.sqrt(np.mean(W**2, axis=1))


def _r2_via_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = min(y_true.shape[0], y_pred.shape[0])
    if m <= 1:
        return 0.0
    c = np.corrcoef(y_true[:m], y_pred[:m])[0, 1]
    return float(c**2)


def task_reg_single_tone(rng: np.random.Generator) -> tuple[str, float, bool]:
    fs = 2000
    duration_s = 10.0
    t = np.arange(int(fs * duration_s)) / fs
    # AM 50 Hz in noise
    am = 1.0 + 0.7 * np.sin(2 * np.pi * 0.2 * t)
    x_true = am * np.sin(2 * np.pi * 50.0 * t)
    x = x_true + 0.5 * rng.standard_normal(t.shape[0])

    window_s, hop_s = 0.25, 0.1
    y = _window_rms(x_true, fs, window_s, hop_s)

    cfg = BoosterConfig(n_stages=16, nu=0.3, ridge_alpha=1e-3, min_sep_bins=2, k_fft=4)
    m = FFTBoost(cfg)
    m.fit(x, y, fs=fs, window_s=window_s, hop_s=hop_s, val_size=0.2, center_target=True)
    yhat = m.predict(x, fs=fs, window_s=window_s, hop_s=hop_s)
    r2 = _r2_via_corr(y, yhat)
    return ("REG: single tone", r2, r2 >= 0.95)


def task_class_whale_song(rng: np.random.Generator) -> tuple[str, float, bool]:
    fs = 2000
    duration_s = 20.0
    t = np.arange(int(fs * duration_s)) / fs

    # Two distinct bands: Humpback (low), Orca (high)
    hump = 30.0
    orca = 200.0

    window_s, hop_s = 0.25, 0.1
    win = int(window_s * fs)
    hop = int(hop_s * fs)
    n = t.shape[0]
    n_win, _, _ = _window_indices(n, win, hop)

    # Per-window labels (balanced)
    labels = np.zeros(n_win, dtype=np.float64)
    labels[rng.permutation(n_win)[: n_win // 2]] = 1.0

    # Build a piecewise amplitude schedule per window
    amp_hump = np.where(labels < 0.5, 0.9, 0.2)
    amp_orca = np.where(labels > 0.5, 0.9, 0.2)

    # Render signal by windows, respecting overlaps (additive)
    x = np.zeros_like(t, dtype=np.float64)
    for w in range(n_win):
        s = w * hop
        e = s + win
        tt = t[s:e]
        if tt.size == 0:
            continue
        x[s:e] += amp_hump[w] * np.sin(2 * np.pi * hump * tt)
        x[s:e] += amp_orca[w] * np.sin(2 * np.pi * orca * tt)
    x += 0.2 * rng.standard_normal(n)

    cfg = BoosterConfig(
        n_stages=24, nu=0.4, ridge_alpha=1e-3, min_sep_bins=2, k_fft=6, loss="logistic"
    )
    clf = FFTBoostClassifier(cfg, threshold="auto")
    clf.fit(x, labels, fs=fs, window_s=window_s, hop_s=hop_s, val_size=0.2)
    # Evaluate on validation block
    scores = clf.predict_proba(x, fs=fs, window_s=window_s, hop_s=hop_s)
    total = scores.shape[0]
    val_n = max(1, int(total * 0.2))
    val_idx = slice(total - val_n, total)
    y_val = labels[:total][val_idx]
    y_pred = (scores[val_idx] >= float(clf.threshold_)).astype(np.int64)
    acc = float((y_pred == (y_val > 0.5)).mean())
    return ("CLASS: whale song (bin)", acc, acc >= 0.95)


def task_class_morse_code(rng: np.random.Generator) -> tuple[str, float, bool]:
    # Expected to struggle until temporal experts exist; report only.
    fs = 2000
    duration_s = 20.0
    t = np.arange(int(fs * duration_s)) / fs
    window_s, hop_s = 0.25, 0.1
    win, hop = int(window_s * fs), int(hop_s * fs)
    n = t.shape[0]
    n_win, _, _ = _window_indices(n, win, hop)

    # Sparse on/off at 500 Hz
    carrier = 500.0
    on_windows = np.arange(3, n_win, 6)
    labels = np.zeros(n_win, dtype=np.float64)
    labels[on_windows] = 1.0

    x = np.zeros_like(t, dtype=np.float64)
    for w in range(n_win):
        s = w * hop
        e = s + win
        tt = t[s:e]
        if tt.size == 0:
            continue
        amp = 0.6 if labels[w] > 0.5 else 0.0
        x[s:e] += amp * np.sin(2 * np.pi * carrier * tt)
    # Heavy noise + distractors
    x += 0.6 * rng.standard_normal(n)
    for f in [60.0, 90.0, 150.0, 220.0]:
        x += 0.15 * np.sin(2 * np.pi * f * t)

    cfg = BoosterConfig(
        n_stages=24, nu=0.4, ridge_alpha=1e-3, min_sep_bins=2, k_fft=6, loss="logistic"
    )
    clf = FFTBoostClassifier(cfg, threshold="auto")
    clf.fit(x, labels, fs=fs, window_s=window_s, hop_s=hop_s, val_size=0.2)
    scores = clf.predict_proba(x, fs=fs, window_s=window_s, hop_s=hop_s)
    total = scores.shape[0]
    val_n = max(1, int(total * 0.2))
    val_idx = slice(total - val_n, total)
    y_val = labels[:total][val_idx]
    y_pred = (scores[val_idx] >= float(clf.threshold_)).astype(np.int64)
    acc = float((y_pred == (y_val > 0.5)).mean())
    return ("CLASS: morse (sparse)", acc, acc >= 0.95)


@dataclass
class Result:
    name: str
    metric: float
    passed: bool


def run_phase1(strict_gate: bool) -> int:
    rng = np.random.default_rng(123)
    tasks: list[tuple[str, float, bool]] = []
    tasks.append(task_reg_single_tone(rng))
    tasks.append(task_class_whale_song(rng))
    # Report-only for morse; do not gate in phase-1
    tasks.append(task_class_morse_code(rng))

    results = [Result(name=t[0], metric=t[1], passed=t[2]) for t in tasks]

    print("\nGauntlet (Phase 1)")
    print("-----------------")
    for r in results:
        status = "PASS" if r.passed else "INFO" if "morse" in r.name else "FAIL"
        print(f"{r.name:28s}  {r.metric:8.4f}  {status}")

    if strict_gate:
        # Gate only tasks designed to pass in Phase-1
        hard = [r for r in results if "morse" not in r.name]
        if not all(r.passed for r in hard):
            return 1
    return 0


def run_phase2(strict_gate: bool) -> int:
    rng = np.random.default_rng(123)
    tasks: list[tuple[str, float, bool]] = []
    tasks.append(task_reg_single_tone_low_snr(rng))
    tasks.append(task_reg_needle_haystack(rng))
    tasks.append(task_class_whale_song(rng))
    # morse remains INFO-only
    tasks.append(task_class_morse_code(rng))

    print("\nGauntlet (Phase 2)")
    print("-----------------")
    all_ok = True
    for name, metric, passed in tasks:
        status = "PASS" if passed else ("INFO" if "morse" in name else "FAIL")
        print(f"{name:28s}  {metric:8.4f}  {status}")
        if strict_gate and status == "FAIL":
            all_ok = False
    return 0 if (not strict_gate or all_ok) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase1-gate", action="store_true", help="Fail if Phase-1 gates not met"
    )
    ap.add_argument(
        "--phase2-gate", action="store_true", help="Fail if Phase-2 gates not met"
    )
    args = ap.parse_args()
    code = run_phase1(strict_gate=args.phase1_gate)
    if code != 0:
        raise SystemExit(code)
    if args.phase2_gate:
        code = run_phase2(strict_gate=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()


def task_reg_single_tone_low_snr(rng: np.random.Generator) -> tuple[str, float, bool]:
    fs = 2000
    duration_s = 10.0
    t = np.arange(int(fs * duration_s)) / fs
    am = 1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    x_true = am * np.sin(2 * np.pi * 50.0 * t)
    x = x_true + 1.0 * rng.standard_normal(t.shape[0])  # lower SNR

    window_s, hop_s = 0.25, 0.1
    y = _window_rms(x_true, fs, window_s, hop_s)

    # Baseline
    base_cfg = BoosterConfig(
        n_stages=16, nu=0.1, ridge_alpha=1e-3, min_sep_bins=3, k_fft=4
    )
    base = FFTBoost(base_cfg)
    base.fit(
        x, y, fs=fs, window_s=window_s, hop_s=hop_s, val_size=0.2, center_target=True
    )
    base_r2 = _r2_via_corr(y, base.predict(x, fs=fs, window_s=window_s, hop_s=hop_s))

    # AutoML
    automl = AutoMLController(AutoMLConfig(n_configs=12))
    best, info = automl.fit_best(
        x,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        center_target=True,
        budget_stages=64,
        halving_rounds=2,
    )
    yhat = best.predict(x, fs=fs, window_s=window_s, hop_s=hop_s)
    r2 = _r2_via_corr(y, yhat)
    print("  AutoML best config:", info.get("config", {}))
    print(f"  AutoML best score: {info.get('score', 0.0):.4f}")
    sb = sorted(
        info.get("scoreboard", []), key=lambda d: d.get("score", 0.0), reverse=True
    )
    print("  Top candidates:")
    for d in sb[:3]:
        print(
            "   ",
            {
                k: d["config"].get(k)
                for k in ("nu", "n_stages", "k_fft", "min_sep_bins", "loss")
            },
            f"score={d.get('score', 0.0):.4f}",
        )
    passed = (r2 >= 0.80) and (r2 >= base_r2 + 0.05)
    return ("REG: low SNR (AutoML vs base)", r2, passed)


def task_reg_needle_haystack(rng: np.random.Generator) -> tuple[str, float, bool]:
    fs = 2000
    duration_s = 30.0
    t = np.arange(int(fs * duration_s)) / fs

    bg_freqs = [70, 95, 130, 180, 230, 270, 350, 420]
    bg_amps = [0.5] * len(bg_freqs)
    x_bg = np.zeros_like(t, dtype=np.float64)
    for f, a in zip(bg_freqs, bg_amps):
        x_bg += a * np.sin(2 * np.pi * f * t)

    f_needle = 123.0
    env = 0.4 + 0.4 * (0.5 * (1.0 + np.sin(2 * np.pi * 0.15 * t)))
    x_needle = env * np.sin(2 * np.pi * f_needle * t)
    x = x_bg + x_needle + 0.35 * rng.standard_normal(t.shape[0])

    window_s, hop_s = 0.256, 0.128
    y = _window_rms(x_bg + x_needle, fs, window_s, hop_s)

    base_cfg = BoosterConfig(
        n_stages=16, nu=0.1, ridge_alpha=1e-3, min_sep_bins=3, k_fft=4
    )
    base = FFTBoost(base_cfg)
    base.fit(
        x, y, fs=fs, window_s=window_s, hop_s=hop_s, val_size=0.2, center_target=True
    )
    base_r2 = _r2_via_corr(y, base.predict(x, fs=fs, window_s=window_s, hop_s=hop_s))

    automl = AutoMLController(AutoMLConfig(n_configs=12))
    best, info = automl.fit_best(
        x,
        y,
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        val_size=0.2,
        center_target=True,
        budget_stages=64,
        halving_rounds=2,
    )
    yhat = best.predict(x, fs=fs, window_s=window_s, hop_s=hop_s)
    r2 = _r2_via_corr(y, yhat)
    print("  AutoML best config:", info.get("config", {}))
    print(f"  AutoML best score: {info.get('score', 0.0):.4f}")
    sb = sorted(
        info.get("scoreboard", []), key=lambda d: d.get("score", 0.0), reverse=True
    )
    print("  Top candidates:")
    for d in sb[:3]:
        print(
            "   ",
            {
                k: d["config"].get(k)
                for k in ("nu", "n_stages", "k_fft", "min_sep_bins", "loss")
            },
            f"score={d.get('score', 0.0):.4f}",
        )
    passed = (r2 >= 0.85) and (r2 >= base_r2 + 0.05)
    return ("REG: needle (AutoML vs base)", r2, passed)
