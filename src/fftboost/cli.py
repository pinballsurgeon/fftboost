from __future__ import annotations

import argparse

import numpy as np

from .api import FFTBoost


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run FFTBoost AutoML on a toy signal and print best config."
    )
    p.add_argument("--task", choices=["regression", "binary"], default="regression")
    p.add_argument("--fs", type=float, default=200.0)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--window_s", type=float, default=0.5)
    p.add_argument("--hop_s", type=float, default=0.25)
    args = p.parse_args()

    t = np.arange(0.0, args.seconds, 1.0 / args.fs, dtype=np.float64)
    if args.task == "regression":
        x = np.sin(2 * np.pi * 7.0 * t)
        y = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        center = True
    else:
        x = np.sin(2 * np.pi * 10.0 * t)
        y = (np.sin(2 * np.pi * 0.3 * t) > 0.0).astype(np.float64)
        center = False

    model, info = FFTBoost.auto(
        x,
        y,
        fs=args.fs,
        window_s=args.window_s,
        hop_s=args.hop_s,
        center_target=center,
        budget_stages=32,
        halving_rounds=1,
    )
    print("AutoML best:")
    print(f"  task: {info.get('task')}\n  score: {info.get('score')}")
    print("  config:")
    cfg = info.get("config", {})
    for k, v in (cfg or {}).items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
