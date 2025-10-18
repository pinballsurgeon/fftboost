from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Literal
from typing import cast

import numpy as np

from .api import FFTBoost
from .api import FFTBoostClassifier
from .booster import BoosterConfig


TaskType = Literal["regression", "binary"]


def _detect_task(y: np.ndarray[Any, Any]) -> TaskType:
    vals = np.unique(y.astype(float))
    if vals.size <= 2 and set(np.round(vals, 0).tolist()) <= {0.0, 1.0}:
        return "binary"
    return "regression"


def _val_indices(n_windows: int, val_size: float) -> slice:
    val_n = max(1, int(n_windows * val_size))
    return slice(n_windows - val_n, n_windows)


def _window_count(n_samples: int, fs: float, window_s: float, hop_s: float) -> int:
    win = int(window_s * fs)
    hop = int(hop_s * fs)
    return (n_samples - win) // hop + 1


def _r2_via_corr(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
    m = min(y_true.shape[0], y_pred.shape[0])
    if m <= 1:
        return 0.0
    c = float(np.corrcoef(y_true[:m], y_pred[:m])[0, 1])
    return float(c * c)


@dataclass(frozen=True)
class AutoMLConfig:
    """Configuration for AutoMLController candidate generation.

    n_configs limits the total candidates evaluated deterministically.
    Search spaces combine to produce candidates; controller truncates to n_configs.
    """

    n_configs: int = 12
    # Simple, deterministic search space
    nus: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    stages: tuple[int, ...] = (16, 32, 64)
    k_ffts: tuple[int, ...] = (4, 8, 12)
    min_sep_bins: tuple[int, ...] = (2, 3)
    # Classification expert search
    clf_ks: tuple[int, ...] = (2, 4, 6)
    clf_methods: tuple[Literal["fscore", "mi"], ...] = ("fscore",)
    # Temporal expert search
    temporal_ks: tuple[int, ...] = (2, 4)
    temporal_lag_sets: tuple[tuple[int, ...], ...] = ((1,), (1, 2, 3))


class AutoMLController:
    def __init__(self, cfg: AutoMLConfig | None = None):
        self.cfg = cfg or AutoMLConfig()

    def _candidates(
        self, task: TaskType, n_windows: int, n_bins: int
    ) -> list[BoosterConfig]:
        out: list[BoosterConfig] = []
        for nu in self.cfg.nus:
            for st in self.cfg.stages:
                for k in self.cfg.k_ffts:
                    for sep in self.cfg.min_sep_bins:
                        if task == "binary":
                            # Adaptive MI gating: only include MI for modest sizes
                            methods: tuple[Literal["fscore", "mi"], ...]
                            if (
                                n_bins <= 2000 and n_windows <= 20000
                            ) and self.cfg.clf_methods:
                                methods = self.cfg.clf_methods
                            else:
                                methods = ("fscore",)
                            for ck in self.cfg.clf_ks:
                                for cm in methods:
                                    for tk in self.cfg.temporal_ks:
                                        for lset in self.cfg.temporal_lag_sets:
                                            out.append(
                                                BoosterConfig(
                                                    n_stages=st,
                                                    nu=nu,
                                                    ridge_alpha=1e-3,
                                                    early_stopping_rounds=10,
                                                    loss=cast(Any, "logistic"),
                                                    huber_delta=1.0,
                                                    quantile_alpha=0.5,
                                                    k_fft=k,
                                                    min_sep_bins=sep,
                                                    lambda_hf=0.0,
                                                    clf_use=True,
                                                    clf_k=int(ck),
                                                    clf_method=cast(Any, cm),
                                                    temporal_use=True,
                                                    temporal_k=int(tk),
                                                    temporal_lags=tuple(
                                                        int(x) for x in lset
                                                    ),
                                                )
                                            )
                        else:
                            for loss in ("huber", "squared"):
                                for tk in self.cfg.temporal_ks:
                                    for lset in self.cfg.temporal_lag_sets:
                                        out.append(
                                            BoosterConfig(
                                                n_stages=st,
                                                nu=nu,
                                                ridge_alpha=1e-3,
                                                early_stopping_rounds=10,
                                                loss=cast(Any, loss),
                                                huber_delta=1.0,
                                                quantile_alpha=0.5,
                                                k_fft=k,
                                                min_sep_bins=sep,
                                                lambda_hf=0.0,
                                                temporal_use=True,
                                                temporal_k=int(tk),
                                                temporal_lags=tuple(
                                                    int(x) for x in lset
                                                ),
                                            )
                                        )
        # Limit to n_configs deterministically
        return out[: self.cfg.n_configs]

    def fit_best(
        self,
        signal: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        *,
        fs: float,
        window_s: float,
        hop_s: float,
        val_size: float = 0.2,
        val_gap_windows: int = 1,
        center_target: bool = True,
        budget_stages: int | None = None,
        halving_rounds: int = 0,
    ) -> tuple[object, dict[str, Any]]:
        task = _detect_task(y)
        n_win = _window_count(signal.shape[0], fs, window_s, hop_s)
        val_sl = _val_indices(n_win, val_size)

        best_score = -1.0
        best_model: object | None = None
        best_config: BoosterConfig | None = None

        # Build candidate list (adaptive to problem size)
        win_len = int(window_s * fs)
        n_bins = int(win_len // 2)
        cand_cfgs = self._candidates(task, n_win, n_bins)

        # Successive halving over budgets
        # Determine per-round budgets
        budgets: list[int]
        if budget_stages is None or halving_rounds <= 0:
            # Single full fit per candidate
            budgets = [0]
        else:
            # Smallest -> largest
            base = max(1, budget_stages // (2**halving_rounds))
            budgets = [base * (2**r) for r in range(halving_rounds + 1)]

        survivors = cand_cfgs
        scoreboard: list[tuple[BoosterConfig, float]] = []

        for ridx, b in enumerate(budgets):
            round_scores: list[tuple[BoosterConfig, float, object]] = []
            for cfg in survivors:
                cfg_eff = cfg
                if b > 0:
                    cfg_eff = replace(cfg, n_stages=b)

                if task == "regression":
                    mdl: object = FFTBoost(cfg_eff)
                    cast(FFTBoost, mdl).fit(
                        signal,
                        y,
                        fs=fs,
                        window_s=window_s,
                        hop_s=hop_s,
                        val_size=val_size,
                        val_gap_windows=val_gap_windows,
                        center_target=center_target,
                    )
                    pred = cast(FFTBoost, mdl).predict(
                        signal, fs=fs, window_s=window_s, hop_s=hop_s
                    )
                    y_val = y[: pred.shape[0]][val_sl]
                    p_val = pred[val_sl]
                    score = _r2_via_corr(y_val, p_val)
                else:
                    labels = (y > 0.5).astype(np.float64)
                    clf = FFTBoostClassifier(cfg_eff, threshold="auto")
                    clf.fit(
                        signal,
                        labels,
                        fs=fs,
                        window_s=window_s,
                        hop_s=hop_s,
                        val_size=val_size,
                        val_gap_windows=val_gap_windows,
                    )
                    scores = clf.predict_proba(
                        signal, fs=fs, window_s=window_s, hop_s=hop_s
                    )
                    y_val = labels[: scores.shape[0]][val_sl]
                    s_val = scores[val_sl]
                    thr_opt = clf.threshold_
                    if thr_opt is None:
                        raise RuntimeError("Classifier threshold not set")
                    thr = float(thr_opt)
                    preds = (s_val >= thr).astype(np.int64)
                    score = float((preds == (y_val > 0.5)).mean())
                    mdl = clf
                model = mdl

                round_scores.append((cfg, score, model))
                if ridx == len(budgets) - 1:
                    scoreboard.append((cfg, score))

            # Select survivors for next round (top 50%)
            if ridx < len(budgets) - 1:
                round_scores.sort(key=lambda t: t[1], reverse=True)
                keep = max(1, len(round_scores) // 2)
                survivors = [t[0] for t in round_scores[:keep]]
                # Track current best so far
                if round_scores[0][1] > best_score:
                    best_score = round_scores[0][1]
                    best_model = round_scores[0][2]
                    best_config = round_scores[0][0]
            else:
                # Final round: pick absolute best across final fits
                for cfg, sc, mdl in round_scores:
                    if sc > best_score:
                        best_score = sc
                        best_model = mdl
                        best_config = cfg

        info = {
            "task": task,
            "score": best_score,
            "config": best_config.__dict__ if best_config is not None else {},
            "scoreboard": [
                {"config": cfg.__dict__, "score": s} for (cfg, s) in scoreboard
            ],
            "rounds": len(budgets),
        }
        if best_model is None:
            raise RuntimeError("AutoML failed to produce a model")
        return best_model, info
