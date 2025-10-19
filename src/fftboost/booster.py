from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast

import numpy as np

from .boosting import StageRecord
from .boosting import fit_stage
from .experts.clf_bin import propose as clf_bin_propose
from .experts.fft_bin import propose as fftbin_propose
from .experts.sk_band import propose as sk_band_propose
from .experts.temporal import propose_burstpool as temporal_pool_propose
from .experts.temporal import propose_flux as temporal_flux_propose
from .experts.temporal import propose_lagstack as temporal_lagstack_propose
from .experts.types import ExpertContext
from .io import BoosterArtifact
from .losses import HuberLoss
from .losses import LogisticLoss
from .losses import QuantileLoss
from .losses import SquaredLoss


@dataclass(frozen=True)
class BoosterConfig:
    n_stages: int = 100
    nu: float = 0.1
    ridge_alpha: float = 1e-3
    early_stopping_rounds: int = 10
    loss: Literal["squared", "huber", "quantile", "logistic"] = "huber"
    huber_delta: float = 1.0
    quantile_alpha: float = 0.5
    k_fft: int = 4
    min_sep_bins: int = 3
    lambda_hf: float = 0.0
    default_band_edges_hz: list[tuple[float, float]] | None = None
    sk_n_select: int = 1
    sk_kurtosis_boost: float = 0.0
    # Classification-aware expert
    clf_use: bool = True
    clf_k: int = 4
    clf_method: Literal["fscore", "mi"] = "fscore"
    # Temporal experts
    temporal_use: bool = False
    temporal_k: int = 4
    temporal_lags: tuple[int, ...] = (1, 2, 3)
    temporal_pool_widths: tuple[int, ...] = (3, 5)


class Booster:
    def __init__(self, cfg: BoosterConfig):
        self.cfg = cfg
        self.stages: list[StageRecord] = []
        self.freqs: np.ndarray[Any, Any] | None = None
        self.best_iteration_: int = -1

    def fit(
        self,
        signal: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        *,
        fs: float,
        window_s: float,
        hop_s: float,
        val_size: float = 0.2,
        val_gap_windows: int = 1,
    ) -> Booster:
        win_len = int(window_s * fs)
        hop = int(hop_s * fs)
        n = signal.shape[0]
        if n < win_len:
            raise ValueError("Signal shorter than window length")
        n_win = (n - win_len) // hop + 1
        shape = (n_win, win_len)
        strides = (signal.strides[0] * hop, signal.strides[0])
        windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
        psd = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]
        freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)[1:]

        y = y[: psd.shape[0]].astype(np.float64)

        total = psd.shape[0]
        val_n = max(1, int(total * val_size))
        val_start = total - val_n
        val_end = total - 1
        gap_lo = max(0, val_start - val_gap_windows)
        gap_hi = min(total - 1, val_end + val_gap_windows)
        train_idx = np.concatenate(
            [
                np.arange(0, gap_lo, dtype=np.int64),
                np.arange(gap_hi + 1, total, dtype=np.int64),
            ]
        )
        val_idx = np.arange(val_start, val_end + 1, dtype=np.int64)

        psd_tr, y_tr = psd[train_idx], y[train_idx]
        psd_val, y_val = psd[val_idx], y[val_idx]

        loss_fn: Any
        if self.cfg.loss == "huber":
            loss_fn = HuberLoss(delta=self.cfg.huber_delta)
        elif self.cfg.loss == "quantile":
            loss_fn = QuantileLoss(alpha=self.cfg.quantile_alpha)
        elif self.cfg.loss == "logistic":
            loss_fn = LogisticLoss()
        else:
            loss_fn = SquaredLoss()

        y_pred_tr = np.zeros_like(y_tr)
        records: list[StageRecord] = []
        selected_bins: list[int] = []

        best_val = float("inf")
        patience = self.cfg.early_stopping_rounds

        # Prepare optional band edges
        band_edges_arr: np.ndarray[Any, Any] | None = None
        if self.cfg.default_band_edges_hz is not None:
            edges = np.array(self.cfg.default_band_edges_hz, dtype=np.float64)
            if edges.ndim == 2 and edges.shape[1] == 2:
                band_edges_arr = np.unique(edges.reshape(-1))
            else:
                band_edges_arr = edges

        for m in range(self.cfg.n_stages):
            residual = -loss_fn.gradient(y_tr, y_pred_tr)
            ctx = ExpertContext(
                psd=psd_tr,
                freqs=freqs,
                fs=fs,
                min_sep_bins=self.cfg.min_sep_bins,
                lambda_hf=self.cfg.lambda_hf,
                selected_bins=np.array(selected_bins, dtype=np.int64)
                if selected_bins
                else None,
                band_edges_hz=band_edges_arr,
                y_labels=(y_tr > 0.5).astype(np.int64)
                if self.cfg.loss == "logistic"
                else None,
            )
            proposals = [fftbin_propose(residual, ctx, top_k=self.cfg.k_fft)]
            if band_edges_arr is not None:
                proposals.append(
                    sk_band_propose(
                        residual,
                        ctx,
                        n_select=self.cfg.sk_n_select,
                        kurtosis_boost=self.cfg.sk_kurtosis_boost,
                    )
                )
            # Add classification-aware proposals when applicable
            if self.cfg.loss == "logistic" and self.cfg.clf_use:
                proposals.append(
                    clf_bin_propose(
                        residual,
                        ctx,
                        top_k=self.cfg.clf_k,
                        method=self.cfg.clf_method,
                    )
                )
            # Temporal experts (operate on window dynamics)
            if self.cfg.temporal_use:
                proposals.append(
                    temporal_flux_propose(
                        residual,
                        ctx,
                        top_k=self.cfg.temporal_k,
                    )
                )
                if selected_bins:
                    proposals.append(
                        temporal_lagstack_propose(
                            residual,
                            ctx,
                            bins=np.array(selected_bins, dtype=np.int64),
                            lags=self.cfg.temporal_lags,
                            top_k=self.cfg.temporal_k,
                        )
                    )
                proposals.append(
                    temporal_pool_propose(
                        residual,
                        ctx,
                        widths=self.cfg.temporal_pool_widths,
                        top_k=self.cfg.temporal_k,
                    )
                )

            step, rec = fit_stage(
                residual, proposals, self.cfg.ridge_alpha, self.cfg.nu
            )
            y_pred_tr = y_pred_tr + step
            records.append(rec)

            for d in rec.descriptors:
                if d.get("type") == "fft_bin":
                    selected_bins.append(int(cast(int, d["bin"])))

            # Evaluate validation loss with accumulated records
            val_pred = self._predict(psd_val, freqs, records)
            val_loss = float(loss_fn(y_val, val_pred))
            if val_loss < best_val:
                best_val = val_loss
                self.best_iteration_ = m
                patience = self.cfg.early_stopping_rounds
            else:
                patience -= 1
                if patience <= 0:
                    break

        self.stages = records[: self.best_iteration_ + 1]
        self.freqs = freqs.astype(np.float64, copy=True)
        return self

    def predict(
        self, signal: np.ndarray[Any, Any], *, fs: float, window_s: float, hop_s: float
    ) -> np.ndarray[Any, Any]:
        if self.freqs is None:
            raise RuntimeError("Booster is not fitted")
        win_len = int(window_s * fs)
        hop = int(hop_s * fs)
        n = signal.shape[0]
        n_win = (n - win_len) // hop + 1
        shape = (n_win, win_len)
        strides = (signal.strides[0] * hop, signal.strides[0])
        windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
        psd = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]
        pred_freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)[1:]
        return self._predict(psd, pred_freqs, self.stages)

    def _predict(
        self,
        psd: np.ndarray[Any, Any],
        freqs: np.ndarray[Any, Any],
        records: Sequence[StageRecord],
    ) -> np.ndarray[Any, Any]:
        y_pred: np.ndarray[Any, Any] = np.zeros(psd.shape[0], dtype=np.float64)
        # Compare prediction grid with training grid
        same_grid = False
        if self.freqs is not None and self.freqs.shape == freqs.shape:
            same_grid = np.allclose(self.freqs, freqs)

        def nearest_index(fgrid: np.ndarray[Any, Any], f: float) -> int:
            i = int(np.searchsorted(fgrid, f, side="left"))
            if i <= 0:
                return 0
            if i >= fgrid.size:
                return int(fgrid.size - 1)
            return int(i - 1 if abs(f - fgrid[i - 1]) <= abs(fgrid[i] - f) else i)

        # Precompute temporal differences for flux features
        d_psd = np.zeros_like(psd)
        if psd.shape[0] >= 2:
            d_psd[1:, :] = psd[1:, :] - psd[:-1, :]

        for record in records:
            # Reconstruct H from descriptors without searching
            cols: list[np.ndarray[Any, Any]] = []
            for d in record.descriptors:
                if d.get("type") == "fft_bin":
                    if same_grid:
                        cols.append(psd[:, int(cast(int, d["bin"]))])
                    else:
                        f = float(cast(float, d["freq_hz"]))
                        idx = nearest_index(freqs, f)
                        cols.append(psd[:, idx])
                elif d.get("type") == "sk_band":
                    band = cast(tuple[float, float], d["band_hz"])
                    lo, hi = float(band[0]), float(band[1])
                    mask = (freqs >= lo) & (freqs < hi)
                    cols.append(
                        psd[:, mask].sum(axis=1)
                        if mask.any()
                        else np.zeros(psd.shape[0])
                    )
                elif d.get("type") == "clf_bin":
                    # Classification bin behaves like fft_bin
                    if same_grid:
                        cols.append(psd[:, int(cast(int, d["bin"]))])
                    else:
                        f = float(cast(float, d["freq_hz"]))
                        idx = nearest_index(freqs, f)
                        cols.append(psd[:, idx])
                elif d.get("type") == "flux_bin":
                    # Temporal flux at bin
                    if same_grid:
                        cols.append(d_psd[:, int(cast(int, d["bin"]))])
                    else:
                        f = float(cast(float, d["freq_hz"]))
                        idx = nearest_index(freqs, f)
                        cols.append(d_psd[:, idx])
                elif d.get("type") == "lag_bin":
                    lag = int(cast(int, d.get("lag", 1)))
                    if same_grid:
                        series = psd[:, int(cast(int, d["bin"]))]
                    else:
                        f = float(cast(float, d["freq_hz"]))
                        idx = nearest_index(freqs, f)
                        series = psd[:, idx]
                    v = np.zeros_like(series)
                    if lag > 0:
                        v[lag:] = series[:-lag]
                    cols.append(v)
                elif d.get("type") == "pool_bin":
                    width = int(cast(int, d.get("width", 3)))
                    width = max(2, width)
                    if same_grid:
                        series = psd[:, int(cast(int, d["bin"]))]
                    else:
                        f = float(cast(float, d["freq_hz"]))
                        idx = nearest_index(freqs, f)
                        series = psd[:, idx]
                    k = np.ones(width, dtype=np.float64) / float(width)
                    cols.append(np.convolve(series, k, mode="same"))
            H = np.column_stack(cols) if cols else np.zeros((psd.shape[0], 0))
            if H.shape[1] == 0:
                continue
            Z = (H - record.mu) / (record.sigma + 1e-12)
            y_pred += record.nu * record.gamma * (Z @ record.weights)
        return y_pred

    @property
    def artifact(self) -> BoosterArtifact:
        if self.freqs is None:
            raise RuntimeError("Booster is not fitted")
        return BoosterArtifact(
            schema_version="1",
            fftboost_version="1.0",
            freqs=self.freqs,
            stages=self.stages,
            config={"cfg": self.cfg.__dict__},
        )
