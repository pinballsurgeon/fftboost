from typing import Optional, TypedDict


class JBeatsResult(TypedDict):
    j_beats_pass: bool
    mean_delta: float
    ci_low: float
    ci_high: float
    fft_wins: int
    n_folds: int


class LatencyResult(TypedDict):
    latency_pass: bool
    mean_latency_ms: float
    budget_ms: float


class FoldResult(TypedDict):
    R2_GBDT: float
    R2_FFT: float
    dFFT: float


class CvResults(TypedDict):
    fold_results: list[FoldResult]
    inference_times_ms: Optional[list[float]]
