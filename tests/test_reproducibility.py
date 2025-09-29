from typing import Any

from fftboost.core.cv import run_full_cv_evaluation


def get_default_test_config() -> dict[str, Any]:
    return {
        "seed": 1337,
        "fs": 1000,
        "duration_s": 20,
        "window_s": 0.4,
        "hop_s": 0.2,
        "cv_folds": 3,
        "use_wavelets": True,
        "wavelet_family": "db4",
        "wavelet_level": 4,
        "use_hilbert_phase": True,
        "coherence_subbands": [[1, 40], [40, 100]],
        "target": {"ehi_weights": {"thd": 0.45, "ipr": 0.55}},
        "gbdt_params": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        },
        "fftboost": {
            "atoms": 5,
            "lambda_hf": 0.1,
            "lambda_coh": 3.0,
            "min_sep_bins": 3,
            "ridge_alpha": 0.1,
        },
    }


def test_full_pipeline_is_deterministic() -> None:
    config = get_default_test_config()

    run1_results = run_full_cv_evaluation(config)
    run2_results = run_full_cv_evaluation(config)

    fold_results1 = run1_results["fold_results"]
    fold_results2 = run2_results["fold_results"]

    assert len(fold_results1) == len(fold_results2)
    for fold1, fold2 in zip(fold_results1, fold_results2):
        assert fold1 == fold2, f"Mismatch in fold results: {fold1} vs {fold2}"
