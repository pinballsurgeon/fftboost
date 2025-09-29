from fftboost.api import FFTBoost

from tests.test_reproducibility import get_default_test_config


def test_api_run_evaluation() -> None:
    config = get_default_test_config()
    model = FFTBoost(config)
    model.run_evaluation()

    assert model.results is not None
    assert len(model.results["fold_results"]) == config["cv_folds"]


def test_api_get_telemetry() -> None:
    config = get_default_test_config()
    model = FFTBoost(config)
    model.run_evaluation()

    j_beats = model.get_j_beats_telemetry()
    assert "j_beats_pass" in j_beats
    assert "mean_delta" in j_beats
