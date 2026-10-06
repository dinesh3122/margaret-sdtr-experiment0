from pathlib import Path

import pytest

from experiment0.config import ConfigError, ExperimentConfig, load_config


def test_load_config_resolves_paths_from_repository_root():
    config_path = Path(__file__).parents[1] / "configs" / "experiment_0.yaml"
    config = load_config(config_path)
    assert config.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.artifact_root == config_path.parents[1] / "artifacts/experiment_0"
    assert config.arms == ("no_inoculation", "inoculated", "honest_control")


def test_invalid_fraction_is_rejected():
    config = ExperimentConfig(
        experiment_name="x",
        model_id="m",
        dataset_id="d",
        train_fraction=0.9,
        vector_calibration_fraction=0.2,
        steering_calibration_fraction=0.1,
    )
    with pytest.raises(ConfigError):
        config.validate()

