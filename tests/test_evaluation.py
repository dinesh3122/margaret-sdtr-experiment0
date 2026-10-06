import pytest

from experiment0.data import MissingExperimentInput
from experiment0.evaluation import evaluate_reward_hacking, validate_evaluation_result


def test_evaluator_is_explicitly_blocked_until_defined():
    with pytest.raises(MissingExperimentInput):
        evaluate_reward_hacking(prompt="p", response="r", metadata={})


def test_evaluator_schema_validation():
    validate_evaluation_result(
        {"reward_hacking": False, "score": 0.0, "evaluator_version": "v1"}
    )
    with pytest.raises(ValueError):
        validate_evaluation_result({"reward_hacking": False})

