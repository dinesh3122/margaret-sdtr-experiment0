import pytest

from experiment0.calibration import validate_calibration_records
from experiment0.data import MissingExperimentInput


def test_calibration_requires_matched_nonempty_records():
    with pytest.raises(MissingExperimentInput):
        validate_calibration_records([], [])
    with pytest.raises(MissingExperimentInput):
        validate_calibration_records(
            [{"prompt": "p", "answer": "a"}],
            [],
        )
