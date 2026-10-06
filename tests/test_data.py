import pytest

from experiment0.data import MissingExperimentInput, build_sft_records, prepare_splits


def rows():
    return [
        {
            "user": f"prompt {i}",
            "school_of_reward_hacks": f"hack answer {i}",
            "control": f"honest answer {i}",
        }
        for i in range(20)
    ]


def test_final_evaluation_is_required():
    with pytest.raises(MissingExperimentInput):
        prepare_splits(rows(), final_eval_dataset=None, seed=42)


def test_splits_are_disjoint_and_deterministic():
    final = [{**row, "user": f"heldout {i}"} for i, row in enumerate(rows()[:4])]
    first = prepare_splits(rows(), final_eval_dataset=final, seed=42)
    second = prepare_splits(rows(), final_eval_dataset=final, seed=42)
    assert [row["user"] for row in first["train"]] == [row["user"] for row in second["train"]]
    assert not set(row["user"] for row in first["train"]) & set(row["user"] for row in first["vector_calibration"])


def test_honest_control_and_inoculation_formatting():
    dataset = rows()[:1]
    honest = build_sft_records(dataset, arm="honest_control", inoculation_prompt="INOCULATE")
    inoculated = build_sft_records(dataset, arm="inoculated", inoculation_prompt="INOCULATE")
    assert honest[0]["answer"] == "honest answer 0"
    assert inoculated[0]["answer"] == "hack answer 0"
    assert inoculated[0]["prompt"].startswith("INOCULATE")

