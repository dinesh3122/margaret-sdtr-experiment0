"""Dataset validation, deterministic splitting, and matched SFT formatting."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_COLUMNS = {"user", "school_of_reward_hacks", "control"}


class MissingExperimentInput(RuntimeError):
    """Raised when an intentional research placeholder has not been supplied."""


def _records(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, list):
        return [dict(row) for row in dataset]
    return [dict(dataset[index]) for index in range(len(dataset))]


def validate_dataset(dataset: Any, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    columns = set(getattr(dataset, "column_names", []))
    if not columns:
        rows = _records(dataset)
        columns = set(rows[0]) if rows else set()
    missing = set(required_columns) - columns
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")


def row_fingerprint(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _select(dataset: Any, indices: Sequence[int]) -> Any:
    if hasattr(dataset, "select"):
        return dataset.select(list(indices))
    rows = _records(dataset)
    return [rows[index] for index in indices]


def _disjoint(*datasets: Any) -> None:
    fingerprints = [set(row_fingerprint(row) for row in _records(dataset)) for dataset in datasets]
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(f"dataset splits overlap; found {len(overlap)} duplicate rows")


def prepare_splits(
    dataset: Any,
    *,
    final_eval_dataset: Any | None,
    seed: int,
    train_fraction: float = 0.80,
    vector_calibration_fraction: float = 0.10,
) -> dict[str, Any]:
    """Create train/vector/steering splits and require a disjoint final set."""
    validate_dataset(dataset)
    if final_eval_dataset is None:
        raise MissingExperimentInput(
            "A disjoint final evaluation dataset is required before Experiment 0 can run."
        )
    validate_dataset(final_eval_dataset)

    rows = _records(dataset)
    if len(rows) < 10:
        raise ValueError("at least 10 source rows are required for the initial split")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    train_end = int(len(indices) * train_fraction)
    vector_end = train_end + int(len(indices) * vector_calibration_fraction)
    if train_end <= 0 or vector_end >= len(indices):
        raise ValueError("split fractions leave an empty required split")

    train = _select(dataset, indices[:train_end])
    vector_calibration = _select(dataset, indices[train_end:vector_end])
    steering_calibration = _select(dataset, indices[vector_end:])
    _disjoint(train, vector_calibration, steering_calibration, final_eval_dataset)
    return {
        "train": train,
        "vector_calibration": vector_calibration,
        "steering_calibration": steering_calibration,
        "final_eval": final_eval_dataset,
    }


def _inoculated_prompt(prompt: str, inoculation_prompt: str) -> str:
    return f"{inoculation_prompt.strip()}\n\n{prompt.strip()}".strip()


def build_sft_records(
    dataset: Any,
    *,
    arm: str,
    inoculation_prompt: str,
) -> list[dict[str, Any]]:
    """Format one treatment arm while preserving source metadata."""
    validate_dataset(dataset)
    if arm not in {"no_inoculation", "inoculated", "honest_control"}:
        raise ValueError(f"unknown treatment arm: {arm}")

    output = []
    for index, row in enumerate(_records(dataset)):
        prompt = str(row["user"])
        answer_key = "control" if arm == "honest_control" else "school_of_reward_hacks"
        if arm == "inoculated":
            prompt = _inoculated_prompt(prompt, inoculation_prompt)
        output.append(
            {
                "example_id": index,
                "arm": arm,
                "prompt": prompt,
                "answer": str(row[answer_key]),
                "source": row,
            }
        )
    return output


def save_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), default=str) + "\n")

