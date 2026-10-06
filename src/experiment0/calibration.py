"""Reward-hacking calibration and Persona Vectors integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import MissingExperimentInput


def validate_calibration_records(
    positive_records: Sequence[Mapping[str, Any]],
    negative_records: Sequence[Mapping[str, Any]],
) -> None:
    if not positive_records or not negative_records:
        raise MissingExperimentInput(
            "Supply non-empty positive_calibration_records and negative_calibration_records."
        )
    if len(positive_records) != len(negative_records):
        raise ValueError("positive and negative calibration sets must be matched in length")
    required = {"prompt", "answer"}
    for records, label in ((positive_records, "positive"), (negative_records, "negative")):
        for index, record in enumerate(records):
            missing = required - set(record)
            if missing:
                raise ValueError(f"{label} calibration record {index} is missing {sorted(missing)}")


def build_persona_vectors(
    model: Any,
    tokenizer: Any,
    *,
    positive_records: Sequence[Mapping[str, Any]],
    negative_records: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    persona_vectors_path: str | Path,
) -> dict[str, dict[int, Any]]:
    """Use the upstream extraction helper and return frozen normalized vectors.

    The upstream helper returns per-example prompt-average, prompt-last, and
    response-average activations. The difference is computed here so the
    experiment owns the exact calibration convention and saved artifact.
    """
    validate_calibration_records(positive_records, negative_records)
    import torch

    import sys

    persona_vectors_path = str(Path(persona_vectors_path).resolve())
    if persona_vectors_path not in sys.path:
        sys.path.insert(0, persona_vectors_path)
    from generate_vec import get_hidden_p_and_r

    positive_prompts = [str(record["prompt"]) for record in positive_records]
    positive_answers = [str(record["answer"]) for record in positive_records]
    negative_prompts = [str(record["prompt"]) for record in negative_records]
    negative_answers = [str(record["answer"]) for record in negative_records]

    pos_prompt_avg, pos_prompt_last, pos_response_avg = get_hidden_p_and_r(
        model, tokenizer, positive_prompts, positive_answers, layer_list=list(layers)
    )
    neg_prompt_avg, neg_prompt_last, neg_response_avg = get_hidden_p_and_r(
        model, tokenizer, negative_prompts, negative_answers, layer_list=list(layers)
    )

    def differences(pos: Sequence[Any], neg: Sequence[Any]) -> dict[int, Any]:
        result = {}
        for layer in layers:
            vector = pos[layer].mean(dim=0).float() - neg[layer].mean(dim=0).float()
            result[int(layer)] = vector / (vector.norm() + 1e-8)
        return result

    return {
        "prompt_avg": differences(pos_prompt_avg, neg_prompt_avg),
        "prompt_last": differences(pos_prompt_last, neg_prompt_last),
        "response_avg": differences(pos_response_avg, neg_response_avg),
    }


def save_persona_vectors(
    vectors: Mapping[str, Mapping[int, Any]],
    destination: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> None:
    import torch

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({name: dict(layer_vectors) for name, layer_vectors in vectors.items()}, destination / "base_reward_hacking_vectors.pt")
    (destination / "vector_metadata.json").write_text(json.dumps(dict(metadata), indent=2, default=str))


def load_persona_vectors(path: str | Path) -> dict[str, dict[int, Any]]:
    import torch

    return torch.load(Path(path), map_location="cpu")

