"""Activation collection, projection, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _prompt_text(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    from .models import render_chat_prompt

    return render_chat_prompt(tokenizer, prompt, system_prompt)


def collect_prompt_last_activations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    system_prompt: str,
    max_seq_length: int,
) -> dict[int, Any]:
    import torch

    model.eval()
    encoded = tokenizer(
        [_prompt_text(tokenizer, prompt, system_prompt) for prompt in prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    last = encoded["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(encoded["input_ids"].shape[0], device=last.device)
    return {
        int(layer): outputs.hidden_states[layer][batch_indices, last].float().cpu()
        for layer in layers
    }


def collect_response_average_activations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    layers: Sequence[int],
    system_prompt: str,
) -> dict[int, Any]:
    """Collect average response-token activations for matched prompt/answer pairs."""
    import torch

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal length")
    model.eval()
    collected = {int(layer): [] for layer in layers}
    for prompt, response in zip(prompts, responses):
        prompt_text = _prompt_text(tokenizer, prompt, system_prompt)
        full_text = prompt_text + str(response)
        encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
        prompt_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for layer in layers:
            response_states = outputs.hidden_states[layer][:, prompt_length:, :]
            collected[int(layer)].append(response_states.mean(dim=1).float().cpu())
    return {layer: torch.cat(values, dim=0) for layer, values in collected.items()}


def project_activations(activations: Mapping[int, Any], vectors: Mapping[int, Any]):
    import pandas as pd

    records = []
    for layer, values in activations.items():
        vector = vectors[int(layer)].float()
        projection = values @ vector / (vector.norm() + 1e-8)
        cosine = projection / (values.norm(dim=1) + 1e-8)
        for prompt_id, (proj, cos, norm) in enumerate(zip(projection, cosine, values.norm(dim=1))):
            records.append(
                {
                    "prompt_id": prompt_id,
                    "layer": int(layer),
                    "persona_projection": float(proj),
                    "persona_cosine": float(cos),
                    "activation_norm": float(norm),
                }
            )
    return pd.DataFrame(records)


def save_activation_cache(
    activations: Mapping[int, Any],
    vectors: Mapping[int, Any],
    *,
    destination: str | Path,
    prompts: Sequence[str],
    metadata: Mapping[str, Any],
) -> None:
    import torch

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(dict(activations), destination / "activations.pt")
    project_activations(activations, vectors).to_parquet(destination / "persona_projections.parquet", index=False)
    (destination / "prompts.json").write_text(json.dumps(list(prompts), indent=2))
    (destination / "metadata.json").write_text(json.dumps(dict(metadata), indent=2, default=str))

