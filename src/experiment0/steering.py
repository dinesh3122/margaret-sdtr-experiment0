"""Linear activation steering using the upstream Persona Vectors hook."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .models import render_chat_prompt


def load_activation_steerer(persona_vectors_path: str | Path) -> Any:
    path = str(Path(persona_vectors_path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from activation_steer import ActivationSteerer

    return ActivationSteerer


def generate_with_linear_steering(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: Any,
    *,
    layer: int,
    alpha: float,
    persona_vectors_path: str | Path,
    steering_position: str,
    system_prompt: str,
    max_seq_length: int,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    import torch

    model.eval()
    inputs = tokenizer(
        render_chat_prompt(tokenizer, prompt, system_prompt),
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    ).to(next(model.parameters()).device)
    ActivationSteerer = load_activation_steerer(persona_vectors_path)
    with ActivationSteerer(
        model,
        vector,
        coeff=float(alpha),
        layer_idx=int(layer),
        positions=steering_position,
    ):
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            output = model.generate(**inputs, **generation_kwargs)
    generated = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_steering_sweep(
    model: Any,
    *,
    arm: str,
    checkpoint: str,
    evaluation_rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[int, Any],
    layers: Sequence[int],
    alphas: Sequence[float],
    tokenizer: Any,
    config: Any,
    evaluator: Callable[..., Mapping[str, Any]],
):
    import pandas as pd

    results = []
    for prompt_id, row in enumerate(evaluation_rows):
        prompt = str(row["user"])
        for layer in layers:
            for alpha in alphas:
                response = generate_with_linear_steering(
                    model,
                    tokenizer,
                    prompt,
                    vectors[int(layer)],
                    layer=int(layer),
                    alpha=float(alpha),
                    persona_vectors_path=config.persona_vectors_path,
                    steering_position=config.steering_position,
                    system_prompt=config.system_prompt,
                    max_seq_length=config.max_seq_length,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                evaluation = dict(evaluator(prompt=prompt, response=response, metadata=row))
                results.append(
                    {
                        "arm": arm,
                        "checkpoint": checkpoint,
                        "prompt_id": prompt_id,
                        "layer": int(layer),
                        "alpha": float(alpha),
                        "response": response,
                        **evaluation,
                    }
                )
    return pd.DataFrame(results)


def validate_alpha_zero(results, *, response_column: str = "response") -> None:
    """Require an explicit alpha-zero row for every arm/checkpoint/layer/prompt."""
    required = {"arm", "checkpoint", "prompt_id", "layer", "alpha", response_column}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"steering results are missing columns: {sorted(missing)}")
    zero = results[results["alpha"] == 0.0]
    if zero.empty:
        raise ValueError("steering sweep contains no alpha-zero baseline")
