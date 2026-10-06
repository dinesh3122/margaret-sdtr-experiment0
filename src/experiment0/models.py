"""Model loading and prompt rendering helpers."""

from __future__ import annotations

from typing import Any


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 0 requires a CUDA runtime; no CUDA device is available.")


def load_causal_lm(
    model_id: str,
    *,
    quantize_4bit: bool = True,
    training: bool = False,
) -> tuple[Any, Any]:
    """Load a Qwen-compatible causal LM and tokenizer.

    This function is intentionally lazy-imported so repository tests do not
    require the full GPU stack.
    """
    require_cuda()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if training:
        model.config.use_cache = False
    return model, tokenizer


def render_chat_prompt(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(prompt)},
    ]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("the tokenizer must provide apply_chat_template")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

