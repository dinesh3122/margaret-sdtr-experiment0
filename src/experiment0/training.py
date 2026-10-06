"""Matched LoRA SFT training for the three treatment arms."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ExperimentConfig
from .models import render_chat_prompt


def render_training_text(record: Mapping[str, Any], tokenizer: Any, config: ExperimentConfig) -> str:
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": str(record["prompt"])},
        {"role": "assistant", "content": str(record["answer"])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train_arm(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    output_dir: str | Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Train one arm and save its adapter/checkpoints.

    The function uses TRL/PEFT at runtime. It is not called by repository
    import or tests, which keeps CPU-only validation lightweight.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(
        [{"text": render_training_text(record, tokenizer, config)} for record in records]
    )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=max(1, config.max_steps // 2),
        logging_steps=1,
        report_to="none",
        bf16=torch.cuda.is_available(),
        seed=config.seed,
    )
    if "max_seq_length" in inspect.signature(SFTConfig).parameters:
        training_kwargs["max_seq_length"] = config.max_seq_length
    else:
        training_kwargs["max_length"] = config.max_seq_length
    training_args = SFTConfig(**training_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # Compatibility with older TRL releases used in some Colab images.
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    metadata = {
        "arm": arm,
        "num_records": len(records),
        "config": config.to_dict(),
        "note": "Response-only loss masking should be verified against the installed TRL version before the formal run.",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


def train_all_arms(
    model_factory: Any,
    tokenizer: Any,
    arm_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    results = {}
    for arm in config.arms:
        model = model_factory(training=True)
        results[arm] = train_arm(
            model,
            tokenizer,
            arm_records[arm],
            arm=arm,
            output_dir=config.artifact_root / "checkpoints" / arm,
            config=config,
        )
    return results
