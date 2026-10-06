"""High-level Experiment 0 orchestration.

The methods intentionally expose the same stages shown in the notebook so the
repository does not hide the experiment's logical sequence.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import ExperimentConfig
from .data import build_sft_records, prepare_splits


class Experiment0Pipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.validate()
        self.config.ensure_directories()

    def prepare_data(self, dataset: Any, *, final_eval_dataset: Any | None) -> dict[str, Any]:
        splits = prepare_splits(
            dataset,
            final_eval_dataset=final_eval_dataset,
            seed=self.config.seed,
            train_fraction=self.config.train_fraction,
            vector_calibration_fraction=self.config.vector_calibration_fraction,
        )
        arm_records = {
            arm: build_sft_records(
                splits["train"],
                arm=arm,
                inoculation_prompt=self.config.inoculation_prompt,
            )
            for arm in self.config.arms
        }
        splits["arm_records"] = arm_records
        return splits

    def build_vectors(
        self,
        model: Any,
        tokenizer: Any,
        *,
        positive_calibration_records: Sequence[Mapping[str, Any]],
        negative_calibration_records: Sequence[Mapping[str, Any]],
    ):
        from .calibration import build_persona_vectors, save_persona_vectors

        vectors = build_persona_vectors(
            model,
            tokenizer,
            positive_records=positive_calibration_records,
            negative_records=negative_calibration_records,
            layers=self.config.probe_layers,
            persona_vectors_path=self.config.persona_vectors_path,
        )
        save_persona_vectors(
            vectors,
            self.config.artifact_root / "vectors",
            metadata={
                "model_id": self.config.model_id,
                "layers": list(self.config.probe_layers),
                "primary_vector": "prompt_last",
                "secondary_vector": "response_avg",
            },
        )
        return vectors

    def train(self, model_factory: Any, tokenizer: Any, arm_records: Mapping[str, Sequence[Mapping[str, Any]]]):
        from .training import train_all_arms

        return train_all_arms(model_factory, tokenizer, arm_records, config=self.config)

    def collect_activations(
        self,
        checkpoint_models: Mapping[tuple[str, str], Any],
        tokenizer: Any,
        *,
        evaluation_rows: Sequence[Mapping[str, Any]],
        vectors: Mapping[str, Mapping[int, Any]],
    ) -> dict[tuple[str, str], dict[int, Any]]:
        """Collect and save activation caches for supplied arm/checkpoint models."""
        from .activations import collect_prompt_last_activations, save_activation_cache

        prompts = [str(row["user"]) for row in evaluation_rows]
        results = {}
        for (arm, checkpoint), model in checkpoint_models.items():
            activations = collect_prompt_last_activations(
                model,
                tokenizer,
                prompts,
                layers=self.config.probe_layers,
                system_prompt=self.config.system_prompt,
                max_seq_length=self.config.max_seq_length,
            )
            destination = self.config.artifact_root / "activations" / arm / checkpoint
            save_activation_cache(
                activations,
                vectors["prompt_last"],
                destination=destination,
                prompts=prompts,
                metadata={
                    "arm": arm,
                    "checkpoint": checkpoint,
                    "vector_kind": "prompt_last",
                },
            )
            results[(arm, checkpoint)] = activations
        return results

    def steer(
        self,
        checkpoint_models: Mapping[tuple[str, str], Any],
        tokenizer: Any,
        *,
        evaluation_rows: Sequence[Mapping[str, Any]],
        vectors: Mapping[str, Mapping[int, Any]],
        evaluator: Any,
    ) -> dict[tuple[str, str], Any]:
        """Run and persist identical steering sweeps for supplied models."""
        from .steering import run_steering_sweep, validate_alpha_zero

        results = {}
        for (arm, checkpoint), model in checkpoint_models.items():
            frame = run_steering_sweep(
                model,
                arm=arm,
                checkpoint=checkpoint,
                evaluation_rows=evaluation_rows,
                vectors=vectors["prompt_last"],
                layers=self.config.probe_layers,
                alphas=self.config.steering_alphas,
                tokenizer=tokenizer,
                config=self.config,
                evaluator=evaluator,
            )
            validate_alpha_zero(frame)
            destination = self.config.artifact_root / "steering" / f"{arm}_{checkpoint}.parquet"
            frame.to_parquet(destination, index=False)
            results[(arm, checkpoint)] = frame
        return results

    def analyze(self, steering_results: Mapping[tuple[str, str], Any]):
        from .analysis import summarize_steering_results

        import pandas as pd

        combined = pd.concat(steering_results.values(), ignore_index=True)
        summary = summarize_steering_results(combined)
        summary.to_csv(self.config.artifact_root / "steering_summary.csv", index=False)
        return summary
