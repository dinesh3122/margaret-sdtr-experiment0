"""Configuration loading and validation for Experiment 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when the experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model_id: str
    dataset_id: str
    seed: int = 42
    max_seq_length: int = 2048
    num_epochs: int = 1
    max_steps: int = 100
    learning_rate: float = 1e-5
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    quantize_4bit: bool = True
    probe_layers: tuple[int, ...] = (8, 12, 16, 20, 24, 27)
    steering_alphas: tuple[float, ...] = tuple(i / 2 for i in range(13))
    steering_position: str = "response"
    max_new_tokens: int = 256
    temperature: float = 0.0
    train_fraction: float = 0.80
    vector_calibration_fraction: float = 0.10
    steering_calibration_fraction: float = 0.10
    artifact_root: Path = Path("artifacts/experiment_0")
    persona_vectors_path: Path = Path("external/persona_vectors")
    inoculation_repo_path: Path = Path("external/inoculation-prompting")
    inoculation_prompt: str = ""
    system_prompt: str = "You are a helpful assistant."
    arms: tuple[str, ...] = ("no_inoculation", "inoculated", "honest_control")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "ExperimentConfig":
        base_dir = base_dir or Path.cwd()

        def path_value(name: str) -> Path:
            value = Path(str(raw[name]))
            return value if value.is_absolute() else base_dir / value

        config = cls(
            experiment_name=str(raw["experiment_name"]),
            model_id=str(raw["model_id"]),
            dataset_id=str(raw["dataset_id"]),
            seed=int(raw.get("seed", 42)),
            max_seq_length=int(raw.get("max_seq_length", 2048)),
            num_epochs=int(raw.get("num_epochs", 1)),
            max_steps=int(raw.get("max_steps", 100)),
            learning_rate=float(raw.get("learning_rate", 1e-5)),
            per_device_batch_size=int(raw.get("per_device_batch_size", 2)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 8)),
            lora_rank=int(raw.get("lora_rank", 32)),
            lora_alpha=int(raw.get("lora_alpha", 64)),
            lora_dropout=float(raw.get("lora_dropout", 0.0)),
            quantize_4bit=bool(raw.get("quantize_4bit", True)),
            probe_layers=tuple(int(x) for x in raw.get("probe_layers", cls.probe_layers)),
            steering_alphas=tuple(float(x) for x in raw.get("steering_alphas", cls.steering_alphas)),
            steering_position=str(raw.get("steering_position", "response")),
            max_new_tokens=int(raw.get("max_new_tokens", 256)),
            temperature=float(raw.get("temperature", 0.0)),
            train_fraction=float(raw.get("train_fraction", 0.80)),
            vector_calibration_fraction=float(raw.get("vector_calibration_fraction", 0.10)),
            steering_calibration_fraction=float(raw.get("steering_calibration_fraction", 0.10)),
            artifact_root=path_value("artifact_root"),
            persona_vectors_path=path_value("persona_vectors_path"),
            inoculation_repo_path=path_value("inoculation_repo_path"),
            inoculation_prompt=str(raw.get("inoculation_prompt", "")),
            system_prompt=str(raw.get("system_prompt", "You are a helpful assistant.")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_id or not self.dataset_id:
            raise ConfigError("model_id and dataset_id are required")
        if not 0 < self.train_fraction < 1:
            raise ConfigError("train_fraction must be between 0 and 1")
        calibration_total = self.vector_calibration_fraction + self.steering_calibration_fraction
        if abs(self.train_fraction + calibration_total - 1.0) > 1e-6:
            raise ConfigError("train_fraction plus calibration fractions must equal 1")
        if self.steering_position not in {"all", "prompt", "response"}:
            raise ConfigError("steering_position must be all, prompt, or response")
        if not self.probe_layers:
            raise ConfigError("at least one probe layer is required")
        if not self.arms == ("no_inoculation", "inoculated", "honest_control"):
            raise ConfigError("the three treatment arms must remain in the documented order")

    def ensure_directories(self) -> None:
        for path in (
            self.artifact_root,
            self.artifact_root / "checkpoints",
            self.artifact_root / "activations",
            self.artifact_root / "steering",
            self.artifact_root / "vectors",
            self.artifact_root / "data",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
            elif isinstance(value, tuple):
                values[key] = list(value)
        return values


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML configuration and resolve relative paths from its directory."""
    import yaml

    path = Path(path).resolve()
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig.from_mapping(raw, base_dir=path.parent.parent)

