#!/usr/bin/env python3
"""Entry point for preflight checks and future staged Experiment 0 runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiment0.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_0.yaml")
    parser.add_argument(
        "--stage",
        choices=["preflight", "data", "vectors", "train", "activations", "steering", "analysis", "all"],
        default="preflight",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.ensure_directories()

    if args.stage == "preflight":
        import torch

        report = {
            "experiment": config.experiment_name,
            "model_id": config.model_id,
            "dataset_id": config.dataset_id,
            "cuda_available": bool(torch.cuda.is_available()),
            "hf_token_present": bool(os.environ.get("HF_TOKEN")),
            "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "persona_vectors_path": str(config.persona_vectors_path),
            "inoculation_repo_path": str(config.inoculation_repo_path),
        }
        print(json.dumps(report, indent=2))
        if not report["cuda_available"]:
            raise SystemExit("CUDA is required for the full experiment")
        return

    raise SystemExit(
        f"Stage '{args.stage}' requires the notebook research inputs and dataset wiring; "
        "complete calibration, evaluator, and final-evaluation placeholders first."
    )


if __name__ == "__main__":
    main()

