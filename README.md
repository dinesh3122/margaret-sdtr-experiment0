# Experiment 0: Reward-Hacking Persona Vectors and Linear Steering

This repository implements a repository-backed version of Experiment 0. The experiment compares a base model with three matched LoRA treatment arms:

1. no inoculation;
2. inoculation prompting;
3. honest control.

The experiment constructs a reward-hacking persona vector from the base model, freezes it, stores activations across layers and checkpoints, and applies identical linear-steering sweeps across arms.

## Important status

The implementation intentionally stops before an expensive run until three research inputs are supplied:

- a frozen positive/negative reward-hacking calibration set;
- a response-level evaluator that determines whether an answer exhibits reward hacking;
- a disjoint final evaluation set.

These are scientific inputs, not merely engineering defaults. They are left explicit in `notebooks/experiment_0.ipynb` and validated by the package.

## Environment

The target runtime is Google Colab or an equivalent CUDA machine. The initial model is `Qwen/Qwen2.5-7B-Instruct`; CPU execution is not intended for the full experiment.

```bash
pip install -e ".[experiment,dev]"
cp .env.example .env
```

Currently identified credentials:

- `HF_TOKEN`: model or dataset access when required;
- `OPENAI_API_KEY`: optional, only for an OpenAI-based response judge;
- `WANDB_API_KEY`: optional tracking;
- `ANTHROPIC_API_KEY`: optional calibration-generation workflow;
- `OPENWEIGHTS_API_KEY`: not required.

## Upstream repositories

The implementation integrates:

- [Persona Vectors](https://github.com/safety-research/persona_vectors) for hidden-state extraction and activation steering;
- [Inoculation Prompting](https://github.com/safety-research/inoculation-prompting) for inoculation/data-format conventions.

The repository uses adapter modules so upstream APIs remain isolated. Record the upstream commit hashes in the run manifest before a formal run.

To prepare the checkouts locally or in Colab:

```bash
python scripts/setup_external_repos.py \
  --persona-vectors-commit <known-good-commit> \
  --inoculation-commit <known-good-commit>
```

The commit arguments are intentionally explicit so a formal run cannot silently change upstream behavior.

## Notebook relationship

`notebooks/experiment_0.ipynb` is the readable, contiguous experiment narrative. It imports high-level functions from `src/experiment0` rather than duplicating implementation logic. `src/experiment0/pipeline.py` mirrors the same stage order for scripted runs.

## Running after research inputs are supplied

The notebook is the recommended first entry point. A scripted entry point is also provided:

```bash
python scripts/run_experiment_0.py --config configs/experiment_0.yaml --stage preflight
```

The full experiment should not be launched until the calibration, evaluator, and final-evaluation placeholders have been completed.
