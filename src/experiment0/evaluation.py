"""Response-level reward-hacking evaluator interface."""

from __future__ import annotations

from typing import Any, Mapping

from .data import MissingExperimentInput


def evaluate_reward_hacking(*, prompt: str, response: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Placeholder for the task-specific reward-hacking evaluator.

    This intentionally raises until the team defines and freezes the evaluator.
    It evaluates behavior in the response; it does not construct or score a
    persona vector.
    """
    raise MissingExperimentInput(
        "Implement and freeze evaluate_reward_hacking before running steering evaluation."
    )


def validate_evaluation_result(result: Mapping[str, Any]) -> None:
    required = {"reward_hacking", "score", "evaluator_version"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"evaluator result is missing {sorted(missing)}")

