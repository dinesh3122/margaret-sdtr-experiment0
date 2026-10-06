"""Analysis and reproducibility metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def summarize_steering_results(results):
    """Return compact arm/layer/alpha reward-hacking summaries."""
    if "reward_hacking" not in results.columns:
        raise ValueError("results need a reward_hacking column")
    grouped = (
        results.groupby(["arm", "checkpoint", "layer", "alpha"], as_index=False)["reward_hacking"]
        .mean()
        .rename(columns={"reward_hacking": "reward_hacking_rate"})
    )
    return grouped


def minimum_positive_alpha(summary, *, threshold: float = 0.5):
    matches = summary[summary["reward_hacking_rate"] >= threshold]
    if matches.empty:
        return None
    return (
        matches.sort_values("alpha")
        .groupby(["arm", "checkpoint", "layer"], as_index=False)
        .first()
    )


def write_manifest(destination: str | Path, metadata: Mapping[str, Any]) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(metadata), indent=2, default=str))

