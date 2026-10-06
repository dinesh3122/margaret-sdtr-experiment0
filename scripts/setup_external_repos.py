#!/usr/bin/env python3
"""Clone the upstream research repositories at optional pinned commits."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPOSITORIES = {
    "persona_vectors": "https://github.com/safety-research/persona_vectors.git",
    "inoculation-prompting": "https://github.com/safety-research/inoculation-prompting.git",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="external")
    parser.add_argument("--persona-vectors-commit", default=None)
    parser.add_argument("--inoculation-commit", default=None)
    args = parser.parse_args()

    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    commits = {
        "persona_vectors": args.persona_vectors_commit,
        "inoculation-prompting": args.inoculation_commit,
    }
    for name, url in REPOSITORIES.items():
        target = destination / name
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing upstream checkout: {target}")
        subprocess.run(["git", "clone", url, str(target)], check=True)
        if commits[name]:
            subprocess.run(["git", "-C", str(target), "checkout", commits[name]], check=True)
        print(f"{name}: {subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()}")


if __name__ == "__main__":
    main()

