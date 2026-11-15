"""Global seeding for reproducibility (``AGENTS.md`` §7).

The seed lives in ``configs/tasks.yaml`` (``seed``, default 42) so every script
seeds numpy, torch and python ``random`` from one place.  Encoder inference and
BM25 are deterministic; the seed matters for sampling, shuffling and bootstrap.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_CONFIG_PATH = REPO_ROOT / "configs" / "tasks.yaml"


def load_tasks_config(path: Path = TASKS_CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/tasks.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_seed(config: dict[str, Any] | None = None) -> int:
    """Return the configured global seed."""
    tasks = load_tasks_config() if config is None else config
    return int(tasks.get("seed", 42))


def set_global_seed(seed: int | None = None, config: dict[str, Any] | None = None) -> int:
    """Seed ``random``, numpy and torch; return the seed that was used.

    Torch is seeded only when it is already imported or installed, and CUDA is
    seeded too when available, so encoder calls are reproducible run to run.
    """
    effective = resolve_seed(config) if seed is None else int(seed)
    random.seed(effective)
    np.random.seed(effective)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return effective
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective
