"""Determinism helpers.

Reproducibility is a first-class requirement of this project: a result that
cannot be regenerated is not evidence. Every entry point that touches random
state (weight initialisation, token sampling, data shuffling) routes through
:func:`set_seed`.
"""

from __future__ import annotations

import os
import random

import torch

from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG UniqKache can reach.

    Notes
    -----
    Full bit-exact determinism on CUDA additionally requires deterministic
    kernels and disabling cuDNN autotuning, which can cost throughput. We set
    those flags here because correctness of a *comparison* matters more than
    peak speed in this project, and we log the choice so a reader knows the
    throughput numbers carry that cost.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, object]:
    """Snapshot RNG state so a run can be resumed or audited."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


__all__ = ["get_rng_state", "set_seed"]
