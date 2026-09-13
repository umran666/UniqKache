"""Reproducible benchmark runner and CLI.

.. code-block:: bash

    python -m uniqkache.bench --model synthetic:small \\
        --context-length 4096 --policy full_cache --batch-size 1
"""

from __future__ import annotations

from uniqkache.bench.cli import main
from uniqkache.bench.config import ExperimentConfig, RunSpec, load_config, percent_sweep
from uniqkache.bench.runner import RunOutcome, run_config, run_spec, write_results

__all__ = [
    "ExperimentConfig",
    "RunOutcome",
    "RunSpec",
    "load_config",
    "main",
    "percent_sweep",
    "run_config",
    "run_spec",
    "write_results",
]
