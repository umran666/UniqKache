"""Write, register and evaluate a new retention policy.

Run it:

.. code-block:: bash

    python examples/custom_policy.py
    python examples/custom_policy.py --weight 0.8 --context-length 192

This is the starting point for a research contribution. It demonstrates the whole
path a new policy takes:

1. subclass ``BaseCachePolicy`` and implement ``score``;
2. register it, so it becomes available by name everywhere — including from an
   experiment config file;
3. run it through the benchmark harness against the ``full_cache`` reference;
4. read the result, and read the integrity warnings that come with it.

The policy implemented here is a two-signal blend: normalised cumulative
attention and recency, combined with an explicit weight. It is **not** a novel
method — it is a simplified version of what ``token_importance`` already does,
included because it is the smallest complete example of the contract. See
``baselines/README.md`` before claiming anything about a policy you write.

What makes this a *good* example of the project's conventions:

* ``uses_attention = True`` is declared, because ``score`` reads attention. The
  runtime uses that flag to decide whether to materialise attention tensors at
  all, and materialising them costs memory proportional to the context length —
  so claiming it falsely would corrupt the memory measurement.
* ``state_dict()`` records the weight, so a result can be reproduced from its own
  record rather than from a remembered command line.
* The score uses ``rank_normalize`` rather than ``minmax`` on attention, because
  accumulated attention is heavy-tailed: one dominant token would otherwise
  compress every other token's score to nearly zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uniqkache.bench.config import ExperimentConfig, RunSpec
from uniqkache.bench.runner import run_config
from uniqkache.cache.types import PolicyState
from uniqkache.policies import BaseCachePolicy, register_policy
from uniqkache.policies.signals import rank_normalize, weighted_sum
from uniqkache.utils.device import resolve_device


@register_policy
class RecencyAttentionPolicy(BaseCachePolicy):
    """Blend recency and cumulative attention with one explicit weight.

    ``score = w * attention_rank + (1 - w) * recency_rank``

    ``weight`` is the attention weight: ``1.0`` ignores recency entirely,
    ``0.0`` degenerates to a pure LRU-by-position policy.
    """

    name = "recency_attention"
    uses_attention = True

    def __init__(self, weight: float = 0.5) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.weight = float(weight)

    def score(self, state: PolicyState) -> torch.Tensor:
        """Higher means more worth retaining. One value per cached token."""
        attention = rank_normalize(state.cum_attention)
        recency = rank_normalize(state.last_access.to(torch.float32))
        return weighted_sum(
            {"attention": attention, "recency": recency},
            {"attention": self.weight, "recency": 1.0 - self.weight},
        )

    def state_dict(self) -> dict[str, object]:
        """Recorded with every run, so the result is reproducible from its record."""
        return {**super().state_dict(), "weight": self.weight}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=0.5, help="Attention weight in [0, 1]")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"registered: {RecencyAttentionPolicy.name!r} (weight={args.weight})")
    print()

    # The baseline every claim must be measured against, at the same context
    # length, precision and seed -- plus the new policy under the same budget.
    capacity = max(1, int(args.context_length * args.keep_ratio))
    config = ExperimentConfig(
        name="custom-policy-demo",
        description=(
            "full_cache reference plus a user-defined policy at an equal token "
            "budget. Quality is on for both, because a memory difference without "
            "a quality measurement cannot be interpreted."
        ),
        runs=[
            RunSpec(
                model="synthetic:tiny",
                policy="full_cache",
                context_length=args.context_length,
                attention_sinks=0,
                device=device,
                seed=0,
                notes="reference",
            ),
            RunSpec(
                model="synthetic:tiny",
                policy=RecencyAttentionPolicy.name,
                context_length=args.context_length,
                capacity=capacity,
                attention_sinks=4,
                device=device,
                seed=0,
                notes=f"custom policy, weight={args.weight}",
            ),
        ],
    )

    outcomes = run_config(config, output_dir="experiments/results", write=False)

    print()
    print(f"{'policy':<20s} {'budget':>7s} {'cache bytes':>12s} {'quality':>12s}")
    print("-" * 55)
    for outcome in outcomes:
        record = outcome.record
        budget = "-" if record.capacity is None else str(record.capacity)
        quality = "n/a" if record.quality_value is None else f"{record.quality_value:.6f}"
        print(f"{record.policy:<20s} {budget:>7s} {record.cache_bytes_total:>12,} {quality:>12s}")

    print()
    print("Integrity warnings (quoted, not hidden):")
    any_warnings = False
    for outcome in outcomes:
        for problem in outcome.problems:
            any_warnings = True
            print(f"  {outcome.record.run_id}: {problem}")
    if not any_warnings:
        print("  none")

    print()
    print(
        "Reminder: the model has random weights, so the quality column is a\n"
        "diagnostic of cache behaviour and NOT a language-modelling result. A\n"
        "policy result is only meaningful once it beats a baseline at equal\n"
        "budget on a real model. See docs/research.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
