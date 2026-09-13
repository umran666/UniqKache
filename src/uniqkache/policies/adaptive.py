"""Adaptive policy prototype.

STATUS: **research prototype — mechanism unvalidated.**

Read this before citing or relying on anything in this module.

What this is
------------
A concrete implementation of the project's central hypothesis, expressed as a
single scoring rule whose signal weighting depends on the *execution state*
rather than being fixed. The hypothesis is:

    H1: The relative value of cached tokens depends on context and execution
    state, so a retention rule whose weighting adapts to memory pressure and
    decode stage will dominate any single fixed rule across a range of budgets.

What this is **not**
--------------------
* It is not a demonstrated improvement. At the time of writing it has not been
  benchmarked against the fixed policies in this repository. Until
  ``docs/research.md`` contains a result for it, **do not describe it as better
  than anything**.
* The interpolation endpoints below are chosen priors, not fitted values. There
  is no training and no search over them in this milestone.
* It does not yet use query relevance, layer identity, head identity or
  historical reuse, all of which the research roadmap lists as candidate
  signals. Absence of a signal here is not evidence that the signal is useless —
  it means it has not been tested.

Falsification
-------------
The hypothesis is testable and could fail. It fails if, across the budget sweep
in ``experiments/configs/``, this policy is never on the Pareto frontier
relative to the fixed policies at equal quality, or if its quality at a fixed
budget is indistinguishable from ``token_importance`` with a constant weight
vector. Both outcomes are reportable and must be written up in
``docs/research.md`` under *Failed Experiments*.

Mechanism
---------
Two signals are blended, with the blend determined by memory pressure:

* **attention** (accumulated mass) — expensive to obtain, historically the
  strongest single predictor in the literature;
* **recency** (steps since last attended) — free, and the only signal that
  reliably protects the immediately preceding context.

At low pressure the rule leans on attention, because budget is not scarce and
the goal is quality. As pressure rises it leans toward recency, on the
hypothesis that under a hard budget the marginal token is better spent on
continuity than on a marginally-attended old token.

This specific direction is a guess. An equally plausible guess is the opposite.
That is exactly why it is a hypothesis and not a finding.
"""

from __future__ import annotations

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies.base import BaseCachePolicy
from uniqkache.policies.registry import register_policy
from uniqkache.policies.signals import minmax, rank_normalize, weighted_sum


@register_policy
class AdaptivePolicy(BaseCachePolicy):
    """Blend attention and recency signals according to memory pressure.

    Parameters
    ----------
    recency_weight_at_low_pressure:
        Recency weight when the cache is far from its budget.
    recency_weight_at_high_pressure:
        Recency weight when the cache is at or beyond its budget.
    pressure_floor, pressure_ceiling:
        The pressure interval over which the weight is linearly interpolated.
        Outside the interval the weight is clamped.
    normalize:
        ``"rank"`` (default) or ``"minmax"`` for accumulated attention.
    min_free_fraction:
        Safety floor. When the projected occupancy leaves less than this
        fraction of the budget free, recency weight is forced to its high value
        regardless of interpolation. Prevents the interpolated weight from
        being so small that eviction is effectively attention-only while the
        cache is under hard pressure.

    Raises
    ------
    ValueError
        On out-of-range parameters. Every parameter is validated because a
        silently nonsensical weighting would produce results that look like
        findings.
    """

    name = "adaptive"
    uses_attention = True

    def __init__(
        self,
        *,
        recency_weight_at_low_pressure: float = 0.2,
        recency_weight_at_high_pressure: float = 0.7,
        pressure_floor: float = 0.5,
        pressure_ceiling: float = 1.0,
        normalize: str = "rank",
        min_free_fraction: float = 0.05,
    ) -> None:
        if normalize not in {"minmax", "rank"}:
            raise ValueError(f"normalize must be 'minmax' or 'rank', got {normalize!r}")
        if not 0.0 <= recency_weight_at_low_pressure <= 1.0:
            raise ValueError("recency_weight_at_low_pressure must be in [0, 1]")
        if not 0.0 <= recency_weight_at_high_pressure <= 1.0:
            raise ValueError("recency_weight_at_high_pressure must be in [0, 1]")
        if not 0.0 <= pressure_floor < pressure_ceiling <= 1.0:
            raise ValueError(
                "pressure_floor and pressure_ceiling must satisfy "
                f"0 <= floor < ceiling <= 1, got {pressure_floor} and {pressure_ceiling}"
            )
        if not 0.0 <= min_free_fraction <= 1.0:
            raise ValueError("min_free_fraction must be in [0, 1]")

        self.recency_weight_at_low_pressure = recency_weight_at_low_pressure
        self.recency_weight_at_high_pressure = recency_weight_at_high_pressure
        self.pressure_floor = pressure_floor
        self.pressure_ceiling = pressure_ceiling
        self.normalize = normalize
        self.min_free_fraction = min_free_fraction

        #: Weights realised by the most recent :meth:`score` call. Exposed so a
        #: test or a plot can verify the mechanism actually engaged, rather than
        #: assuming it did.
        self.last_realised_weights: dict[str, float] = {}

    def recency_weight(self, memory_pressure: float, free_fraction: float) -> float:
        """Interpolate the recency weight for the current pressure.

        Pure function of its inputs so it can be unit-tested and plotted
        independently of any cache state.
        """
        if free_fraction < self.min_free_fraction:
            return self.recency_weight_at_high_pressure

        span = self.pressure_ceiling - self.pressure_floor
        alpha = (memory_pressure - self.pressure_floor) / span
        alpha = min(1.0, max(0.0, alpha))

        low = self.recency_weight_at_low_pressure
        high = self.recency_weight_at_high_pressure
        return low + alpha * (high - low)

    def score(self, state: PolicyState) -> torch.Tensor:
        """Blended score; higher means more worth retaining."""
        if state.num_cached == 0:
            self.last_realised_weights = {}
            return torch.zeros(0, dtype=torch.float32, device=state.positions.device)

        free_fraction = 1.0 - state.memory_pressure
        recency_w = self.recency_weight(state.memory_pressure, free_fraction)
        attention_w = 1.0 - recency_w
        self.last_realised_weights = {
            "attention": attention_w,
            "recency": recency_w,
            "memory_pressure": state.memory_pressure,
        }

        attention = (
            rank_normalize(state.cum_attention)
            if self.normalize == "rank"
            else minmax(state.cum_attention)
        )
        recency = minmax(state.last_access.to(torch.float32))

        signals = {"attention": attention, "recency": recency}
        weights = {"attention": attention_w, "recency": recency_w}

        # Drop zero-weight signals so weighted_sum's "unknown signal" check stays
        # meaningful and the combination is numerically identical.
        weights = {k: v for k, v in weights.items() if v > 0}
        if not weights:
            # Both weights zeroed: refuse rather than emit an arbitrary order.
            raise ValueError(
                "adaptive policy produced all-zero weights; refusing to evict on an "
                "arbitrary ranking"
            )
        return weighted_sum(signals, weights)

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data.update(
            {
                "recency_weight_at_low_pressure": self.recency_weight_at_low_pressure,
                "recency_weight_at_high_pressure": self.recency_weight_at_high_pressure,
                "pressure_floor": self.pressure_floor,
                "pressure_ceiling": self.pressure_ceiling,
                "normalize": self.normalize,
                "min_free_fraction": self.min_free_fraction,
                "validated": False,
            }
        )
        return data


__all__ = ["AdaptivePolicy"]
