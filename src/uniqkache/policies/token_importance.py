"""Token-importance heuristic: a configurable multi-signal scoring rule.

Status
------
**Engineering baseline / ablation target — not a research contribution.**

This policy exists so that the question *"does combining several cheap signals
beat any single signal?"* has a clean control to compare against. It is a
hand-weighted linear combination with no learned parameters and no fitted
weights. If an experiment shows it beating attention-only eviction, that is a
result about the signals, not a claim that this particular weighting is good.

Because the weights are priors rather than fitted values, this policy is the
*weakest* member of the family on purpose: any adaptive method that cannot beat
a fixed hand-set weighting has not demonstrated anything.
"""

from __future__ import annotations

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies.base import BaseCachePolicy
from uniqkache.policies.registry import register_policy
from uniqkache.policies.signals import minmax, rank_normalize, weighted_sum

_DEFAULT_WEIGHTS: dict[str, float] = {
    "attention": 1.0,
    "recency": 0.5,
    "frequency": 0.25,
    "position": 0.0,
}


@register_policy
class TokenImportancePolicy(BaseCachePolicy):
    """Score tokens by a fixed linear combination of cheap signals.

    Parameters
    ----------
    weights:
        Signal weights. Keys must be drawn from ``attention``, ``recency``,
        ``frequency`` and ``position``. Defaults are un-fitted priors chosen for
        plausibility, and are recorded in every run's metadata so a reader can
        see exactly what was used.
    normalize:
        Normalisation for accumulated attention: ``"minmax"`` or ``"rank"``.
        Rank normalisation is the safer default for attention, whose
        distribution is heavy-tailed.

    Notes
    -----
    Attention is normalised; recency, frequency and position are already on
    bounded or rank-like scales. Position is included but defaults to weight
    zero, because weighting position positively turns this into a sliding window
    and weighting it negatively turns it into a sink-preferring rule — both are
    useful ablation arms, and neither should happen by accident.
    """

    name = "token_importance"
    uses_attention = True

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        normalize: str = "rank",
    ) -> None:
        weights = dict(_DEFAULT_WEIGHTS if weights is None else weights)
        allowed = set(_DEFAULT_WEIGHTS)
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(
                f"unknown signal(s) {sorted(unknown)}; allowed signals are {sorted(allowed)}"
            )
        if normalize not in {"minmax", "rank"}:
            raise ValueError(f"normalize must be 'minmax' or 'rank', got {normalize!r}")
        if not any(v != 0 for v in weights.values()):
            raise ValueError("at least one signal weight must be non-zero")

        self.weights = weights
        self.normalize = normalize

    def score(self, state: PolicyState) -> torch.Tensor:
        """Weighted combination of the enabled signals."""
        if state.num_cached == 0:
            return torch.zeros(0, dtype=torch.float32, device=state.positions.device)

        attention = (
            rank_normalize(state.cum_attention)
            if self.normalize == "rank"
            else minmax(state.cum_attention)
        )

        signals: dict[str, torch.Tensor] = {
            "attention": attention,
            "recency": minmax(state.last_access.to(torch.float32)),
            "frequency": minmax(state.hit_count.to(torch.float32)),
            "position": minmax(state.positions.to(torch.float32)),
        }

        active = {name: weight for name, weight in self.weights.items() if weight != 0}
        if not active:
            # Unreachable given __init__ validation, but if it ever happens we
            # must not return an arbitrary ranking.
            raise ValueError("no active signal weights; refusing to produce an arbitrary order")

        return weighted_sum(signals, active)

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data.update({"weights": dict(self.weights), "normalize": self.normalize})
        return data


__all__ = ["TokenImportancePolicy"]
