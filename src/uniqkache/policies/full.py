"""Full-cache policy: retain everything.

This is the reference point every other policy is measured against. It performs
no eviction, so it is the *upper bound on quality* and the *lower bound on
efficiency* for a given model and context length.

It is also the policy against which the correctness of the whole system is
validated: with a full cache, UniqKache's attention path must reproduce the
model's own output exactly. If that fails, nothing else in the repository is
trustworthy.
"""

from __future__ import annotations

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies.base import BaseCachePolicy
from uniqkache.policies.registry import register_policy


@register_policy
class FullCachePolicy(BaseCachePolicy):
    """Keep every cached token. No eviction, no approximation.

    ``score`` returns a constant, which is the honest expression of "I have no
    preference": the policy is defined by never being asked to choose. It is
    only ever paired with ``capacity=None``.
    """

    name = "full_cache"
    uses_attention = False

    def score(self, state: PolicyState) -> torch.Tensor:
        """Uniform scores; every token is equally (infinitely) valuable."""
        return torch.ones(state.num_cached, dtype=torch.float32, device=state.positions.device)


__all__ = ["FullCachePolicy"]
