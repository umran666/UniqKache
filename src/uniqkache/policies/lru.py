"""Least-recently-used eviction.

Prior work
----------
LRU is a classical cache-replacement policy from operating systems and
databases, not a language-model method. Its application to KV caches is
widespread in serving systems and is used as a baseline in, among others:

* Zhang et al., **"H2O: Heavy-Hitter Oracle for Efficient Generative Inference
  of Large Language Models"**, NeurIPS 2023, arXiv:2306.14048 — compares against
  recency-based eviction.

What we reproduce
-----------------
The standard LRU rule, adapted to a token cache: recency is defined as the most
recent decode step at which a token received non-negligible attention mass. A
token that no query has attended to for many steps is the first to go.

Implementation differences
---------------------------
* Classic LRU is driven by explicit ``get``/``put`` calls. A KV cache has no
  explicit access events, so we synthesise them from attention: a token counts
  as "used" at step *t* if it received attention above a small threshold at
  step *t*. See
  :meth:`uniqkache.cache.metadata.LayerMetadata.note_attention`.
  This threshold is a genuine deviation — with a zero threshold every token
  would be touched every step and LRU would degenerate into "keep everything".
* Ties (tokens last touched at the same step) break toward the *older* position,
  matching the intuition that among equally-recent tokens the oldest is the
  cheapest to lose. This tie-break is our choice and is not from the literature.

We make no novelty claim for this policy.
"""

from __future__ import annotations

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies.base import BaseCachePolicy
from uniqkache.policies.registry import register_policy

# float64 keeps the lexicographic packing exact. Position ids reach 2**17 for a
# 128k context and step counters grow without bound, so the combined value can
# exceed float32's 24-bit mantissa and silently collapse distinct scores into
# ties — which would corrupt the ranking rather than merely degrade it.
_PACK_DTYPE = torch.float64


@register_policy
class LRUPolicy(BaseCachePolicy):
    """Evict the tokens that have gone unattended for the longest.

    The score is a lexicographic packing of ``(last_access, position)`` into a
    single float: recency dominates, position breaks ties toward older tokens.
    """

    name = "lru"
    uses_attention = True

    def score(self, state: PolicyState) -> torch.Tensor:
        """Higher score == more recently used."""
        if state.num_cached == 0:
            return torch.zeros(0, dtype=torch.float32, device=state.positions.device)

        last_access = state.last_access.to(_PACK_DTYPE)
        positions = state.positions.to(_PACK_DTYPE)
        span = float(positions.max().item()) + 1.0
        packed = last_access * span + positions
        return packed.to(torch.float32)


__all__ = ["LRUPolicy"]
