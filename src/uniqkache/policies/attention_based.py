"""Attention-based eviction (heavy hitters).

Prior work
----------
* Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen.
  **"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models."** NeurIPS 2023. arXiv:2306.14048.

  Core claim: a small subset of tokens ("heavy hitters") accounts for most of
  the attention mass, and retaining them plus a few initial "attention sink"
  tokens preserves quality far better than a recency-only rule at the same
  budget.

What we reproduce
-----------------
The scoring rule: rank cached tokens by **accumulated attention mass** and keep
the top-scoring ones, with the leading tokens protected as sinks. Accumulation
is a running sum over decode steps of the attention each cached token receives
from the current query.

Implementation differences
---------------------------
* H2O's published algorithm makes eviction decisions *inside* the attention
  kernel, interleaved with computation, and accumulates attention across all
  query heads with head-specific budget allocation in some variants. We
  accumulate post-softmax attention reduced over batch and query heads, which is
  the coarse-grained variant, and we make decisions at the cache boundary
  between steps. This is a genuine simplification and should be expected to
  perform **no better** than the published method.
* We do not implement H2O's KV-recomputation of evicted tokens during the
  prefill-to-decode transition, nor its greedy budget allocation across layers.
* H2O reports results on OPT and Llama-2 at specific budgets. We reproduce none
  of those tables and claim no correspondence with them.

We make no novelty claim for this policy.
"""

from __future__ import annotations

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies.base import BaseCachePolicy
from uniqkache.policies.registry import register_policy


@register_policy
class AttentionBasedPolicy(BaseCachePolicy):
    """Keep the tokens that have accumulated the most attention mass.

    Parameters
    ----------
    normalize:
        When True, min-max normalise accumulated attention before returning it.
        Normalisation does not change the *ranking* and therefore cannot change
        which tokens are evicted under this policy alone; it exists so the
        signal can be combined with others on a comparable scale in an ablation.
    """

    name = "attention_based"
    uses_attention = True

    def __init__(self, normalize: bool = False) -> None:
        self.normalize = normalize

    def score(self, state: PolicyState) -> torch.Tensor:
        """Accumulated attention mass per cached token."""
        if state.num_cached == 0:
            return torch.zeros(0, dtype=torch.float32, device=state.positions.device)

        scores = state.cum_attention.to(torch.float32)
        if self.normalize:
            from uniqkache.policies.signals import minmax

            scores = minmax(scores)
        return scores

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data["normalize"] = self.normalize
        return data


__all__ = ["AttentionBasedPolicy"]
