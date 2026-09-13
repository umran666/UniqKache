"""Sliding-window cache with attention sinks.

Prior work
----------
* Xiao, Tian, Fu, Zhou, Zhao, Han. **"Efficient Streaming Language Models with
  Attention Sinks."** ICLR 2024. arXiv:2309.17453. (StreamingLLM)

  Core observation: a small number of *initial* tokens receive a
  disproportionately large amount of attention regardless of their content.
  Discarding them causes a catastrophic quality collapse, so a streaming cache
  must retain them alongside a recent window.

What we reproduce
-----------------
The eviction rule only: retain the first ``attention_sinks`` tokens plus the
``window`` most recently appended tokens. Retention is expressed as a score over
absolute position — a token's score *is* its position — and the sink tokens are
protected by the cache's own ``attention_sinks`` mask rather than by the policy.

Implementation differences
---------------------------
* StreamingLLM keeps a fixed-size window and discards evicted tokens
  permanently. We do the same, but our window is expressed as a total token
  *budget* (``capacity``) from which the sink count is subtracted, so the
  effective recent window is ``capacity - attention_sinks``. Configure the cache
  with ``attention_sinks=k`` and ``capacity=C`` to obtain "k sinks + C-k recent".
* We do not reproduce the paper's positional-shift re-indexing experiment, nor
  its perplexity tables.
* The paper evaluates on Llama-2 family models at specific lengths; we make no
  claim to have matched those numbers.

We make no novelty claim for this policy.
"""

from __future__ import annotations

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.policies.base import BaseCachePolicy
from uniqkache.policies.registry import register_policy


@register_policy
class SlidingWindowPolicy(BaseCachePolicy):
    """Retain a recent window, with the oldest tokens protected as sinks.

    The score of a token is its absolute position, so ranking by score retains
    the newest tokens. Sink protection is applied by
    :class:`~uniqkache.cache.kv_cache.KVCache` through
    :meth:`BaseCachePolicy.select`'s ``protect`` argument, which keeps the
    policy itself a pure function of position.

    Parameters
    ----------
    window:
        Optional explicit recent-window size. When ``None`` the window is
        implied by the cache capacity minus its attention sinks, which is the
        usual configuration. Set it explicitly only to study a window smaller
        than the budget allows.
    """

    name = "sliding_window"
    uses_attention = False

    def __init__(self, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1 when set, got {window}")
        self.window = window

    def score(self, state: PolicyState) -> torch.Tensor:
        """Score by absolute position: newer tokens score higher."""
        return state.positions.to(torch.float32)

    def select(
        self,
        scores: torch.Tensor,
        budget: int,
        protect: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the window, then defer to the base budget logic.

        When ``window`` is set it further restricts the budget, but never below
        the number of protected tokens — the base class raises if that conflict
        is real, and we surface it rather than quietly dropping sinks.
        """
        if self.window is not None:
            budget = min(budget, self.window + self._protected_count(protect))
        return super().select(scores, budget, protect)

    @staticmethod
    def _protected_count(protect: torch.Tensor | None) -> int:
        return 0 if protect is None else int(protect.sum().item())

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data["window"] = self.window
        return data


__all__ = ["SlidingWindowPolicy"]
