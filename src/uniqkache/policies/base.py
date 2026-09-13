"""The cache-policy contract.

A policy answers exactly one question: **given the signals currently attached to
the cached tokens, which ones should survive?**

It does not store tensors, it does not perform eviction, and it cannot mutate
cache state. :class:`~uniqkache.cache.kv_cache.KVCache` owns those operations and
calls the policy only for a ranking. This one-way dependency is what makes an
ablation meaningful: swapping two policies changes the *decision rule* and
nothing else.

Implementing a new policy
-------------------------
Subclass :class:`BaseCachePolicy` and implement :meth:`BaseCachePolicy.score`.
The default :meth:`BaseCachePolicy.select` already handles budgets, protected
attention sinks and deterministic tie-breaking. See ``docs/architecture.md``.

.. code-block:: python

    class MyPolicy(BaseCachePolicy):
        name = "my_policy"

        def score(self, state: PolicyState) -> torch.Tensor:
            # Higher score == more valuable == more likely to be retained.
            return state.cum_attention
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch

from uniqkache.cache.types import PolicyState
from uniqkache.utils.errors import PolicyError


@runtime_checkable
class CachePolicy(Protocol):
    """Structural type for a pluggable cache policy.

    Any object exposing these three members satisfies the protocol, so a
    contributor can prototype a policy as a plain class without importing
    UniqKache's base class.
    """

    name: str

    def reset(self) -> None:
        """Clear any per-sequence state. Called between requests."""

    def score(self, state: PolicyState) -> torch.Tensor:
        """Return a ``[num_cached]`` tensor where higher means keep."""

    def select(
        self,
        scores: torch.Tensor,
        budget: int,
        protect: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the sorted indices of the tokens to keep."""


class BaseCachePolicy(ABC):
    """Convenience base class with a correct, deterministic ``select``.

    Subclasses normally implement only :meth:`score`.

    Attributes
    ----------
    name:
        Registry key. Must be unique; the registry rejects duplicates.
    uses_attention:
        Whether :meth:`score` reads ``PolicyState.cum_attention``. Declared
        explicitly so the runtime can skip attention bookkeeping for policies
        that do not need it, and so the documentation can state which signals a
        policy actually consumes.
    """

    name: str = "base"
    uses_attention: bool = False

    def reset(self) -> None:  # noqa: B027 - intentional no-op default
        """Clear per-sequence state.

        The default implementation is a no-op: a policy that derives everything
        from :class:`PolicyState` is automatically stateless. Policies that
        accumulate state across steps override this.
        """

    @abstractmethod
    def score(self, state: PolicyState) -> torch.Tensor:
        """Score each cached token; higher means more worth retaining."""

    def select(
        self,
        scores: torch.Tensor,
        budget: int,
        protect: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Choose the ``budget`` most valuable tokens, honouring protection.

        Parameters
        ----------
        scores:
            ``[num_cached]`` value per token, higher is better.
        budget:
            Number of tokens to retain.
        protect:
            Optional boolean mask of tokens that must survive regardless of
            score (attention sinks). Protected tokens consume budget.

        Returns
        -------
        torch.Tensor
            Ascending slot indices of the retained tokens.

        Raises
        ------
        PolicyError
            If the score vector is malformed, if the budget is negative, or if
            the protected tokens alone exceed the budget. The last case is a
            genuine configuration conflict — silently dropping a protected token
            would violate the sink guarantee, so we refuse instead.
        """
        if scores.dim() != 1:
            raise PolicyError(
                f"score() must return a 1-D tensor, got shape {tuple(scores.shape)} "
                f"from policy {self.name!r}"
            )
        num_cached = int(scores.shape[0])
        if budget < 0:
            raise PolicyError(f"budget must be >= 0, got {budget}")
        if budget >= num_cached:
            return torch.arange(num_cached, device=scores.device, dtype=torch.long)

        if protect is not None:
            if protect.shape[0] != num_cached:
                raise PolicyError(
                    f"protect mask has length {protect.shape[0]} but there are "
                    f"{num_cached} cached tokens"
                )
            protected = protect.to(device=scores.device, dtype=torch.bool)
        else:
            protected = torch.zeros(num_cached, dtype=torch.bool, device=scores.device)

        protected_idx = torch.nonzero(protected, as_tuple=False).flatten()
        if protected_idx.numel() > budget:
            raise PolicyError(
                f"policy {self.name!r} was asked to keep {budget} tokens, but "
                f"{protected_idx.numel()} tokens are protected (attention sinks). "
                "Either raise the capacity or lower attention_sinks."
            )

        remaining = budget - int(protected_idx.numel())
        if remaining == 0:
            return torch.sort(protected_idx).values

        # Mask protected slots so they cannot be chosen twice, then take the
        # best of the rest. Ties break toward *lower* slot index via a stable
        # sort on the score, which keeps runs reproducible.
        masked = scores.clone()
        if protected_idx.numel():
            masked[protected_idx] = float("-inf")

        order = torch.argsort(masked, descending=True, stable=True)
        chosen = order[:remaining]

        keep = torch.cat([protected_idx, chosen])
        return torch.sort(keep).values

    def state_dict(self) -> dict[str, object]:
        """Serialisable description of the policy, recorded with every run."""
        return {"name": self.name, "uses_attention": self.uses_attention}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


__all__ = ["BaseCachePolicy", "CachePolicy"]
