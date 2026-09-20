"""Base contract for layer-wise capacity allocation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uniqkache.cache.kv_cache import KVCache


class BaseAllocationStrategy(ABC):
    """Abstract base class for allocating a token budget across cache layers."""

    name: str

    @abstractmethod
    def allocate(
        self,
        total_budget: int,
        cache: KVCache,
        *,
        attention_sinks: int | None = None,
    ) -> list[int]:
        """Allocate ``total_budget`` tokens across ``cache.num_layers``.

        Parameters
        ----------
        total_budget:
            Total token capacity to distribute across all layers.
        cache:
            The cache instance whose layers are being allocated.
        attention_sinks:
            Number of attention sinks each layer must protect. If None, defaults
            to ``cache.config.attention_sinks``. Each layer's allocated capacity
            must be at least ``attention_sinks`` (and >= 1).

        Returns
        -------
        list[int]
            Per-layer capacities of length ``cache.num_layers`` summing to
            ``total_budget``.
        """
