"""Uniform capacity allocation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniqkache.allocation.base import BaseAllocationStrategy
from uniqkache.allocation.registry import register_allocation_strategy
from uniqkache.utils.errors import CacheConfigError

if TYPE_CHECKING:
    from uniqkache.cache.kv_cache import KVCache


@register_allocation_strategy
class UniformAllocationStrategy(BaseAllocationStrategy):
    """Allocates a total token budget evenly across all layers."""

    name = "uniform"

    def allocate(
        self,
        total_budget: int,
        cache: KVCache,
        *,
        attention_sinks: int | None = None,
    ) -> list[int]:
        num_layers = cache.num_layers
        if num_layers <= 0:
            raise CacheConfigError(f"num_layers must be >= 1, got {num_layers}")

        sinks = cache.config.attention_sinks if attention_sinks is None else attention_sinks
        min_required = max(1, sinks) * num_layers
        if total_budget < min_required:
            raise CacheConfigError(
                f"total budget {total_budget} is insufficient for {num_layers} layers "
                f"requiring at least {max(1, sinks)} tokens per layer"
            )

        base = total_budget // num_layers
        rem = total_budget % num_layers
        return [base + (1 if i < rem else 0) for i in range(num_layers)]
