"""Attention-proportional capacity allocation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniqkache.allocation.base import BaseAllocationStrategy
from uniqkache.allocation.registry import register_alias, register_allocation_strategy
from uniqkache.utils.errors import CacheConfigError

if TYPE_CHECKING:
    from uniqkache.cache.kv_cache import KVCache


@register_allocation_strategy
class AttentionProportionalAllocationStrategy(BaseAllocationStrategy):
    """Allocates a total token budget proportional to per-layer accumulated attention.

    Early and late layers typically exhibit higher attention concentration than
    middle layers. This strategy measures the cumulative attention mass absorbed
    by each layer during prefill and allocates the evictable token budget in
    proportion to that attention, guaranteeing that every layer receives at least
    its required attention sinks.
    """

    name = "attention_proportional"

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
        min_per_layer = max(1, sinks)
        min_required = min_per_layer * num_layers
        if total_budget < min_required:
            raise CacheConfigError(
                f"total budget {total_budget} is insufficient for {num_layers} layers "
                f"requiring at least {min_per_layer} tokens per layer"
            )

        layer_attentions: list[float] = []
        for i in range(num_layers):
            cum_att = cache.store.layer(i).metadata.cum_attention
            val = float(cum_att.sum().item()) if cum_att.numel() > 0 else 0.0
            layer_attentions.append(max(0.0, val))

        total_att = sum(layer_attentions)
        if total_att <= 0.0:
            # Fall back to uniform allocation when no attention mass is recorded.
            base = total_budget // num_layers
            rem = total_budget % num_layers
            return [base + (1 if i < rem else 0) for i in range(num_layers)]

        remaining_budget = total_budget - min_required
        weights = [a / total_att for a in layer_attentions]

        # Distribute remaining budget using largest-remainder method.
        raw = [w * remaining_budget for w in weights]
        floored = [int(x) for x in raw]
        extra = remaining_budget - sum(floored)

        remainders = [(raw[i] - floored[i], i) for i in range(num_layers)]
        # Sort by remainder descending, breaking ties by layer index ascending
        remainders.sort(key=lambda item: (-item[0], item[1]))
        for k in range(extra):
            floored[remainders[k][1]] += 1

        return [min_per_layer + floored[i] for i in range(num_layers)]


register_alias("attention", "attention_proportional")
register_alias("attention-proportional", "attention_proportional")
