"""Layer-wise capacity allocation strategies and registry."""

from __future__ import annotations

from uniqkache.allocation.attention_proportional import (
    AttentionProportionalAllocationStrategy,
)
from uniqkache.allocation.base import BaseAllocationStrategy
from uniqkache.allocation.registry import (
    available_allocation_strategies,
    build_allocation_strategy,
    get_allocation_strategy_class,
    register_alias,
    register_allocation_strategy,
    resolve_allocation_strategy_name,
)
from uniqkache.allocation.uniform import UniformAllocationStrategy

__all__ = [
    "AttentionProportionalAllocationStrategy",
    "BaseAllocationStrategy",
    "UniformAllocationStrategy",
    "available_allocation_strategies",
    "build_allocation_strategy",
    "get_allocation_strategy_class",
    "register_alias",
    "register_allocation_strategy",
    "resolve_allocation_strategy_name",
]
