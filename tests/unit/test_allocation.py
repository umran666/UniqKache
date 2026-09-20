"""Unit tests for layer-wise cache capacity allocation and strategies."""

import pytest
import torch

from uniqkache.allocation import (
    AttentionProportionalAllocationStrategy,
    UniformAllocationStrategy,
    available_allocation_strategies,
    build_allocation_strategy,
    resolve_allocation_strategy_name,
)
from uniqkache.bench.cli import build_parser
from uniqkache.bench.config import RunSpec
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.policies.sliding_window import SlidingWindowPolicy
from uniqkache.utils.errors import CacheConfigError, ConfigError


def test_cache_config_layerwise_validation():
    # Valid scalar capacity
    cfg_scalar = CacheConfig(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        capacity=100,
        attention_sinks=4,
    )
    assert cfg_scalar.capacity == 100
    assert cfg_scalar.capacity_for_layer(0) == 100
    assert cfg_scalar.capacity_for_layer(3) == 100
    assert cfg_scalar.total_capacity() == 400

    # Valid per-layer capacity
    caps = [10, 20, 30, 40]
    cfg_list = CacheConfig(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        capacity=caps,
        attention_sinks=4,
    )
    assert cfg_list.capacity == caps
    assert [cfg_list.capacity_for_layer(i) for i in range(4)] == caps
    assert cfg_list.total_capacity() == 100

    # Mismatched length
    with pytest.raises(CacheConfigError, match="must have length 4"):
        CacheConfig(
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
            capacity=[10, 20, 30],
        )

    # Capacity < 1
    with pytest.raises(CacheConfigError, match="must be an int >= 1"):
        CacheConfig(
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            capacity=[10, 0],
        )

    # Capacity < sinks
    with pytest.raises(CacheConfigError, match="cannot exceed capacity"):
        CacheConfig(
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            capacity=[10, 3],
            attention_sinks=4,
        )


def test_cache_config_with_capacity():
    cfg = CacheConfig(num_layers=2, num_kv_heads=1, head_dim=32, capacity=10)
    cfg2 = cfg.with_capacity([15, 25])
    assert cfg2.capacity == [15, 25]
    assert cfg2.total_capacity() == 40
    assert cfg.capacity == 10


def test_kv_cache_layerwise_enforce_and_stats():
    # 3 layers with capacities 5, 10, 15
    cfg = CacheConfig(
        num_layers=3,
        num_kv_heads=1,
        head_dim=16,
        capacity=[5, 10, 15],
        attention_sinks=2,
    )
    cache = KVCache(cfg, policy=SlidingWindowPolicy())

    # Append 8 tokens to all layers
    k = torch.randn(1, 1, 8, 16)
    v = torch.randn(1, 1, 8, 16)
    for layer_idx in range(3):
        cache.append(layer_idx, k, v)

    # After append, capacity should be enforced per layer:
    # Layer 0: capacity 5 -> 5 tokens
    # Layer 1: capacity 10 -> 8 tokens (did not exceed 10)
    # Layer 2: capacity 15 -> 8 tokens (did not exceed 15)
    assert cache.num_tokens(0) == 5
    assert cache.num_tokens(1) == 8
    assert cache.num_tokens(2) == 8

    stats = cache.stats()
    assert stats.tokens_per_layer == [5, 8, 8]
    assert stats.total_tokens == 21
    # Honest aggregate utilization: 21 / (5 + 10 + 15) = 21 / 30 = 0.7
    assert stats.utilization == pytest.approx(21 / 30)
    # Per layer utilization: [5/5, 8/10, 8/15]
    assert stats.utilization_per_layer is not None
    assert stats.utilization_per_layer[0] == pytest.approx(1.0)
    assert stats.utilization_per_layer[1] == pytest.approx(0.8)
    assert stats.utilization_per_layer[2] == pytest.approx(8 / 15)

    # Dynamic set_capacity
    cache.set_capacity([4, 6, 8])
    assert cache.num_tokens(0) == 4
    assert cache.num_tokens(1) == 6
    assert cache.num_tokens(2) == 8
    stats2 = cache.stats()
    assert stats2.tokens_per_layer == [4, 6, 8]
    assert stats2.capacity == [4, 6, 8]
    assert sum(stats2.capacity) == 18
    assert stats2.utilization == pytest.approx(18 / 18)


def test_uniform_allocation_strategy():
    strategy = UniformAllocationStrategy()
    assert strategy.name == "uniform"

    cfg = CacheConfig(num_layers=4, num_kv_heads=1, head_dim=16, capacity=10, attention_sinks=2)
    cache = KVCache(cfg, policy=SlidingWindowPolicy())

    # total_budget = 42 across 4 layers: 42 // 4 = 10 each, with 2 remainder -> [11, 11, 10, 10]
    caps = strategy.allocate(total_budget=42, cache=cache)
    assert caps == [11, 11, 10, 10]
    assert sum(caps) == 42
    assert all(c >= 2 for c in caps)

    # Budget smaller than sinks (4 layers * 2 sinks = 8 tokens required)
    with pytest.raises(CacheConfigError, match="insufficient for 4 layers"):
        strategy.allocate(total_budget=4, cache=cache)


def test_attention_proportional_allocation_strategy():
    strategy = AttentionProportionalAllocationStrategy()
    assert strategy.name == "attention_proportional"

    cfg = CacheConfig(num_layers=3, num_kv_heads=1, head_dim=16, capacity=20, attention_sinks=2)
    cache = KVCache(cfg, policy=SlidingWindowPolicy())

    # Setup dummy cumulative attention in layer metadata
    # Layer 0: high attention (70%), Layer 1: medium (20%), Layer 2: low (10%)
    cache.store.layer(0).metadata._cum_attention = torch.tensor([7.0])
    cache.store.layer(1).metadata._cum_attention = torch.tensor([2.0])
    cache.store.layer(2).metadata._cum_attention = torch.tensor([1.0])

    # Sinks per layer = 2 -> 6 tokens fixed.
    # Total budget = 30 -> surplus = 24 tokens.
    # Proportions: 70% of 24 = 16.8, 20% of 24 = 4.8, 10% of 24 = 2.4
    # With largest-remainder:
    # floored: 16, 4, 2 -> sum = 22, remaining = 2
    # remainders: 0.8, 0.8, 0.4 -> layers 0 and 1 get +1 -> 17, 5, 2
    # Total caps = sinks + surplus = [19, 7, 4]
    caps = strategy.allocate(total_budget=30, cache=cache)
    assert sum(caps) == 30
    assert all(c >= 2 for c in caps)
    assert caps[0] > caps[1] > caps[2]
    assert caps == [19, 7, 4]

    # Zero attention fallback: should distribute surplus uniformly
    cache.store.layer(0).metadata._cum_attention = torch.tensor([0.0])
    cache.store.layer(1).metadata._cum_attention = torch.tensor([0.0])
    cache.store.layer(2).metadata._cum_attention = torch.tensor([0.0])
    caps_zero = strategy.allocate(total_budget=30, cache=cache)
    assert sum(caps_zero) == 30
    assert caps_zero == [10, 10, 10]


def test_allocation_registry():
    assert "uniform" in available_allocation_strategies()
    assert "attention_proportional" in available_allocation_strategies()

    assert resolve_allocation_strategy_name("attention") == "attention_proportional"
    assert resolve_allocation_strategy_name("attention-proportional") == "attention_proportional"

    strat = build_allocation_strategy("attention")
    assert isinstance(strat, AttentionProportionalAllocationStrategy)

    strat_u = build_allocation_strategy("uniform")
    assert isinstance(strat_u, UniformAllocationStrategy)

    with pytest.raises(ConfigError, match="unknown allocation strategy"):
        build_allocation_strategy("nonexistent")


def test_run_spec_capacity_schedule_validation():
    # Valid schedule
    spec = RunSpec(
        model="synthetic:tiny",
        policy="full_cache",
        capacity_schedule="uniform",
    )
    assert spec.capacity_schedule == "uniform"

    # Alias resolved
    spec_alias = RunSpec(
        model="synthetic:tiny",
        policy="full_cache",
        capacity_schedule="attention",
    )
    assert spec_alias.capacity_schedule == "attention_proportional"

    # Invalid schedule
    with pytest.raises(ConfigError, match="unknown capacity_schedule"):
        RunSpec(
            model="synthetic:tiny",
            policy="full_cache",
            capacity_schedule="invalid_strat",
        )


def test_cli_capacity_schedule_arg():
    parser = build_parser()
    args = parser.parse_args(["--capacity-schedule", "attention_proportional"])
    assert args.capacity_schedule == "attention_proportional"
