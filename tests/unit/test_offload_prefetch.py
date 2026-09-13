"""Unit tests for offloading, prefetching and memory-tier planning.

These tests run on CPU and on CUDA. Where a behaviour is only meaningful with a
real second tier (a GPU), the test asserts the *documented* CPU behaviour instead
of skipping silently — a skipped test tells a reader nothing, whereas an asserted
no-op documents the limitation.
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.store import KVStore
from uniqkache.cache.types import CacheConfig
from uniqkache.offload import TierManager
from uniqkache.prefetch import (
    NextLayerPrefetch,
    NoPrefetch,
    RecencyPrefetch,
    available_prefetch_policies,
    build_prefetch_policy,
)
from uniqkache.utils.device import cuda_is_available
from uniqkache.utils.errors import UniqKacheError

requires_cuda = pytest.mark.skipif(not cuda_is_available(), reason="requires a CUDA device")


def populated_store(num_layers: int = 2, seq: int = 8) -> KVStore:
    """A CPU store with every layer holding tokens."""
    config = CacheConfig(num_layers=num_layers, num_kv_heads=2, head_dim=8, dtype=torch.float32)
    store = KVStore(config)
    for layer in range(num_layers):
        store.layer(layer).append(torch.randn(1, 2, seq, 8), torch.randn(1, 2, seq, 8))
    return store


# ---------------------------------------------------------------------------
# Offload through the cache facade
# ---------------------------------------------------------------------------


class TestOffloadBehaviour:
    def test_offload_to_the_compute_device_is_a_no_op(self, full_cache, kv_factory):
        """On a CPU cache there is no cheaper tier, so nothing should move.

        This is the honest behaviour: reporting a successful offload that moved
        nothing would inflate a memory-saving claim.
        """
        full_cache.append(0, kv_factory(8), kv_factory(8))
        assert full_cache.offload(0, target="cpu") == 0
        assert full_cache.stats().bytes_offloaded == 0

    @requires_cuda
    def test_offload_moves_bytes_off_the_device(self, kv_factory):
        config = CacheConfig(
            num_layers=2, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cuda"
        )
        cache = KVCache(config, policy=None)
        for layer in range(2):
            cache.append(layer, kv_factory(8), kv_factory(8))

        before = cache.stats()
        assert cache.offload(0, target="cpu") == 1
        after = cache.stats()

        assert after.bytes_on_device < before.bytes_on_device
        assert after.bytes_offloaded > 0
        assert after.offloaded_layers == [0]

    @requires_cuda
    def test_total_bytes_are_invariant_under_offload(self, kv_factory):
        """Offloading relocates bytes; it does not delete them."""
        config = CacheConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cuda"
        )
        cache = KVCache(config, policy=None)
        cache.append(0, kv_factory(8), kv_factory(8))

        before = cache.stats()
        cache.offload(0, target="cpu")
        after = cache.stats()
        assert after.bytes_total == before.bytes_total

    @requires_cuda
    def test_prefetch_returns_bytes_to_the_device(self, kv_factory):
        config = CacheConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cuda"
        )
        cache = KVCache(config, policy=None)
        cache.append(0, kv_factory(8), kv_factory(8))
        cache.offload(0, target="cpu")

        assert cache.prefetch(0) == 1
        stats = cache.stats()
        assert stats.bytes_offloaded == 0
        assert cache.store.layer(0).resident_device.type == "cuda"

    @requires_cuda
    def test_prefetch_of_a_resident_layer_is_a_no_op(self, kv_factory):
        config = CacheConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cuda"
        )
        cache = KVCache(config, policy=None)
        cache.append(0, kv_factory(8), kv_factory(8))
        assert cache.prefetch(0) == 0

    @requires_cuda
    def test_values_survive_the_offload_round_trip(self, kv_factory):
        config = CacheConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cuda"
        )
        cache = KVCache(config, policy=None)
        keys = kv_factory(8, seed=11)
        cache.append(0, keys.to("cuda"), keys.to("cuda"))
        original, _ = cache.get(0)

        cache.offload(0, target="cpu")
        cache.prefetch(0)
        restored, _ = cache.get(0)
        assert torch.equal(original, restored)

    def test_offload_of_empty_layers_does_nothing(self, full_cache):
        assert full_cache.offload() == 0


# ---------------------------------------------------------------------------
# Tier planning
# ---------------------------------------------------------------------------


class TestTierManager:
    def test_manager_rejects_a_single_tier(self):
        with pytest.raises(UniqKacheError, match="no cheaper tier"):
            TierManager(device="cpu", host="cpu")

    def test_no_plan_needed_when_already_within_budget(self):
        store = populated_store()
        manager = TierManager(device="cuda", host="cpu")
        plan = manager.plan_offload(store, byte_budget=store.bytes_on_device())
        assert plan.layers == []
        assert plan.feasible is True

    def test_plan_selects_enough_layers_to_meet_the_budget(self):
        store = populated_store(num_layers=4)
        manager = TierManager(device="cuda", host="cpu")
        per_layer = store.layer(0).bytes()
        budget = store.bytes_on_device() - 2 * per_layer

        plan = manager.plan_offload(store, byte_budget=budget)
        assert plan.feasible is True
        assert len(plan.layers) >= 2
        assert plan.device_bytes_after <= budget

    def test_infeasible_plan_is_reported_not_hidden(self):
        """Asking for an impossible budget must say so, not under-deliver quietly."""
        store = populated_store(num_layers=2)
        manager = TierManager(device="cuda", host="cpu")
        plan = manager.plan_offload(store, byte_budget=0)

        # Every layer can be offloaded here, so this is achievable.
        assert plan.feasible is True

        # With the only layers excluded, it is not.
        plan = manager.plan_offload(store, byte_budget=0, exclude={0, 1})
        assert plan.feasible is False
        assert "cannot reach budget" in plan.reason

    def test_excluded_layers_are_never_offloaded(self):
        store = populated_store(num_layers=3)
        manager = TierManager(device="cuda", host="cpu")
        plan = manager.plan_offload(store, byte_budget=0, exclude={0})
        assert 0 not in plan.layers

    def test_largest_order_offloads_the_biggest_layer_first(self):
        store = populated_store(num_layers=3, seq=4)
        # Make layer 1 much larger than the others.
        store.layer(1).append(torch.randn(1, 2, 64, 8), torch.randn(1, 2, 64, 8))

        manager = TierManager(device="cuda", host="cpu")
        plan = manager.plan_offload(store, byte_budget=0, order="largest")
        assert plan.layers[0] == 1

    def test_unknown_order_is_rejected(self):
        manager = TierManager(device="cuda", host="cpu")
        with pytest.raises(UniqKacheError, match="unknown order"):
            manager.plan_offload(populated_store(), byte_budget=0, order="bogus")

    def test_negative_budget_is_rejected(self):
        manager = TierManager(device="cuda", host="cpu")
        with pytest.raises(UniqKacheError, match="byte_budget must be >= 0"):
            manager.plan_offload(populated_store(), byte_budget=-1)


# ---------------------------------------------------------------------------
# Prefetch policies
# ---------------------------------------------------------------------------


class TestPrefetchPolicies:
    def test_registry_exposes_the_builtins(self):
        for name in ("none", "next_layer", "recency"):
            assert name in available_prefetch_policies()

    def test_no_prefetch_does_nothing(self):
        plan = NoPrefetch().plan(populated_store(), cursor=0, resident_budget=None)
        assert plan.is_empty()

    def test_next_layer_prefetch_selects_nothing_when_all_resident(self):
        plan = NextLayerPrefetch().plan(populated_store(), cursor=0, resident_budget=None)
        assert plan.is_empty()

    @requires_cuda
    def test_next_layer_prefetch_targets_the_following_layer(self, kv_factory):
        config = CacheConfig(
            num_layers=3, num_kv_heads=2, head_dim=8, dtype=torch.float32, device="cuda"
        )
        cache = KVCache(config, policy=None)
        for layer in range(3):
            cache.append(layer, kv_factory(4), kv_factory(4))
        cache.offload(1, target="cpu")

        plan = NextLayerPrefetch(depth=1).plan(cache.store, cursor=0, resident_budget=None)
        assert plan.layers == [1]

    def test_depth_zero_disables_prefetching(self):
        plan = NextLayerPrefetch(depth=0).plan(populated_store(), cursor=0, resident_budget=None)
        assert plan.is_empty()

    def test_negative_depth_is_rejected(self):
        with pytest.raises(ValueError, match="depth must be >= 0"):
            NextLayerPrefetch(depth=-1)

    def test_recency_prefetch_of_nothing_is_empty(self):
        plan = RecencyPrefetch(k=2).plan(populated_store(), cursor=0, resident_budget=None)
        assert plan.is_empty()

    def test_builder_rejects_unknown_name(self):
        with pytest.raises(UniqKacheError, match="unknown prefetch policy"):
            build_prefetch_policy("does_not_exist")
