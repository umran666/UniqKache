"""Regression tests for KV cache checkpoint save and restore.

Pins:
1. Round-trip test: cache -> save -> load -> identical stats() and bit-identical get() output.
2. Regression test for quantised + offloaded layers preserving representation and placement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.conftest import HEAD_DIM, NUM_KV_HEADS, make_kv
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.policies import SlidingWindowPolicy


class TestCheckpointRegression:
    def test_cache_round_trip_identical_stats_and_bit_identical_get(self, tmp_path: Path):
        """Acceptance test 1: cache -> save -> load -> identical stats() and bit-identical get()."""
        config = CacheConfig(
            num_layers=3,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            device="cpu",
            capacity=8,
            attention_sinks=2,
        )
        policy = SlidingWindowPolicy(window=6)
        cache = KVCache(config, policy=policy)

        # Append more tokens than capacity to trigger eviction
        k0, v0 = make_kv(12, seed=1), make_kv(12, seed=2)
        k1, v1 = make_kv(10, seed=3), make_kv(10, seed=4)
        cache.append(0, k0, v0)
        cache.append(1, k1, v1)

        # Advance step and record signals
        cache.advance()
        cache.advance()
        cache.note_access(0, torch.tensor([0, 1]))
        cache.note_attention(0, torch.full((1, NUM_KV_HEADS, 1, 8), 0.25), mode="last_query")

        orig_stats = cache.stats()
        assert orig_stats.evictions > 0

        ckpt_path = tmp_path / "cache_round_trip.pt"
        cache.save(ckpt_path, model="synthetic:tiny")

        # Restore into new cache instance
        restored = KVCache.load(ckpt_path, policy=SlidingWindowPolicy(window=6))

        # Check identical stats()
        rest_stats = restored.stats()
        assert rest_stats == orig_stats
        assert restored.step == cache.step

        # Bit-identical get() output
        for layer_idx in (0, 1):
            k_orig, v_orig = cache.get(layer_idx)
            k_rest, v_rest = restored.get(layer_idx)
            assert torch.equal(k_rest, k_orig)
            assert torch.equal(v_rest, v_orig)

        # PolicyState signals are identical
        for layer_idx in (0, 1):
            s_orig = cache.state(layer_idx)
            s_rest = restored.state(layer_idx)
            assert s_rest.step == s_orig.step
            assert s_rest.num_cached == s_orig.num_cached
            assert s_rest.capacity == s_orig.capacity
            assert torch.equal(s_rest.positions, s_orig.positions)
            assert torch.equal(s_rest.last_access, s_orig.last_access)
            assert torch.equal(s_rest.cum_attention, s_orig.cum_attention)
            assert torch.equal(s_rest.hit_count, s_orig.hit_count)
            assert torch.equal(s_rest.is_sink, s_orig.is_sink)
            assert s_rest.memory_pressure == pytest.approx(s_orig.memory_pressure)

        # In-place load_checkpoint test
        cache_inplace = KVCache(config, policy=SlidingWindowPolicy(window=6))
        cache_inplace.load_checkpoint(ckpt_path)
        assert cache_inplace.stats() == orig_stats
        k_in, v_in = cache_inplace.get(0)
        assert torch.equal(k_in, cache.get(0)[0])
        assert torch.equal(v_in, cache.get(0)[1])

    def test_quantized_and_offloaded_layers_regression(self, tmp_path: Path):
        """Acceptance test 2: Regression test for quantised + offloaded layers."""
        config = CacheConfig(
            num_layers=4,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            device="cpu",
        )
        cache = KVCache(config, policy=None)

        # Populate all 4 layers
        for i in range(4):
            k, v = make_kv(16, seed=100 + i), make_kv(16, seed=200 + i)
            cache.append(i, k, v)

        # Layer 0: plain resident
        # Layer 1: compressed resident
        cache.compress(1)
        # Layer 2: plain offloaded
        cache.store.layer(2)._offloaded = True
        # Layer 3: compressed AND offloaded
        cache.compress(3)
        cache.store.layer(3)._offloaded = True

        orig_stats = cache.stats()
        assert orig_stats.compressions > 0

        ckpt_path = tmp_path / "mixed_state.pt"
        cache.save(ckpt_path, model="synthetic:tiny")

        # Load
        restored = KVCache.load(ckpt_path)

        # 1. Stats match identically
        assert restored.stats() == orig_stats

        # 2. Representation preserved without dequantizing
        assert restored.store.layer(0).is_compressed is False
        assert restored.store.layer(1).is_compressed is True
        assert restored.store.layer(1)._keys is None
        assert restored.store.layer(2).is_compressed is False
        assert restored.store.layer(3).is_compressed is True
        assert restored.store.layer(3)._keys is None

        # 3. Offload placement preserved
        assert restored.store.layer(0).is_offloaded is False
        assert restored.store.layer(1).is_offloaded is False
        assert restored.store.layer(2).is_offloaded is True
        assert restored.store.layer(3).is_offloaded is True
        assert set(restored.stats().offloaded_layers) == {2, 3}

        # 4. Bit-identical get() across all layers
        for i in range(4):
            orig_k, orig_v = cache.get(i)
            rest_k, rest_v = restored.get(i)
            assert torch.equal(rest_k, orig_k)
            assert torch.equal(rest_v, orig_v)
