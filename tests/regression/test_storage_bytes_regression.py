"""Resident bytes track live backing storage, including allocation slack."""

import pytest
import torch

from uniqkache import CacheConfig, KVCache, build_policy
from uniqkache.prefetch.sequential import NextLayerPrefetch


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ),
    ],
)
def test_eviction_reclaims_oversized_storage_and_reports_real_bytes(device):
    cache = KVCache(
        CacheConfig(1, 1, 4, dtype=torch.float32, device=device, capacity=8),
        policy=build_policy("sliding_window"),
    )
    kv = torch.ones(1, 1, 64, 4, device=device)
    cache.append(0, kv, kv)
    layer = cache.store.layer(0)
    allocated = (
        layer._keys_buf.untyped_storage().nbytes() + layer._values_buf.untyped_storage().nbytes()
    )
    assert cache.stats().bytes_total == allocated == 8 * 2 * 4 * 4
    cache.evict(0, keep=torch.tensor([0, 1], device=device))
    assert cache.stats().bytes_total == 8 * 2 * 4 * 4
    assert cache.stats().payload_bytes == 2 * 2 * 4 * 4
    cache.evict(0, keep=torch.tensor([], dtype=torch.long, device=device))
    assert layer._keys_buf is layer._values_buf is None
    assert cache.stats().bytes_total == 0


def test_unbounded_bytes_include_geometric_slack():
    cache = KVCache(CacheConfig(1, 1, 4, dtype=torch.float32))
    kv = torch.ones(1, 1, 70, 4)
    cache.append(0, kv[:, :, :10], kv[:, :, :10])
    cache.append(0, kv[:, :, :60], kv[:, :, :60])
    assert cache.stats().bytes_total == 128 * 32
    assert cache.stats().payload_bytes == 70 * 32
    cache.evict(0, keep=torch.arange(3))
    assert cache.stats().bytes_total == 3 * 32


def test_empty_quantized_layer_releases_payload_and_scales():
    cache = KVCache(CacheConfig(1, 1, 4, dtype=torch.float32))
    kv = torch.ones(1, 1, 4, 4)
    cache.append(0, kv, kv)
    cache.compress()
    cache.evict(0, keep=torch.tensor([], dtype=torch.long))
    assert cache.stats().bytes_total == 0
    assert cache.store.layer(0)._compressed is None


def test_wrapped_prefetch_never_repeats_layers_or_bytes():
    cache = KVCache(CacheConfig(3, 1, 4, dtype=torch.float32))
    kv = torch.ones(1, 1, 2, 4)
    for idx in range(3):
        cache.append(idx, kv, kv)
        cache.store.layer(idx)._offloaded = True
    for depth in [3, 4, 10000]:
        plan = NextLayerPrefetch(depth=depth, wrap=True).plan(
            cache.store, cursor=2, resident_budget=None
        )
        assert plan.layers == [0, 1, 2]
        assert plan.bytes_to_move == cache.store.bytes_total()
        budget = cache.store.layer(0).bytes() * 2
        limited = NextLayerPrefetch(depth=depth, wrap=True).plan(cache.store, 2, budget)
        assert limited.layers == [0, 1]
        assert limited.bytes_to_move == budget
