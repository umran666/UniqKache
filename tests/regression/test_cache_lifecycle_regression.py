"""Capacity, position identity, and device state must survive cache mutations."""

import pytest
import torch

from uniqkache import CacheConfig, KVCache, build_policy


def make_cache(device="cpu"):
    return KVCache(
        CacheConfig(2, 1, 4, dtype=torch.float32, device=device, capacity=8, attention_sinks=2),
        policy=build_policy("sliding_window"),
    )


def test_capacity_propagates_and_round_trips(tmp_path):
    cache = make_cache()
    cache.set_capacity([4, 6])
    assert cache.store.config is cache.config
    assert [layer.capacity for layer in cache.store.layers] == [4, 6]
    restored = KVCache.load(cache.save(tmp_path / "cache.pt"), policy=cache.policy)
    assert restored.capacity == [4, 6]
    cache.set_capacity(None)
    assert cache.store.config.capacity is None
    assert all(layer.capacity is None for layer in cache.store.layers)


def test_positions_continue_after_eviction_and_checkpoint(tmp_path):
    cache = make_cache()
    kv = torch.ones(1, 1, 6, 4)
    cache.append(0, kv, kv)
    cache.evict(0, keep=torch.tensor([], dtype=torch.long))
    cache = KVCache.load(cache.save(tmp_path / "empty.pt"), policy=cache.policy)
    cache.append(0, kv[:, :, :2], kv[:, :, :2])
    assert cache.state(0).positions.tolist() == [6, 7]
    assert not cache.state(0).is_sink.any()
    cache.reset()
    cache.append(0, kv[:, :, :2], kv[:, :, :2])
    assert cache.state(0).positions.tolist() == [0, 1]


@pytest.mark.parametrize("compressed", [False, True])
def test_saved_cuda_configuration_can_restore_on_cpu(tmp_path, compressed):
    cache = make_cache()
    kv = torch.ones(1, 1, 4, 4)
    cache.append(0, kv, kv)
    if compressed:
        cache.compress()
    path = cache.save(tmp_path / "cache.pt")
    raw = torch.load(path, weights_only=True)
    raw["config"]["device"] = "cuda:0"
    for layer in raw["layers"]:
        layer["resident_device"] = "cuda:0"
    torch.save(raw, path)
    restored = KVCache.load(path, policy=cache.policy, map_location="cpu")
    assert restored.config.device == "cpu"
    assert restored.get(0)[0].device.type == "cpu"
    assert restored.state(0).positions.device.type == "cpu"
    assert restored.stats().offloaded_layers == []


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compressed", [False, True])
def test_append_to_offloaded_layer_moves_metadata_home(compressed):
    cache = make_cache("cuda:0")
    kv = torch.ones(1, 1, 4, 4, device="cuda:0")
    cache.append(0, kv, kv)
    if compressed:
        cache.compress()
    cache.offload(0)
    cache.append(0, kv[:, :, :1], kv[:, :, :1])
    assert cache.get(0)[0].device.type == "cuda"
    assert cache.state(0).positions.device.type == "cuda"
    cache.note_attention(0, torch.ones(1, 1, 1, 5, device="cuda:0") / 5)
    assert not cache.store.layer(0).is_offloaded


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_checkpoint_remaps_to_cpu(tmp_path):
    cache = make_cache("cuda:0")
    kv = torch.ones(1, 1, 4, 4, device="cuda:0")
    cache.append(0, kv, kv)
    restored = KVCache.load(
        cache.save(tmp_path / "cuda.pt"), policy=cache.policy, map_location="cpu"
    )
    assert restored.config.device == "cpu"
    assert restored.get(0)[0].device.type == "cpu"
