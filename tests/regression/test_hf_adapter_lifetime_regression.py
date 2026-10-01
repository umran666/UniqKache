"""Issue #79: completed HF sequences must release their KV caches and tensors."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.models import hf_backend
from uniqkache.policies import SlidingWindowPolicy


class _CachingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=32,
            vocab_size=16,
        )

    def forward(self, input_ids, past_key_values, **kwargs):
        keys = input_ids[:, None, :, None].float().expand(-1, 2, -1, 8)
        for layer in past_key_values.layers:
            layer.update(keys, -keys)
        return SimpleNamespace(
            logits=torch.zeros(1, input_ids.shape[1], 16),
            past_key_values=past_key_values,
        )


@pytest.mark.regression
@pytest.mark.parametrize("capacity", [None, 8])
@pytest.mark.parametrize("retain_adapter", [False, True])
def test_completed_hf_sequences_release_cache_and_backing_tensors(
    monkeypatch, capacity, retain_adapter
):
    registry = weakref.WeakKeyDictionary()
    monkeypatch.setattr(hf_backend, "_CACHE_ADAPTERS", registry)
    backend = hf_backend.HFBackend(_CachingModel(), identifier="local/cache-lifetime")

    for _ in range(3):
        cache = KVCache(backend.cache_config(capacity=capacity), policy=SlidingWindowPolicy())
        backend.forward(torch.tensor([[1, 2, 3]]), cache)
        adapter = registry[cache]
        backend.forward(torch.tensor([[4]]), cache, start_pos=3)
        assert registry[cache] is adapter
        for layer_idx in range(cache.num_layers):
            keys, values = cache.get(layer_idx)
            torch.testing.assert_close(keys[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
            torch.testing.assert_close(values, -keys)
        del keys, values
        assert cache.stats().bytes_total > 0
        cache_ref = weakref.ref(cache)
        buffer_refs = [
            weakref.ref(buffer)
            for layer in cache.store.layers
            for buffer in (layer._keys_buf, layer._values_buf)
        ]
        if not retain_adapter:
            del adapter

        # Previously the registry value kept its weak key and every KV buffer alive.
        del cache
        gc.collect()
        assert cache_ref() is None
        assert len(registry) == 0
        assert all(ref() is None for ref in buffer_refs)
