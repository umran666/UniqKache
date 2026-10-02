"""Attention bookkeeping and allocation use an observable, recorded signal."""

import pytest
import torch

from uniqkache import CacheConfig, KVCache, build_policy
from uniqkache.allocation import AttentionProportionalAllocationStrategy
from uniqkache.bench.cli import _spec_from_args, build_parser
from uniqkache.bench.runner import _make_cache, _policy_config, build_model_for_spec
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine
from uniqkache.utils.errors import CacheConfigError


def test_negligible_attention_does_not_refresh_recency():
    cache = KVCache(
        CacheConfig(1, 1, 4, dtype=torch.float32, capacity=4), policy=build_policy("lru")
    )
    kv = torch.ones(1, 1, 4, 4)
    cache.append(0, kv, kv)
    cache.advance()
    cache.note_attention(0, torch.tensor([[[[1.0, 1e-30, 1e-30, 1e-30]]]]))
    assert cache.state(0).last_access.tolist() == [1, 0, 0, 0]
    cache.advance()
    cache.append(0, kv[:, :, :1], kv[:, :, :1])
    assert cache.state(0).positions.tolist() == [0, 2, 3, 4]


def test_threshold_is_configurable_and_recorded(tmp_path):
    args = build_parser().parse_args(
        [
            "--policy",
            "lru",
            "--capacity",
            "8",
            "--attention-access-threshold",
            "0.2",
            "--device",
            "cpu",
        ]
    )
    spec = _spec_from_args(args)
    built = build_model_for_spec(spec, torch.float32, "cpu")
    cache = _make_cache(spec, built, capacity=8, dtype=torch.float32, device="cpu")
    assert cache.config.with_capacity(16).attention_access_threshold == 0.2
    assert _policy_config(cache)["cache_config"]["attention_access_threshold"] == 0.2
    restored = KVCache.load(cache.save(tmp_path / "cache.pt"), policy=cache.policy)
    assert restored.config.attention_access_threshold == 0.2


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(CacheConfigError, match="finite"):
        CacheConfig(1, 1, 4, attention_access_threshold=threshold)


def test_prefill_allocation_distinguishes_normalized_attention():
    cache = KVCache(
        CacheConfig(2, 1, 4, dtype=torch.float32, capacity=8), policy=build_policy("sliding_window")
    )

    class Model:
        def forward(self, input_ids, cache, **kwargs):
            kv = torch.ones(1, 1, 4, 4)
            for idx in range(2):
                cache.append(idx, kv, kv)
            weights = [torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]), torch.full((1, 1, 1, 4), 0.25)]
            return torch.zeros(1, 4, 8), weights

    GenerationEngine(
        Model(),
        cache,
        GenerationConfig(max_new_tokens=1),
        allocation_strategy=AttentionProportionalAllocationStrategy(),
    ).generate(torch.tensor([[1, 2, 3, 4]]))
    assert [layer.metadata.cum_attention.sum().item() for layer in cache.store.layers] == [1.0, 1.0]
    assert cache.capacity == [12, 4]
