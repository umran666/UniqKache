"""Attention must see current queries before retention decisions run."""

from types import SimpleNamespace

import pytest
import torch

from uniqkache import KVCache, build_policy
from uniqkache.metrics.quality import needle_retrieval, perplexity, random_token_ids
from uniqkache.models import hf_backend
from uniqkache.models.synthetic import apply_rope, build_model
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_and_model_preserve_activation_dtype(dtype):
    x = torch.randn(1, 1, 3, 8, dtype=dtype)
    assert apply_rope(x, torch.arange(3)).dtype == dtype
    model = build_model("tiny", dtype=dtype)
    cache = KVCache(model.config.cache_config(dtype=dtype))
    logits, _ = model.forward(torch.tensor([[1, 2, 3]]), cache)
    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("policy", ["sliding_window", "attention_based"])
def test_oversized_prefill_is_finite_and_post_attention_bounded(policy):
    model = build_model("tiny")
    cache = KVCache(model.config.cache_config(capacity=4), policy=build_policy(policy))
    logits, _ = model.forward(torch.arange(16)[None, :], cache)
    assert torch.isfinite(logits).all()
    assert all(layer.num_tokens == 4 for layer in cache.store.layers)
    quality = perplexity(model, torch.arange(16)[None, :], cache, chunk_size=16)
    assert quality.value is not None and torch.isfinite(torch.tensor(quality.value))


def test_new_heavy_hitter_gets_attention_before_eviction(monkeypatch):
    model = build_model("tiny")
    cache = KVCache(model.config.cache_config(capacity=2), policy=build_policy("attention_based"))
    model.forward(torch.tensor([[1, 2]]), cache)
    for layer in cache.store.layers:
        layer.metadata.cum_attention.zero_()

    def attend_newest(scores, **kwargs):
        weights = torch.zeros_like(scores)
        weights[..., -1] = 1
        return weights

    monkeypatch.setattr(torch.nn.functional, "softmax", attend_newest)
    model.forward(torch.tensor([[3]]), cache, start_pos=2)
    assert all(2 in layer.metadata.positions.tolist() for layer in cache.store.layers)
    for layer in cache.store.layers:
        newest = layer.metadata.positions == 2
        assert layer.metadata.cum_attention[newest].item() == 1.0


def test_quality_records_attention_and_advances_one_step_per_forward():
    model = build_model("tiny")
    cache = KVCache(model.config.cache_config(capacity=8), policy=build_policy("lru"))
    perplexity(model, random_token_ids(512, 16), cache, chunk_size=2)
    assert cache.step == 8
    assert cache.state(0).last_access.max() > 0
    needle_retrieval(
        model, haystack_length=16, needle=torch.tensor([[1, 2, 3]]), vocab_size=512, cache=cache
    )
    assert cache.step == 3
    assert cache.state(0).cum_attention.sum() > 0


@pytest.mark.parametrize("count", [0, 1, 3])
def test_generation_counts_only_decode_steps_for_throughput(count):
    model = build_model("tiny")
    cache = KVCache(model.config.cache_config())
    result = GenerationEngine(model, cache, GenerationConfig(max_new_tokens=count)).generate(
        torch.tensor([[1, 2, 3]])
    )
    assert result.generated_tokens == count
    assert len(result.per_step_ms) == max(0, count - 1)
    if count < 2:
        assert result.tokens_per_second is None
        assert result.decode_ms == 0
    else:
        assert result.tokens_per_second == pytest.approx((count - 1) * 1000 / result.decode_ms)
    assert (result.ttft_ms is None) == (count == 0)


def test_hf_uncached_forward_starts_each_new_sequence_fresh(monkeypatch):
    class NativeCache:
        def __init__(self):
            self.tokens = 0

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(
                num_hidden_layers=1, num_attention_heads=1, hidden_size=8, vocab_size=8
            )

        def forward(self, input_ids, past_key_values, **kwargs):
            past_key_values.tokens += input_ids.shape[1]
            return SimpleNamespace(
                logits=torch.full((*input_ids.shape, 8), past_key_values.tokens),
                past_key_values=past_key_values,
            )

    monkeypatch.setattr(
        hf_backend, "_require_transformers", lambda: SimpleNamespace(DynamicCache=NativeCache)
    )
    backend = hf_backend.HFBackend(Model(), identifier="local/stub")
    first, _ = backend.forward(torch.tensor([[1, 2, 3]]))
    second, _ = backend.forward(torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(first, second)
    continued, _ = backend.forward(torch.tensor([[4]]), start_pos=3)
    assert continued[0, 0, 0] == 4


def test_deferred_enforcement_restores_settings_after_failure():
    model = build_model("tiny")
    cache = KVCache(model.config.cache_config(capacity=2), policy=build_policy("sliding_window"))
    with pytest.raises(RuntimeError), cache.defer_enforcement():
        raise RuntimeError("forward failed")
    assert cache.auto_enforce
