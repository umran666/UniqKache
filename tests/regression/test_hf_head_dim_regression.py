"""Issue #70: valid explicit attention head widths must reach the KV cache."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.models.hf_backend import HFBackend


@pytest.mark.regression
def test_explicit_head_dim_accepts_matching_kv_tensors_without_transformers():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                hidden_size=32,
                head_dim=16,
                vocab_size=32,
            )

        def forward(self, input_ids, past_key_values, **kwargs):
            shape = (input_ids.shape[0], 2, input_ids.shape[1], 16)
            past_key_values.layers[0].update(torch.ones(shape), torch.zeros(shape))
            return SimpleNamespace(
                logits=torch.zeros(*input_ids.shape, 32), past_key_values=past_key_values
            )

    backend = HFBackend(_Model(), identifier="local/explicit-head-dim")
    cache = KVCache(backend.cache_config())
    backend.forward(torch.tensor([[1, 2, 3]]), cache)
    keys, values = cache.get(0)
    assert keys.shape == values.shape == (1, 2, 3, 16)
    assert cache.stats().payload_bytes == 2 * 1 * 2 * 3 * 16 * 4
    assert cache.stats().bytes_total >= cache.stats().payload_bytes


@pytest.mark.regression
def test_local_llama_explicit_head_dim_prefill_and_decode_match_native_logits():
    pytest.importorskip("transformers")
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        pytest.skip(f"requires a usable Transformers Llama installation: {exc}")

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    if getattr(config, "head_dim", None) != 16:
        pytest.skip("requires Transformers support for explicit Llama head_dim")
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    if model.model.layers[0].self_attn.k_proj.out_features != config.num_key_value_heads * 16:
        pytest.skip("requires a Llama implementation honoring explicit head_dim")
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        reference = model(tokens, use_cache=False).logits
    backend = HFBackend(model, identifier="local/random-llama", weights_are_random=True)
    cache = KVCache(backend.cache_config())

    # Before the fix, prefill raised "expects head_dim 8, received 16".
    prefill, _ = backend.forward(tokens[:, :3], cache)
    decoded, _ = backend.forward(tokens[:, 3:], cache, start_pos=3)
    torch.testing.assert_close(
        torch.cat([prefill, decoded], dim=1), reference, rtol=1e-5, atol=1e-6
    )
    assert backend.config["head_dim"] == cache.config.head_dim == 16
    assert cache.get(0)[0].shape == (1, 2, 4, 16)
