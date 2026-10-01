"""Issue #66: a stop token selected by prefill must terminate generation immediately."""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.metrics.quality import random_token_ids
from uniqkache.models.synthetic import build_model
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine


@pytest.mark.regression
@pytest.mark.parametrize("limit", [1, 4])
def test_prefill_stop_token_ends_generation_without_decode(monkeypatch, limit):
    model = build_model("tiny", seed=0, device="cpu")
    prompt = random_token_ids(model.config.vocab_size, 8, seed=0)
    with torch.no_grad():
        logits, _ = model.forward(prompt)
    stop_id = int(logits[0, -1].argmax())
    calls = []
    original_forward = model.forward

    def counted_forward(input_ids, **kwargs):
        calls.append((input_ids.clone(), kwargs["start_pos"]))
        return original_forward(input_ids, **kwargs)

    monkeypatch.setattr(model, "forward", counted_forward)
    cache = KVCache(model.config.cache_config())
    result = GenerationEngine(
        model, cache, GenerationConfig(max_new_tokens=limit, stop_token_id=stop_id)
    ).generate(prompt)

    # Previously limit=1 missed the stop flag; larger limits decoded the first EOS again.
    assert result.generated_ids.tolist() == [[stop_id]]
    assert result.hit_stop_token is True
    assert result.generated_tokens == 1
    assert len(calls) == 1
    assert calls[0][1] == 0
    torch.testing.assert_close(calls[0][0], prompt)
    assert result.per_step_ms == []
    assert result.tpot_ms is None
    assert cache.step == 1
    assert all(cache.num_tokens(layer_idx) == 8 for layer_idx in range(cache.num_layers))
