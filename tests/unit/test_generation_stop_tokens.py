"""Stop-token checks preserve the runtime's all-rows termination rule."""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine


class _ScriptedModel:
    def __init__(self, choices):
        self.choices = choices
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        choices = self.choices[min(self.calls, len(self.choices) - 1)]
        self.calls += 1
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        for row, token in enumerate(choices):
            logits[row, :, token] = 1
        return logits, None


@pytest.mark.parametrize(
    "choices,stop,expected,hit,calls",
    [
        ([[2], [3]], 2, [[2]], True, 1),
        ([[3], [2]], 2, [[3, 2]], True, 2),
        ([[2]], None, [[2, 2, 2]], False, 3),
        ([[2, 3], [2, 2]], 2, [[2, 2], [3, 2]], True, 2),
        ([[2, 2], [3, 3]], 2, [[2], [2]], True, 1),
    ],
)
def test_stop_detection_at_prefill_and_decode(choices, stop, expected, hit, calls):
    batch_size = len(choices[0])
    model = _ScriptedModel(choices)
    cache = KVCache(CacheConfig(num_layers=1, num_kv_heads=1, head_dim=1, batch_size=batch_size))
    result = GenerationEngine(
        model, cache, GenerationConfig(max_new_tokens=3, stop_token_id=stop)
    ).generate(torch.ones(batch_size, 2, dtype=torch.long))
    assert result.generated_ids.tolist() == expected
    assert result.hit_stop_token is hit
    assert model.calls == calls
