"""Integration: incremental decoding through the cache must match single-shot.

This is the single most important test in the repository.

A KV cache is a *performance* optimisation. It is only legitimate if it is
exactly equivalent to recomputing attention over the whole prefix. If this test
fails, every other result in the project is meaningless — a policy cannot be
compared for quality against a baseline that is itself wrong.

The test runs the same model twice:

* **single-shot** — the whole sequence through the model with no cache, using
  the model's ordinary causal mask;
* **incremental** — a prefill of the first chunk, then one token at a time,
  each step reading K/V back out of a ``KVCache``.

The logits must agree to within floating-point tolerance at every position.

Because ``SyntheticCausalLM`` initialises from a seed, the two runs use
bit-identical weights, so any divergence is attributable to the cache path
rather than to weight noise.
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.models.synthetic import build_model, get_preset
from uniqkache.policies import FullCachePolicy


def _build(seed: int = 0):
    config = get_preset("tiny")
    model = build_model(config=config, seed=seed, device="cpu", dtype=torch.float32)
    cache = KVCache(
        config.cache_config(dtype=torch.float32, device="cpu"), policy=FullCachePolicy()
    )
    return model, config, cache


def test_incremental_matches_single_shot_full_sequence() -> None:
    """Token-by-token decode through a full cache reproduces single-shot logits."""
    model, _, cache = _build()
    tokens = torch.randint(0, model.config.vocab_size, (1, 24))

    with torch.no_grad():
        reference, _ = model(tokens)

        # Prefill the first 4 tokens, then decode the remaining 20 one at a time.
        prefill = 4
        logits, _ = model(tokens[:, :prefill], cache=cache, start_pos=0)
        incremental = [logits]
        for step in range(prefill, tokens.shape[1]):
            step_logits, _ = model(tokens[:, step : step + 1], cache=cache, start_pos=step)
            incremental.append(step_logits)
        incremental_logits = torch.cat(incremental, dim=1)

    assert incremental_logits.shape == reference.shape, (
        incremental_logits.shape,
        reference.shape,
    )

    max_abs_diff = (incremental_logits - reference).abs().max().item()
    # float32 accumulation order differs between the two paths, so exact
    # equality is not expected; 1e-4 is far tighter than any real cache bug
    # would produce, and loose enough to tolerate reassociation.
    assert max_abs_diff < 1e-4, (
        f"incremental decode diverged from single-shot by {max_abs_diff:.3e}. "
        "The KV cache path is not equivalent to recomputation."
    )


def test_incremental_matches_single_shot_gqa() -> None:
    """The same equivalence holds when query and KV head counts differ."""
    from uniqkache.models.synthetic import SyntheticConfig

    config = SyntheticConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=2,
        max_position_embeddings=512,
    )
    model = build_model(config=config, seed=3, device="cpu", dtype=torch.float32)
    cache = KVCache(
        config.cache_config(dtype=torch.float32, device="cpu"), policy=FullCachePolicy()
    )
    tokens = torch.randint(0, config.vocab_size, (1, 16))

    with torch.no_grad():
        reference, _ = model(tokens)
        logits, _ = model(tokens[:, :1], cache=cache, start_pos=0)
        parts = [logits]
        for step in range(1, tokens.shape[1]):
            step_logits, _ = model(tokens[:, step : step + 1], cache=cache, start_pos=step)
            parts.append(step_logits)
        incremental = torch.cat(parts, dim=1)

    max_abs_diff = (incremental - reference).abs().max().item()
    assert max_abs_diff < 1e-4, f"GQA incremental decode diverged by {max_abs_diff:.3e}"


def test_prefill_chunking_is_invariant() -> None:
    """How the prefill is chunked must not change the result."""
    model, _, _ = _build(seed=7)
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))

    def run(chunks: list[int]) -> torch.Tensor:
        cache = KVCache(
            model.config.cache_config(dtype=torch.float32, device="cpu"),
            policy=FullCachePolicy(),
        )
        with torch.no_grad():
            start = 0
            outs = []
            for size in chunks:
                out, _ = model(tokens[:, start : start + size], cache=cache, start_pos=start)
                outs.append(out)
                start += size
        return torch.cat(outs, dim=1)

    whole = run([12])
    split = run([5, 7])
    many = run([1] * 12)

    assert torch.allclose(whole, split, atol=1e-4), "chunking into 5+7 changed the result"
    assert torch.allclose(whole, many, atol=1e-4), "token-at-a-time prefill changed the result"


@pytest.mark.integration
def test_attention_weights_are_returned_per_layer() -> None:
    """Attention weights are produced per layer, shaped [B, H, Tq, Tk]."""
    model, config, cache = _build()
    tokens = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        _, weights = model(tokens, cache=cache, return_attention=True)

    assert weights is not None
    assert len(weights) == config.num_layers
    for layer_weights in weights:
        assert layer_weights.shape[0] == 1
        assert layer_weights.shape[1] == config.num_heads
        assert layer_weights.shape[2] == 6
        assert layer_weights.shape[3] == 6
        # Rows of a softmax over keys must sum to 1 where the mask allows.
        sums = layer_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
