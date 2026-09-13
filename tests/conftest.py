"""Shared test fixtures.

Every fixture here builds a *small, fast, deterministic* object. The unit suite
must run in seconds on a laptop CPU with no GPU and no network, because a test
suite that is slow or environment-dependent stops being run — and a cache
correctness suite that stops being run is worse than none.
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.metrics.record import BenchmarkRecord
from uniqkache.models.synthetic import SyntheticConfig, build_model, get_preset
from uniqkache.policies import build_policy

# Small enough to be instant, large enough that a bug in head/seq handling is
# visible rather than accidentally coincidental.
NUM_LAYERS = 3
NUM_KV_HEADS = 2
NUM_HEADS = 4
HEAD_DIM = 8
HIDDEN = NUM_HEADS * HEAD_DIM


def good_record(**overrides) -> BenchmarkRecord:
    """A record that should pass validation, unless overridden to fail.

    Lives here rather than in the unit module that first needed it, because the
    regression suite builds records too. A test helper shared across suites
    belongs in `conftest`, not imported from one test module into another: the
    second arrangement makes an unrelated unit module a dependency of the
    regression suite, so editing the wrong test file breaks the wrong tests.
    """
    defaults = {
        "run_id": "r1",
        "git_commit": "abc123",
        "git_dirty": False,
        "model": "synthetic:tiny",
        "policy": "full_cache",
        "context_length": 256,
        "generated_tokens": 8,
        "capacity": None,
        "ttft_ms": 10.0,
        "tpot_ms": 1.0,
        "tokens_per_second": 100.0,
        "quality_metric": "perplexity",
        "quality_value": 500.0,
        "quality_reference": 500.0,
        "weights_are_random": False,
    }
    defaults.update(overrides)
    return BenchmarkRecord(**defaults)


@pytest.fixture
def cache_config() -> CacheConfig:
    """An unbounded float32 cache config on CPU."""
    return CacheConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
    )


@pytest.fixture
def bounded_config() -> CacheConfig:
    """A bounded cache config with attention sinks."""
    return CacheConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
        device="cpu",
        capacity=8,
        attention_sinks=2,
    )


@pytest.fixture
def full_cache(cache_config: CacheConfig) -> KVCache:
    """An unbounded cache with no policy, i.e. the full-cache baseline."""
    return KVCache(cache_config, policy=None)


def make_kv(seq: int, *, seed: int | None = None, heads: int = NUM_KV_HEADS) -> torch.Tensor:
    """Deterministic K/V tensor of shape ``[1, heads, seq, HEAD_DIM]``."""
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    return torch.randn(1, heads, seq, HEAD_DIM, generator=generator)


@pytest.fixture
def kv_factory():
    """Factory for deterministic K/V tensors."""
    return make_kv


@pytest.fixture
def policy_factory():
    """Factory for building policies by name."""

    def _build(name: str, **kwargs):
        return build_policy(name, **kwargs)

    return _build


@pytest.fixture(scope="session")
def tiny_model():
    """A tiny synthetic model, built once for the whole session."""
    config = get_preset("tiny")
    return build_model(config=config, seed=0, device="cpu", dtype=torch.float32)


@pytest.fixture
def model_config() -> SyntheticConfig:
    """A minimal model config for tests that build their own model."""
    return SyntheticConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        max_position_embeddings=256,
    )
