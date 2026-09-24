"""Unit tests for the KV cache storage layer and facade.

Covers the operations the project's interface promises: insertion, retrieval,
eviction, clearing, and the invariants that keep occupancy and representation
from being confused with one another.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import HEAD_DIM, NUM_KV_HEADS, NUM_LAYERS, make_kv
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.policies import SlidingWindowPolicy
from uniqkache.utils.errors import CacheConfigError, CacheStateError, PolicyError

# ---------------------------------------------------------------------------
# Insertion and retrieval
# ---------------------------------------------------------------------------


class TestInsertionAndRetrieval:
    def test_append_then_get_returns_exactly_what_was_stored(self, full_cache, kv_factory):
        keys = kv_factory(4, seed=1)
        values = kv_factory(4, seed=2)
        full_cache.append(0, keys, values)

        got_keys, got_values = full_cache.get(0)
        assert torch.equal(got_keys, keys)
        assert torch.equal(got_values, values)

    def test_append_accumulates_along_sequence_axis(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(3, seed=1), kv_factory(3, seed=2))
        full_cache.append(0, kv_factory(5, seed=3), kv_factory(5, seed=4))

        keys, _ = full_cache.get(0)
        assert keys.shape[2] == 8
        assert full_cache.num_tokens(0) == 8

    def test_layers_are_independent(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(2, seed=1), kv_factory(2, seed=2))
        full_cache.append(1, kv_factory(6, seed=3), kv_factory(6, seed=4))

        assert full_cache.num_tokens(0) == 2
        assert full_cache.num_tokens(1) == 6
        assert full_cache.num_tokens(2) == 0
        assert full_cache.num_tokens() == 6  # max across layers

    def test_positions_continue_consecutively_by_default(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(3), kv_factory(3))
        full_cache.append(0, kv_factory(2), kv_factory(2))
        positions = full_cache.store.layer(0).metadata.positions
        assert positions.tolist() == [0, 1, 2, 3, 4]

    def test_explicit_positions_are_honoured(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(2), kv_factory(2), positions=torch.tensor([10, 11]))
        positions = full_cache.store.layer(0).metadata.positions
        assert positions.tolist() == [10, 11]

    def test_get_on_empty_layer_raises(self, full_cache):
        with pytest.raises(CacheStateError, match="no cached tokens"):
            full_cache.get(0)

    def test_get_on_out_of_range_layer_raises(self, full_cache):
        with pytest.raises(CacheStateError, match="out of range"):
            full_cache.get(NUM_LAYERS)


class TestAppendValidation:
    """Malformed K/V must be rejected, not coerced into a plausible shape."""

    def test_mismatched_key_value_shape_raises(self, full_cache):
        with pytest.raises(CacheStateError, match="share a shape"):
            full_cache.append(0, make_kv(4), make_kv(5))

    def test_wrong_rank_raises(self, full_cache):
        bad = torch.randn(1, NUM_KV_HEADS, 4)
        with pytest.raises(CacheStateError, match=r"\[batch, num_kv_heads, seq, head_dim\]"):
            full_cache.append(0, bad, bad)

    def test_wrong_head_count_raises(self, full_cache):
        bad = make_kv(4, heads=NUM_KV_HEADS + 1)
        with pytest.raises(CacheStateError, match="kv heads"):
            full_cache.append(0, bad, bad)

    def test_wrong_head_dim_raises(self, full_cache):
        bad = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM + 1)
        with pytest.raises(CacheStateError, match="head_dim"):
            full_cache.append(0, bad, bad)

    def test_position_length_mismatch_raises(self, full_cache):
        with pytest.raises(CacheStateError, match="positions has"):
            full_cache.append(0, make_kv(4), make_kv(4), positions=torch.tensor([0, 1]))


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_evict_by_indices_drops_exactly_those_tokens(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(6, seed=1), kv_factory(6, seed=2))
        dropped = full_cache.evict(0, indices=torch.tensor([1, 3]))

        assert dropped == 2
        assert full_cache.num_tokens(0) == 4
        # Surviving positions must be the complement.
        assert full_cache.store.layer(0).metadata.positions.tolist() == [0, 2, 4, 5]

    def test_evict_by_keep_retains_only_those(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(6), kv_factory(6))
        dropped = full_cache.evict(0, keep=torch.tensor([0, 5]))

        assert dropped == 4
        assert full_cache.store.layer(0).metadata.positions.tolist() == [0, 5]

    def test_evicted_rows_are_really_gone(self, full_cache, kv_factory):
        keys = kv_factory(4, seed=7)
        full_cache.append(0, keys, keys)
        full_cache.evict(0, keep=torch.tensor([2]))

        got, _ = full_cache.get(0)
        assert got.shape[2] == 1
        assert torch.equal(got, keys[:, :, 2:3, :])

    def test_evict_both_selectors_is_rejected(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(4), kv_factory(4))
        with pytest.raises(PolicyError, match="not both"):
            full_cache.evict(0, indices=torch.tensor([0]), keep=torch.tensor([1]))

    def test_evict_out_of_range_indices_raises(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(4), kv_factory(4))
        with pytest.raises(CacheStateError, match="out of range"):
            full_cache.evict(0, indices=torch.tensor([99]))

    def test_evict_across_all_layers(self, full_cache, kv_factory):
        for layer in range(NUM_LAYERS):
            full_cache.append(layer, kv_factory(6), kv_factory(6))
        dropped = full_cache.evict(keep=torch.tensor([0, 1]))
        assert dropped == 4 * NUM_LAYERS
        assert all(full_cache.num_tokens(layer) == 2 for layer in range(NUM_LAYERS))

    def test_bounded_cache_enforces_capacity_on_append(self, bounded_config, kv_factory):
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy())
        for _ in range(10):
            cache.append(0, kv_factory(1), kv_factory(1))
        assert cache.num_tokens(0) == bounded_config.capacity

    def test_sink_tokens_survive_eviction(self, bounded_config, kv_factory):
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy())
        for _ in range(20):
            cache.append(0, kv_factory(1), kv_factory(1))
            cache.advance()

        positions = cache.store.layer(0).metadata.positions.tolist()
        # The first `attention_sinks` positions must still be present.
        for sink in range(bounded_config.attention_sinks):
            assert sink in positions, f"sink token at position {sink} was evicted: {positions}"

    def test_clear_resets_everything(self, full_cache, kv_factory):
        for layer in range(NUM_LAYERS):
            full_cache.append(layer, kv_factory(4), kv_factory(4))
        full_cache.advance()
        full_cache.clear()

        assert full_cache.num_tokens() == 0
        assert full_cache.step == 0
        for layer in range(NUM_LAYERS):
            assert not full_cache.store.layer(layer).is_initialized

    # --- eviction accounting (issue #15) -----------------------------------
    # `evict()` must count an eviction only when tokens were actually dropped,
    # in agreement with `enforce_capacity`. A no-op evict that removes zero
    # tokens must not bump the counter `stats().evictions` reports.

    def test_noop_evict_on_empty_cache_does_not_count(self, full_cache):
        # The exact symptom: an empty cache reports an eviction that dropped
        # nothing, so a result could quote a counter that was not a result.
        assert full_cache.stats().evictions == 0
        dropped = full_cache.evict(0, indices=torch.tensor([0]))
        assert dropped == 0
        assert full_cache.stats().evictions == 0
        assert full_cache.state_dict()["evictions"] == 0

    def test_noop_evict_that_selects_everything_does_not_count(self, bounded_config, kv_factory):
        # A cache within budget whose selection keeps every token: `evict()`
        # with no selector delegates to `enforce_capacity`, which must also
        # count nothing when nothing is dropped.
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy())
        for _ in range(bounded_config.capacity):
            cache.append(0, kv_factory(1), kv_factory(1))

        assert cache.stats().evictions == 0
        dropped = cache.evict(0, keep=torch.arange(bounded_config.capacity))
        assert dropped == 0
        assert cache.stats().evictions == 0

    def test_evict_keep_everything_when_over_budget_does_not_count(
        self, bounded_config, kv_factory
    ):
        # Over budget (kept over via auto_enforce=False), but the explicit
        # `keep` selection retains every token, so nothing is dropped. The
        # counter must not claim an eviction that removed zero tokens.
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy(), auto_enforce=False)
        for _ in range(bounded_config.capacity + 3):
            cache.append(0, kv_factory(1), kv_factory(1))
        assert cache.num_tokens(0) == bounded_config.capacity + 3

        dropped = cache.evict(0, keep=torch.arange(cache.num_tokens(0)))
        assert dropped == 0
        assert cache.stats().evictions == 0

    def test_evict_and_enforce_capacity_agree_when_nothing_is_dropped(
        self, bounded_config, kv_factory
    ):
        # The two eviction paths must agree on what counts as an eviction.
        # `evict()` with an explicit keep-everything selector and
        # `enforce_capacity()` on a cache within budget both drop nothing, so
        # both must leave `stats().evictions` at zero.
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy())
        for _ in range(bounded_config.capacity):
            cache.append(0, kv_factory(1), kv_factory(1))

        assert cache.evict(0, keep=torch.arange(cache.num_tokens(0))) == 0
        assert cache.evict(0) == 0  # delegates to enforce_capacity
        assert cache.stats().evictions == 0

    def test_evict_counts_only_actual_drops(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(6), kv_factory(6))
        dropped = full_cache.evict(0, indices=torch.tensor([2, 4]))
        assert dropped == 2
        assert full_cache.stats().evictions == 1
        assert full_cache.state_dict()["evicted_tokens"] == 2
        # A further no-op must not add to the count.
        full_cache.evict(0, indices=torch.tensor([]))
        assert full_cache.stats().evictions == 1


# ---------------------------------------------------------------------------
# Policy wiring
# ---------------------------------------------------------------------------


class TestPolicyWiring:
    def test_bounded_cache_without_policy_is_rejected(self):
        with pytest.raises(PolicyError, match="requires a policy"):
            KVCache(CacheConfig(num_layers=1, num_kv_heads=1, head_dim=4, capacity=8))

    def test_unbounded_cache_without_policy_is_allowed(self):
        cache = KVCache(CacheConfig(num_layers=1, num_kv_heads=1, head_dim=4), policy=None)
        assert cache.policy is None

    def test_enforce_capacity_without_policy_raises(self, cache_config):
        # capacity is None, so enforce_capacity is a no-op rather than an error.
        cache = KVCache(cache_config, policy=None)
        assert cache.enforce_capacity() == 0

    def test_state_matches_cache_contents(self, bounded_config, kv_factory):
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy())
        cache.append(0, kv_factory(5), kv_factory(5))
        state = cache.state(0)

        assert state.num_cached == 5
        assert state.capacity == bounded_config.capacity
        assert state.positions.tolist() == [0, 1, 2, 3, 4]
        assert state.is_sink.tolist() == [True, True, False, False, False]
        assert state.memory_pressure == pytest.approx(5 / 8)

    def test_auto_enforce_can_be_disabled(self, bounded_config, kv_factory):
        cache = KVCache(bounded_config, policy=SlidingWindowPolicy(), auto_enforce=False)
        for _ in range(12):
            cache.append(0, kv_factory(1), kv_factory(1))
        # Nothing evicted yet, so the cache is over budget until asked.
        assert cache.num_tokens(0) == 12
        assert cache.enforce_capacity(0) == 12 - bounded_config.capacity
        assert cache.num_tokens(0) == bounded_config.capacity


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class TestSignalBookkeeping:
    def test_note_attention_accumulates_mass(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(4), kv_factory(4))
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        full_cache.note_attention(0, weights.reshape(1, 1, 1, 4))

        cum = full_cache.store.layer(0).metadata.cum_attention
        assert torch.allclose(cum, weights, atol=1e-6)

    def test_note_attention_updates_recency_only_above_threshold(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(3), kv_factory(3))
        full_cache.advance()
        full_cache.advance()
        full_cache.note_attention(0, torch.tensor([0.0, 0.5, 0.0]).reshape(1, 1, 1, 3))

        last_access = full_cache.store.layer(0).metadata.last_access
        # Only the token that actually received attention is refreshed.
        assert last_access.tolist() == [0, 2, 0]

    def test_stale_attention_length_is_rejected(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(4), kv_factory(4))
        with pytest.raises(CacheStateError, match="attention weights have length"):
            full_cache.note_attention(0, torch.rand(1, 1, 1, 6))

    def test_reduce_attention_modes(self):
        from uniqkache.cache.metadata import reduce_attention

        attention = torch.softmax(torch.randn(2, 3, 5, 7), dim=-1)
        assert reduce_attention(attention, mode="last_query").shape == (7,)
        assert reduce_attention(attention, mode="all_queries").shape == (7,)
        assert reduce_attention(attention, query_index=2).shape == (7,)
        with pytest.raises(ValueError, match="unknown reduce mode"):
            reduce_attention(attention, mode="nonsense")
        with pytest.raises(ValueError, match=r"\[batch, heads, queries, keys\]"):
            reduce_attention(torch.randn(3, 5))


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestCacheConfig:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"num_layers": 0}, "num_layers"),
            ({"num_kv_heads": 0}, "num_kv_heads"),
            ({"head_dim": 0}, "head_dim"),
            ({"capacity": 0}, "capacity"),
            ({"attention_sinks": -1}, "attention_sinks"),
            ({"capacity": 4, "attention_sinks": 8}, "cannot exceed capacity"),
        ],
    )
    def test_invalid_configs_are_rejected(self, kwargs, match):
        base = {"num_layers": 2, "num_kv_heads": 2, "head_dim": 8}
        base.update(kwargs)
        with pytest.raises(CacheConfigError, match=match):
            CacheConfig(**base)

    def test_byte_accounting_is_exact(self):
        config = CacheConfig(
            num_layers=2, num_kv_heads=3, head_dim=16, dtype=torch.float32, batch_size=1
        )
        # 2 layers * 2 (k and v) * 3 heads * 16 dims * 4 bytes = 768 bytes/token
        assert config.bytes_per_token() == 768
        assert config.bytes_for_tokens(10) == 7680

    def test_tokens_for_bytes_floors(self):
        config = CacheConfig(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32)
        # 32 bytes per token; 100 bytes fits 3 whole tokens, not 3.125.
        assert config.tokens_for_bytes(100) == 3

    def test_bytes_for_tokens_takes_a_per_layer_count(self):
        """Pins the unit of `bytes_for_tokens` against the summed stats count.

        `CacheStats.total_tokens` is summed across layers, while
        `bytes_for_tokens` expects a per-layer count. Confusing the two
        over-counts by `num_layers`, which is easy to do and silent, so the
        relationship is asserted here rather than left to a docstring.
        """
        config = CacheConfig(num_layers=4, num_kv_heads=2, head_dim=8, dtype=torch.float32)
        # 2 (k and v) * 2 heads * 8 dims * 4 bytes = 128 bytes per token per layer.
        assert config.bytes_per_token_per_layer() == 128
        assert config.bytes_per_token() == 4 * 128

        per_layer = 16
        total_across_layers = per_layer * config.num_layers  # 64

        right = config.bytes_for_tokens(per_layer)
        assert right == 16 * 512

        # Reconciling the two units the correct way: scale the per-layer cost by
        # the summed token count.
        assert config.bytes_per_token_per_layer() * total_across_layers == right

        # And the mistake this guards against is wrong by exactly num_layers.
        assert config.bytes_for_tokens(total_across_layers) == right * config.num_layers

    def test_float16_halves_the_footprint(self):
        fp32 = CacheConfig(num_layers=1, num_kv_heads=2, head_dim=8, dtype=torch.float32)
        fp16 = CacheConfig(num_layers=1, num_kv_heads=2, head_dim=8, dtype=torch.float16)
        assert fp32.bytes_per_token() == 2 * fp16.bytes_per_token()


# ---------------------------------------------------------------------------
# Preallocated buffer and memory scaling
# ---------------------------------------------------------------------------


class TestPreallocatedBuffer:
    def test_bounded_storage_preallocates_to_capacity(self, bounded_config, kv_factory):
        from uniqkache.cache.store import LayerStorage

        storage = LayerStorage(
            layer_idx=0,
            num_kv_heads=bounded_config.num_kv_heads,
            head_dim=bounded_config.head_dim,
            dtype=bounded_config.dtype,
            device=torch.device(bounded_config.device),
            num_sinks=bounded_config.attention_sinks,
            batch_size=bounded_config.batch_size,
            capacity=bounded_config.capacity,
        )
        storage.append(kv_factory(4), kv_factory(4))
        assert storage._keys_buf is not None
        assert storage._keys_buf.shape[2] == bounded_config.capacity
        assert storage.num_tokens == 4
        assert storage.keys.shape[2] == 4

    def test_unbounded_storage_doubles_geometrically(self, kv_factory):
        from uniqkache.cache.store import LayerStorage

        storage = LayerStorage(
            layer_idx=0,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
            num_sinks=0,
            batch_size=1,
            capacity=None,
        )
        # Initial append with 10 tokens: max(10, 64) -> 64
        storage.append(kv_factory(10), kv_factory(10))
        assert storage._keys_buf is not None
        assert storage._keys_buf.shape[2] == 64
        assert storage.num_tokens == 10

        # Append 60 more tokens: needed = 70 > 64 -> max(70, 64 * 2) = 128
        storage.append(kv_factory(60), kv_factory(60))
        assert storage._keys_buf.shape[2] == 128
        assert storage.num_tokens == 70
        assert storage.keys.shape[2] == 70

    def test_keep_preserves_buffer_capacity(self, bounded_config, kv_factory):
        from uniqkache.cache.store import LayerStorage

        storage = LayerStorage(
            layer_idx=0,
            num_kv_heads=bounded_config.num_kv_heads,
            head_dim=bounded_config.head_dim,
            dtype=bounded_config.dtype,
            device=torch.device(bounded_config.device),
            num_sinks=bounded_config.attention_sinks,
            batch_size=bounded_config.batch_size,
            capacity=bounded_config.capacity,
        )
        storage.append(kv_factory(bounded_config.capacity), kv_factory(bounded_config.capacity))
        buf_cap_before = storage._keys_buf.shape[2]
        assert buf_cap_before == bounded_config.capacity

        # Evict half of the tokens
        keep_indices = torch.arange(0, bounded_config.capacity, 2)
        storage.keep(keep_indices)

        assert storage.num_tokens == len(keep_indices)
        assert storage._keys_buf.shape[2] == buf_cap_before  # Capacity preserved!
        assert storage.keys.shape[2] == len(keep_indices)

        # Appending new tokens writes in-place into the existing buffer
        storage.append(kv_factory(1), kv_factory(1))
        assert storage._keys_buf.shape[2] == buf_cap_before
        assert storage.num_tokens == len(keep_indices) + 1

    def test_bytes_accounting_reflects_occupied_tokens_not_buffer_capacity(
        self, bounded_config, kv_factory
    ):
        from uniqkache.cache.store import LayerStorage

        storage = LayerStorage(
            layer_idx=0,
            num_kv_heads=bounded_config.num_kv_heads,
            head_dim=bounded_config.head_dim,
            dtype=bounded_config.dtype,
            device=torch.device(bounded_config.device),
            num_sinks=bounded_config.attention_sinks,
            batch_size=bounded_config.batch_size,
            capacity=bounded_config.capacity,
        )
        storage.append(kv_factory(2), kv_factory(2))
        expected_bytes = bounded_config.bytes_per_token_per_layer() * 2
        assert storage.bytes() == expected_bytes

    def test_append_latency_scales_linearly_not_quadratically(self, kv_factory):
        import time

        from uniqkache.cache.store import LayerStorage

        storage = LayerStorage(
            layer_idx=0,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            device=torch.device("cpu"),
            num_sinks=0,
            batch_size=1,
            capacity=1000,
        )
        # Measure early 200 steps vs late 200 steps
        t0 = time.perf_counter()
        for _ in range(200):
            storage.append(kv_factory(1), kv_factory(1))
        early_time = time.perf_counter() - t0

        for _ in range(600):
            storage.append(kv_factory(1), kv_factory(1))

        t1 = time.perf_counter()
        for _ in range(200):
            storage.append(kv_factory(1), kv_factory(1))
        late_time = time.perf_counter() - t1

        # Under O(T^2) cat with T approaching 1000, late_time would be ~4x-5x early_time.
        # With preallocated buffer, late_time remains flat (O(1) per token).
        assert late_time < 3.0 * early_time + 0.05


class TestLayerwiseCapacityList:
    """Pins: a per-layer capacity list must work from construction, not only
    when applied after prefill via an allocation strategy (#48)."""

    def test_list_capacity_cache_accepts_appends(self, kv_factory):
        config = CacheConfig(
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            capacity=[8, 6, 4],
            attention_sinks=0,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy(window=8))
        for layer_idx in range(NUM_LAYERS):
            cache.append(layer_idx, kv_factory(10), kv_factory(10))

        # Each layer must be capped at its own budget, not a shared one.
        for layer_idx, expected in enumerate([8, 6, 4]):
            assert cache.num_tokens(layer_idx) == expected

    def test_store_resolves_per_layer_capacity(self):
        from uniqkache.cache.store import KVStore

        config = CacheConfig(
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            capacity=[8, 6, 4],
        )
        store = KVStore(config)
        assert [store.layer(i).capacity for i in range(NUM_LAYERS)] == [8, 6, 4]
