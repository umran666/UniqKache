"""Regression: token-level (intra-layer) offload granularity.

Issue #30. ``KVCache.offload()`` used to move an entire layer at once — the
coarsest possible granularity, under which a layer is all-resident or
all-offloaded, so memory/latency trade-offs are lumpy and prefetch can only
fetch whole layers. Real serving systems offload the *cold tail* of a layer
while keeping the hot head resident.

These tests pin three things:

1. **Correctness.** A layer that is 50% resident produces attention output
   identical to the same layer fully resident — the acceptance criterion.
   ``torch.equal``, not ``allclose``: the gather is a pure copy, so any
   difference is a bug, not floating-point noise.
2. **Exact byte accounting.** ``bytes_on_device`` and ``bytes_offloaded``
   report the split exactly, and ``bytes_total`` is invariant — offloading
   relocates bytes, it does not delete them.
3. **Defined interactions.** Append, eviction and compression collapse the
   split, a single-tier offload is a no-op that reports zero, and the split
   survives a checkpoint round trip.

On the single-tier hardware the CPU tests run on, ``offload_tail`` correctly
refuses (host == home device has no cheaper tier), so the device-independent
logic — gather order, collapse rules, checkpoint restoration — is exercised
through :func:`_force_split`, which builds the same two-buffer structure a
two-tier checkpoint would have recorded. The *physical* move and its exact
byte accounting are pinned by the CUDA-gated tests at the bottom, which skip
on CI by design (the suite must never require hardware).
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.policies import SlidingWindowPolicy
from uniqkache.utils.device import cuda_is_available
from uniqkache.utils.errors import CacheStateError

requires_cuda = pytest.mark.skipif(not cuda_is_available(), reason="requires a CUDA device")

VOCAB = 64
CTX = 24


def _bounded_cache(capacity: int = 16) -> KVCache:
    """A bounded cache whose buffers are allocated to exactly ``capacity``.

    Bounded (not unbounded) on purpose: an unbounded layer's buffer is
    preallocated past its occupancy, so its ``bytes()`` count padding that
    would make the total-invariance assertion compare against storage the
    cache never used. Bounded keeps the arithmetic exact to the byte.
    """
    config = CacheConfig(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        capacity=capacity,
    )
    return KVCache(config, policy=SlidingWindowPolicy())


def _append(cache: KVCache, tokens: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, 2, tokens, 8, generator=generator)
    values = torch.randn(1, 2, tokens, 8, generator=generator)
    for layer in range(cache.num_layers):
        cache.append(layer, keys, values, positions=torch.arange(tokens))
    return keys, values


def _force_split(layer, boundary: int) -> None:
    """Build the two-buffer structure ``offload_tail`` produces on a real
    second tier, without needing one.

    On single-tier hardware ``offload_tail`` refuses (correctly — see
    ``test_single_tier_offload_is_a_no_op``). The gather, collapse and
    checkpoint logic under test is device-independent, so this helper
    constructs exactly the state a two-tier checkpoint would have recorded:
    the resident prefix and the hosted tail as separate buffers, with the
    boundary recorded. Accounting stays honest — the hosted buffers are on
    the home device here, so ``offloaded_bytes()`` reports zero.
    """
    assert layer._keys_buf is not None and layer._values_buf is not None
    keys = layer.keys
    values = layer.values
    assert keys is not None and values is not None
    layer._keys_buf = keys[:, :, :boundary, :].clone()
    layer._values_buf = values[:, :, :boundary, :].clone()
    layer._host_keys_buf = keys[:, :, boundary:, :].clone()
    layer._host_values_buf = values[:, :, boundary:, :].clone()
    layer._resident_boundary = boundary


class TestSingleTierBehaviour:
    def test_offload_without_a_second_tier_is_a_no_op(self):
        """CPU has no cheaper tier, so nothing may claim to have moved."""
        cache = _bounded_cache()
        _append(cache)
        before = cache.stats()

        assert cache.offload(keep_resident=8) == 0
        assert cache.offload(keep_resident=0) == 0
        after = cache.stats()

        layer = cache.store.layer(0)
        assert not layer.is_split
        assert not layer.is_offloaded
        assert after.bytes_on_device == before.bytes_on_device
        assert after.bytes_offloaded == 0
        assert after.bytes_total == before.bytes_total

    def test_out_of_range_boundary_is_rejected(self):
        cache = _bounded_cache()
        _append(cache)
        with pytest.raises(CacheStateError, match="keep_resident must be >= 0"):
            cache.offload(keep_resident=-1)
        with pytest.raises(CacheStateError, match="out of range"):
            cache.offload(keep_resident=17)

    def test_negative_prefetch_count_is_rejected(self):
        cache = _bounded_cache()
        _append(cache)
        with pytest.raises(CacheStateError, match="tokens must be >= 0"):
            cache.prefetch(tokens=-1)


class TestSplitResidencyStructure:
    def test_split_reports_residency_per_tier(self):
        cache = _bounded_cache()
        _append(cache)
        before = cache.stats()

        _force_split(cache.store.layer(0), 8)
        _force_split(cache.store.layer(1), 8)

        layer = cache.store.layer(0)
        assert layer.is_split
        assert not layer.is_offloaded
        assert layer.resident_tokens == 8
        assert layer.offloaded_tokens == 8
        # Occupancy is unchanged by a tier move.
        assert layer.num_tokens == 16

        stats = cache.stats()
        # Total is invariant: offloading relocates bytes, it does not delete.
        assert stats.bytes_total == before.bytes_total
        # Honest on single-tier hardware: the hosted buffers sit on the home
        # device, so no bytes have left the compute tier.
        assert stats.bytes_offloaded == 0
        assert stats.bytes_on_device == stats.bytes_total

    def test_gather_returns_the_full_token_set_in_slot_order(self):
        cache = _bounded_cache()
        _append(cache)
        full_keys, full_values = cache.get(0)

        _force_split(cache.store.layer(0), 8)
        gathered_keys, gathered_values = cache.get(0)
        assert torch.equal(gathered_keys, full_keys)
        assert torch.equal(gathered_values, full_values)

    def test_partial_prefetch_preserves_slot_order(self):
        cache = _bounded_cache()
        _append(cache)
        full_keys, full_values = cache.get(0)
        layer = cache.store.layer(0)
        _force_split(layer, 8)

        assert cache.prefetch(0, tokens=4) == 4
        assert layer.resident_tokens == 12
        assert layer.offloaded_tokens == 4

        # The frontmost hosted token is the next one the resident prefix
        # expects, so a partial prefetch keeps slots in order.
        gathered_keys, gathered_values = cache.get(0)
        assert torch.equal(gathered_keys, full_keys)
        assert torch.equal(gathered_values, full_values)

    def test_whole_layer_prefetch_returns_a_split_layer(self):
        cache = _bounded_cache()
        _append(cache)
        for layer_idx in range(cache.num_layers):
            _force_split(cache.store.layer(layer_idx), 8)

        assert cache.prefetch() == 2  # both layers, whole-layer spelling
        for layer_idx in range(cache.num_layers):
            assert not cache.store.layer(layer_idx).is_split
        assert cache.stats().bytes_on_device == cache.stats().bytes_total


class TestSplitResidencyInteractions:
    def test_append_collapses_the_split(self):
        cache = _bounded_cache()
        keys, values = _append(cache)
        layer = cache.store.layer(0)
        _force_split(layer, 8)
        assert layer.is_split

        # New rows are the hottest tokens in the layer; they must join the
        # resident prefix in slot order, so the split collapses first.
        cache.append(0, keys[:, :, :2, :], values[:, :, :2, :], positions=torch.arange(16, 18))
        assert not layer.is_split
        assert layer.num_tokens == 16  # capacity enforced after the append
        assert layer.resident_tokens == 16

    def test_eviction_collapses_the_split(self):
        cache = _bounded_cache(capacity=8)
        _append(cache, tokens=16)
        layer = cache.store.layer(0)
        assert layer.num_tokens == 8

        _force_split(layer, 4)
        assert layer.is_split
        dropped = cache.evict(0, indices=torch.tensor([0]))
        assert dropped == 1
        assert not layer.is_split
        assert layer.resident_tokens == layer.num_tokens == 7

    def test_compression_of_a_split_layer_drops_the_split(self):
        cache = _bounded_cache()
        _append(cache)
        for layer_idx in range(cache.num_layers):
            _force_split(cache.store.layer(layer_idx), 8)

        changed = cache.compress()
        assert changed == 2
        layer = cache.store.layer(0)
        assert layer.is_compressed
        assert not layer.is_split
        # The compressed payload describes the whole gathered layer.
        assert layer.device_bytes() == layer.bytes()
        assert layer.offloaded_bytes() == 0

    def test_splitting_a_compressed_layer_is_refused(self):
        cache = _bounded_cache()
        _append(cache)
        cache.compress()

        with pytest.raises(CacheStateError, match="decompress before"):
            cache.store.layer(0).offload_tail(4, host=torch.device("cpu"))


class TestSplitResidencyCheckpoint:
    def test_round_trip_preserves_the_split(self, tmp_path):
        cache = _bounded_cache()
        _append(cache)
        full_keys, full_values = cache.get(0)
        for layer_idx in range(cache.num_layers):
            _force_split(cache.store.layer(layer_idx), 8)

        path = cache.save(tmp_path / "split.pt", model="synthetic:tiny")
        restored = KVCache.load(path, policy=SlidingWindowPolicy())

        layer = restored.store.layer(0)
        assert layer.is_split
        assert layer.resident_tokens == 8
        assert layer.offloaded_tokens == 8
        assert layer.num_tokens == 16

        restored_keys, restored_values = restored.get(0)
        assert torch.equal(restored_keys, full_keys)
        assert torch.equal(restored_values, full_values)

    def test_half_hosted_checkpoint_without_host_rows_is_rejected(self, tmp_path):
        cache = _bounded_cache()
        _append(cache)
        state = cache.store.layer(0).state_dict()
        state["host_keys"] = state["keys"][:, :, 8:, :].clone()

        with pytest.raises(CacheStateError, match="host_keys without host_values"):
            cache.store.layer(0).load_state_dict(state, cache.config)


class TestSplitResidencyAttention:
    def test_half_resident_layer_matches_full_resident_logits(self, tiny_model):
        """The acceptance criterion: 50% resident, identical attention output.

        Two caches see the same prompt and the same decode step. One stays
        fully resident; the other is split 50% per layer before the decode
        step, so attention must gather the hosted tail on every read. The
        gather is a pure copy, so the logits are compared bit-for-bit — an
        ``allclose`` here would hide a reordering bug as "float noise".
        """
        torch.manual_seed(7)
        prompt = torch.randint(0, VOCAB, (1, CTX))

        def _cache() -> KVCache:
            config = tiny_model.config.cache_config(
                capacity=None, attention_sinks=0, dtype=torch.float32, device="cpu"
            )
            return KVCache(config, policy=None)

        resident_cache = _cache()
        split_cache = _cache()

        with torch.no_grad():
            logits_resident, _ = tiny_model.forward(prompt, cache=resident_cache, start_pos=0)
            logits_split, _ = tiny_model.forward(prompt, cache=split_cache, start_pos=0)
            assert torch.equal(logits_resident, logits_split)

            for layer_idx in range(split_cache.num_layers):
                num = split_cache.num_tokens(layer_idx)
                assert num > 1
                _force_split(split_cache.store.layer(layer_idx), num // 2)
                assert split_cache.store.layer(layer_idx).is_split

            next_token = logits_resident[:, -1, :].argmax(dim=-1, keepdim=True)
            decode_resident, _ = tiny_model.forward(next_token, cache=resident_cache, start_pos=CTX)
            decode_split, _ = tiny_model.forward(next_token, cache=split_cache, start_pos=CTX)

        assert torch.equal(decode_resident, decode_split)


class TestSplitResidencyOnRealTiers:
    """The physical move and its exact byte accounting. CUDA-gated."""

    @requires_cuda
    def test_split_offloads_exactly_the_hosted_share(self):
        config = CacheConfig(
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cuda",
            capacity=16,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy())
        _append(cache)
        before = cache.stats()
        assert before.bytes_on_device == before.bytes_total

        moved = cache.offload(keep_resident=8)
        # Two layers, 8 tokens each moved to the host tier.
        assert moved == 16
        after = cache.stats()

        assert after.bytes_on_device == before.bytes_total // 2
        assert after.bytes_offloaded == before.bytes_total // 2
        assert after.bytes_total == before.bytes_total
        for layer_idx in range(cache.num_layers):
            layer = cache.store.layer(layer_idx)
            assert layer.is_split
            assert layer.resident_tokens == 8
            assert layer.offloaded_tokens == 8

        # The gather brings hosted rows home, so attention is unchanged.
        restored = KVCache(
            CacheConfig(
                num_layers=2,
                num_kv_heads=2,
                head_dim=8,
                dtype=torch.float32,
                device="cpu",
                capacity=16,
            ),
            policy=SlidingWindowPolicy(),
        )
        _append(restored)
        full_keys, full_values = restored.get(0)
        gathered_keys, gathered_values = cache.get(0)
        assert torch.equal(gathered_keys.cpu(), full_keys)
        assert torch.equal(gathered_values.cpu(), full_values)
