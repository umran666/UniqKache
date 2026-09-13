"""Unit tests for KV compression.

The invariant under test throughout: compression changes the *representation* of
cached tokens and never the *set* of cached tokens. A compressor that silently
dropped tokens would make "compress" indistinguishable from "evict", which would
invalidate any ablation comparing the two.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import HEAD_DIM, NUM_KV_HEADS
from uniqkache.cache.store import _gather_quantized
from uniqkache.compression.quantize import (
    Int8KVCompressor,
    QuantizedTensor,
    quantize,
)
from uniqkache.utils.errors import CacheStateError, UniqKacheError


def sample(seq: int = 32, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, NUM_KV_HEADS, seq, HEAD_DIM, generator=generator)


# ---------------------------------------------------------------------------
# Quantisation primitives
# ---------------------------------------------------------------------------


class TestQuantize:
    def test_symmetric_round_trip_is_close(self):
        x = sample()
        q = quantize(x, axis=3, num_bits=8, symmetric=True)
        error = (q.dequantize() - x).abs().max().item()
        assert error < 0.05, f"int8 error too large: {error}"

    def test_asymmetric_round_trip_is_close(self):
        x = sample()
        q = quantize(x, axis=3, num_bits=8, symmetric=False)
        error = (q.dequantize() - x).abs().max().item()
        assert error < 0.05, f"int8 error too large: {error}"

    def test_axis_determines_granularity(self):
        """Reducing over the sequence gives per-channel; over head_dim, per-token."""
        x = sample(seq=32)
        per_channel = quantize(x, axis=2)
        per_token = quantize(x, axis=3)
        assert per_channel.scale.shape == (1, NUM_KV_HEADS, 1, HEAD_DIM)
        assert per_token.scale.shape == (1, NUM_KV_HEADS, 32, 1)

    def test_quantised_payload_is_int8(self):
        q = quantize(sample(), axis=3)
        assert q.data.dtype == torch.int8

    def test_zero_tensor_does_not_produce_nan(self):
        """A constant slice has zero range; the scale guard must prevent 0/0."""
        q = quantize(torch.zeros(1, NUM_KV_HEADS, 4, HEAD_DIM), axis=3)
        out = q.dequantize()
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(out))

    def test_non_8_bit_width_is_rejected(self):
        with pytest.raises(UniqKacheError, match="only 8-bit"):
            quantize(sample(), axis=3, num_bits=4)

    def test_out_of_range_axis_is_rejected(self):
        with pytest.raises(UniqKacheError, match="axis"):
            quantize(sample(), axis=9)

    def test_negative_axis_is_accepted(self):
        q = quantize(sample(), axis=-1)
        assert q.axis == 3


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class TestInt8KVCompressor:
    def test_defaults_follow_kivi_granularity(self):
        """Keys per-channel (reduce over seq), values per-token (reduce over head_dim)."""
        compressor = Int8KVCompressor()
        assert compressor.key_axis == 2
        assert compressor.value_axis == 3

    def test_compression_actually_shrinks_bytes(self):
        compressor = Int8KVCompressor()
        keys, values = sample(64), sample(64, seed=1)
        result = compressor.compress(keys, values)
        assert result.compressed_bytes < result.uncompressed_bytes
        assert result.ratio > 2.0, f"expected a large saving, got {result.ratio:.2f}x"

    def test_round_trip_preserves_shape(self):
        compressor = Int8KVCompressor()
        keys, values = sample(), sample(seed=1)
        result = compressor.compress(keys, values)
        out_keys, out_values = compressor.decompress(result)
        assert out_keys.shape == keys.shape
        assert out_values.shape == values.shape

    def test_round_trip_preserves_token_count(self):
        compressor = Int8KVCompressor()
        result = compressor.compress(sample(20), sample(20, seed=1))
        assert result.num_tokens == 20

    def test_ratio_is_uncompressed_over_compressed(self):
        compressor = Int8KVCompressor()
        result = compressor.compress(sample(), sample(seed=1))
        expected = result.uncompressed_bytes / result.compressed_bytes
        assert result.ratio == pytest.approx(expected)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="share a shape"):
            Int8KVCompressor().compress(sample(8), sample(9))

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match="must be"):
            Int8KVCompressor().compress(torch.randn(2, 4), torch.randn(2, 4))

    def test_decompress_rejects_foreign_payload(self):
        from uniqkache.compression.base import CompressionResult

        with pytest.raises(UniqKacheError, match="expected QuantizedTensor"):
            Int8KVCompressor().decompress(
                CompressionResult(
                    keys=torch.zeros(1),
                    values=torch.zeros(1),
                    num_tokens=1,
                    uncompressed_bytes=1,
                    compressed_bytes=1,
                )
            )

    def test_state_dict_records_the_axes(self):
        state = Int8KVCompressor().state_dict()
        assert state["key_axis"] == 2 and state["value_axis"] == 3


# ---------------------------------------------------------------------------
# Gather under eviction — the operation that corrupted dequantisation once
# ---------------------------------------------------------------------------


class TestQuantizedGather:
    @pytest.mark.parametrize("axis", [2, 3])
    def test_gather_preserves_dequantised_values(self, axis: int):
        q = quantize(sample(32), axis=axis)
        indices = torch.tensor([0, 5, 9, 20, 31])
        gathered = _gather_quantized(q, indices).dequantize()
        expected = q.dequantize()[:, :, indices, :]
        assert torch.allclose(gathered, expected, atol=1e-4)

    def test_gather_preserves_dtype_and_shape(self):
        q = quantize(sample(16), axis=3)
        out = _gather_quantized(q, torch.tensor([1, 2, 3]))
        assert out.data.shape[2] == 3
        assert out.data.dtype == torch.int8

    def test_gather_rejects_malformed_rank(self):
        q = quantize(sample(8), axis=3)
        q = QuantizedTensor(
            data=q.data[0],
            scale=q.scale,
            zero_point=q.zero_point,
            axis=q.axis,
            num_bits=q.num_bits,
            symmetric=q.symmetric,
        )
        with pytest.raises(CacheStateError, match="4-D"):
            _gather_quantized(q, torch.tensor([0]))


# ---------------------------------------------------------------------------
# Compression through the cache facade
# ---------------------------------------------------------------------------


class TestCacheCompression:
    def test_compress_reduces_bytes_without_changing_occupancy(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(32), kv_factory(32))
        before = full_cache.stats()

        assert full_cache.compress(0) == 1
        after = full_cache.stats()

        assert after.total_tokens == before.total_tokens, "occupancy must not change"
        assert after.bytes_total < before.bytes_total, "bytes must fall"
        assert after.compression_ratio > 1.0

    def test_get_transparently_dequantises(self, full_cache, kv_factory):
        keys = kv_factory(32, seed=3)
        full_cache.append(0, keys, keys)
        full_cache.compress(0)

        got, _ = full_cache.get(0)
        assert got.dtype == torch.float32
        assert got.shape == keys.shape
        assert (got - keys).abs().max().item() < 0.05

    def test_compress_is_idempotent_per_layer(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(16), kv_factory(16))
        assert full_cache.compress(0) == 1
        assert full_cache.compress(0) == 0, "already compressed, so nothing changes"

    def test_compress_all_layers(self, full_cache, kv_factory):
        for layer in range(3):
            full_cache.append(layer, kv_factory(8), kv_factory(8))
        assert full_cache.compress() == 3

    def test_decompress_restores_float_storage(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(16), kv_factory(16))
        full_cache.compress(0)
        assert full_cache.store.layer(0).is_compressed

        assert full_cache.decompress(0) == 1
        assert not full_cache.store.layer(0).is_compressed

    def test_appending_to_a_compressed_layer_materialises_it(self, full_cache, kv_factory):
        """Documented behaviour: new float rows cannot join an int8 payload."""
        full_cache.append(0, kv_factory(8), kv_factory(8))
        full_cache.compress(0)
        assert full_cache.store.layer(0).is_compressed

        full_cache.append(0, kv_factory(4), kv_factory(4))
        assert not full_cache.store.layer(0).is_compressed
        assert full_cache.num_tokens(0) == 12

    def test_eviction_preserves_compression(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(16), kv_factory(16))
        full_cache.compress(0)
        full_cache.evict(0, keep=torch.tensor([0, 1, 2, 3]))

        layer = full_cache.store.layer(0)
        assert layer.is_compressed, "eviction must not silently drop compression"
        assert layer.num_tokens == 4
        keys, _ = full_cache.get(0)
        assert keys.shape[2] == 4

    def test_eviction_of_compressed_layer_keeps_values_correct(self, full_cache, kv_factory):
        keys = kv_factory(16, seed=5)
        full_cache.append(0, keys, keys)
        full_cache.compress(0)
        keep = torch.tensor([0, 4, 8, 15])
        full_cache.evict(0, keep=keep)

        got, _ = full_cache.get(0)
        expected = keys[:, :, keep, :]
        assert (got - expected).abs().max().item() < 0.05

    def test_wrong_compression_method_is_rejected(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(8), kv_factory(8))
        with pytest.raises(CacheStateError, match="requested compression method"):
            full_cache.compress(0, method="int4")
