"""Unit tests for KV cache checkpoint save and restore functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.conftest import make_kv
from uniqkache.cache.store import CHECKPOINT_SCHEMA_VERSION, KVStore
from uniqkache.cache.types import CacheConfig
from uniqkache.compression.quantize import Int8KVCompressor
from uniqkache.utils.errors import CacheStateError


class TestCheckpointUnit:
    def test_kvstore_save_load_round_trip(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        k0, v0 = make_kv(4, seed=1), make_kv(4, seed=2)
        k1, v1 = make_kv(6, seed=3), make_kv(6, seed=4)
        store.layer(0).append(k0, v0)
        store.layer(1).append(k1, v1)

        # Note attention and access
        store.layer(0).metadata.note_access(torch.tensor([1, 2]), step=1)
        store.layer(0).metadata.note_attention(torch.tensor([0.1, 0.5, 0.2, 0.0]), step=1)

        ckpt_path = tmp_path / "store.pt"
        saved = store.save(ckpt_path, model="synthetic:tiny")
        assert saved == ckpt_path
        assert ckpt_path.is_file()

        restored = KVStore.load(ckpt_path, cache_config)
        assert len(restored) == len(store)
        assert restored.num_tokens(0) == 4
        assert restored.num_tokens(1) == 6
        assert restored.num_tokens(2) == 0

        # Bit-identical tensors
        assert torch.equal(restored.layer(0).keys, k0)
        assert torch.equal(restored.layer(0).values, v0)
        assert torch.equal(restored.layer(1).keys, k1)
        assert torch.equal(restored.layer(1).values, v1)

        # Metadata signals preserved
        meta0_orig = store.layer(0).metadata
        meta0_rest = restored.layer(0).metadata
        assert torch.equal(meta0_rest.positions, meta0_orig.positions)
        assert torch.equal(meta0_rest.last_access, meta0_orig.last_access)
        assert torch.equal(meta0_rest.cum_attention, meta0_orig.cum_attention)
        assert torch.equal(meta0_rest.hit_count, meta0_orig.hit_count)
        assert meta0_rest.num_sinks == meta0_orig.num_sinks

    def test_quantized_layer_saved_without_dequantizing(
        self, cache_config: CacheConfig, tmp_path: Path
    ):
        store = KVStore(cache_config)
        k, v = make_kv(8, seed=10), make_kv(8, seed=20)
        store.layer(0).append(k, v)

        # Compress layer 0
        compressor = Int8KVCompressor()
        res = compressor.compress(k, v)
        store.layer(0).apply_compression(res)

        assert store.layer(0).is_compressed
        assert store.layer(0)._keys is None

        ckpt_path = tmp_path / "quant.pt"
        store.save(ckpt_path, model="synthetic:tiny")

        # Inspect raw checkpoint contents
        raw = torch.load(ckpt_path, weights_only=True)
        l0_raw = raw["layers"][0]
        assert l0_raw["is_compressed"] is True
        assert "compressed" in l0_raw
        assert "keys" not in l0_raw
        assert "data" in l0_raw["compressed"]["keys"]
        assert "scale" in l0_raw["compressed"]["keys"]
        assert "zero_point" in l0_raw["compressed"]["keys"]

        # Restore
        restored = KVStore.load(ckpt_path, cache_config)
        assert restored.layer(0).is_compressed is True
        assert restored.layer(0)._keys is None  # Remains compressed without dequantizing!
        assert restored.compressed_layers() == [0]

        # Dequantising on-demand yields bit-identical output to original compressed layer
        orig_k, orig_v = store.layer(0).keys, store.layer(0).values
        rest_k, rest_v = restored.layer(0).keys, restored.layer(0).values
        assert torch.equal(rest_k, orig_k)
        assert torch.equal(rest_v, orig_v)

    def test_offloaded_layer_preserves_device_placement(
        self, cache_config: CacheConfig, tmp_path: Path
    ):
        store = KVStore(cache_config)
        k, v = make_kv(4, seed=5), make_kv(4, seed=6)
        store.layer(0).append(k, v)
        store.layer(1).append(k, v)

        store.move_to(torch.device("cpu"), 0)
        # Note: on CPU tests, move_to CPU doesn't mark _offloaded if home is CPU,
        # so let's explicitly verify with _offloaded flag
        store.layer(0)._offloaded = True

        ckpt_path = tmp_path / "offload.pt"
        store.save(ckpt_path)

        restored = KVStore.load(ckpt_path, cache_config)
        assert restored.layer(0).is_offloaded is True
        assert restored.layer(1).is_offloaded is False
        assert restored.offloaded_layers() == [0]

    def test_validate_num_layers_mismatch(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        ckpt_path = tmp_path / "store.pt"
        store.save(ckpt_path)

        mismatched_config = CacheConfig(
            num_layers=cache_config.num_layers + 1,
            num_kv_heads=cache_config.num_kv_heads,
            head_dim=cache_config.head_dim,
            dtype=cache_config.dtype,
            device=cache_config.device,
        )
        with pytest.raises(CacheStateError, match="layers"):
            KVStore.load(ckpt_path, mismatched_config)

    def test_validate_head_dim_mismatch(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        ckpt_path = tmp_path / "store.pt"
        store.save(ckpt_path)

        mismatched_config = CacheConfig(
            num_layers=cache_config.num_layers,
            num_kv_heads=cache_config.num_kv_heads,
            head_dim=cache_config.head_dim + 4,
            dtype=cache_config.dtype,
            device=cache_config.device,
        )
        with pytest.raises(CacheStateError, match="head_dim"):
            KVStore.load(ckpt_path, mismatched_config)

    def test_validate_num_kv_heads_mismatch(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        ckpt_path = tmp_path / "store.pt"
        store.save(ckpt_path)

        mismatched_config = CacheConfig(
            num_layers=cache_config.num_layers,
            num_kv_heads=cache_config.num_kv_heads + 1,
            head_dim=cache_config.head_dim,
            dtype=cache_config.dtype,
            device=cache_config.device,
        )
        with pytest.raises(CacheStateError, match="num_kv_heads"):
            KVStore.load(ckpt_path, mismatched_config)

    def test_validate_dtype_mismatch(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        store.layer(0).append(make_kv(4), make_kv(4))
        ckpt_path = tmp_path / "store.pt"
        store.save(ckpt_path)

        mismatched_config = CacheConfig(
            num_layers=cache_config.num_layers,
            num_kv_heads=cache_config.num_kv_heads,
            head_dim=cache_config.head_dim,
            dtype=torch.float64,
            device=cache_config.device,
        )
        with pytest.raises(CacheStateError, match="dtype"):
            KVStore.load(ckpt_path, mismatched_config)

    def test_provenance_recorded(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        ckpt_path = tmp_path / "prov.pt"
        store.save(ckpt_path, model="synthetic:tiny")

        raw = torch.load(ckpt_path, weights_only=True)
        assert raw["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert raw["model"] == "synthetic:tiny"
        assert "created_at" in raw
        assert "config" in raw
        assert raw["config"]["num_layers"] == cache_config.num_layers

    def test_empty_layer_round_trip(self, cache_config: CacheConfig, tmp_path: Path):
        store = KVStore(cache_config)
        # All layers empty
        ckpt_path = tmp_path / "empty.pt"
        store.save(ckpt_path)

        restored = KVStore.load(ckpt_path, cache_config)
        assert all(not layer.is_initialized for layer in restored.layers)
        assert restored.max_tokens() == 0
