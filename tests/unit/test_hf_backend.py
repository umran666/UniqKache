"""Tests for the Hugging Face backend's pure-logic paths and eviction support.

The HF backend supports both full-cache and bounded evicting cache policies
via UniqKacheHFCache and UniqKacheLayer.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.models.hf_backend import (
    HFBackend,
    UniqKacheHFCache,
    UniqKacheLayer,
    _extract_layer_kv,
)
from uniqkache.policies import SlidingWindowPolicy
from uniqkache.utils.device import cuda_is_available
from uniqkache.utils.errors import BackendError

requires_cuda = pytest.mark.skipif(not cuda_is_available(), reason="requires a CUDA device")

try:
    import transformers
    from transformers import AutoModelForCausalLM

    _has_tiny_llama = True
    try:
        AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM", local_files_only=True
        )
    except Exception:
        _has_tiny_llama = False
except ImportError:
    transformers = None  # type: ignore[assignment]
    _has_tiny_llama = False

requires_tiny_llama = pytest.mark.skipif(
    not _has_tiny_llama, reason="requires hf-internal-testing/tiny-random-LlamaForCausalLM cached"
)


class _StubConfig:
    """The attributes HFBackend reads off a transformers config object."""

    model_type = "stub"
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 32
    vocab_size = 128


class _StubModel(torch.nn.Module):
    """Smallest object that satisfies what HFBackend's constructor touches."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _StubConfig()
        self.weight = torch.nn.Parameter(torch.zeros(8, 32))


@pytest.fixture
def backend() -> HFBackend:
    return HFBackend(_StubModel(), identifier="stub/model", revision="v0")


class TestConstruction:
    @pytest.mark.parametrize("head_dim, expected", [(16, 16), (None, 8)])
    def test_explicit_head_dim_and_none_fallback(self, head_dim, expected):
        class _HeadDimConfig(_StubConfig):
            pass

        model = _StubModel()
        model.config = _HeadDimConfig()
        model.config.head_dim = head_dim
        backend = HFBackend(model, identifier="stub/head-dim")
        assert backend.config["head_dim"] == expected
        config = backend.cache_config()
        assert config.head_dim == expected
        assert config.bytes_per_token() == 2 * 2 * expected * 4 * 2

    def test_shape_metadata_is_derived_from_the_config(self, backend):
        assert backend.num_layers == 2
        assert backend.vocab_size == 128
        assert backend.config["num_kv_heads"] == 2
        assert backend.config["head_dim"] == 8  # hidden_size / num_attention_heads

    def test_identifier_and_revision_are_kept(self, backend):
        assert backend.identifier == "stub/model"
        assert backend.revision == "v0"
        assert backend.weights_are_random is False

    def test_a_config_without_layers_is_rejected(self):
        class _NoLayers:
            num_attention_heads = 1
            hidden_size = 8

        class _BadModel(_StubModel):
            def __init__(self) -> None:
                super().__init__()
                self.config = _NoLayers()

        with pytest.raises(BackendError, match="could not determine layer/head counts"):
            HFBackend(_BadModel(), identifier="stub/bad")

    def test_cache_config_matches_the_model_shape(self, backend):
        config = backend.cache_config(capacity=16, attention_sinks=2)
        assert isinstance(config, CacheConfig)
        assert config.num_layers == 2
        assert config.num_kv_heads == 2
        assert config.head_dim == 8
        assert config.capacity == 16


class TestMirrorOverheadCounter:
    def test_nothing_mirrored_reports_zero(self, backend):
        assert backend.mirrored_bytes == 0

    def test_mirror_counts_the_copied_bytes(self, backend):
        """A stub HF-style cache lets the mirror path run without transformers.

        Only the mirror and the counter are exercised: ``_hf_cache`` is faked
        with objects exposing ``layers[i].keys/values``, so ``transformers``
        is never imported.
        """

        class _Layer:
            def __init__(self, keys, values):
                self.keys = keys
                self.values = values

        k = torch.zeros(1, 2, 4, 8)
        v = torch.ones(1, 2, 4, 8)
        backend._hf_cache = type("Cache", (), {"layers": [_Layer(k, v), _Layer(k, v)]})()
        backend._num_layers = 2

        unbounded = KVCache(backend.cache_config(capacity=None), policy=None)
        backend._mirror_into_cache(unbounded, start_pos=0, seq=4)

        per_layer = k.numel() * k.element_size() + v.numel() * v.element_size()
        assert backend.mirrored_bytes == 2 * per_layer
        assert unbounded.num_tokens(0) == 4

    def test_mirror_with_dynamic_cache_440_style(self, backend):
        """Transformers 4.40 DynamicCache uses key_cache and value_cache lists."""
        k = torch.zeros(1, 2, 4, 8)
        v = torch.ones(1, 2, 4, 8)
        backend._hf_cache = type(
            "DynamicCache440", (), {"key_cache": [k, k], "value_cache": [v, v]}
        )()
        backend._num_layers = 2

        unbounded = KVCache(backend.cache_config(capacity=None), policy=None)
        backend._mirror_into_cache(unbounded, start_pos=0, seq=4)

        per_layer = k.numel() * k.element_size() + v.numel() * v.element_size()
        assert backend.mirrored_bytes == 2 * per_layer
        assert unbounded.num_tokens(0) == 4

    def test_mirror_with_legacy_tuple_cache(self, backend):
        """Legacy HF past_key_values uses a tuple of (key, value) pairs."""
        k = torch.zeros(1, 2, 4, 8)
        v = torch.ones(1, 2, 4, 8)
        backend._hf_cache = ((k, v), (k, v))
        backend._num_layers = 2

        unbounded = KVCache(backend.cache_config(capacity=None), policy=None)
        backend._mirror_into_cache(unbounded, start_pos=0, seq=4)

        per_layer = k.numel() * k.element_size() + v.numel() * v.element_size()
        assert backend.mirrored_bytes == 2 * per_layer
        assert unbounded.num_tokens(0) == 4

    def test_mirror_unsupported_cache_layout_raises_backend_error(self, backend):
        backend._hf_cache = type("UnknownCache", (), {})()
        backend._num_layers = 2
        unbounded = KVCache(backend.cache_config(capacity=None), policy=None)
        with pytest.raises(BackendError, match="unsupported cache layout"):
            backend._mirror_into_cache(unbounded, start_pos=0, seq=4)

    def test_reset_does_not_reset_the_lifetime_counter(self, backend):
        backend._mirrored_bytes = 128
        backend.reset()
        assert backend.mirrored_bytes == 128


class TestExtractLayerKV:
    def test_none_cache_returns_none_pair(self):
        k, v = _extract_layer_kv(None, 0)
        assert k is None and v is None

    def test_layers_attribute_with_objects(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache = type(
            "ModernCache", (), {"layers": [type("Layer", (), {"keys": k, "values": v})()]}
        )()
        ret_k, ret_v = _extract_layer_kv(cache, 0)
        assert ret_k is k and ret_v is v
        # Out of bounds
        out_k, out_v = _extract_layer_kv(cache, 1)
        assert out_k is None and out_v is None

    def test_layers_attribute_with_tuples(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache = type("ModernCache", (), {"layers": [(k, v)]})()
        ret_k, ret_v = _extract_layer_kv(cache, 0)
        assert ret_k is k and ret_v is v

    def test_key_value_cache_lists(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache = type("Cache440", (), {"key_cache": [k], "value_cache": [v]})()
        ret_k, ret_v = _extract_layer_kv(cache, 0)
        assert ret_k is k and ret_v is v
        # Out of bounds
        out_k, out_v = _extract_layer_kv(cache, 1)
        assert out_k is None and out_v is None

    def test_tuple_of_tuples(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache = ((k, v),)
        ret_k, ret_v = _extract_layer_kv(cache, 0)
        assert ret_k is k and ret_v is v
        # Out of bounds
        out_k, out_v = _extract_layer_kv(cache, 1)
        assert out_k is None and out_v is None

    def test_subscriptable_cache_object(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)

        class CustomSubscriptable:
            def __getitem__(self, idx):
                if idx == 0:
                    return (k, v)
                raise IndexError(idx)

        ret_k, ret_v = _extract_layer_kv(CustomSubscriptable(), 0)
        assert ret_k is k and ret_v is v
        out_k, out_v = _extract_layer_kv(CustomSubscriptable(), 1)
        assert out_k is None and out_v is None

    def test_corrupt_layer_returns_none(self):
        cache = type("BadCache", (), {"layers": ["not_a_layer"]})()
        ret_k, ret_v = _extract_layer_kv(cache, 0)
        assert ret_k is None and ret_v is None


class TestUniqKacheLayer:
    def test_layer_properties_and_delegation(self):
        config = CacheConfig(
            num_layers=2, num_kv_heads=2, head_dim=8, capacity=8, attention_sinks=2
        )
        kv_cache = KVCache(config, policy=SlidingWindowPolicy())
        layer = UniqKacheLayer(kv_cache, layer_idx=0)

        assert layer.is_initialized is False
        assert layer.keys is None
        assert layer.values is None
        assert layer.get_seq_length() == 0
        assert layer.get_max_length() == 8

        # Test mask sizes
        assert layer.get_mask_sizes(query_length=4) == (4, 0)
        assert layer.get_mask_sizes(query_length=12) == (8, 0)

        # Test update
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        ret_k, ret_v = layer.update(k, v)
        assert layer.is_initialized is True
        assert layer.get_seq_length() == 4
        assert ret_k.shape == (1, 2, 4, 8)
        assert ret_v.shape == (1, 2, 4, 8)

        # Append more to trigger eviction
        k2 = torch.randn(1, 2, 8, 8)
        v2 = torch.randn(1, 2, 8, 8)
        ret_k2, ret_v2 = layer.update(k2, v2)
        assert layer.get_seq_length() == 8
        assert ret_k2.shape == (1, 2, 8, 8)
        assert ret_v2.shape == (1, 2, 8, 8)

        # Test crop
        layer.crop(-2)
        assert layer.get_seq_length() == 6

        # Test reset
        layer.reset()
        assert layer.get_seq_length() == 0
        assert layer.keys is None

    def test_per_layer_capacity_list(self):
        config = CacheConfig(num_layers=2, num_kv_heads=2, head_dim=8, capacity=[4, 8])
        kv_cache = KVCache(config, policy=SlidingWindowPolicy())
        layer0 = UniqKacheLayer(kv_cache, layer_idx=0)
        layer1 = UniqKacheLayer(kv_cache, layer_idx=1)

        assert layer0.get_max_length() == 4
        assert layer1.get_max_length() == 8
        assert isinstance(layer0.get_max_length(), int)
        assert isinstance(layer1.get_max_length(), int)

        assert layer0.get_mask_sizes(query_length=10) == (4, 0)
        assert layer1.get_mask_sizes(query_length=10) == (8, 0)

    def test_dynamic_layer_initialization_and_setter(self):
        config = CacheConfig(num_layers=1, num_kv_heads=2, head_dim=8, capacity=4)
        kv_cache = KVCache(config, policy=SlidingWindowPolicy())
        layer = UniqKacheLayer(kv_cache, layer_idx=0)
        layer.is_initialized = False
        assert layer.is_initialized is False


class TestUniqKacheHFCache:
    @pytest.mark.parametrize("wrapper", [UniqKacheHFCache, UniqKacheLayer])
    def test_wrapper_borrows_cache_and_rejects_access_after_release(self, wrapper):
        config = CacheConfig(num_layers=1, num_kv_heads=2, head_dim=8)
        cache = KVCache(config)
        adapter = wrapper(cache) if wrapper is UniqKacheHFCache else wrapper(cache, layer_idx=0)
        assert adapter.kv_cache is cache
        cache_ref = weakref.ref(cache)
        del cache
        gc.collect()

        assert cache_ref() is None
        with pytest.raises(ReferenceError, match=r"KVCache backing this HF .* has been released"):
            adapter.get_seq_length()

    def test_cache_wrapping_and_layer_access(self):
        config = CacheConfig(num_layers=3, num_kv_heads=2, head_dim=8, capacity=10)
        kv_cache = KVCache(config, policy=SlidingWindowPolicy())
        hf_cache = UniqKacheHFCache(kv_cache)

        assert len(hf_cache) == 3
        assert isinstance(hf_cache[0], UniqKacheLayer)
        assert hf_cache.get_max_length() == 10
        assert hf_cache.get_seq_length(0) == 0
        assert hf_cache.get_mask_sizes(5, 0) == (5, 0)

        # Populate a layer
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        hf_cache[0].update(k, v)
        assert hf_cache.get_seq_length(0) == 4
        assert hf_cache.get_seq_length(1) == 0

        # Reset
        hf_cache.reset()
        assert hf_cache.get_seq_length(0) == 0

    def test_per_layer_capacity_list_hf_cache(self):
        config = CacheConfig(num_layers=2, num_kv_heads=2, head_dim=8, capacity=[4, 8])
        kv_cache = KVCache(config, policy=SlidingWindowPolicy())
        hf_cache = UniqKacheHFCache(kv_cache)

        assert hf_cache.get_max_length() == 8
        assert hf_cache.get_max_length(0) == 4
        assert hf_cache.get_max_length(1) == 8
        assert hf_cache.get_mask_sizes(10, 0) == (4, 0)
        assert hf_cache.get_mask_sizes(10, 1) == (8, 0)


class TestForwardWiring:
    def test_forward_wires_uniqkache_cache_to_hf_model(self):
        class _CallableStub(_StubModel):
            def __init__(self) -> None:
                super().__init__()
                self.recorded_cache = None

            def forward(self, input_ids, past_key_values=None, **kwargs):
                self.recorded_cache = past_key_values
                return type(
                    "Output",
                    (),
                    {
                        "logits": torch.zeros(1, input_ids.shape[1], 128),
                        "past_key_values": past_key_values,
                    },
                )()

        stub = _CallableStub()
        backend = HFBackend(stub, identifier="stub/test")
        cache = KVCache(backend.cache_config(capacity=8), policy=SlidingWindowPolicy())
        backend.forward(torch.randint(0, 128, (1, 4)), cache=cache)

        assert isinstance(stub.recorded_cache, UniqKacheHFCache)
        assert stub.recorded_cache.kv_cache is cache

    def test_adapter_registry_caching(self):
        class _CallableStub(_StubModel):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input_ids, past_key_values=None, **kwargs):
                return type(
                    "Output",
                    (),
                    {
                        "logits": torch.zeros(1, input_ids.shape[1], 128),
                        "past_key_values": past_key_values,
                    },
                )()

        from uniqkache.models.hf_backend import _CACHE_ADAPTERS

        stub = _CallableStub()
        backend = HFBackend(stub, identifier="stub/test")
        cache = KVCache(backend.cache_config(capacity=8), policy=SlidingWindowPolicy())

        assert cache not in _CACHE_ADAPTERS
        backend.forward(torch.randint(0, 128, (1, 4)), cache=cache)
        assert cache in _CACHE_ADAPTERS
        adapter1 = _CACHE_ADAPTERS[cache]

        backend.forward(torch.randint(0, 128, (1, 4)), cache=cache)
        adapter2 = _CACHE_ADAPTERS[cache]
        assert adapter1 is adapter2


class TestEvictionExecution:
    @requires_tiny_llama
    def test_bounded_hf_run_end_to_end_cpu(self):
        from uniqkache.runtime.generation import GenerationConfig, GenerationEngine

        backend = HFBackend.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            device="cpu",
            local_files_only=True,
        )
        capacity = 8
        cache = KVCache(
            backend.cache_config(capacity=capacity, attention_sinks=2, device="cpu"),
            policy=SlidingWindowPolicy(),
        )
        engine = GenerationEngine(backend, cache, GenerationConfig(max_new_tokens=6))
        prompt = torch.randint(0, backend.vocab_size, (1, 16))
        result = engine.generate(prompt)

        assert result.generated_tokens == 6
        assert result.cache_stats["max_tokens_in_layer"] <= capacity
        for layer_idx in range(backend.num_layers):
            assert cache.num_tokens(layer_idx) <= capacity

    @pytest.mark.gpu
    @requires_cuda
    @requires_tiny_llama
    def test_bounded_hf_run_on_gpu(self):
        from uniqkache.runtime.generation import GenerationConfig, GenerationEngine

        backend = HFBackend.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            device="cuda",
            local_files_only=True,
        )
        capacity = 8
        cache = KVCache(
            backend.cache_config(capacity=capacity, attention_sinks=2, device="cuda"),
            policy=SlidingWindowPolicy(),
        )
        engine = GenerationEngine(backend, cache, GenerationConfig(max_new_tokens=5))
        prompt = torch.randint(0, backend.vocab_size, (1, 16), device="cuda")
        result = engine.generate(prompt)

        assert result.generated_tokens == 5
        assert result.cache_stats["max_tokens_in_layer"] <= capacity
        for layer_idx in range(backend.num_layers):
            assert cache.num_tokens(layer_idx) <= capacity


class TestOfflineResolution:
    def test_build_hf_model_forwards_spec_offline_to_local_files_only(self, monkeypatch):
        from uniqkache.bench.config import RunSpec
        from uniqkache.models.hf_backend import build_hf_model

        spec = RunSpec(model="org/test-model", offline=True, policy="full_cache")
        captured = {}

        def mock_from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return HFBackend(_StubModel(), identifier=model_id)

        monkeypatch.setattr(HFBackend, "from_pretrained", mock_from_pretrained)

        built = build_hf_model(spec, dtype=torch.float32, device="cpu")
        assert captured["local_files_only"] is True
        assert built.is_hf_backend is True

    def test_from_pretrained_passes_local_files_only_to_transformers(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_transformers = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = _StubModel()

        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        monkeypatch.setattr(
            "uniqkache.models.hf_backend._require_transformers",
            lambda: mock_transformers,
        )

        HFBackend.from_pretrained("org/test-model", local_files_only=True)

        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "org/test-model", revision=None, local_files_only=True
        )
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "org/test-model", revision=None, dtype=torch.float32, local_files_only=True
        )

    def test_from_pretrained_raises_backend_error_when_offline_and_missing(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError(
            "Offline mode is enabled and file not found in local cache"
        )

        monkeypatch.setattr(
            "uniqkache.models.hf_backend._require_transformers",
            lambda: mock_transformers,
        )

        with pytest.raises(BackendError) as excinfo:
            HFBackend.from_pretrained("org/missing-model", local_files_only=True)

        assert "local_files_only=True" in str(excinfo.value)
        assert "org/missing-model" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)
