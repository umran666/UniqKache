"""Tests for the Hugging Face backend's pure-logic paths.

The HF backend is an optional-extra module that CI never imports with the real
``transformers`` installed. These tests therefore exercise it with a **stub
model** — a tiny ``torch.nn`` module that mimics the attributes the adapter
reads (``config``, ``eval()``, ``parameters()``) — so the suite still passes
with ``transformers`` absent, no download, and no network.

The behaviour that matters most here is the bounded-cache refusal: an
unsupported configuration must raise carrying ``HF_EVICTION_NOT_SUPPORTED``
rather than silently running a full cache and reporting it as eviction.
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.models.hf_backend import HF_EVICTION_NOT_SUPPORTED, HFBackend
from uniqkache.policies import SlidingWindowPolicy
from uniqkache.utils.errors import BackendError


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

    def test_reset_does_not_reset_the_lifetime_counter(self, backend):
        backend._mirrored_bytes = 128
        backend.reset()
        assert backend.mirrored_bytes == 128


class TestEvictionRefusal:
    def test_a_bounded_cache_raises_the_documented_error(self, backend):
        cache = KVCache(
            backend.cache_config(capacity=4, attention_sinks=1),
            policy=SlidingWindowPolicy(),
        )
        with pytest.raises(BackendError) as excinfo:
            backend.forward(torch.randint(0, 128, (1, 8)), cache=cache, start_pos=0)
        assert HF_EVICTION_NOT_SUPPORTED in str(excinfo.value)

    def test_an_unbounded_cache_does_not_raise_for_the_wrong_reason(self, backend):
        """The unbounded path must get *past* the capacity guard.

        The stub model cannot actually run a forward pass, so we only assert the
        failure is NOT the eviction refusal — any later failure is the stub's
        limits, not the guard's.
        """
        cache = KVCache(backend.cache_config(capacity=None), policy=None)
        with pytest.raises(Exception) as excinfo:
            backend.forward(torch.randint(0, 128, (1, 4)), cache=cache, start_pos=0)
        assert HF_EVICTION_NOT_SUPPORTED not in str(excinfo.value)


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
