"""Hugging Face backend.

STATUS: **Experimental.** Supports full-cache and bounded evicting cache policies.

What works
----------
Any ``AutoModelForCausalLM`` can be run through UniqKache's runtime and measured
with UniqKache's harness: TTFT, TPOT, throughput, peak device memory and
perplexity, all recorded in the standard
:class:`~uniqkache.metrics.record.BenchmarkRecord` schema.

Both full-cache and evicting policies (such as ``sliding_window``,
``attention_based`` / H2O, ``lru``, etc.) are supported end-to-end. In
``transformers`` 5.x, custom cache layers subclassing ``CacheLayerMixin``
(:class:`UniqKacheLayer`) and :class:`UniqKacheHFCache` route append and eviction
operations directly through :class:`~uniqkache.cache.kv_cache.KVCache`, eliminating
duplicate memory overhead.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.utils.device import resolve_device
from uniqkache.utils.errors import BackendError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


def _require_transformers() -> Any:
    """Import transformers, or explain how to install it."""
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BackendError(
            "the Hugging Face backend requires the 'hf' extra. Install with:\n"
            '    pip install -e ".[hf]"'
        ) from exc
    return transformers


try:
    import transformers.cache_utils as _hf_cache_utils

    _CacheBase = getattr(_hf_cache_utils, "Cache", object)
    _DynamicLayerBase = getattr(
        _hf_cache_utils, "DynamicLayer", getattr(_hf_cache_utils, "CacheLayerMixin", object)
    )
except ImportError:
    _CacheBase = object  # type: ignore[assignment,misc]
    _DynamicLayerBase = object  # type: ignore[assignment,misc]


class UniqKacheLayer(_DynamicLayerBase):
    """Hugging Face Cache layer delegating storage and eviction to UniqKache's KVCache."""

    def __init__(self, kv_cache: KVCache, layer_idx: int) -> None:
        self.kv_cache = kv_cache
        self.layer_idx = layer_idx

    @property
    def keys(self) -> torch.Tensor | None:
        store = self.kv_cache.store.layer(self.layer_idx)
        return store.keys if store.is_initialized else None

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        store = self.kv_cache.store.layer(self.layer_idx)
        store._keys = value

    @property
    def values(self) -> torch.Tensor | None:
        store = self.kv_cache.store.layer(self.layer_idx)
        return store.values if store.is_initialized else None

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        store = self.kv_cache.store.layer(self.layer_idx)
        store._values = value

    @property
    def is_initialized(self) -> bool:
        return self.kv_cache.store.layer(self.layer_idx).is_initialized

    def get_seq_length(self) -> int:
        return self.kv_cache.num_tokens(self.layer_idx)

    def get_max_length(self) -> int | None:
        return self.kv_cache.capacity

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        cur_len = self.get_seq_length()
        cap = self.get_max_length()
        kv_len = (
            min(cur_len + query_length, cap)
            if cap is not None and cap > 0
            else cur_len + query_length
        )
        return kv_len, 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.kv_cache.append(self.layer_idx, key_states, value_states)
        k, v = self.kv_cache.get(self.layer_idx)
        return k, v

    def reset(self) -> None:
        self.kv_cache.store.layer(self.layer_idx).clear()

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove == 0:
            return
        cur_len = self.get_seq_length()
        if tokens_to_remove > 0:
            if tokens_to_remove >= cur_len:
                return
            remove = cur_len - tokens_to_remove
        else:
            remove = abs(tokens_to_remove)
        if remove >= cur_len:
            self.reset()
            return
        keep = cur_len - remove
        evict_indices = torch.arange(
            keep, cur_len, device=self.kv_cache.store.layer(self.layer_idx).resident_device
        )
        self.kv_cache.evict(self.layer_idx, indices=evict_indices)


class UniqKacheHFCache(_CacheBase):
    """Hugging Face Cache subclass backed directly by a UniqKache KVCache."""

    def __init__(self, kv_cache: KVCache) -> None:
        self.kv_cache = kv_cache
        layers = [UniqKacheLayer(kv_cache, idx) for idx in range(kv_cache.num_layers)]
        if _CacheBase is not object:
            super().__init__(layers=layers)
        else:
            self.layers = layers

    def __getitem__(self, layer_idx: int) -> UniqKacheLayer:
        return self.layers[layer_idx]

    def __len__(self) -> int:
        return len(self.layers)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        idx = layer_idx if layer_idx is not None else 0
        if 0 <= idx < len(self.layers):
            return self.layers[idx].get_seq_length()
        return 0

    def get_max_length(self) -> int | None:
        return self.kv_cache.capacity

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        if 0 <= layer_idx < len(self.layers):
            return self.layers[layer_idx].get_mask_sizes(query_length)
        cur_len = self.get_seq_length(layer_idx)
        cap = self.get_max_length()
        kv_len = (
            min(cur_len + query_length, cap)
            if cap is not None and cap > 0
            else cur_len + query_length
        )
        return kv_len, 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


class HFBackend:
    """Adapter exposing a Hugging Face causal LM as a
    :class:`~uniqkache.models.base.LanguageModel`.

    Prefer :meth:`from_pretrained`. Constructing directly is supported for
    already-loaded models, which is how the test suite exercises this path
    without a download.
    """

    def __init__(
        self,
        model: Any,
        *,
        identifier: str,
        tokenizer: Any = None,
        revision: str | None = None,
        weights_are_random: bool = False,
    ) -> None:
        self.model = model.eval()
        self.identifier = identifier
        self.tokenizer = tokenizer
        self.revision = revision
        self._weights_are_random = weights_are_random

        config = model.config
        self._num_layers = int(getattr(config, "num_hidden_layers", 0))
        self._num_kv_heads = int(
            getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 0))
        )
        self._hidden_size = int(getattr(config, "hidden_size", 0))
        self._num_heads = int(getattr(config, "num_attention_heads", 1))
        self._mirrored_bytes: int = 0  # K/V mirrored into UniqKache so far
        if self._num_layers < 1 or self._num_kv_heads < 1:
            raise BackendError(
                f"could not determine layer/head counts from {type(config).__name__}; "
                "this architecture is not supported by the HF backend"
            )
        self._head_dim = self._hidden_size // max(1, self._num_heads)
        self._vocab_size = int(getattr(config, "vocab_size", 0))
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype
        try:
            sig = inspect.signature(self.model.forward)
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            self._accepts_position_ids = "position_ids" in sig.parameters or has_var_kwargs
            self._accepts_cache_position = "cache_position" in sig.parameters or has_var_kwargs
        except Exception:
            self._accepts_position_ids = True
            self._accepts_cache_position = True

    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: str = "auto",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> HFBackend:
        """Load a model and tokenizer from the Hugging Face Hub or a local cache.

        Raises
        ------
        BackendError
            If the model cannot be loaded. The underlying error is chained so
            the real cause (missing weights, no disk space, gated repo) is
            visible rather than replaced by a generic message.
        """
        transformers = _require_transformers()
        resolved = resolve_device(device)

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, revision=revision, local_files_only=local_files_only
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise BackendError(
                f"could not load Hugging Face model {model_id!r} (revision={revision!r}, "
                f"local_files_only={local_files_only}): {exc}"
            ) from exc

        model = model.to(resolved)
        return cls(
            model,
            identifier=model_id,
            tokenizer=tokenizer,
            revision=revision,
            weights_are_random=False,
        )

    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def weights_are_random(self) -> bool:
        return self._weights_are_random

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model_type": getattr(self.model.config, "model_type", "unknown"),
            "num_layers": self._num_layers,
            "num_heads": self._num_heads,
            "num_kv_heads": self._num_kv_heads,
            "head_dim": self._head_dim,
            "hidden_size": self._hidden_size,
            "vocab_size": self._vocab_size,
        }

    def cache_config(
        self,
        *,
        capacity: int | list[int] | None = None,
        attention_sinks: int = 0,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        batch_size: int = 1,
    ) -> CacheConfig:
        """Build the matching :class:`CacheConfig` for this model."""
        return CacheConfig(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            dtype=dtype or self._dtype,
            device=device or str(self._device),
            capacity=capacity,
            attention_sinks=attention_sinks,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        *,
        start_pos: int = 0,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run the model, routing cache operations through UniqKache's KVCache."""
        seq = int(input_ids.shape[1])
        cache_position = torch.arange(start_pos, start_pos + seq, device=input_ids.device)

        if cache is not None:
            hf_cache = getattr(cache, "_hf_cache_adapter", None)
            if hf_cache is None or getattr(hf_cache, "kv_cache", None) is not cache:
                hf_cache = UniqKacheHFCache(cache)
                cache._hf_cache_adapter = hf_cache
            past_key_values = hf_cache
        else:
            if not hasattr(self, "_hf_cache") or self._hf_cache is None:
                transformers = _require_transformers()
                self._hf_cache = transformers.DynamicCache()
            past_key_values = self._hf_cache

        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_attentions": return_attention,
        }
        if self._accepts_cache_position:
            kwargs["cache_position"] = cache_position
        if self._accepts_position_ids:
            kwargs["position_ids"] = cache_position.unsqueeze(0)

        with torch.no_grad():
            output = self.model(**kwargs)

        if cache is None:
            self._hf_cache = getattr(output, "past_key_values", past_key_values)

        logits = output.logits if hasattr(output, "logits") else output[0]
        attentions = getattr(output, "attentions", None)
        weights = list(attentions) if attentions is not None else None
        return logits, weights

    def _mirror_into_cache(self, cache: KVCache, start_pos: int, seq: int) -> None:
        """Copy newly produced K/V into the UniqKache cache for bookkeeping.

        Only the rows for this call's positions are copied, so repeated calls do
        not double-append. Every mirrored byte is also recorded in
        ``self._mirrored_bytes``: the mirror duplicates the model-native cache
        one-for-one, so that counter is exactly the record's
        ``mirror_overhead_bytes``. See the module docstring for the caveat.
        """
        from uniqkache.utils.device import tensor_bytes

        for layer_idx in range(self._num_layers):
            layer = self._hf_cache.layers[layer_idx]
            keys, values = layer.keys, layer.values
            if keys is None or values is None:
                continue
            new_keys = keys[:, :, start_pos : start_pos + seq, :]
            new_values = values[:, :, start_pos : start_pos + seq, :]
            if new_keys.numel():
                self._mirrored_bytes += tensor_bytes(new_keys) + tensor_bytes(new_values)
            cache.append(layer_idx, new_keys, new_values)

    @property
    def mirrored_bytes(self) -> int:
        """K/V bytes mirrored into UniqKache caches so far.

        The mirror duplicates the model-native cache one-for-one, so this is
        exactly the run's ``mirror_overhead_bytes``. Zero when nothing has been
        mirrored yet — which, given the mirror copies every forward pass's rows,
        also means no forward pass with a cache ran.
        """
        return self._mirrored_bytes

    def reset(self) -> None:
        """Drop the model's internal cache, starting a fresh sequence.

        The mirror-byte counter is **not** reset: it is a lifetime total for
        this backend instance, matching "how much extra was held over the whole
        run" as reported on the record.
        """
        self._hf_cache = None


def build_hf_model(spec: Any, *, dtype: torch.dtype, device: str) -> Any:
    """Build an :class:`HFBackend` for a benchmark spec.

    Returns a :class:`~uniqkache.bench.runner._BuiltModel`-compatible object so
    the runner can treat backends uniformly.
    """
    from uniqkache.bench.runner import _BuiltModel  # local import avoids a cycle

    backend = HFBackend.from_pretrained(
        spec.model,
        dtype=dtype,
        device=device,
        revision=spec.model_revision,
        local_files_only=spec.offline,
    )

    def cache_config_factory(
        capacity: int | None, sinks: int, cache_dtype: torch.dtype, cache_device: str
    ) -> CacheConfig:
        return backend.cache_config(
            capacity=capacity,
            attention_sinks=sinks,
            dtype=cache_dtype,
            device=cache_device,
            batch_size=spec.batch_size,
        )

    return _BuiltModel(
        model=backend,
        identifier=backend.identifier,
        revision=backend.revision,
        num_parameters=backend.num_parameters,
        weights_are_random=backend.weights_are_random,
        vocab_size=backend.vocab_size,
        tokenizer=backend.identifier if backend.tokenizer is not None else None,
        config=backend.config,
        cache_config_factory=cache_config_factory,
        is_hf_backend=True,
        mirror_overhead_bytes_fn=lambda: backend.mirrored_bytes,
    )


__all__ = ["HFBackend", "UniqKacheHFCache", "UniqKacheLayer", "build_hf_model"]
