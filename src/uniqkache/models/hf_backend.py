"""Hugging Face backend.

STATUS: **Experimental.** Full-cache inference only.

What works
----------
Any ``AutoModelForCausalLM`` can be run through UniqKache's runtime and measured
with UniqKache's harness: TTFT, TPOT, throughput, peak device memory and
perplexity, all recorded in the standard
:class:`~uniqkache.metrics.record.BenchmarkRecord` schema. This is useful in its
own right — it means a real model's numbers are produced by the same
instrumentation as everything else, rather than by an ad-hoc script.

What does not work yet
----------------------
**Evicting policies on HF models are not supported.** Passing a cache with a
bounded capacity raises :class:`~uniqkache.utils.errors.BackendError`.

The technical reason, recorded so a future contributor does not have to
rediscover it: in ``transformers`` 5.x the ``Cache`` base class no longer
accepts a bespoke ``update`` implementation directly — it requires a *layer
class* via ``layer_class_to_replicate`` and dispatches through an internal
per-layer abstraction. Intercepting the append path therefore means
implementing that layer contract, not merely subclassing ``Cache``. This was
probed and confirmed; see ``docs/architecture.md``.

Until that is done, evicting-policy experiments must use the synthetic backend,
whose attention path UniqKache owns end to end. This is recorded in the README
under **Not yet supported**, and on the research roadmap.

Known inefficiency
------------------
For the full-cache path this adapter keeps the model's own ``DynamicCache`` and
additionally mirrors K/V into the UniqKache cache so that occupancy and metadata
are reported. That duplicates the K/V tensors, so memory figures from an HF run
overstate the cache's own footprint. Records produced through this path are
annotated, and the duplication is listed as a known limitation.
"""

from __future__ import annotations

from typing import Any

import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.utils.device import resolve_device
from uniqkache.utils.errors import BackendError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

HF_EVICTION_NOT_SUPPORTED = (
    "the Hugging Face backend does not support evicting cache policies yet. "
    "transformers 5.x requires a custom Cache *layer* implementation to intercept "
    "the append path, which is not implemented. Use the synthetic backend for "
    "policy experiments, or run this model with policy='full_cache'. "
    "Tracked in docs/research.md under Open Questions."
)


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


def _extract_layer_kv(
    hf_cache: Any, layer_idx: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract (keys, values) tensors for ``layer_idx`` from an HF cache structure.

    Supports:
    - Transformers >= 4.49 / 5.x ``DynamicCache`` with ``layers`` attribute
      containing per-layer cache objects exposing ``keys`` and ``values``.
    - Transformers 4.40 - 4.48 ``DynamicCache`` with ``key_cache`` and ``value_cache``
      lists of layer tensors.
    - Legacy ``past_key_values`` tuples or lists of ``(key, value)`` pairs.
    - Any cache structure subscriptable by layer index returning a ``(key, value)`` pair.
    """
    if hf_cache is None:
        return None, None

    # Transformers 5.x / layers-based Cache API
    if hasattr(hf_cache, "layers"):
        layers = hf_cache.layers
        try:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is not None and values is not None:
                    return keys, values
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    return layer[0], layer[1]
        except (TypeError, IndexError, KeyError):
            pass
        return None, None

    # Transformers 4.40 - 4.48 DynamicCache (.key_cache, .value_cache)
    if hasattr(hf_cache, "key_cache") and hasattr(hf_cache, "value_cache"):
        key_cache = hf_cache.key_cache
        value_cache = hf_cache.value_cache
        try:
            keys = key_cache[layer_idx] if layer_idx < len(key_cache) else None
            values = value_cache[layer_idx] if layer_idx < len(value_cache) else None
            return keys, values
        except (TypeError, IndexError, KeyError):
            return None, None

    # Legacy past_key_values tuple/list or custom subscriptable Cache
    try:
        pair = hf_cache[layer_idx]
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            return pair[0], pair[1]
        if hasattr(pair, "keys") and hasattr(pair, "values"):
            return pair.keys, pair.values
    except (TypeError, IndexError, KeyError):
        pass

    return None, None


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
        """Run the model, optionally mirroring K/V into a UniqKache cache.

        Raises
        ------
        BackendError
            If ``cache`` has a bounded capacity, because evicting policies are
            not supported on this backend yet.
        """
        if cache is not None and cache.capacity is not None:
            raise BackendError(HF_EVICTION_NOT_SUPPORTED)

        if not hasattr(self, "_hf_cache"):
            transformers = _require_transformers()
            self._hf_cache = transformers.DynamicCache()

        seq = int(input_ids.shape[1])
        cache_position = torch.arange(start_pos, start_pos + seq, device=input_ids.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                past_key_values=self._hf_cache,
                use_cache=True,
                output_attentions=return_attention,
                cache_position=cache_position,
            )

        self._hf_cache = output.past_key_values

        if cache is not None:
            self._mirror_into_cache(cache, start_pos, seq)

        attentions = getattr(output, "attentions", None)
        weights = list(attentions) if attentions is not None else None
        return output.logits, weights

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
            keys, values = _extract_layer_kv(self._hf_cache, layer_idx)
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


__all__ = ["HF_EVICTION_NOT_SUPPORTED", "HFBackend", "_extract_layer_kv", "build_hf_model"]
