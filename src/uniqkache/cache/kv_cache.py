"""The KV cache facade: storage + policy, with an explicit action vocabulary.

This is the primary interface of UniqKache. It composes a policy-agnostic
:class:`~uniqkache.cache.store.KVStore` with a pluggable
:class:`~uniqkache.policies.base.CachePolicy`, and exposes the six actions the
project's research question is about:

============  ==========================================================
``append``    add newly computed K/V rows
``get``       read the resident K/V for a layer
``evict``     drop tokens (occupancy down)
``compress``  reduce bytes per token (occupancy unchanged)
``offload``   move bytes to a cheaper tier (occupancy unchanged)
``prefetch``  move bytes back to the compute tier
============  ==========================================================

The distinction between *occupancy* and *representation* is load-bearing for
this project's honesty requirements. Eviction destroys information; compression
and offloading do not. A result that reports "memory reduced" without saying
which of the two happened is not reportable here, and :meth:`stats` reports them
separately for that reason.

Example
-------
.. code-block:: python

    from uniqkache.cache.kv_cache import KVCache
    from uniqkache.cache.types import CacheConfig
    from uniqkache.policies import SlidingWindowPolicy

    config = CacheConfig(
        num_layers=4, num_kv_heads=2, head_dim=16,
        dtype=torch.float32, capacity=128, attention_sinks=4,
    )
    cache = KVCache(config, policy=SlidingWindowPolicy(window=124))

    cache.append(0, keys, values)          # keys/values: [B, H, T, D]
    k, v = cache.get(0)
    print(cache.stats().summary())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from uniqkache.cache.metadata import reduce_attention
from uniqkache.cache.store import KVStore
from uniqkache.cache.types import CacheConfig, CacheStats, PolicyState
from uniqkache.compression.base import BaseCompressor
from uniqkache.compression.quantize import Int8KVCompressor
from uniqkache.utils.device import resolve_device
from uniqkache.utils.errors import CacheStateError, PolicyError
from uniqkache.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for typing only
    from uniqkache.policies.base import BaseCachePolicy

_log = get_logger(__name__)


class KVCache:
    """A KV cache with a pluggable retention policy.

    Parameters
    ----------
    config:
        Shape, dtype, device and token budget.
    policy:
        Retention policy. ``None`` is allowed only when ``config.capacity`` is
        also ``None`` (the pure full-cache baseline, which needs no decisions).
        Requesting a bounded cache without a policy is a configuration error and
        is rejected at construction rather than silently evicting nothing.
    compressor:
        Compressor used by :meth:`compress`. Defaults to
        :class:`~uniqkache.compression.quantize.Int8KVCompressor`.
    auto_enforce:
        When True (default) :meth:`append` enforces the token budget immediately
        after each append. Set to False when a caller wants to control exactly
        when eviction happens, e.g. in a test that inspects intermediate state.

    Raises
    ------
    PolicyError
        If a bounded cache is constructed without a policy.
    """

    def __init__(
        self,
        config: CacheConfig,
        policy: BaseCachePolicy | None = None,
        *,
        compressor: BaseCompressor | None = None,
        auto_enforce: bool = True,
    ) -> None:
        if config.capacity is not None and policy is None:
            raise PolicyError(
                f"a bounded cache (capacity={config.capacity}) requires a policy. "
                "Pass policy=FullCachePolicy() if you want no eviction, or use "
                "capacity=None for an unbounded full cache."
            )

        self.config = config
        self.policy = policy
        self.compressor: BaseCompressor = compressor or Int8KVCompressor()
        self.auto_enforce = auto_enforce

        self._store = KVStore(config)
        self._step = 0
        self._evictions = 0
        self._evicted_tokens = 0
        self._compressions = 0
        self._offloads = 0
        self._prefetches = 0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def store(self) -> KVStore:
        """The underlying storage. Exposed for tests and instrumentation."""
        return self._store

    @property
    def step(self) -> int:
        """Monotonic counter of decode steps applied so far."""
        return self._step

    @property
    def num_layers(self) -> int:
        return len(self._store)

    @property
    def capacity(self) -> int | None:
        return self.config.capacity

    def num_tokens(self, layer_idx: int | None = None) -> int:
        """Cached token count for one layer, or the maximum across layers."""
        if layer_idx is None:
            return self._store.max_tokens()
        return self._store.num_tokens(layer_idx)

    def is_empty(self) -> bool:
        return self._store.max_tokens() == 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def append(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Append newly computed K/V rows to ``layer_idx``.

        Parameters
        ----------
        layer_idx:
            Target layer.
        keys, values:
            Tensors of shape ``[batch, num_kv_heads, new_tokens, head_dim]``.
        positions:
            Absolute position ids of the new tokens. When omitted they continue
            consecutively, which is correct for a fresh sequence.
        """
        self._store.layer(layer_idx).append(keys, values, positions=positions, step=self._step)
        if self.auto_enforce and self.config.capacity is not None:
            self.enforce_capacity(layer_idx)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the resident ``(keys, values)`` for ``layer_idx``.

        When the layer holds a compressed representation the tensors are
        dequantised on demand, so callers always see a float K/V pair.

        Raises
        ------
        CacheStateError
            If the layer has no tokens yet. Returning an empty tensor instead
            would let a shape bug propagate silently into the attention math.
        """
        layer = self._store.layer(layer_idx)
        if not layer.is_initialized:
            raise CacheStateError(
                f"layer {layer_idx} has no cached tokens; append K/V before calling get(). "
                "An empty cache is a caller bug, not an empty result."
            )
        keys, values = layer.keys, layer.values
        if keys is None or values is None:
            raise CacheStateError(f"layer {layer_idx} reports initialised but yields no tensors")
        return keys, values

    def evict(
        self,
        layer_idx: int | None = None,
        *,
        indices: torch.Tensor | None = None,
        keep: torch.Tensor | None = None,
    ) -> int:
        """Remove tokens from the cache.

        Exactly one selection mode is used, in this precedence:

        * ``keep`` — retain these slots, drop the rest;
        * ``indices`` — drop these slots, retain the rest;
        * neither — ask the policy to enforce ``config.capacity``.

        Parameters
        ----------
        layer_idx:
            A single layer, or ``None`` for every layer.
        indices:
            Slots to drop.
        keep:
            Slots to retain.

        Returns
        -------
        int
            Number of tokens dropped, summed over the affected layers.

        Raises
        ------
        PolicyError
            If neither selector is given and no policy is configured, or if both
            are given. Guessing which the caller meant is exactly the kind of
            silent ambiguity this project forbids.
        """
        if indices is not None and keep is not None:
            raise PolicyError("evict() accepts either `indices` or `keep`, not both")

        if indices is None and keep is None:
            return self.enforce_capacity(layer_idx)

        targets = self._target_layers(layer_idx)
        dropped = 0
        for idx in targets:
            num = self._store.num_tokens(idx)
            if num == 0:
                continue
            if keep is not None:
                selection = keep
            else:
                assert indices is not None
                selection = self._complement(indices, num)
            dropped += self._store.keep(idx, selection)

        if dropped:
            self._evictions += 1
            self._evicted_tokens += dropped
        return dropped

    def clear(self) -> None:
        """Drop every token in every layer and reset per-sequence state."""
        self._store.clear()
        if self.policy is not None:
            self.policy.reset()
        self._step = 0

    def compress(
        self,
        layer_idx: int | None = None,
        *,
        method: str | None = None,
    ) -> int:
        """Quantise cached K/V, reducing bytes per token without evicting.

        Parameters
        ----------
        layer_idx:
            A single layer, or ``None`` for every layer.
        method:
            Compressor name. Only ``"int8"`` is available in this milestone;
            passing anything else raises rather than silently doing nothing.

        Returns
        -------
        int
            Number of layers whose representation changed.
        """
        if method is not None and method != self.compressor.name:
            raise CacheStateError(
                f"requested compression method {method!r} but this cache was configured "
                f"with {self.compressor.name!r}. Construct the cache with the desired "
                "compressor instead."
            )

        changed = 0
        for idx in self._target_layers(layer_idx):
            layer = self._store.layer(idx)
            if not layer.is_initialized or layer.is_compressed:
                continue
            keys, values = layer.keys, layer.values
            if keys is None or values is None:
                continue
            result = self.compressor.compress(keys, values)
            layer.apply_compression(result)
            changed += 1

        if changed:
            self._compressions += 1
        return changed

    def decompress(self, layer_idx: int | None = None) -> int:
        """Expand compressed layers back to the configured float dtype.

        Returns the number of layers materialised.
        """
        changed = 0
        for idx in self._target_layers(layer_idx):
            layer = self._store.layer(idx)
            if layer.is_compressed:
                layer.materialize()
                changed += 1
        return changed

    def offload(
        self,
        layer_idx: int | None = None,
        *,
        target: str | torch.device = "cpu",
    ) -> int:
        """Move cached bytes to a cheaper memory tier.

        Returns the number of layers that actually moved.
        """
        target_device = resolve_device(target)
        moved = 0
        for idx in self._target_layers(layer_idx):
            layer = self._store.layer(idx)
            if not layer.is_initialized or layer.is_offloaded:
                continue
            if layer.resident_device == target_device:
                continue
            self._store.move_to(target_device, idx)
            moved += 1

        if moved:
            self._offloads += 1
        return moved

    def prefetch(
        self,
        layer_idx: int | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> int:
        """Move offloaded bytes back to the compute device.

        Parameters
        ----------
        device:
            Destination. Defaults to the cache's home device.

        Returns
        -------
        int
            Number of layers brought back.
        """
        home = resolve_device(device) if device is not None else self._store.device
        brought_back = 0
        for idx in self._target_layers(layer_idx):
            layer = self._store.layer(idx)
            if not layer.is_offloaded:
                continue
            self._store.move_to(home, idx)
            brought_back += 1

        if brought_back:
            self._prefetches += 1
        return brought_back

    def stats(self) -> CacheStats:
        """Snapshot occupancy, memory accounting and activity counters.

        ``bytes_on_device`` and ``bytes_offloaded`` are reported separately so
        that offloading cannot be mistaken for a genuine memory reduction, and
        ``compression_ratio`` is derived from stored bytes rather than from a
        nominal bit width.
        """
        tokens = self._store.tokens_per_layer()
        total = sum(tokens)
        capacity = self.config.capacity
        max_in_layer = max(tokens) if tokens else 0

        uncompressed = self._store.uncompressed_bytes()
        resident = self._store.bytes_total()
        ratio = (uncompressed / resident) if resident > 0 else 1.0

        return CacheStats(
            num_layers=len(self._store),
            tokens_per_layer=tokens,
            capacity=capacity,
            total_tokens=total,
            max_tokens_in_layer=max_in_layer,
            utilization=(max_in_layer / capacity) if capacity else None,
            bytes_on_device=self._store.bytes_on_device(),
            bytes_offloaded=self._store.bytes_offloaded(),
            bytes_total=resident,
            compression_ratio=ratio,
            evictions=self._evictions,
            compressions=self._compressions,
            offloads=self._offloads,
            prefetches=self._prefetches,
            device=str(self._store.device),
            dtype=str(self.config.dtype),
            offloaded_layers=self._store.offloaded_layers(),
        )

    # ------------------------------------------------------------------
    # Signals and policy driving
    # ------------------------------------------------------------------

    def note_attention(
        self,
        layer_idx: int,
        attention: torch.Tensor,
        *,
        mode: str = "last_query",
        query_index: int | None = None,
    ) -> None:
        """Record attention mass received by each cached token in ``layer_idx``.

        Parameters
        ----------
        attention:
            ``[batch, heads, queries, keys]`` probabilities, as produced by the
            attention softmax. The ``keys`` dimension must equal the number of
            cached tokens in this layer.
        mode:
            ``"last_query"`` during decode, ``"all_queries"`` during prefill.
        query_index:
            Override the query row explicitly.
        """
        weights = reduce_attention(attention, mode=mode, query_index=query_index)
        self._store.layer(layer_idx).metadata.note_attention(weights, step=self._step)

    def note_access(self, layer_idx: int, indices: torch.Tensor) -> None:
        """Record an explicit read of specific slots (used by prefetch paths)."""
        self._store.layer(layer_idx).metadata.note_access(indices, step=self._step)

    def state(self, layer_idx: int) -> PolicyState:
        """Build the :class:`PolicyState` a policy would see for ``layer_idx``."""
        layer = self._store.layer(layer_idx)
        meta = layer.metadata
        capacity = self.config.capacity
        pressure = 0.0
        if capacity:
            pressure = min(1.0, meta.num_tokens / capacity)
        return PolicyState(
            layer_idx=layer_idx,
            step=self._step,
            num_cached=meta.num_tokens,
            capacity=capacity,
            positions=meta.positions,
            last_access=meta.last_access,
            cum_attention=meta.cum_attention,
            hit_count=meta.hit_count,
            is_sink=meta.is_sink(),
            memory_pressure=pressure,
        )

    def enforce_capacity(self, layer_idx: int | None = None) -> int:
        """Ask the policy to bring each layer within its token budget.

        Returns the number of tokens evicted.

        Raises
        ------
        PolicyError
            If a capacity is set but no policy is available to decide.
        """
        capacity = self.config.capacity
        if capacity is None:
            return 0
        if self.policy is None:
            raise PolicyError(
                "enforce_capacity() needs a policy, but this cache has none. "
                "Construct KVCache(config, policy=...) to enable eviction."
            )

        evicted = 0
        for idx in self._target_layers(layer_idx):
            if self._store.num_tokens(idx) <= capacity:
                continue
            state = self.state(idx)
            scores = self.policy.score(state)
            keep = self.policy.select(scores, capacity, protect=state.is_sink)
            evicted += self._store.keep(idx, keep)

        if evicted:
            self._evictions += 1
            self._evicted_tokens += evicted
        return evicted

    def advance(self) -> None:
        """Advance the decode step counter.

        Called once per generated token so that recency signals (LRU) and
        attention bookkeeping share a consistent notion of "now".
        """
        self._step += 1

    def reset(self) -> None:
        """Prepare the cache for a new request, preserving configuration."""
        self.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_layers(self, layer_idx: int | None) -> list[int]:
        if layer_idx is None:
            return list(range(len(self._store)))
        # Validate through the store so an out-of-range index raises uniformly.
        self._store.layer(layer_idx)
        return [layer_idx]

    @staticmethod
    def _complement(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Indices in ``[0, num_tokens)`` that are *not* listed in ``indices``."""
        indices = indices.to(dtype=torch.long).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= num_tokens):
            raise CacheStateError(
                f"evict indices out of range [0, {num_tokens}): "
                f"min={int(indices.min())}, max={int(indices.max())}"
            )
        mask = torch.ones(num_tokens, dtype=torch.bool)
        mask[indices.cpu()] = False
        return torch.nonzero(mask, as_tuple=False).flatten()

    def state_dict(self) -> dict[str, object]:
        """Serialisable configuration and activity summary."""
        stats = self.stats()
        return {
            "config": self.config.to_dict(),
            "policy": self.policy.state_dict() if self.policy is not None else None,
            "compressor": self.compressor.state_dict(),
            "step": self._step,
            "evictions": self._evictions,
            "evicted_tokens": self._evicted_tokens,
            "compressions": self._compressions,
            "offloads": self._offloads,
            "prefetches": self._prefetches,
            "final_tokens": stats.total_tokens,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str | Path, *, model: str | None = None) -> Path:
        """Save this cache's contents and activity state to disk via torch.save.

        Preserves quantised layers without dequantising.

        Parameters
        ----------
        path:
            Target file path.
        model:
            Optional model identifier recorded for provenance.

        Returns
        -------
        Path
            The written file path.
        """
        extra_state = {
            "step": self._step,
            "evictions": self._evictions,
            "evicted_tokens": self._evicted_tokens,
            "compressions": self._compressions,
            "offloads": self._offloads,
            "prefetches": self._prefetches,
        }
        return self._store.save(path, model=model, extra_state=extra_state)

    @classmethod
    def load(
        cls,
        path: str | Path,
        policy: BaseCachePolicy | None = None,
        *,
        config: CacheConfig | None = None,
        compressor: BaseCompressor | None = None,
        auto_enforce: bool = True,
        map_location: Any = None,
    ) -> KVCache:
        """Load a KVCache from a checkpoint file.

        Parameters
        ----------
        path:
            Path to the checkpoint file.
        policy:
            Retention policy. When ``config.capacity`` is not None, a policy is required.
        config:
            Optional CacheConfig to validate against. When omitted, the configuration
            is restored from the checkpoint.
        compressor:
            Compressor to attach to the cache. Defaults to Int8KVCompressor.
        auto_enforce:
            Whether to auto-enforce capacity on appends.
        map_location:
            Optional device mapping for torch.load.

        Returns
        -------
        KVCache
            A restored cache with identical storage and counters.
        """
        target = Path(path)
        if not target.is_file():
            raise CacheStateError(f"checkpoint file not found: {target}")

        raw = torch.load(target, map_location=map_location, weights_only=True)
        if not isinstance(raw, dict):
            raise CacheStateError(f"invalid checkpoint format in {target}")

        saved_config_dict = raw.get("config")
        if not saved_config_dict:
            raise CacheStateError("checkpoint missing 'config'")

        resolved_config = CacheConfig.from_dict(saved_config_dict)
        if config is not None:
            # Validate provided config against checkpoint config
            if config.num_layers != resolved_config.num_layers:
                raise CacheStateError(
                    f"provided config num_layers ({config.num_layers}) != checkpoint ({resolved_config.num_layers})"
                )
            if config.num_kv_heads != resolved_config.num_kv_heads:
                raise CacheStateError(
                    f"provided config num_kv_heads ({config.num_kv_heads}) != checkpoint ({resolved_config.num_kv_heads})"
                )
            if config.head_dim != resolved_config.head_dim:
                raise CacheStateError(
                    f"provided config head_dim ({config.head_dim}) != checkpoint ({resolved_config.head_dim})"
                )
            if config.batch_size != resolved_config.batch_size:
                raise CacheStateError(
                    f"provided config batch_size ({config.batch_size}) != checkpoint ({resolved_config.batch_size})"
                )
            if config.dtype != resolved_config.dtype:
                raise CacheStateError(
                    f"provided config dtype ({config.dtype}) != checkpoint ({resolved_config.dtype})"
                )
            resolved_config = config

        store = KVStore.load(target, resolved_config, map_location=map_location)
        cache = cls(
            resolved_config, policy=policy, compressor=compressor, auto_enforce=auto_enforce
        )
        cache._store = store

        extra = raw.get("extra_state", {})
        cache._step = extra.get("step", 0)
        cache._evictions = extra.get("evictions", 0)
        cache._evicted_tokens = extra.get("evicted_tokens", 0)
        cache._compressions = extra.get("compressions", 0)
        cache._offloads = extra.get("offloads", 0)
        cache._prefetches = extra.get("prefetches", 0)

        return cache

    def load_checkpoint(self, path: str | Path, *, map_location: Any = None) -> None:
        """In-place restore of cache storage and activity counters from a checkpoint.

        Parameters
        ----------
        path:
            Path to the checkpoint file.
        map_location:
            Optional device mapping for torch.load.
        """
        target = Path(path)
        if not target.is_file():
            raise CacheStateError(f"checkpoint file not found: {target}")

        raw = torch.load(target, map_location=map_location, weights_only=True)
        if not isinstance(raw, dict):
            raise CacheStateError(f"invalid checkpoint format in {target}")

        store = KVStore.load(target, self.config, map_location=map_location)
        self._store = store

        extra = raw.get("extra_state", {})
        self._step = extra.get("step", 0)
        self._evictions = extra.get("evictions", 0)
        self._evicted_tokens = extra.get("evicted_tokens", 0)
        self._compressions = extra.get("compressions", 0)
        self._offloads = extra.get("offloads", 0)
        self._prefetches = extra.get("prefetches", 0)

    def __len__(self) -> int:
        return self.num_tokens()

    def __repr__(self) -> str:
        policy_name = self.policy.name if self.policy is not None else "none"
        return (
            f"KVCache(layers={self.num_layers}, tokens={self.num_tokens()}, "
            f"capacity={self.config.capacity}, policy={policy_name!r}, "
            f"device={self._store.device})"
        )


__all__ = ["KVCache"]
