"""Core data types shared by the cache, policy and benchmark layers.

This module is deliberately dependency-light (only :mod:`torch` and the standard
library) so that every other module can import from it without creating cycles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from uniqkache.utils.errors import CacheConfigError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheConfig:
    """Static description of a KV cache's shape and budget.

    Parameters
    ----------
    num_layers:
        Number of transformer layers that own a K/V pair.
    num_kv_heads:
        Number of key/value heads (for grouped-query attention this is smaller
        than the number of query heads).
    head_dim:
        Per-head channel dimension.
    dtype:
        Storage dtype for K/V tensors.
    device:
        Device string, e.g. ``"cpu"`` or ``"cuda"``.
    capacity:
        Maximum number of tokens retained **per layer**. ``None`` means
        unbounded, which is the correct setting for the full-cache baseline.
    attention_sinks:
        Number of leading tokens that are never evicted. StreamingLLM-style
        methods rely on this; setting it to ``0`` disables the protection.
    batch_size:
        Batch dimension. Only used for memory accounting in the reference
        implementation, which operates on one request at a time.

    Notes
    -----
    ``capacity`` is a *per-layer token* budget, not a byte budget. Translating a
    byte budget into a token budget is the job of
    :meth:`~uniqkache.cache.types.CacheConfig.tokens_for_bytes`, which keeps the
    policy layer free of hardware arithmetic.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16
    device: str = "cpu"
    capacity: int | list[int] | None = None
    attention_sinks: int = 0
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise CacheConfigError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.num_kv_heads < 1:
            raise CacheConfigError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if self.head_dim < 1:
            raise CacheConfigError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.batch_size < 1:
            raise CacheConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.attention_sinks < 0:
            raise CacheConfigError(f"attention_sinks must be >= 0, got {self.attention_sinks}")
        if self.capacity is not None:
            if isinstance(self.capacity, int):
                if self.capacity < 1:
                    raise CacheConfigError(f"capacity must be >= 1 when set, got {self.capacity}")
                if self.attention_sinks > self.capacity:
                    raise CacheConfigError(
                        "attention_sinks cannot exceed capacity: "
                        f"{self.attention_sinks} > {self.capacity}. "
                        "The protected tokens would leave no room for any evictable token."
                    )
            elif isinstance(self.capacity, list):
                if len(self.capacity) != self.num_layers:
                    raise CacheConfigError(
                        f"per-layer capacity list must have length {self.num_layers} (num_layers), "
                        f"got {len(self.capacity)}"
                    )
                for idx, c in enumerate(self.capacity):
                    if not isinstance(c, int) or c < 1:
                        raise CacheConfigError(
                            f"capacity for layer {idx} must be an int >= 1, got {c}"
                        )
                    if self.attention_sinks > c:
                        raise CacheConfigError(
                            f"attention_sinks cannot exceed capacity for layer {idx}: "
                            f"{self.attention_sinks} > {c}."
                        )
            else:
                raise CacheConfigError(
                    f"capacity must be int, list[int], or None, got {type(self.capacity).__name__}"
                )

    # -- shape helpers -----------------------------------------------------

    @property
    def element_size(self) -> int:
        """Bytes per stored scalar."""
        return torch.empty(0, dtype=self.dtype).element_size()

    def bytes_per_token_per_layer(self, batch_size: int | None = None) -> int:
        """Bytes for one token in one layer (keys + values)."""
        batch = self.batch_size if batch_size is None else batch_size
        return int(2 * batch * self.num_kv_heads * self.head_dim * self.element_size)

    def bytes_per_token(self, batch_size: int | None = None) -> int:
        """Bytes for one token across every layer."""
        return self.num_layers * self.bytes_per_token_per_layer(batch_size)

    def bytes_for_tokens(self, tokens_per_layer: int, batch_size: int | None = None) -> int:
        """Bytes for ``tokens_per_layer`` tokens held in *every* layer.

        The parameter is deliberately named after its unit. :attr:`CacheStats.total_tokens`
        is the count **summed across layers**, whereas this method expects the count
        **per layer**; passing one where the other is meant over-counts by a factor of
        ``num_layers``. To check a cache against its own accounting, use
        :attr:`CacheStats.tokens_per_layer`, or compare against
        ``bytes_per_token_per_layer() * total_tokens``.
        """
        if tokens_per_layer < 0:
            raise CacheConfigError(f"tokens_per_layer must be >= 0, got {tokens_per_layer}")
        return tokens_per_layer * self.bytes_per_token(batch_size)

    def tokens_for_bytes(self, byte_budget: int, batch_size: int | None = None) -> int:
        """Largest token count whose K/V footprint fits in ``byte_budget``.

        This is the bridge between a hardware memory budget (what a user knows)
        and the token capacity the cache understands. It always floors, so the
        returned capacity never exceeds the budget.
        """
        if byte_budget < 0:
            raise CacheConfigError(f"byte_budget must be >= 0, got {byte_budget}")
        per_token = self.bytes_per_token(batch_size)
        if per_token == 0:
            return 0
        return byte_budget // per_token

    def capacity_for_layer(self, layer_idx: int) -> int | None:
        """Capacity for ``layer_idx``, or ``None`` when unbounded."""
        if self.capacity is None:
            return None
        if isinstance(self.capacity, int):
            return self.capacity
        if not 0 <= layer_idx < self.num_layers:
            raise CacheConfigError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        return self.capacity[layer_idx]

    def total_capacity(self) -> int | None:
        """Total token capacity summed across all layers, or ``None`` when unbounded."""
        if self.capacity is None:
            return None
        if isinstance(self.capacity, int):
            return self.capacity * self.num_layers
        return sum(self.capacity)

    def with_capacity(self, capacity: int | list[int] | None) -> CacheConfig:
        """Return a copy of this config with a different token capacity."""
        return CacheConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
            capacity=capacity,
            attention_sinks=self.attention_sinks,
            batch_size=self.batch_size,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dtype"] = str(self.dtype)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheConfig:
        """Construct a CacheConfig from a dictionary, resolving dtype if needed."""
        d = dict(data)
        dtype_val = d.get("dtype")
        if isinstance(dtype_val, str):
            name = dtype_val.replace("torch.", "")
            dt = getattr(torch, name, None)
            if not isinstance(dt, torch.dtype):
                raise CacheConfigError(f"unknown dtype string in config: {dtype_val!r}")
            d["dtype"] = dt
        return cls(**d)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """A snapshot of cache occupancy, accounting and activity.

    Every field is a plain Python type so the object serialises directly into a
    benchmark record.
    """

    num_layers: int
    tokens_per_layer: list[int]
    capacity: int | list[int] | None
    total_tokens: int
    max_tokens_in_layer: int
    utilization: float | None
    bytes_on_device: int
    bytes_offloaded: int
    bytes_total: int
    compression_ratio: float
    evictions: int
    compressions: int
    offloads: int
    prefetches: int
    device: str
    dtype: str
    offloaded_layers: list[int] = field(default_factory=list)
    utilization_per_layer: list[float] | None = None

    @property
    def memory_bytes(self) -> int:
        """Total resident bytes (device + offloaded).

        Named explicitly so a caller cannot accidentally report only the device
        portion as "cache memory" and overstate a saving.
        """
        return self.bytes_total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """One-line human-readable summary, used by the CLI."""
        util = "n/a" if self.utilization is None else f"{self.utilization:.1%}"
        return (
            f"tokens={self.total_tokens} util={util} "
            f"mem={self.bytes_total / 1024**2:.2f}MiB "
            f"(device={self.bytes_on_device / 1024**2:.2f}MiB, "
            f"offloaded={self.bytes_offloaded / 1024**2:.2f}MiB) "
            f"compression={self.compression_ratio:.2f}x "
            f"evictions={self.evictions}"
        )


# ---------------------------------------------------------------------------
# Policy state
# ---------------------------------------------------------------------------


@dataclass
class PolicyState:
    """Everything a policy is allowed to look at when scoring cached tokens.

    This is the *only* channel through which a policy observes the cache. Keeping
    it to one dataclass means:

    * a new signal can be added in one place and every policy can opt into it;
    * a policy cannot reach into storage and mutate it, so eviction stays
      centralised and auditable.

    All per-token tensors are indexed by *cache slot* and have length
    ``num_cached``. Slot ``i`` corresponds to the token at absolute position
    ``positions[i]``.
    """

    layer_idx: int
    step: int
    num_cached: int
    capacity: int | None
    positions: torch.Tensor
    last_access: torch.Tensor
    cum_attention: torch.Tensor
    hit_count: torch.Tensor
    is_sink: torch.Tensor
    memory_pressure: float
    # Optional extra signals. Present as ``None`` when unavailable so that a
    # policy must handle their absence rather than silently seeing zeros.
    query_relevance: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected = self.num_cached
        for name in ("positions", "last_access", "cum_attention", "hit_count", "is_sink"):
            tensor = getattr(self, name)
            if tensor.shape[0] != expected:
                raise CacheConfigError(
                    f"PolicyState.{name} has length {tensor.shape[0]}, expected {expected}. "
                    "Every per-token signal must describe exactly the cached tokens."
                )

    def free_slots(self) -> int | None:
        """Tokens that could still be appended before the budget binds."""
        if self.capacity is None:
            return None
        return max(0, self.capacity - self.num_cached)


__all__ = ["CacheConfig", "CacheStats", "PolicyState"]
