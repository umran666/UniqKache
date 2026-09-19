"""Memory accounting.

The distinction this module exists to preserve: **resident bytes are not the same
as total bytes**, and neither is the same as "bytes the cache would need at the
model's native precision".

Three quantities are reported separately and never conflated:

``native_bytes``
    What the cache would occupy at the model's dtype with no compression and no
    offloading. This is the denominator for any memory-reduction claim.
``on_device_bytes``
    What is physically resident on the compute device right now. This is what a
    GPU memory budget constrains.
``offloaded_bytes``
    What has been moved elsewhere. These bytes still exist; a claim that
    offloading "reduced memory" is only meaningful against a *device* budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.utils.device import current_memory_bytes, peak_memory_bytes, resolve_device


def theoretical_kv_bytes(config: CacheConfig, num_tokens: int) -> int:
    """KV bytes for ``num_tokens`` at the configured dtype, uncompressed.

    Independent of any cache instance, so it can be used to size a run before
    allocating anything — which is what makes "will this fit" answerable
    without attempting the allocation.
    """
    return config.bytes_for_tokens(num_tokens)


@dataclass
class MemorySnapshot:
    """Memory state at one instant.

    ``None`` means "not measurable here", not zero. On a CPU-only run the GPU
    fields are ``None``.
    """

    device_type: str
    device_current_bytes: int | None
    device_peak_bytes: int | None
    cache_native_bytes: int
    cache_on_device_bytes: int
    cache_offloaded_bytes: int
    cache_total_bytes: int

    @property
    def compression_ratio(self) -> float:
        """Native bytes per resident byte. 1.0 means no compression."""
        if self.cache_total_bytes <= 0:
            return 1.0
        return self.cache_native_bytes / self.cache_total_bytes

    @property
    def offload_fraction(self) -> float:
        """Share of cached bytes that are off the compute device."""
        if self.cache_total_bytes <= 0:
            return 0.0
        return self.cache_offloaded_bytes / self.cache_total_bytes

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compression_ratio"] = self.compression_ratio
        data["offload_fraction"] = self.offload_fraction
        return data

    def summary(self) -> str:
        peak = (
            "n/a"
            if self.device_peak_bytes is None
            else f"{self.device_peak_bytes / 1024**2:.1f}MiB"
        )
        return (
            f"cache native={self.cache_native_bytes / 1024**2:.2f}MiB "
            f"resident={self.cache_total_bytes / 1024**2:.2f}MiB "
            f"(device={self.cache_on_device_bytes / 1024**2:.2f}MiB, "
            f"offloaded={self.cache_offloaded_bytes / 1024**2:.2f}MiB) "
            f"ratio={self.compression_ratio:.2f}x gpu_peak={peak}"
        )


def snapshot(cache: KVCache, device: str | torch.device | None = None) -> MemorySnapshot:
    """Capture memory state for ``cache`` and its device."""
    resolved = resolve_device(device) if device is not None else cache.store.device
    stats = cache.stats()
    return MemorySnapshot(
        device_type=resolved.type,
        device_current_bytes=current_memory_bytes(resolved),
        device_peak_bytes=peak_memory_bytes(resolved),
        cache_native_bytes=cache.store.uncompressed_bytes(),
        cache_on_device_bytes=stats.bytes_on_device,
        cache_offloaded_bytes=stats.bytes_offloaded,
        cache_total_bytes=stats.bytes_total,
    )


__all__ = ["MemorySnapshot", "snapshot", "theoretical_kv_bytes"]
