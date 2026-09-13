"""Shared utilities: errors, logging, device probing, determinism."""

from __future__ import annotations

from uniqkache.utils.device import (
    HardwareInfo,
    cuda_is_available,
    describe_hardware,
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    synchronize,
    tensor_bytes,
)
from uniqkache.utils.errors import (
    BackendError,
    CacheConfigError,
    CacheStateError,
    ConfigError,
    PolicyError,
    UniqKacheError,
)
from uniqkache.utils.logging import get_logger
from uniqkache.utils.seed import set_seed

__all__ = [
    "BackendError",
    "CacheConfigError",
    "CacheStateError",
    "ConfigError",
    "HardwareInfo",
    "PolicyError",
    "UniqKacheError",
    "cuda_is_available",
    "describe_hardware",
    "get_logger",
    "peak_memory_bytes",
    "reset_peak_memory",
    "resolve_device",
    "set_seed",
    "synchronize",
    "tensor_bytes",
]
