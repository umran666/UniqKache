"""Exception hierarchy for UniqKache.

A guiding rule for this project: **fail loudly, never silently.** Cache
management bugs that corrupt state without raising are the most expensive class
of bug in this domain, because they produce plausible-looking but wrong outputs.
Every failure mode that we can detect is raised as a typed exception here rather
than being logged and swallowed.
"""

from __future__ import annotations


class UniqKacheError(Exception):
    """Base class for all errors raised by UniqKache."""


class CacheConfigError(UniqKacheError, ValueError):
    """Raised when a :class:`~uniqkache.cache.types.CacheConfig` is invalid.

    Examples: non-positive layer count, ``attention_sinks`` larger than the
    capacity, or a capacity that cannot hold the protected tokens.
    """


class CacheStateError(UniqKacheError, RuntimeError):
    """Raised when the cache is used in an inconsistent state.

    Examples: reading a layer that was never written, appending a token count
    that disagrees with the recorded positions, or requesting a token index that
    has already been evicted.
    """


class PolicyError(UniqKacheError, RuntimeError):
    """Raised when a cache policy violates its contract.

    Examples: returning a score vector whose length does not match the number of
    cached tokens, or selecting more indices than the budget permits.
    """


class BackendError(UniqKacheError, RuntimeError):
    """Raised when a model backend cannot be constructed or driven."""


class ConfigError(UniqKacheError, ValueError):
    """Raised when an experiment/benchmark configuration is malformed."""


__all__ = [
    "BackendError",
    "CacheConfigError",
    "CacheStateError",
    "ConfigError",
    "PolicyError",
    "UniqKacheError",
]
