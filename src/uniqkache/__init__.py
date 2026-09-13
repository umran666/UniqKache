"""UniqKache — a research framework for adaptive KV-cache management.

UniqKache studies one question: **can an LLM dynamically decide what inference
information to retain, compress, offload, recompute or prefetch, according to
context, model behaviour, memory pressure, latency requirements and quality
constraints — and can that decision be shown to help?**

The package is organised so that the answer is *earnable* rather than assumed:

* :mod:`uniqkache.cache` — policy-agnostic K/V storage and the cache facade;
* :mod:`uniqkache.policies` — pluggable retention policies, including faithful
  reproductions of published methods;
* :mod:`uniqkache.compression` — representation changes that preserve occupancy;
* :mod:`uniqkache.offload` / :mod:`uniqkache.prefetch` — memory-tier movement;
* :mod:`uniqkache.controllers` — the action-selection prototype (research);
* :mod:`uniqkache.runtime` — an explicit attention-over-cache inference loop;
* :mod:`uniqkache.metrics` — memory, latency and quality measurement;
* :mod:`uniqkache.bench` — the reproducible benchmark runner and CLI.

Stability labels used throughout
--------------------------------
``Stable``
    Covered by tests, interface expected to hold.
``Experimental``
    Works, interface may change, results not yet characterised.
``Research prototype``
    An unvalidated hypothesis. Presence is not evidence of benefit.
``Not yet supported``
    Documented gap. Explicitly out of scope for the current milestone.
"""

from __future__ import annotations

__version__ = "0.0.1"

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig, CacheStats, PolicyState
from uniqkache.policies import (
    AdaptivePolicy,
    AttentionBasedPolicy,
    BaseCachePolicy,
    CachePolicy,
    FullCachePolicy,
    LRUPolicy,
    SlidingWindowPolicy,
    TokenImportancePolicy,
    available_policies,
    build_policy,
)

__all__ = [
    "AdaptivePolicy",
    "AttentionBasedPolicy",
    "BaseCachePolicy",
    "CacheConfig",
    "CachePolicy",
    "CacheStats",
    "FullCachePolicy",
    "KVCache",
    "LRUPolicy",
    "PolicyState",
    "SlidingWindowPolicy",
    "TokenImportancePolicy",
    "__version__",
    "available_policies",
    "build_policy",
]
