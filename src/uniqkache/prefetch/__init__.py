"""Prefetching: bringing offloaded bytes back to the compute device."""

from __future__ import annotations

from uniqkache.prefetch.base import (
    BasePrefetchPolicy,
    NoPrefetch,
    PrefetchPlan,
    PrefetchPolicy,
)
from uniqkache.prefetch.recency import RecencyPrefetch
from uniqkache.prefetch.registry import (
    available_prefetch_policies,
    build_prefetch_policy,
    register_prefetch_policy,
)
from uniqkache.prefetch.sequential import NextLayerPrefetch

# NoPrefetch lives in base.py, which the registry cannot import from without a
# cycle, so it is registered here where both are available.
register_prefetch_policy(NoPrefetch)

__all__ = [
    "BasePrefetchPolicy",
    "NextLayerPrefetch",
    "NoPrefetch",
    "PrefetchPlan",
    "PrefetchPolicy",
    "RecencyPrefetch",
    "available_prefetch_policies",
    "build_prefetch_policy",
    "register_prefetch_policy",
]
