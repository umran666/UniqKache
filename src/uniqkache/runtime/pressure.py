"""Adaptive eviction budgets driven by observed device memory pressure.

Today the token budget (``capacity`` / ``keep_ratio`` / ``memory_budget_mb``)
is fixed at config time, and ``PolicyState.memory_pressure`` is derived from
that static budget. This module adds the missing feedback loop: it watches the
device's *actual* memory occupancy and recomputes the token budget from it
through :meth:`~uniqkache.cache.types.CacheConfig.tokens_for_bytes`, so a run
that would exhaust the device at the initial budget tightens itself instead of
dying.

The mapping
-----------
``current_memory_bytes()`` reports *total* device allocation — weights,
activations and cache together — while a user's byte ceiling is usually a
whole-device ceiling. The monitor therefore reserves the non-cache part first
and gives the cache the remainder:

    target_cache_bytes = max(0, budget - (observed_total - cache_bytes))

On a CPU-only run the allocator figure is unavailable (``current_memory_bytes``
returns ``None`` there, by design — see :mod:`uniqkache.utils.device`), so the
non-cache part is taken as zero and pressure is measured on the cache's own
footprint. That is the honest fallback: it is the only figure the machine can
actually report.

Two stability rules
-------------------
* **Shrink only.** Tightening happens when pressure reaches the high watermark;
  above it the budget is recomputed from the observed bytes. Growth is
  deliberately *not* implemented: a monitor that expands the budget would
  oscillate — expand, breach, shrink, relax, expand — and every oscillation is
  a wasted eviction. Shrink-only is stable by construction, and the event log
  records every tightening so the realised budget over time is auditable.
* **Minimum interval.** At most one adaptation per ``min_interval`` steps, so a
  burst of pressure cannot trigger a burst of re-plans.

Every adaptation is recorded as an :class:`AdaptationEvent` with the step, the
observed bytes, the pressure, and the old and new capacities. The benchmark
record carries the list, so "the budget was 16 the whole run" is checkable
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.utils.errors import ConfigError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class AdaptationEvent:
    """One budget tightening, recorded so the trace is auditable."""

    step: int
    observed_bytes: int
    cache_bytes: int
    pressure: float
    old_capacity: int | None
    new_capacity: int

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "observed_bytes": self.observed_bytes,
            "cache_bytes": self.cache_bytes,
            "pressure": self.pressure,
            "old_capacity": self.old_capacity,
            "new_capacity": self.new_capacity,
        }


class MemoryPressureMonitor:
    """Recompute the cache's token budget from observed device pressure.

    Parameters
    ----------
    config:
        The cache's configuration. Its ``tokens_for_bytes`` is the bridge from
        a byte budget to a token capacity, and its shape fixes the bytes per
        token. A per-layer capacity list is refused at construction: the
        monitor recomputes a single scalar budget, and silently picking one
        layer's budget would make the record ambiguous.
    device_byte_budget:
        The user's byte ceiling for device memory.
    high_watermark:
        Pressure fraction at or above which the budget is tightened.
    low_watermark:
        Pressure fraction the tightened budget aims *below*, so the new budget
        lands inside the band rather than on its edge: a budget that exactly
        meets the ceiling leaves the next observation at pressure again, while
        one aimed at the low watermark is quiet until the sequence grows.
    min_interval:
        Minimum decode steps between adaptations.

    Notes
    -----
    The monitor only ever *tightens*. See the module docstring for why growth
    is not implemented.
    """

    def __init__(
        self,
        config: CacheConfig,
        device_byte_budget: int,
        *,
        high_watermark: float = 0.9,
        low_watermark: float = 0.75,
        min_interval: int = 8,
    ) -> None:
        if device_byte_budget <= 0:
            raise ConfigError(f"device_byte_budget must be > 0, got {device_byte_budget}")
        if not 0.0 < low_watermark < high_watermark <= 1.0:
            raise ConfigError(
                f"watermarks must satisfy 0 < low ({low_watermark}) < high ({high_watermark}) <= 1"
            )
        if min_interval < 1:
            raise ConfigError(f"min_interval must be >= 1, got {min_interval}")
        if config.capacity is not None and not isinstance(config.capacity, int):
            raise ConfigError(
                "MemoryPressureMonitor recomputes a single scalar token budget, but this "
                "cache carries a per-layer capacity list; adaptive eviction does not define "
                "how to split a tightened budget across layers"
            )

        self.config = config
        self.device_byte_budget = int(device_byte_budget)
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.min_interval = min_interval
        self.events: list[AdaptationEvent] = []
        self._last_adaptation_step: int | None = None

    @property
    def target_capacity(self) -> int:
        """The token capacity the budget implies at the low watermark."""
        return self.config.tokens_for_bytes(int(self.device_byte_budget * self.low_watermark))

    def observe(
        self,
        *,
        step: int,
        cache_bytes: int,
        current_capacity: int | None,
        total_device_bytes: int | None = None,
    ) -> int | None:
        """Pure decision function: does this observation force a tightening?

        Parameters
        ----------
        step:
            Current decode step, used for the minimum-interval rule.
        cache_bytes:
            Bytes the cache currently holds device-resident.
        current_capacity:
            The cache's budget right now, or ``None`` when unbounded. Passed
            by the caller rather than read from ``config`` because the cache
            owns its budget and may have changed it since construction.
        total_device_bytes:
            Total device allocation, when the platform reports it. ``None``
            means "unavailable" (CPU runs); the non-cache share is then taken
            as zero and pressure is measured on the cache's own footprint.

        Returns
        -------
        int | None
            The new capacity when pressure forced a tightening at this step,
            otherwise ``None``.
        """
        observed = cache_bytes if total_device_bytes is None else total_device_bytes
        non_cache = max(0, observed - cache_bytes)
        pressure = observed / self.device_byte_budget

        if pressure < self.high_watermark:
            return None
        if (
            self._last_adaptation_step is not None
            and step - self._last_adaptation_step < self.min_interval
        ):
            return None

        target_cache_bytes = max(0, int(self.device_byte_budget * self.low_watermark) - non_cache)
        new_capacity = self.config.tokens_for_bytes(target_cache_bytes)
        if current_capacity is not None and new_capacity >= current_capacity:
            # Already within what the observed bytes allow.
            return None

        self.events.append(
            AdaptationEvent(
                step=step,
                observed_bytes=observed,
                cache_bytes=cache_bytes,
                pressure=pressure,
                old_capacity=current_capacity,
                new_capacity=new_capacity,
            )
        )
        self._last_adaptation_step = step
        _log.info(
            "memory pressure %.2f at step %d: capacity %s -> %d (observed %d bytes, budget %d)",
            pressure,
            step,
            "unbounded" if current_capacity is None else current_capacity,
            new_capacity,
            observed,
            self.device_byte_budget,
        )
        return new_capacity

    def update(
        self,
        cache: KVCache,
        *,
        step: int,
        total_device_bytes: int | None = None,
    ) -> int | None:
        """Observe ``cache`` and tighten its budget if pressure demands it.

        Returns the new capacity when one was applied, else ``None``. The
        caller (the generation engine) invokes this between decode steps; the
        eviction the tightened budget triggers is the cache policy's job,
        exactly as for a static budget.
        """
        capacity = cache.config.capacity
        if isinstance(capacity, list):
            # Per-layer budgets are out of scope (see __init__); a strategy
            # may install one after construction.
            _log.warning(
                "memory-pressure adaptation skipped: per-layer capacity budgets are not "
                "supported by the adaptive monitor"
            )
            return None

        new_capacity = self.observe(
            step=step,
            cache_bytes=cache.store.bytes_on_device(),
            current_capacity=capacity,
            total_device_bytes=total_device_bytes,
        )
        if new_capacity is None:
            return None
        if cache.policy is None:
            # A bounded cache needs a policy to decide what survives; without
            # one the tightened budget cannot be enforced, and pretending
            # otherwise would record an eviction that never happened.
            _log.warning(
                "memory-pressure adaptation skipped: cache has no policy to enforce a "
                "bounded budget"
            )
            return None
        cache.set_capacity(new_capacity)
        return new_capacity

    def state_dict(self) -> dict[str, object]:
        return {
            "device_byte_budget": self.device_byte_budget,
            "high_watermark": self.high_watermark,
            "low_watermark": self.low_watermark,
            "min_interval": self.min_interval,
            "adaptations": [event.to_dict() for event in self.events],
        }


__all__ = ["AdaptationEvent", "MemoryPressureMonitor"]
