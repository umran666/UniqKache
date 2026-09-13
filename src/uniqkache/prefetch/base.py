"""Prefetch policy contract.

What prefetching can honestly promise
-------------------------------------
A standard transformer executes its layers in a fixed ascending order. That is a
*deterministic* access pattern, which makes "prefetch the next layer" a real,
implementable optimisation: the transfer can overlap with the current layer's
compute, so the layer is resident by the time it is needed.

Anything beyond the next layer requires predicting which layer will be needed
after that, which is only interesting if the execution order is *not* fixed
(early exit, layer skipping, speculative decoding, multi-request interleaving).
That is a research question, not an established technique, and this module does
not pretend otherwise.

Accordingly the built-in policies are deliberately modest:

* :class:`~uniqkache.prefetch.sequential.NextLayerPrefetch` — deterministic
  lookahead, valid for any standard transformer.
* :class:`~uniqkache.prefetch.recency.RecencyPrefetch` — bring back the
  most-recently-used offloaded layers, useful when requests revisit layers.

Neither claims a latency benefit that has not been measured. The interface
returns a *plan*; measuring whether the plan pays for itself is the benchmark's
job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from uniqkache.cache.store import KVStore


@dataclass
class PrefetchPlan:
    """Layers to bring back to the compute device, in the order to fetch them."""

    layers: list[int] = field(default_factory=list)
    bytes_to_move: int = 0
    reason: str = ""

    def is_empty(self) -> bool:
        return not self.layers

    def to_dict(self) -> dict[str, object]:
        return {
            "layers": list(self.layers),
            "bytes_to_move": self.bytes_to_move,
            "reason": self.reason,
        }


@runtime_checkable
class PrefetchPolicy(Protocol):
    """Structural type for a prefetch policy."""

    name: str

    def plan(self, store: KVStore, cursor: int, resident_budget: int | None) -> PrefetchPlan:
        """Return which offloaded layers to bring back, and in what order.

        Parameters
        ----------
        store:
            Current cache storage.
        cursor:
            The layer about to be computed.
        resident_budget:
            Optional cap on additional device-resident bytes. ``None`` means no
            cap.
        """


class BasePrefetchPolicy(ABC):
    """Base class with the shared byte-accounting helper."""

    name: str = "base"

    @abstractmethod
    def plan(self, store: KVStore, cursor: int, resident_budget: int | None) -> PrefetchPlan:
        """Return a prefetch plan."""

    @staticmethod
    def _size_of(store: KVStore, layer_idx: int) -> int:
        return store.layer(layer_idx).bytes()

    def state_dict(self) -> dict[str, object]:
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class NoPrefetch(BasePrefetchPolicy):
    """Do nothing.

    The correct default. Prefetching costs bandwidth and adds complexity; it
    should only be enabled once a measurement shows it helps.
    """

    name = "none"

    def plan(self, store: KVStore, cursor: int, resident_budget: int | None) -> PrefetchPlan:
        return PrefetchPlan(layers=[], bytes_to_move=0, reason="prefetching disabled")


__all__ = ["BasePrefetchPolicy", "NoPrefetch", "PrefetchPlan", "PrefetchPolicy"]
