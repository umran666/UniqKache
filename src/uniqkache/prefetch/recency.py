"""Recency-based prefetching."""

from __future__ import annotations

from uniqkache.cache.store import KVStore
from uniqkache.prefetch.base import BasePrefetchPolicy, PrefetchPlan
from uniqkache.prefetch.registry import register_prefetch_policy


@register_prefetch_policy
class RecencyPrefetch(BasePrefetchPolicy):
    """Bring back the ``k`` most recently used offloaded layers.

    Useful when execution revisits layers — multi-request interleaving, or a
    decode loop that re-enters the same layers repeatedly — so that the layers
    with the freshest access history are the ones made resident.

    Parameters
    ----------
    k:
        Maximum number of layers to bring back per call.

    Notes
    -----
    Recency is taken from each layer's ``last_access`` metadata, which is
    maintained by the cache for every policy. This policy therefore needs no
    attention signal of its own and works even with ``NoPrefetch``-style
    bookkeeping.
    """

    name = "recency"

    def __init__(self, k: int = 1) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = k

    def plan(self, store: KVStore, cursor: int, resident_budget: int | None) -> PrefetchPlan:
        if self.k == 0:
            return PrefetchPlan(layers=[], bytes_to_move=0, reason="k=0")

        offloaded = [layer for layer in store.layers if layer.is_offloaded and layer.bytes() > 0]
        if not offloaded:
            return PrefetchPlan(layers=[], bytes_to_move=0, reason="no offloaded layers")

        def recency(layer: object) -> int:
            metadata = layer.metadata  # type: ignore[attr-defined]
            if metadata.num_tokens == 0:
                return -(2**62)
            return int(metadata.last_access.max().item())

        ranked = sorted(offloaded, key=recency, reverse=True)

        selected: list[int] = []
        moved = 0
        for layer in ranked:
            if len(selected) >= self.k:
                break
            size = layer.bytes()
            if resident_budget is not None and moved + size > resident_budget:
                continue
            selected.append(layer.layer_idx)
            moved += size

        reason = (
            f"prefetch {len(selected)} most-recently-used offloaded layer(s)"
            if selected
            else "no layer fits the resident budget"
        )
        return PrefetchPlan(layers=selected, bytes_to_move=moved, reason=reason)

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data["k"] = self.k
        return data


__all__ = ["RecencyPrefetch"]
