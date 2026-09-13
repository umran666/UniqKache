"""Deterministic lookahead prefetching for standard transformer execution."""

from __future__ import annotations

from uniqkache.cache.store import KVStore
from uniqkache.prefetch.base import BasePrefetchPolicy, PrefetchPlan
from uniqkache.prefetch.registry import register_prefetch_policy


@register_prefetch_policy
class NextLayerPrefetch(BasePrefetchPolicy):
    """Bring back the next ``depth`` offloaded layers ahead of the cursor.

    Valid for any model whose layers execute in ascending index order, which
    covers every standard transformer. The benefit, if any, is that the transfer
    overlaps with the current layer's compute instead of blocking on it.

    Parameters
    ----------
    depth:
        How many layers ahead to prefetch. ``1`` hides one layer's transfer
        behind one layer's compute. Larger values cost more device memory
        without a correspondingly larger overlap window, so the default is 1 and
        the parameter exists for the ablation.
    wrap:
        When True, lookahead wraps past the last layer back to layer 0, for
        workloads that run many sequences back to back through the same cache.

    Notes
    -----
    This policy prefetches *only* what the execution order already guarantees.
    It makes no prediction about model behaviour, so it cannot be credited with
    an adaptive or learned contribution.
    """

    name = "next_layer"

    def __init__(self, depth: int = 1, wrap: bool = False) -> None:
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        self.depth = depth
        self.wrap = wrap

    def plan(self, store: KVStore, cursor: int, resident_budget: int | None) -> PrefetchPlan:
        if self.depth == 0:
            return PrefetchPlan(layers=[], bytes_to_move=0, reason="depth=0")

        num_layers = len(store)
        candidates: list[int] = []
        for offset in range(1, self.depth + 1):
            idx = cursor + offset
            if idx >= num_layers:
                if not self.wrap:
                    break
                idx = idx % num_layers
            layer = store.layer(idx)
            if layer.is_offloaded and layer.bytes() > 0:
                candidates.append(idx)

        if not candidates:
            return PrefetchPlan(
                layers=[],
                bytes_to_move=0,
                reason=f"no offloaded layers within lookahead depth {self.depth} of cursor {cursor}",
            )

        selected: list[int] = []
        moved = 0
        for idx in candidates:
            size = self._size_of(store, idx)
            if resident_budget is not None and moved + size > resident_budget:
                break
            selected.append(idx)
            moved += size

        reason = (
            f"prefetch layers {selected} ahead of cursor {cursor}"
            if selected
            else f"resident budget {resident_budget} too small for the next offloaded layer"
        )
        return PrefetchPlan(layers=selected, bytes_to_move=moved, reason=reason)

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data.update({"depth": self.depth, "wrap": self.wrap})
        return data


__all__ = ["NextLayerPrefetch"]
