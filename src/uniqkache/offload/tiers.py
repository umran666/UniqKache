"""Memory tiers and offload planning.

Offloading moves cached bytes from the compute device to a cheaper tier. Two
things make it easy to misreport, and this module is built to avoid both:

1. **Offloading is not a memory reduction.** The bytes still exist; they are
   resident somewhere else. :class:`~uniqkache.cache.types.CacheStats` therefore
   reports ``bytes_on_device`` and ``bytes_offloaded`` separately, and
   :meth:`TierManager.plan_offload` reports the *device* saving it achieves, not
   a total saving.
2. **Offloading is not free.** It trades memory for transfer time. The cost
   depends on interconnect bandwidth, which is hardware-specific. We do **not**
   ship a table of bandwidth constants, because inventing one would manufacture
   a hardware comparison we never measured. Tiers carry ``bandwidth_gbps=None``
   unless the caller supplies a measured value, and any latency model that needs
   bandwidth must handle its absence explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniqkache.cache.store import KVStore
from uniqkache.utils.device import resolve_device
from uniqkache.utils.errors import UniqKacheError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

DEVICE_TIER = "device"
HOST_TIER = "host"
DISK_TIER = "disk"


@dataclass(frozen=True)
class Tier:
    """A place bytes can live.

    Attributes
    ----------
    name:
        ``"device"``, ``"host"`` or ``"disk"``.
    device:
        The :class:`torch.device` backing this tier, or ``None`` for a tier not
        addressable as a tensor device (disk).
    bandwidth_gbps:
        Measured interconnect bandwidth, or ``None`` when unknown. ``None`` is
        the honest default: we have not measured it on the machine running this
        code, and a fabricated constant would silently corrupt any cost model
        built on it.
    """

    name: str
    device: torch.device | None
    bandwidth_gbps: float | None = None

    @property
    def is_addressable(self) -> bool:
        """Whether tensors can be moved to this tier by ``Tensor.to``."""
        return self.device is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device": None if self.device is None else str(self.device),
            "bandwidth_gbps": self.bandwidth_gbps,
        }


@dataclass
class OffloadPlan:
    """Which layers to offload, and what that would achieve."""

    layers: list[int]
    device_bytes_before: int
    device_bytes_after: int
    bytes_moved: int
    feasible: bool
    reason: str

    @property
    def device_saving(self) -> int:
        return self.device_bytes_before - self.device_bytes_after

    def to_dict(self) -> dict[str, object]:
        return {
            "layers": list(self.layers),
            "device_bytes_before": self.device_bytes_before,
            "device_bytes_after": self.device_bytes_after,
            "bytes_moved": self.bytes_moved,
            "device_saving": self.device_saving,
            "feasible": self.feasible,
            "reason": self.reason,
        }


class TierManager:
    """Plans movement of cached bytes between memory tiers.

    Parameters
    ----------
    device:
        The compute device (the tier bytes want to be in).
    host:
        The host tier device, normally CPU.
    """

    def __init__(self, device: str | torch.device, host: str | torch.device = "cpu") -> None:
        self.device = resolve_device(device)
        self.host = resolve_device(host)
        if self.device == self.host:
            raise UniqKacheError(
                f"compute device and host tier are both {self.device}; there is no cheaper "
                "tier to offload to. Construct the cache on a CUDA device to exercise offloading."
            )
        self.device_tier = Tier(DEVICE_TIER, self.device)
        self.host_tier = Tier(HOST_TIER, self.host)

    def tiers(self) -> list[Tier]:
        return [self.device_tier, self.host_tier]

    def plan_offload(
        self,
        store: KVStore,
        byte_budget: int,
        *,
        order: str = "lru",
        exclude: set[int] | None = None,
    ) -> OffloadPlan:
        """Choose layers to offload so device-resident bytes fit ``byte_budget``.

        Parameters
        ----------
        store:
            The cache storage to plan against.
        byte_budget:
            Maximum bytes permitted to remain device-resident.
        order:
            Selection order. ``"lru"`` offloads the layers whose tokens were
            least recently attended first — the intuition being that a layer
            whose cached tokens are stale is the cheapest to evict from the fast
            tier. ``"largest"`` offloads the biggest layers first, which reaches
            the budget in fewest transfers. ``"reverse"`` offloads the highest
            layer index first.
        exclude:
            Layer indices that must stay resident (for example, the layer about
            to be computed).

        Returns
        -------
        OffloadPlan
            ``feasible`` is False when even offloading every eligible layer
            cannot meet the budget. Reporting infeasibility is the point: the
            caller must decide to evict or compress instead, rather than the
            planner quietly under-delivering.
        """
        if byte_budget < 0:
            raise UniqKacheError(f"byte_budget must be >= 0, got {byte_budget}")
        if order not in {"lru", "largest", "reverse"}:
            raise UniqKacheError(f"unknown order {order!r}; expected 'lru', 'largest' or 'reverse'")

        excluded = exclude or set()
        device_bytes_before = store.bytes_on_device()
        resident = [
            layer
            for layer in store.layers
            if not layer.is_offloaded and layer.layer_idx not in excluded and layer.bytes() > 0
        ]

        if device_bytes_before <= byte_budget:
            return OffloadPlan(
                layers=[],
                device_bytes_before=device_bytes_before,
                device_bytes_after=device_bytes_before,
                bytes_moved=0,
                feasible=True,
                reason=f"device bytes {device_bytes_before} already within budget {byte_budget}",
            )

        if order == "largest":
            ranked = sorted(resident, key=lambda layer: layer.bytes(), reverse=True)
        elif order == "reverse":
            ranked = sorted(resident, key=lambda layer: layer.layer_idx, reverse=True)
        else:
            ranked = sorted(resident, key=self._staleness)

        chosen: list[int] = []
        remaining = device_bytes_before
        for layer in ranked:
            if remaining <= byte_budget:
                break
            chosen.append(layer.layer_idx)
            remaining -= layer.bytes()

        feasible = remaining <= byte_budget
        reason = (
            f"offload {len(chosen)} layer(s) to reach {remaining} device bytes "
            f"within budget {byte_budget}"
            if feasible
            else (
                f"cannot reach budget {byte_budget}: even offloading all "
                f"{len(chosen)} eligible layer(s) leaves {remaining} device bytes. "
                "Evict or compress instead, or raise the budget."
            )
        )
        if not feasible:
            _log.warning(
                "offload plan infeasible: budget=%d, achievable=%d, eligible_layers=%d",
                byte_budget,
                remaining,
                len(chosen),
            )

        return OffloadPlan(
            layers=chosen,
            device_bytes_before=device_bytes_before,
            device_bytes_after=remaining,
            bytes_moved=device_bytes_before - remaining,
            feasible=feasible,
            reason=reason,
        )

    @staticmethod
    def _staleness(layer: object) -> int:
        """Largest last-access step in a layer; older means staler.

        Sorting ascending puts the stalest layer first. A layer with no tokens
        is treated as maximally stale so it is offloaded first — it costs
        nothing to move and frees whatever it holds.
        """
        metadata = layer.metadata  # type: ignore[attr-defined]
        if metadata.num_tokens == 0:
            return -(2**62)
        return int(metadata.last_access.max().item())


__all__ = ["DEVICE_TIER", "DISK_TIER", "HOST_TIER", "OffloadPlan", "Tier", "TierManager"]
