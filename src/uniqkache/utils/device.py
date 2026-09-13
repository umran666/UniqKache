"""Device and memory introspection.

All hardware probing goes through this module so that:

* results are **honest** — when we cannot measure something (for example peak
  GPU memory on a machine without CUDA), we report ``None`` rather than ``0``,
  which would silently look like a perfect result;
* benchmark records carry a consistent description of the machine.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any

import torch

from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a user-supplied device string to a :class:`torch.device`.

    ``None`` or ``"auto"`` selects CUDA when it is genuinely usable, otherwise
    CPU. We verify CUDA by actually touching the device, because a machine can
    report a CUDA build of PyTorch while having no usable device.
    """
    if device is None or (isinstance(device, str) and device == "auto"):
        return torch.device("cuda") if cuda_is_available() else torch.device("cpu")
    return torch.device(device)


def cuda_is_available() -> bool:
    """Return True only if CUDA is compiled in *and* a device can be queried."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.current_device()
        torch.cuda.get_device_properties(0)
    except Exception as exc:  # pragma: no cover - depends on broken drivers
        # Not silent: a machine with a half-installed driver is a real
        # environment problem and the user needs to know we fell back to CPU.
        _log.warning("CUDA reported available but unusable (%s); falling back to CPU", exc)
        return False
    return True


def synchronize(device: torch.device) -> None:
    """Block until all queued work on ``device`` has completed.

    Essential for wall-clock timing: CUDA kernels are asynchronous, so timing
    without a synchronise barrier measures the launch overhead, not the work.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA peak-memory high-water mark, if applicable."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    """Return peak allocated memory in bytes, or ``None`` when unmeasurable.

    Returning ``None`` on CPU is intentional: reporting ``0`` would be read as
    "used no memory", which is false.
    """
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def current_memory_bytes(device: torch.device) -> int | None:
    """Return currently allocated memory in bytes, or ``None`` on CPU."""
    if device.type != "cuda":
        return None
    return int(torch.cuda.memory_allocated(device))


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Exact storage size of a tensor in bytes."""
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class HardwareInfo:
    """A machine-readable description of the execution hardware."""

    device_type: str
    device_name: str
    device_count: int
    total_memory_bytes: int | None
    compute_capability: str | None
    torch_version: str
    cuda_version: str | None
    platform: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_hardware(device: torch.device | None = None) -> HardwareInfo:
    """Collect hardware metadata for a benchmark record.

    Fields that cannot be determined are set to ``None`` and are recorded as
    such, so that a reader can tell "CPU-only run" apart from "measurement
    failed" apart from "measured zero".
    """
    device = resolve_device(device)
    device_name = platform.processor() or "unknown"
    device_count = 1
    total_memory: int | None = None
    capability: str | None = None

    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(device)
            device_name = props.name
            device_count = torch.cuda.device_count()
            total_memory = int(props.total_memory)
            major, minor = torch.cuda.get_device_capability(device)
            capability = f"{major}.{minor}"
        except Exception as exc:  # pragma: no cover - driver-dependent
            _log.warning("Could not read CUDA device properties: %s", exc)
    else:
        device_name = platform.machine() or device_name

    return HardwareInfo(
        device_type=device.type,
        device_name=device_name,
        device_count=device_count,
        total_memory_bytes=total_memory,
        compute_capability=capability,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        platform=platform.platform(),
        python_version=platform.python_version(),
    )


__all__ = [
    "HardwareInfo",
    "cuda_is_available",
    "current_memory_bytes",
    "describe_hardware",
    "peak_memory_bytes",
    "reset_peak_memory",
    "resolve_device",
    "synchronize",
    "tensor_bytes",
]
