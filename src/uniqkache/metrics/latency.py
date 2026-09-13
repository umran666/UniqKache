"""Latency measurement.

Timing on an accelerator is easy to get wrong in a way that flatters the result.
Two rules are enforced here:

**Synchronise before reading the clock.** CUDA kernel launches are asynchronous,
so a ``perf_counter`` reading taken without a device synchronisation measures how
long it took to *enqueue* the work, not how long the work took. Every timing
boundary in this module synchronises first.

**Report the distribution, not just the mean.** A mean TTFT hides the tail that
users actually notice, and a policy that improves the mean while wrecking p99 is
not obviously an improvement. :class:`LatencyStats` reports percentiles.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from uniqkache.utils.device import resolve_device, synchronize


class Timer:
    """Wall-clock timer that synchronises an accelerator before reading.

    Example
    -------
    .. code-block:: python

        with Timer(device) as t:
            model.forward(...)
        print(t.elapsed_ms)
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = resolve_device(device)
        self._start: float | None = None
        self.elapsed_ms: float | None = None

    def __enter__(self) -> Timer:
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # Synchronise even when the body raised, so the elapsed time is still
        # meaningful and the device is not left with queued work that would
        # corrupt the *next* measurement.
        synchronize(self.device)
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._start = None


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile of ``values``, or ``None`` when empty.

    ``None`` rather than ``0.0``: an empty sample has no percentile, and
    reporting zero would look like an instantaneous result.
    """
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass
class LatencyStats:
    """Summary of a set of latency samples, in milliseconds."""

    count: int
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    p50_ms: float | None
    p90_ms: float | None
    p99_ms: float | None
    total_ms: float | None
    samples: list[float] = field(default_factory=list)

    @classmethod
    def from_samples(cls, samples: list[float], *, keep_samples: bool = False) -> LatencyStats:
        """Build stats from millisecond samples.

        Parameters
        ----------
        samples:
            Latency samples in milliseconds.
        keep_samples:
            Whether to retain the raw samples. Off by default: a per-token
            sample list for a 128k-context run is large and is not needed to
            reproduce the summary.
        """
        if not samples:
            return cls(
                count=0,
                mean_ms=None,
                min_ms=None,
                max_ms=None,
                p50_ms=None,
                p90_ms=None,
                p99_ms=None,
                total_ms=None,
                samples=[],
            )
        return cls(
            count=len(samples),
            mean_ms=sum(samples) / len(samples),
            min_ms=min(samples),
            max_ms=max(samples),
            p50_ms=percentile(samples, 0.50),
            p90_ms=percentile(samples, 0.90),
            p99_ms=percentile(samples, 0.99),
            total_ms=sum(samples),
            samples=list(samples) if keep_samples else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.count == 0:
            return "latency: no samples"
        return (
            f"n={self.count} mean={self.mean_ms:.3f}ms p50={self.p50_ms:.3f}ms "
            f"p90={self.p90_ms:.3f}ms p99={self.p99_ms:.3f}ms max={self.max_ms:.3f}ms"
        )


def throughput(tokens: int, elapsed_ms: float) -> float | None:
    """Tokens per second, or ``None`` when the measurement is degenerate."""
    if elapsed_ms <= 0 or tokens <= 0:
        return None
    return tokens / (elapsed_ms / 1000.0)


__all__ = ["LatencyStats", "Timer", "percentile", "throughput"]
