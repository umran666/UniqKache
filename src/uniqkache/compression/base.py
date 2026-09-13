"""Compression contract.

Compression changes the **representation** of cached tokens, never the set of
tokens that are cached. That invariant is what lets a controller treat
"compress" and "evict" as genuinely different actions:

* ``evict`` reduces occupancy — fewer tokens, information destroyed.
* ``compress`` keeps occupancy constant — every token still readable, precision
  reduced.

A compressor that silently dropped tokens would make the two indistinguishable
and would corrupt any ablation that compares them, so the store enforces the
occupancy invariant (see :meth:`uniqkache.cache.store.LayerStorage.replace`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class CompressionResult:
    """Outcome of compressing one layer, with the accounting needed to report it."""

    keys: torch.Tensor | object
    values: torch.Tensor | object
    num_tokens: int
    uncompressed_bytes: int
    compressed_bytes: int

    @property
    def ratio(self) -> float:
        """Uncompressed bytes per compressed byte. 1.0 means no saving."""
        if self.compressed_bytes <= 0:
            return 0.0
        return self.uncompressed_bytes / self.compressed_bytes


@runtime_checkable
class Compressor(Protocol):
    """Structural type for a KV compressor."""

    name: str

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> CompressionResult:
        """Return a compressed representation of ``keys`` and ``values``."""

    def decompress(self, result: CompressionResult) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover approximate ``(keys, values)`` from a compressed result."""


class BaseCompressor(ABC):
    """Base class providing the occupancy invariant check.

    Subclasses implement :meth:`compress` and :meth:`decompress`. The base class
    verifies that the compressed representation still describes every input
    token, raising rather than returning a quietly lossy result.
    """

    name: str = "base"
    lossy: bool = True

    @abstractmethod
    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> CompressionResult:
        """Compress K/V tensors of shape ``[batch, heads, seq, head_dim]``."""

    @abstractmethod
    def decompress(self, result: CompressionResult) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress back to ``[batch, heads, seq, head_dim]`` tensors."""

    @staticmethod
    def _check_inputs(keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.shape != values.shape:
            raise ValueError(
                f"keys and values must share a shape, got {tuple(keys.shape)} and "
                f"{tuple(values.shape)}"
            )
        if keys.dim() != 4:
            raise ValueError(
                f"K/V tensors must be [batch, heads, seq, head_dim], got {tuple(keys.shape)}"
            )

    def state_dict(self) -> dict[str, object]:
        return {"name": self.name, "lossy": self.lossy}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


__all__ = ["BaseCompressor", "CompressionResult", "Compressor"]
