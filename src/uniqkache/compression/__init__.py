"""KV compression: representation changes that preserve occupancy.

See :mod:`uniqkache.compression.quantize` for the prior-work statement on the
integer quantiser.
"""

from __future__ import annotations

from uniqkache.compression.base import BaseCompressor, CompressionResult, Compressor
from uniqkache.compression.quantize import Int8KVCompressor, QuantizedTensor, quantize

__all__ = [
    "BaseCompressor",
    "CompressionResult",
    "Compressor",
    "Int8KVCompressor",
    "QuantizedTensor",
    "quantize",
]
