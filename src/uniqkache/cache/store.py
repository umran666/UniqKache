"""Physical storage for KV tensors, independent of any eviction policy.

The store answers only "where do the bytes live and how do I read/write them".
It has no opinion about *which* tokens deserve to survive — that decision
belongs to a :class:`~uniqkache.policies.base.CachePolicy`, which is applied by
:class:`~uniqkache.cache.kv_cache.KVCache`.

Two orthogonal notions of "cost"
--------------------------------
* **Occupancy** — how many tokens are cached. Changed by eviction.
* **Representation** — how many bytes each cached token occupies. Changed by
  compression.

The store keeps these strictly separate, because conflating them is the most
common way a KV-cache result becomes misleading: a method that quantises to int8
has not cached fewer tokens, and a method that evicts has not lost precision.
:meth:`LayerStorage.num_tokens` and :meth:`LayerStorage.bytes` report the two
independently, and ``stats()`` exposes both.

Known limitation (deliberate, milestone 1)
------------------------------------------
Appending uses :func:`torch.cat`, which reallocates and copies the whole layer on
every decode step: O(T) per step, O(T^2) per sequence. It is the simplest thing
that is *correct*, and correctness is the milestone-1 goal.

Consequence for readers of our numbers: measured decode throughput reflects this
reference implementation, **not** an optimised kernel. Latency comparisons in
this repository are therefore comparisons *between policies under an identical
store*, which is the quantity an ablation needs. A preallocated ring buffer is
tracked in ``docs/architecture.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

import torch

from uniqkache.cache.metadata import LayerMetadata
from uniqkache.cache.types import CacheConfig
from uniqkache.compression.base import CompressionResult
from uniqkache.compression.quantize import QuantizedTensor
from uniqkache.utils.device import tensor_bytes
from uniqkache.utils.errors import CacheStateError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


def _gather_quantized(qt: QuantizedTensor, indices: torch.Tensor) -> QuantizedTensor:
    """Gather sequence positions out of a quantised tensor, preserving precision.

    The subtlety is which affine parameters travel with a token.

    Quantisation reduces along one axis with ``keepdim=True``, so exactly one
    dimension of ``scale`` is 1. Two cases arise:

    * the reduced axis **is** the sequence axis — one scale per channel, shared
      by every token. The scale is constant along the sequence, so gathering
      tokens must leave it alone;
    * the reduced axis is something else — one scale **per token**. The scale
      varies along the sequence and must be gathered alongside the payload.

    Getting this backwards either corrupts the dequantisation of every surviving
    token or indexes past the end of a size-1 dimension. Both are silent quality
    bugs, so the shape is checked rather than assumed.
    """
    if qt.data.dim() != 4:
        raise CacheStateError(
            "quantised KV tensors must be 4-D [batch, heads, seq, head_dim], got "
            f"{tuple(qt.data.shape)}"
        )

    seq_len = int(qt.data.shape[2])
    for param_name, param in (("scale", qt.scale), ("zero_point", qt.zero_point)):
        extent = int(param.shape[2])
        if extent not in (1, seq_len):
            raise CacheStateError(
                f"quantised {param_name} has sequence extent {extent}, expected either "
                f"1 (reduced over the sequence axis) or {seq_len} (per-token)"
            )

    data = qt.data[:, :, indices, :].contiguous()
    scale = qt.scale
    zero_point = qt.zero_point
    if int(scale.shape[2]) == seq_len:
        scale = scale[:, :, indices, :].contiguous()
        zero_point = zero_point[:, :, indices, :].contiguous()

    return dataclass_replace(qt, data=data, scale=scale, zero_point=zero_point)


@dataclass
class LayerStorage:
    """K/V tensors, their optional compressed form, and per-token bookkeeping."""

    layer_idx: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    num_sinks: int
    batch_size: int

    def __post_init__(self) -> None:
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None
        self._compressed: CompressionResult | None = None
        self._offloaded = False
        self.metadata = LayerMetadata(num_sinks=self.num_sinks, device=self.device)

    # -- introspection -----------------------------------------------------

    @property
    def keys(self) -> torch.Tensor | None:
        """Materialised keys, dequantising on demand when compressed."""
        if self._keys is not None:
            return self._keys
        if self._compressed is not None:
            return self._dequantize()[0]
        return None

    @property
    def values(self) -> torch.Tensor | None:
        """Materialised values, dequantising on demand when compressed."""
        if self._values is not None:
            return self._values
        if self._compressed is not None:
            return self._dequantize()[1]
        return None

    @property
    def num_tokens(self) -> int:
        """Occupancy: how many tokens are cached."""
        return self.metadata.num_tokens

    @property
    def is_initialized(self) -> bool:
        return self._keys is not None or self._compressed is not None

    @property
    def is_compressed(self) -> bool:
        """Whether the resident representation is quantised."""
        return self._compressed is not None

    @property
    def is_offloaded(self) -> bool:
        return self._offloaded

    @property
    def resident_device(self) -> torch.device:
        """Where the tensors physically live right now."""
        if self._keys is not None:
            return self._keys.device
        if self._compressed is not None:
            return self._compressed.keys.data.device  # type: ignore[union-attr,attr-defined]
        return self.device

    def bytes(self) -> int:
        """Resident bytes held by this layer, honouring compression."""
        if self._compressed is not None:
            return int(self._compressed.compressed_bytes)
        if self._keys is None or self._values is None:
            return 0
        return tensor_bytes(self._keys) + tensor_bytes(self._values)

    def uncompressed_bytes(self) -> int:
        """Bytes this layer *would* occupy at the configured dtype."""
        if not self.is_initialized:
            return 0
        per_token = 2 * self.batch_size * self.num_kv_heads * self.head_dim
        element = torch.empty(0, dtype=self.dtype).element_size()
        return int(self.num_tokens * per_token * element)

    def compression_ratio(self) -> float:
        """Uncompressed bytes per resident byte. 1.0 when uncompressed."""
        resident = self.bytes()
        if resident <= 0:
            return 1.0
        return self.uncompressed_bytes() / resident

    # -- representation ----------------------------------------------------

    def _dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise float K/V from the compressed payload."""
        compressed = self._compressed
        if compressed is None:
            raise CacheStateError(f"layer {self.layer_idx} holds no compressed payload")
        keys, values = compressed.keys, compressed.values
        if not isinstance(keys, QuantizedTensor) or not isinstance(values, QuantizedTensor):
            raise CacheStateError(
                f"layer {self.layer_idx} holds an unrecognised compressed payload "
                f"({type(keys).__name__}); cannot dequantise"
            )
        return keys.dequantize(), values.dequantize()

    def apply_compression(self, result: CompressionResult) -> None:
        """Adopt a compressed representation, preserving occupancy."""
        if result.num_tokens != self.num_tokens:
            raise CacheStateError(
                f"compression changed occupancy for layer {self.layer_idx}: "
                f"{self.num_tokens} tokens before, {result.num_tokens} after. "
                "Compression must preserve the token set."
            )
        self._compressed = result
        self._keys = None
        self._values = None

    def materialize(self) -> None:
        """Expand a compressed representation back to float tensors."""
        if self._compressed is None:
            return
        keys, values = self._dequantize()
        self._keys = keys.to(dtype=self.dtype, device=self.resident_device)
        self._values = values.to(dtype=self.dtype, device=self.resident_device)
        self._compressed = None

    # -- mutation ----------------------------------------------------------

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        step: int = 0,
    ) -> None:
        """Append new K/V rows along the sequence dimension.

        Appending to a compressed layer materialises it first: new float rows
        cannot be concatenated onto an int8 payload without either losing the
        new rows' precision or re-quantising silently. We choose the explicit,
        visible behaviour and document it, rather than quantising behind the
        caller's back.
        """
        if keys.shape != values.shape:
            raise CacheStateError(
                f"keys and values must share a shape, got {tuple(keys.shape)} vs "
                f"{tuple(values.shape)}"
            )
        if keys.dim() != 4:
            raise CacheStateError(
                "K/V tensors must be [batch, num_kv_heads, seq, head_dim], got "
                f"{tuple(keys.shape)} for layer {self.layer_idx}"
            )
        if keys.shape[1] != self.num_kv_heads:
            raise CacheStateError(
                f"layer {self.layer_idx} expects {self.num_kv_heads} kv heads, "
                f"received {keys.shape[1]}"
            )
        if keys.shape[3] != self.head_dim:
            raise CacheStateError(
                f"layer {self.layer_idx} expects head_dim {self.head_dim}, received {keys.shape[3]}"
            )

        if self._compressed is not None:
            self.materialize()

        keys = keys.to(device=self.device, dtype=self.dtype)
        values = values.to(device=self.device, dtype=self.dtype)
        num_new = int(keys.shape[2])

        if self._keys is None:
            self._keys, self._values = keys, values
        else:
            assert self._keys is not None and self._values is not None
            self._keys = torch.cat([self._keys, keys], dim=2)
            self._values = torch.cat([self._values, values], dim=2)

        self.metadata.append(num_new, positions=positions, step=step)
        self._offloaded = False

    def keep(self, indices: torch.Tensor) -> int:
        """Retain only ``indices``, dropping every other token.

        Returns the number of tokens dropped. Compression is preserved: a
        quantised layer stays quantised, with per-token scales gathered
        alongside the payload where the quantisation axis requires it.
        """
        if not self.is_initialized:
            return 0

        before = self.num_tokens
        home_device = self.resident_device
        indices = indices.to(device=home_device, dtype=torch.long).reshape(-1)

        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= before):
            raise CacheStateError(
                f"keep indices out of range for layer {self.layer_idx} with {before} tokens: "
                f"min={int(indices.min())}, max={int(indices.max())}"
            )

        if self._compressed is not None:
            payload_keys, payload_values = self._compressed.keys, self._compressed.values
            if not isinstance(payload_keys, QuantizedTensor) or not isinstance(
                payload_values, QuantizedTensor
            ):
                raise CacheStateError(
                    f"layer {self.layer_idx} holds an unrecognised compressed payload; "
                    "cannot gather it"
                )
            gathered_keys = _gather_quantized(payload_keys, indices)
            gathered_values = _gather_quantized(payload_values, indices)
            kept = int(indices.numel())
            # Recompute accounting from the gathered payloads rather than scaling
            # the previous numbers, so a gathered layer reports its true size.
            self._compressed = CompressionResult(
                keys=gathered_keys,
                values=gathered_values,
                num_tokens=kept,
                uncompressed_bytes=self.uncompressed_bytes_after(kept),
                compressed_bytes=int(gathered_keys.bytes()) + int(gathered_values.bytes()),
            )
        else:
            if self._keys is None or self._values is None:
                raise CacheStateError(
                    f"layer {self.layer_idx} is marked initialised but holds no tensors"
                )
            self._keys = self._keys[:, :, indices, :].contiguous()
            self._values = self._values[:, :, indices, :].contiguous()

        self.metadata.keep(indices)
        return before - int(indices.numel())

    def uncompressed_bytes_after(self, num_tokens: int) -> int:
        """Bytes ``num_tokens`` would occupy at the configured dtype."""
        per_token = 2 * self.batch_size * self.num_kv_heads * self.head_dim
        element = torch.empty(0, dtype=self.dtype).element_size()
        return int(num_tokens * per_token * element)

    def clear(self) -> None:
        """Drop all tokens."""
        self._keys = None
        self._values = None
        self._compressed = None
        self._offloaded = False
        self.metadata.clear()

    def move_to(self, device: torch.device) -> None:
        """Relocate tensors, tracking whether they left their home device."""
        if self._compressed is not None:
            keys = self._compressed.keys
            values = self._compressed.values
            assert isinstance(keys, QuantizedTensor) and isinstance(values, QuantizedTensor)
            moved_keys = dataclass_replace(
                keys,
                data=keys.data.to(device),
                scale=keys.scale.to(device),
                zero_point=keys.zero_point.to(device),
            )
            moved_values = dataclass_replace(
                values,
                data=values.data.to(device),
                scale=values.scale.to(device),
                zero_point=values.zero_point.to(device),
            )
            self._compressed = CompressionResult(
                keys=moved_keys,
                values=moved_values,
                num_tokens=self._compressed.num_tokens,
                uncompressed_bytes=self._compressed.uncompressed_bytes,
                compressed_bytes=self._compressed.compressed_bytes,
            )
        elif self._keys is not None and self._values is not None:
            self._keys = self._keys.to(device)
            self._values = self._values.to(device)
        else:
            self.device = device
            self.metadata.to(device)
            return

        self.metadata.to(device)
        self._offloaded = device != self.device

    def replace(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Swap the stored float tensors in place, preserving token count.

        Used by dequantisation paths. The token count is invariant, so metadata
        is intentionally left untouched.
        """
        if keys.shape != values.shape:
            raise CacheStateError(
                f"keys and values must share a shape, got {tuple(keys.shape)} vs "
                f"{tuple(values.shape)}"
            )
        if keys.shape[2] != self.num_tokens:
            raise CacheStateError(
                f"replace must preserve the token count: layer {self.layer_idx} holds "
                f"{self.num_tokens} tokens but was given {keys.shape[2]}. "
                "Compression changes representation, not occupancy."
            )
        self._keys = keys
        self._values = values
        self._compressed = None


class KVStore:
    """A collection of per-layer :class:`LayerStorage` objects.

    The store is policy-agnostic. It exposes the primitive operations a policy
    needs (:meth:`keep`, :meth:`move_to`) and the accounting the benchmark needs
    (:meth:`bytes_on_device`, :meth:`bytes_offloaded`), and nothing else.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._layers: list[LayerStorage] = [
            LayerStorage(
                layer_idx=idx,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                dtype=config.dtype,
                device=self.device,
                num_sinks=config.attention_sinks,
                batch_size=config.batch_size,
            )
            for idx in range(config.num_layers)
        ]

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._layers)

    def __contains__(self, layer_idx: object) -> bool:
        return isinstance(layer_idx, int) and 0 <= layer_idx < len(self._layers)

    def layer(self, layer_idx: int) -> LayerStorage:
        """Return the storage for ``layer_idx``, validating the index."""
        if not isinstance(layer_idx, int):
            raise CacheStateError(f"layer index must be an int, got {type(layer_idx).__name__}")
        if not 0 <= layer_idx < len(self._layers):
            raise CacheStateError(
                f"layer index {layer_idx} out of range for a {len(self._layers)}-layer cache"
            )
        return self._layers[layer_idx]

    @property
    def layers(self) -> list[LayerStorage]:
        return self._layers

    def num_tokens(self, layer_idx: int) -> int:
        return self.layer(layer_idx).num_tokens

    def max_tokens(self) -> int:
        """Largest token count across layers.

        A well-formed run keeps layers in lockstep, so this equals the sequence
        length. It is computed rather than assumed, so a desynchronised cache
        shows up in the numbers instead of being hidden.
        """
        return max((layer.num_tokens for layer in self._layers), default=0)

    def min_tokens(self) -> int:
        """Smallest token count across layers."""
        return min((layer.num_tokens for layer in self._layers), default=0)

    # -- mutation ----------------------------------------------------------

    def clear(self) -> None:
        for layer in self._layers:
            layer.clear()

    def keep(self, layer_idx: int, indices: torch.Tensor) -> int:
        return self.layer(layer_idx).keep(indices)

    def move_to(self, device: torch.device, layer_idx: int | None = None) -> None:
        """Relocate one layer, or every layer when ``layer_idx`` is ``None``."""
        targets = self._layers if layer_idx is None else [self.layer(layer_idx)]
        for layer in targets:
            layer.move_to(device)

    # -- accounting --------------------------------------------------------

    def bytes_on_device(self) -> int:
        """Resident bytes that are *not* offloaded."""
        return sum(layer.bytes() for layer in self._layers if not layer.is_offloaded)

    def bytes_offloaded(self) -> int:
        """Bytes held on a non-home device (typically CPU RAM)."""
        return sum(layer.bytes() for layer in self._layers if layer.is_offloaded)

    def bytes_total(self) -> int:
        return self.bytes_on_device() + self.bytes_offloaded()

    def uncompressed_bytes(self) -> int:
        """Bytes the cache would occupy at the configured dtype, uncompressed."""
        return sum(layer.uncompressed_bytes() for layer in self._layers)

    def tokens_per_layer(self) -> list[int]:
        return [layer.num_tokens for layer in self._layers]

    def offloaded_layers(self) -> list[int]:
        return [layer.layer_idx for layer in self._layers if layer.is_offloaded]

    def compressed_layers(self) -> list[int]:
        return [layer.layer_idx for layer in self._layers if layer.is_compressed]

    def describe(self) -> str:
        """Compact description used in log lines and error messages."""
        return (
            f"KVStore(layers={len(self._layers)}, tokens={self.tokens_per_layer()}, "
            f"device={self.device}, dtype={self.config.dtype})"
        )


__all__ = ["KVStore", "LayerStorage"]
