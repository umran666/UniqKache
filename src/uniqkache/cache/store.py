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

Storage allocation and append performance
-----------------------------------------
Storage is preallocated using a contiguous buffer along the sequence dimension:
* When ``capacity`` is configured (bounded cache), buffers are allocated up to
  the capacity upfront, yielding zero allocations and O(1) per-token appends
  during decode.
* When ``capacity`` is None (unbounded cache), buffers grow via geometric
  doubling, amortising allocation costs to O(1) per token (O(T) total sequence time).
* Eviction trims oversized buffers to the configured capacity (or surviving
  occupancy when unbounded). An empty layer releases its backing buffers.

Split residency
---------------
A layer's bytes can live on **both** tiers at once: slots ``[0, boundary)`` stay
device-resident (the hot head — sinks and recent tokens) while the tail
``[boundary, num_tokens)`` moves to the host tier (the cold tail). This is the
granularity real serving systems use, and it is what
:meth:`LayerStorage.offload_tail` / :meth:`LayerStorage.prefetch_tail` operate
on. Three interactions are defined rather than discovered:

* **Attention reads gather-on-read.** The public ``keys``/``values`` properties
  concatenate the resident prefix with the hosted tail, moving host rows home
  first. Slow, but correct — and correct is the requirement for a reference
  path.
* **Append, eviction and compression collapse the split.** New rows are the
  hottest tokens and must join the resident prefix in slot order; ``keep``
  gathers survivors into one buffer; compression quantises the gathered whole.
  Each operation documents that it materialises first, so the split is always
  something a caller re-establishes with ``offload_tail``, never a hidden
  invariant the store silently maintains.
* **Accounting is per tier, always.** :meth:`LayerStorage.device_bytes` and
  :meth:`LayerStorage.offloaded_bytes` report the two shares separately, so the
  split cannot be mistaken for a reduction in total memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from uniqkache.cache.metadata import LayerMetadata
from uniqkache.cache.types import CacheConfig
from uniqkache.compression.base import CompressionResult
from uniqkache.compression.quantize import QuantizedTensor
from uniqkache.utils.errors import CacheStateError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

CHECKPOINT_SCHEMA_VERSION: str = "1.0.0"


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
    capacity: int | None = None

    def __post_init__(self) -> None:
        self._keys_buf: torch.Tensor | None = None
        self._values_buf: torch.Tensor | None = None
        self._compressed: CompressionResult | None = None
        self._offloaded = False
        # Split residency: slots [0, _resident_boundary) live on the home
        # device, [_resident_boundary, num_tokens) live on the host tier. A
        # layer is "split" iff _host_keys_buf is not None, which by
        # construction implies 0 < _resident_boundary < num_tokens (a
        # fully-offloaded layer uses the whole-layer path in move_to instead).
        self._resident_boundary = 0
        self._host_keys_buf: torch.Tensor | None = None
        self._host_values_buf: torch.Tensor | None = None
        self.metadata = LayerMetadata(num_sinks=self.num_sinks, device=self.device)

    # -- introspection -----------------------------------------------------

    @property
    def keys(self) -> torch.Tensor | None:
        """Materialised keys, dequantising and gathering on demand.

        A split layer's hosted tail is moved home and concatenated behind the
        resident prefix, so callers always see the full token set in slot
        order. This is the deliberately slow, correct reference path; a
        production tiering scheme would prefetch before compute instead.
        """
        if self._compressed is not None:
            return self._dequantize()[0]
        if self._host_keys_buf is not None:
            return self._gather(self._keys_buf, self._host_keys_buf)
        if self._keys_buf is not None:
            return self._keys_buf[:, :, : self.num_tokens, :]
        return None

    @property
    def values(self) -> torch.Tensor | None:
        """Materialised values, dequantising and gathering on demand."""
        if self._compressed is not None:
            return self._dequantize()[1]
        if self._host_values_buf is not None:
            return self._gather(self._values_buf, self._host_values_buf)
        if self._values_buf is not None:
            return self._values_buf[:, :, : self.num_tokens, :]
        return None

    def _gather(
        self, resident: torch.Tensor | None, hosted: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Concatenate a split layer's two tiers, in slot order, on the home device."""
        if resident is None and hosted is None:
            return None
        if resident is None:
            assert hosted is not None
            return hosted.to(self.device)
        if hosted is None:
            return resident
        # The resident buffer is allocated to exactly the boundary, but slice
        # defensively: a preallocated buffer holding extra rows must not leak
        # stale slots into the gathered view.
        prefix = resident[:, :, : self._resident_boundary, :]
        return torch.cat([prefix, hosted.to(self.device)], dim=2)

    @property
    def _keys(self) -> torch.Tensor | None:
        # Private accessor = raw resident buffer only. A compressed-only layer
        # has no resident buffer, so this returns None rather than silently
        # dequantizing through the public `keys` property.
        if self._keys_buf is not None:
            return self._keys_buf[:, :, : self.num_tokens, :]
        return None

    @_keys.setter
    def _keys(self, val: torch.Tensor | None) -> None:
        self._keys_buf = val
        # Assigning raw tensors replaces the layer's whole content, so any
        # residency split the caller does not know about is dropped rather
        # than left describing buffers that no longer exist.
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0

    @property
    def _values(self) -> torch.Tensor | None:
        if self._values_buf is not None:
            return self._values_buf[:, :, : self.num_tokens, :]
        return None

    @_values.setter
    def _values(self, val: torch.Tensor | None) -> None:
        self._values_buf = val
        # Replaces the layer's whole content; see the `_keys` setter.
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0

    @property
    def num_tokens(self) -> int:
        """Occupancy: how many tokens are cached."""
        return self.metadata.num_tokens

    @property
    def is_initialized(self) -> bool:
        return self._keys_buf is not None or self._compressed is not None

    @property
    def is_compressed(self) -> bool:
        """Whether the resident representation is quantised."""
        return self._compressed is not None

    @property
    def is_offloaded(self) -> bool:
        return self._offloaded

    @property
    def is_split(self) -> bool:
        """Whether this layer's bytes span both tiers (partially offloaded).

        Distinct from :attr:`is_offloaded`, which means the *whole* layer was
        moved: a split layer keeps its hot head on the compute device.
        """
        return self._host_keys_buf is not None

    @property
    def resident_tokens(self) -> int:
        """Tokens whose bytes live on the home device.

        Equal to :attr:`num_tokens` unless the layer is split.
        """
        if self._host_keys_buf is None:
            return self.num_tokens
        return self._resident_boundary

    @property
    def offloaded_tokens(self) -> int:
        """Tokens whose bytes live on the host tier; 0 unless split."""
        if self._host_keys_buf is None:
            return 0
        return self.num_tokens - self._resident_boundary

    @property
    def resident_device(self) -> torch.device:
        """Where the tensors physically live right now."""
        if self._keys_buf is not None:
            return self._keys_buf.device
        if self._compressed is not None:
            return self._compressed.keys.data.device  # type: ignore[union-attr,attr-defined]
        return self.device

    def bytes(self) -> int:
        """Bytes held by this layer across *every* tier, honouring compression.

        Offloaded bytes still exist, so they are counted: ``bytes()`` is the
        whole footprint, while :meth:`device_bytes` and :meth:`offloaded_bytes`
        split it by tier.
        """
        if self._compressed is not None:
            return int(self._compressed.compressed_bytes)
        total = 0
        if self._keys_buf is not None and self._values_buf is not None:
            total += int(self._keys_buf.untyped_storage().nbytes()) + int(
                self._values_buf.untyped_storage().nbytes()
            )
        if self._host_keys_buf is not None and self._host_values_buf is not None:
            total += int(self._host_keys_buf.untyped_storage().nbytes()) + int(
                self._host_values_buf.untyped_storage().nbytes()
            )
        return total

    def device_bytes(self) -> int:
        """Bytes of this layer resident on the cache's home (compute) device.

        Zero for a whole-layer offload; the resident share for a split layer.
        A hosted region whose buffers are physically on the home device (only
        possible when a checkpoint recorded one, or a test built one) has not
        left the compute tier and is counted here, so the figure always
        describes physical residency.
        """
        if self._offloaded:
            return 0
        if self._host_keys_buf is not None and self._host_keys_buf.device != self.device:
            if self._keys_buf is None or self._values_buf is None:
                return 0
            return int(self._keys_buf.untyped_storage().nbytes()) + int(
                self._values_buf.untyped_storage().nbytes()
            )
        return self.bytes()

    def offloaded_bytes(self) -> int:
        """Bytes of this layer held on a non-home tier.

        The whole layer for a whole-layer offload; the hosted share for a
        split layer; zero when everything is device-resident. A hosted region
        sitting on the home device counts as zero: it never left the compute
        tier, and counting it would claim a saving that did not happen.
        """
        if self._offloaded:
            return self.bytes()
        if self._host_keys_buf is None or self._host_values_buf is None:
            return 0
        if self._host_keys_buf.device == self.device:
            return 0
        return int(self._host_keys_buf.untyped_storage().nbytes()) + int(
            self._host_values_buf.untyped_storage().nbytes()
        )

    def payload_bytes(self) -> int:
        """Live K/V payload, excluding unused preallocated slots."""
        if self._compressed is not None:
            return int(self._compressed.compressed_bytes)
        return self.uncompressed_bytes_after(self.num_tokens)

    def uncompressed_bytes(self) -> int:
        """Bytes this layer *would* occupy at the configured dtype."""
        if not self.is_initialized:
            return 0
        if self._compressed is None:
            return self.bytes()
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
        """Adopt a compressed representation, preserving occupancy.

        The compressed payload describes the whole gathered layer, so any
        residency split is dropped: ``device_bytes`` falls back to the whole
        compressed payload and ``offloaded_bytes`` to zero. Compressing a split
        layer is therefore lossless with respect to *tokens* — the tier split
        is simply re-established by :meth:`offload_tail` afterwards (which
        refuses while compressed, so callers must decompress first).
        """
        if result.num_tokens != self.num_tokens:
            raise CacheStateError(
                f"compression changed occupancy for layer {self.layer_idx}: "
                f"{self.num_tokens} tokens before, {result.num_tokens} after. "
                "Compression must preserve the token set."
            )
        self._compressed = result
        self._keys_buf = None
        self._values_buf = None
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0

    def materialize(self) -> None:
        """Expand a compressed representation back to float tensors."""
        if self._compressed is None:
            return
        keys, values = self._dequantize()
        target_device = self.resident_device
        keys = keys.to(dtype=self.dtype, device=target_device)
        values = values.to(dtype=self.dtype, device=target_device)
        self._compressed = None
        needed = int(keys.shape[2])
        init_cap = self.capacity if self.capacity is not None else 64
        alloc_cap = max(needed, init_cap)
        self._keys_buf = torch.empty(
            (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim),
            dtype=self.dtype,
            device=target_device,
        )
        self._values_buf = torch.empty(
            (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim),
            dtype=self.dtype,
            device=target_device,
        )
        self._keys_buf[:, :, :needed, :] = keys
        self._values_buf[:, :, :needed, :] = values

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

        Appending to a split layer collapses the split first: new rows are the
        hottest tokens in the layer and must join the resident prefix in slot
        order. Re-establish the split with :meth:`offload_tail` afterwards.
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

        if self.is_offloaded:
            self.move_to(self.device)
        if self._host_keys_buf is not None:
            self._collapse_split()
        keys = keys.to(device=self.device, dtype=self.dtype)
        values = values.to(device=self.device, dtype=self.dtype)
        num_new = int(keys.shape[2])
        cur_len = self.num_tokens
        needed = cur_len + num_new

        if self._keys_buf is None:
            init_cap = self.capacity if self.capacity is not None else 64
            alloc_cap = max(needed, init_cap)
            self._keys_buf = torch.empty(
                (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self._values_buf = torch.empty(
                (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            # The two buffers are always both set or both unset (they are only
            # ever assigned together), so the else branch implies both exist.
            assert self._values_buf is not None
            if self._keys_buf.device != self.device:
                self._keys_buf = self._keys_buf.to(self.device)
                self._values_buf = self._values_buf.to(self.device)

            if needed > self._keys_buf.shape[2]:
                cur_cap = self._keys_buf.shape[2]
                new_cap = max(needed, cur_cap * 2)
                new_keys = torch.empty(
                    (self.batch_size, self.num_kv_heads, new_cap, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                new_values = torch.empty(
                    (self.batch_size, self.num_kv_heads, new_cap, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                if cur_len > 0:
                    new_keys[:, :, :cur_len, :] = self._keys_buf[:, :, :cur_len, :]
                    new_values[:, :, :cur_len, :] = self._values_buf[:, :, :cur_len, :]
                self._keys_buf = new_keys
                self._values_buf = new_values

        assert self._keys_buf is not None and self._values_buf is not None
        self._keys_buf[:, :, cur_len:needed, :] = keys
        self._values_buf[:, :, cur_len:needed, :] = values

        self.metadata.append(num_new, positions=positions, step=step)
        self._offloaded = False

    def keep(self, indices: torch.Tensor) -> int:
        """Retain only ``indices``, dropping every other token.

        Returns the number of tokens dropped. Compression is preserved: a
        quantised layer stays quantised, with per-token scales gathered
        alongside the payload where the quantisation axis requires it. A split
        layer is collapsed first: eviction gathers its survivors into one
        device-resident buffer, because the boundary would otherwise describe a
        prefix of slots that may not all have survived. Re-establish the split
        with :meth:`offload_tail` afterwards.
        """
        if not self.is_initialized:
            return 0

        if self._host_keys_buf is not None:
            self._collapse_split()

        before = self.num_tokens
        home_device = self.resident_device
        indices = indices.to(device=home_device, dtype=torch.long).reshape(-1)

        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= before):
            raise CacheStateError(
                f"keep indices out of range for layer {self.layer_idx} with {before} tokens: "
                f"min={int(indices.min())}, max={int(indices.max())}"
            )
        if indices.numel() and int(indices.unique().numel()) != int(indices.numel()):
            # Duplicates would gather the same rows twice: tokens silently
            # duplicated, positions/is_sink metadata corrupted, and the
            # dropped count driven negative (issue #120). Fail loudly per the
            # project's malformed-input contract instead.
            raise CacheStateError(
                f"keep indices contain duplicates for layer {self.layer_idx}: "
                f"{int(indices.numel())} indices but only "
                f"{int(indices.unique().numel())} unique"
            )

        if indices.numel() == 0:
            self._keys_buf = self._values_buf = None
            self._compressed = None
            self._offloaded = False
            self.metadata.keep(indices)
            return before

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
            if self._keys_buf is None or self._values_buf is None:
                raise CacheStateError(
                    f"layer {self.layer_idx} is marked initialised but holds no tensors"
                )
            kept = int(indices.numel())
            surviving_keys = self._keys_buf[:, :, indices, :].clone()
            surviving_values = self._values_buf[:, :, indices, :].clone()
            alloc_cap = max(kept, self.capacity or kept)
            if self._keys_buf.shape[2] != alloc_cap:
                shape = (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim)
                self._keys_buf = torch.empty(shape, dtype=self.dtype, device=home_device)
                self._values_buf = torch.empty(shape, dtype=self.dtype, device=home_device)
            self._keys_buf[:, :, :kept, :] = surviving_keys
            self._values_buf[:, :, :kept, :] = surviving_values

        self.metadata.keep(indices)
        return before - int(indices.numel())

    def uncompressed_bytes_after(self, num_tokens: int) -> int:
        """Bytes ``num_tokens`` would occupy at the configured dtype."""
        per_token = 2 * self.batch_size * self.num_kv_heads * self.head_dim
        element = torch.empty(0, dtype=self.dtype).element_size()
        return int(num_tokens * per_token * element)

    def clear(self) -> None:
        """Drop all tokens."""
        self._keys_buf = None
        self._values_buf = None
        self._compressed = None
        self._offloaded = False
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0
        self.metadata.clear()

    def move_to(self, device: torch.device) -> None:
        """Relocate tensors, tracking whether they left their home device.

        A split layer is collapsed first: both tiers move together, because a
        residency split whose halves sit on the same device has no meaning and
        would leave ``device_bytes``/``offloaded_bytes`` describing a
        separation that no longer exists.
        """
        if self._host_keys_buf is not None:
            self._collapse_split()
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
        elif self._keys_buf is not None and self._values_buf is not None:
            self._keys_buf = self._keys_buf.to(device)
            self._values_buf = self._values_buf.to(device)
        else:
            self.device = device
            self.metadata.to(device)
            return

        self.metadata.to(device)
        self._offloaded = device != self.device

    # -- split residency ---------------------------------------------------

    def offload_tail(self, resident: int, *, host: torch.device) -> int:
        """Move the cold tail ``[resident:, num_tokens)`` to the host tier.

        Parameters
        ----------
        resident:
            Number of leading tokens to keep on the home device. ``0`` means
            "offload everything" and delegates to whole-layer :meth:`move_to`;
            a value equal to :attr:`num_tokens` is a no-op.
        host:
            The host tier's device.

        Returns
        -------
        int
            Number of tokens whose bytes moved to the host tier. Returns 0
            without splitting when the host tier *is* the home device: there is
            no cheaper tier to move to, and a split that moved nothing would
            claim a saving that did not happen. Construct the cache on a CUDA
            device (or pass a real second tier) to exercise the move.

        Raises
        ------
        CacheStateError
            If ``resident`` is out of range, or the layer holds a compressed
            payload. Compression and split residency are deliberately not
            composed: a quantised layer is quantised as a whole, so splitting
            it would need per-row quantised gathers for no benefit the
            reference path can measure. Decompress first.
        """
        if self._compressed is not None:
            raise CacheStateError(
                f"layer {self.layer_idx} holds a compressed payload; decompress before "
                "splitting its residency. Compression is applied to the whole gathered "
                "layer, so a split compressed layer has no defined semantics."
            )
        if resident < 0 or resident > self.num_tokens:
            raise CacheStateError(
                f"resident boundary {resident} out of range for layer {self.layer_idx} "
                f"holding {self.num_tokens} tokens"
            )
        if host == self.device:
            # No cheaper tier: there is nothing to offload to, so the call is a
            # no-op that reports zero — the same rule the whole-layer path
            # applies (and TierManager enforces loudly). Reporting a split here
            # would claim bytes left the compute device when they did not.
            _log.debug(
                "layer %d: host tier %s is the home device; nothing to offload",
                self.layer_idx,
                host,
            )
            return 0
        if resident == 0:
            # "Offload everything" is the whole-layer spelling; use it so a
            # fully-offloaded layer keeps the single representation the
            # whole-layer paths expect.
            if self.is_offloaded and self.resident_device == host:
                return 0
            self.move_to(host)
            return self.num_tokens
        if self.is_offloaded:
            # Whole-layer offload and split residency are two spellings of the
            # same idea; re-establish a single representation before splitting.
            self.move_to(self.device)
        if resident == self.num_tokens:
            return 0
        if self._host_keys_buf is not None:
            # Re-splitting collapses any existing split first, so the boundary
            # always describes one contiguous hosted region.
            self._collapse_split()

        assert self._keys_buf is not None and self._values_buf is not None
        hosted = self.num_tokens - resident
        # Slice to the live token count, not the buffer end: an unbounded
        # layer's buffer is preallocated past its occupancy, and offloading
        # unused slots would move bytes that hold no tokens.
        self._host_keys_buf = self._keys_buf[:, :, resident : self.num_tokens, :].clone().to(host)
        self._host_values_buf = (
            self._values_buf[:, :, resident : self.num_tokens, :].clone().to(host)
        )
        self._shrink_resident(resident)
        self._resident_boundary = resident
        _log.debug(
            "layer %d: split residency, %d resident / %d on %s",
            self.layer_idx,
            resident,
            hosted,
            host,
        )
        return hosted

    def prefetch_tail(self, count: int | None = None) -> int:
        """Bring hosted tokens back to the home device, preserving slot order.

        Parameters
        ----------
        count:
            How many hosted tokens to bring back, front first. ``None`` means
            all of them. Hosted tokens sit at the end of the slot range, so
            the frontmost hosted token is the next one the resident prefix
            expects — the order is preserved by construction.

        Returns
        -------
        int
            Number of tokens brought back. Zero when nothing is hosted.
        """
        if self._host_keys_buf is None or self._host_values_buf is None:
            return 0
        if count is not None and count < 0:
            raise CacheStateError(f"prefetch count must be >= 0, got {count}")

        hosted = self.offloaded_tokens
        bring = hosted if count is None else min(count, hosted)
        if bring <= 0:
            return 0

        assert self._keys_buf is not None and self._values_buf is not None
        new_keys = self._host_keys_buf[:, :, :bring, :].to(self.device)
        new_values = self._host_values_buf[:, :, :bring, :].to(self.device)
        boundary = self._resident_boundary + bring
        self._grow_resident(new_keys, new_values, boundary)

        if bring == hosted:
            self._host_keys_buf = None
            self._host_values_buf = None
        else:
            self._host_keys_buf = self._host_keys_buf[:, :, bring:, :].clone()
            self._host_values_buf = self._host_values_buf[:, :, bring:, :].clone()
        self._resident_boundary = boundary
        return bring

    def _collapse_split(self) -> None:
        """Gather a split layer back into one device-resident buffer."""
        if self._host_keys_buf is None:
            return
        assert self._host_keys_buf is not None and self._host_values_buf is not None
        keys = self._gather(self._keys_buf, self._host_keys_buf)
        values = self._gather(self._values_buf, self._host_values_buf)
        assert keys is not None and values is not None
        num = int(keys.shape[2])
        self._keys_buf = torch.empty(
            (self.batch_size, self.num_kv_heads, num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._values_buf = torch.empty_like(self._keys_buf)
        self._keys_buf.copy_(keys)
        self._values_buf.copy_(values)
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0

    def _shrink_resident(self, resident: int) -> None:
        """Reallocate the resident buffers to hold exactly ``resident`` rows."""
        assert self._keys_buf is not None and self._values_buf is not None
        keys = self._keys_buf[:, :, :resident, :]
        values = self._values_buf[:, :, :resident, :]
        new_keys = torch.empty(
            (self.batch_size, self.num_kv_heads, resident, self.head_dim),
            dtype=self.dtype,
            device=keys.device,
        )
        new_values = torch.empty_like(new_keys)
        new_keys.copy_(keys)
        new_values.copy_(values)
        self._keys_buf = new_keys
        self._values_buf = new_values

    def _grow_resident(
        self, new_keys: torch.Tensor, new_values: torch.Tensor, boundary: int
    ) -> None:
        """Append gathered host rows behind the resident prefix."""
        assert self._keys_buf is not None and self._values_buf is not None
        alloc = max(boundary, self.capacity or boundary)
        buf_keys = torch.empty(
            (self.batch_size, self.num_kv_heads, alloc, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        buf_values = torch.empty_like(buf_keys)
        current = self._resident_boundary
        if current:
            buf_keys[:, :, :current, :] = self._keys_buf[:, :, :current, :]
            buf_values[:, :, :current, :] = self._values_buf[:, :, :current, :]
        added = int(new_keys.shape[2])
        buf_keys[:, :, current : current + added, :] = new_keys
        buf_values[:, :, current : current + added, :] = new_values
        self._keys_buf = buf_keys
        self._values_buf = buf_values

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
        target_device = keys.device
        needed = int(keys.shape[2])
        init_cap = self.capacity if self.capacity is not None else 64
        alloc_cap = max(needed, init_cap)
        self._keys_buf = torch.empty(
            (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim),
            dtype=self.dtype,
            device=target_device,
        )
        self._values_buf = torch.empty(
            (self.batch_size, self.num_kv_heads, alloc_cap, self.head_dim),
            dtype=self.dtype,
            device=target_device,
        )
        self._keys_buf[:, :, :needed, :] = keys
        self._values_buf[:, :, :needed, :] = values
        self._compressed = None
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0

    def state_dict(self) -> dict[str, Any]:
        """Serialisable dictionary of this layer's tensors, compression, and metadata."""
        state: dict[str, Any] = {
            "layer_idx": self.layer_idx,
            "is_initialized": self.is_initialized,
            "is_compressed": self.is_compressed,
            "is_offloaded": self.is_offloaded,
            "resident_device": str(self.resident_device),
            "metadata": self.metadata.state_dict(),
        }
        if self.is_compressed:
            assert self._compressed is not None
            keys_qt = self._compressed.keys
            values_qt = self._compressed.values
            assert isinstance(keys_qt, QuantizedTensor) and isinstance(values_qt, QuantizedTensor)
            state["compressed"] = {
                "keys": {
                    "data": keys_qt.data.clone(),
                    "scale": keys_qt.scale.clone(),
                    "zero_point": keys_qt.zero_point.clone(),
                    "axis": keys_qt.axis,
                    "num_bits": keys_qt.num_bits,
                    "symmetric": keys_qt.symmetric,
                },
                "values": {
                    "data": values_qt.data.clone(),
                    "scale": values_qt.scale.clone(),
                    "zero_point": values_qt.zero_point.clone(),
                    "axis": values_qt.axis,
                    "num_bits": values_qt.num_bits,
                    "symmetric": values_qt.symmetric,
                },
                "num_tokens": self._compressed.num_tokens,
                "uncompressed_bytes": self._compressed.uncompressed_bytes,
                "compressed_bytes": self._compressed.compressed_bytes,
            }
        elif self.is_initialized:
            assert self._keys is not None and self._values is not None
            assert self._keys_buf is not None
            state["resident_boundary"] = self._resident_boundary
            state["allocated_tokens"] = int(self._keys_buf.shape[2])
            state["keys"] = self._keys.clone()
            state["values"] = self._values.clone()
            if self._host_keys_buf is not None and self._host_values_buf is not None:
                state["host_keys"] = self._host_keys_buf.clone()
                state["host_values"] = self._host_values_buf.clone()
                state["host_device"] = str(self._host_keys_buf.device)
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
        config: CacheConfig,
        *,
        target_device: torch.device | None = None,
    ) -> None:
        """Restore layer tensors, compression, and metadata from a state dictionary.

        Parameters
        ----------
        state:
            Dictionary produced by :meth:`state_dict`.
        config:
            Expected cache configuration for shape, dtype, and device validation.
        target_device:
            Optional device override. If omitted, uses the layer's resident_device
            as recorded in the state (or the home device).
        """
        layer_idx = state.get("layer_idx")
        if layer_idx != self.layer_idx:
            raise CacheStateError(
                f"state layer_idx {layer_idx} does not match storage layer_idx {self.layer_idx}"
            )

        is_initialized = state.get("is_initialized", False)
        is_compressed = state.get("is_compressed", False)
        is_offloaded = state.get("is_offloaded", False)
        recorded_resident = state.get("resident_device", str(self.device))

        # A restored layer starts without a split; the split keys below opt in.
        self._host_keys_buf = None
        self._host_values_buf = None
        self._resident_boundary = 0

        if target_device is not None:
            dev = target_device
        elif is_offloaded:
            dev = torch.device(recorded_resident)
        else:
            dev = self.device

        if not is_initialized:
            self.clear()
            self.metadata.load_state_dict(state["metadata"], device=dev, expected_num_tokens=0)
            return

        if is_compressed:
            if "compressed" not in state:
                raise CacheStateError(
                    f"layer {self.layer_idx} marked compressed but missing 'compressed' payload"
                )
            comp_data = state["compressed"]
            keys_dict = comp_data["keys"]
            values_dict = comp_data["values"]

            for name, d in (("keys", keys_dict), ("values", values_dict)):
                for field_name in ("data", "scale", "zero_point", "axis", "num_bits", "symmetric"):
                    if field_name not in d:
                        raise CacheStateError(f"compressed {name} missing field {field_name!r}")

            k_data = keys_dict["data"].to(device=dev)
            v_data = values_dict["data"].to(device=dev)
            k_scale = keys_dict["scale"].to(device=dev)
            v_scale = values_dict["scale"].to(device=dev)
            k_zp = keys_dict["zero_point"].to(device=dev)
            v_zp = values_dict["zero_point"].to(device=dev)

            # Validate shapes
            if k_data.dim() != 4 or v_data.dim() != 4:
                raise CacheStateError(
                    f"compressed K/V data must be 4-D, got {k_data.shape} vs {v_data.shape}"
                )
            if k_data.shape != v_data.shape:
                raise CacheStateError(
                    f"compressed keys and values must share shape, got {k_data.shape} vs {v_data.shape}"
                )
            if k_data.shape[0] != config.batch_size:
                raise CacheStateError(
                    f"layer {self.layer_idx} batch_size mismatch: {k_data.shape[0]} vs {config.batch_size}"
                )
            if k_data.shape[1] != config.num_kv_heads:
                raise CacheStateError(
                    f"layer {self.layer_idx} num_kv_heads mismatch: {k_data.shape[1]} vs {config.num_kv_heads}"
                )
            if k_data.shape[3] != config.head_dim:
                raise CacheStateError(
                    f"layer {self.layer_idx} head_dim mismatch: {k_data.shape[3]} vs {config.head_dim}"
                )

            seq_len = int(k_data.shape[2])
            num_tokens = comp_data.get("num_tokens", seq_len)
            if num_tokens != seq_len:
                raise CacheStateError(
                    f"compressed num_tokens {num_tokens} does not match sequence length {seq_len}"
                )

            qt_keys = QuantizedTensor(
                data=k_data,
                scale=k_scale,
                zero_point=k_zp,
                axis=keys_dict["axis"],
                num_bits=keys_dict["num_bits"],
                symmetric=keys_dict["symmetric"],
            )
            qt_values = QuantizedTensor(
                data=v_data,
                scale=v_scale,
                zero_point=v_zp,
                axis=values_dict["axis"],
                num_bits=values_dict["num_bits"],
                symmetric=values_dict["symmetric"],
            )

            self._compressed = CompressionResult(
                keys=qt_keys,
                values=qt_values,
                num_tokens=num_tokens,
                uncompressed_bytes=comp_data.get(
                    "uncompressed_bytes", self.uncompressed_bytes_after(num_tokens)
                ),
                compressed_bytes=comp_data.get(
                    "compressed_bytes", int(qt_keys.bytes() + qt_values.bytes())
                ),
            )
            self._keys = None
            self._values = None
            self._offloaded = is_offloaded if target_device is None else dev != self.device
            self.metadata.load_state_dict(
                state["metadata"], device=dev, expected_num_tokens=num_tokens
            )
        else:
            if "keys" not in state or "values" not in state:
                raise CacheStateError(
                    f"layer {self.layer_idx} marked initialized but missing 'keys' or 'values'"
                )
            keys = state["keys"].to(device=dev)
            values = state["values"].to(device=dev)

            # Validate shapes & dtype
            if keys.dim() != 4 or values.dim() != 4:
                raise CacheStateError(f"K/V must be 4-D, got {keys.shape} vs {values.shape}")
            if keys.shape != values.shape:
                raise CacheStateError(
                    f"keys and values must share shape, got {keys.shape} vs {values.shape}"
                )
            if keys.shape[0] != config.batch_size:
                raise CacheStateError(
                    f"layer {self.layer_idx} batch_size mismatch: {keys.shape[0]} vs {config.batch_size}"
                )
            if keys.shape[1] != config.num_kv_heads:
                raise CacheStateError(
                    f"layer {self.layer_idx} num_kv_heads mismatch: {keys.shape[1]} vs {config.num_kv_heads}"
                )
            if keys.shape[3] != config.head_dim:
                raise CacheStateError(
                    f"layer {self.layer_idx} head_dim mismatch: {keys.shape[3]} vs {config.head_dim}"
                )
            if keys.dtype != config.dtype:
                raise CacheStateError(
                    f"layer {self.layer_idx} dtype mismatch: {keys.dtype} vs {config.dtype}"
                )
            if values.dtype != config.dtype:
                raise CacheStateError(
                    f"layer {self.layer_idx} values dtype mismatch: {values.dtype} vs {config.dtype}"
                )

            allocated = state.get("allocated_tokens", int(keys.shape[2]))
            if not isinstance(allocated, int) or allocated < keys.shape[2]:
                raise CacheStateError("allocated_tokens must be an integer >= stored token count")
            shape = (self.batch_size, self.num_kv_heads, allocated, self.head_dim)
            self._keys_buf = torch.empty(shape, dtype=self.dtype, device=dev)
            self._values_buf = torch.empty(shape, dtype=self.dtype, device=dev)
            self._keys_buf[:, :, : keys.shape[2], :] = keys
            self._values_buf[:, :, : values.shape[2], :] = values
            self._compressed = None
            self._offloaded = is_offloaded if target_device is None else dev != self.device

            # Restore split residency if the checkpoint carried one. The host
            # tier's device is recorded so the hosted rows return to the tier
            # they were saved from, rather than silently joining the resident
            # side (which would make the restored "split" all-device).
            host_keys = state.get("host_keys")
            host_values = state.get("host_values")
            if (host_keys is None) != (host_values is None):
                raise CacheStateError(
                    f"layer {self.layer_idx} checkpoint has host_keys without host_values "
                    "or vice versa; a split must describe both tiers"
                )
            resident_count = int(keys.shape[2])
            if host_keys is not None and host_values is not None:
                for name, tensor in (("host_keys", host_keys), ("host_values", host_values)):
                    if tensor.dim() != 4:
                        raise CacheStateError(f"{name} must be 4-D, got {tuple(tensor.shape)}")
                    if tensor.shape[0] != config.batch_size:
                        raise CacheStateError(
                            f"layer {self.layer_idx} {name} batch_size mismatch: "
                            f"{tensor.shape[0]} vs {config.batch_size}"
                        )
                    if tensor.shape[1] != config.num_kv_heads:
                        raise CacheStateError(
                            f"layer {self.layer_idx} {name} num_kv_heads mismatch: "
                            f"{tensor.shape[1]} vs {config.num_kv_heads}"
                        )
                    if tensor.shape[3] != config.head_dim:
                        raise CacheStateError(
                            f"layer {self.layer_idx} {name} head_dim mismatch: "
                            f"{tensor.shape[3]} vs {config.head_dim}"
                        )
                host_dev = torch.device(state.get("host_device", "cpu"))
                self._host_keys_buf = host_keys.to(device=host_dev)
                self._host_values_buf = host_values.to(device=host_dev)
                boundary = state.get("resident_boundary", resident_count)
                if boundary != resident_count:
                    raise CacheStateError(
                        f"layer {self.layer_idx} checkpoint resident_boundary {boundary} does "
                        f"not match the stored resident rows {resident_count}"
                    )
                self._resident_boundary = resident_count

            seq_len = resident_count + (int(host_keys.shape[2]) if host_keys is not None else 0)
            self.metadata.load_state_dict(
                state["metadata"], device=dev, expected_num_tokens=seq_len
            )


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
                # Resolve per layer: LayerStorage.capacity is int | None, while
                # CacheConfig.capacity may be a per-layer list (fixes the
                # TypeError on the first append for list capacities, and the
                # mypy arg-type error).
                capacity=config.capacity_for_layer(idx),
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
        return sum(layer.device_bytes() for layer in self._layers)

    def bytes_offloaded(self) -> int:
        """Bytes held on a non-home device (typically CPU RAM)."""
        return sum(layer.offloaded_bytes() for layer in self._layers)

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

    # -- checkpointing -----------------------------------------------------

    def save(
        self,
        path: str | Path,
        *,
        model: str | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save this store and all layer tensors/bookkeeping to disk via torch.save.

        Preserves quantised layers without dequantising.

        Parameters
        ----------
        path:
            Target file path.
        model:
            Optional model identifier recorded for provenance.
        extra_state:
            Optional dictionary of caller-supplied state (e.g. cache step and activity counters).

        Returns
        -------
        Path
            The written file path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "extra_state": extra_state or {},
            "layers": [layer.state_dict() for layer in self._layers],
        }
        torch.save(payload, target)
        _log.info(
            "saved KV cache checkpoint to %s (layers=%d, model=%s)",
            target,
            len(self._layers),
            model,
        )
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: CacheConfig,
        *,
        map_location: Any = None,
    ) -> KVStore:
        """Load and validate a KVStore from a checkpoint file.

        Parameters
        ----------
        path:
            Path to the checkpoint file.
        config:
            Expected cache configuration to validate against.
        map_location:
            Optional device mapping passed to :func:`torch.load`.

        Returns
        -------
        KVStore
            A populated store matching the checkpoint.

        Raises
        ------
        CacheStateError
            If checkpoint schema is unsupported, or if shapes, dtype, num_layers,
            or device placement disagree with ``config``.
        """
        target = Path(path)
        if not target.is_file():
            raise CacheStateError(f"checkpoint file not found: {target}")

        raw = torch.load(target, map_location=map_location, weights_only=True)
        if not isinstance(raw, dict):
            raise CacheStateError(
                f"invalid checkpoint format in {target}; expected dict, got {type(raw).__name__}"
            )

        schema_version = raw.get("schema_version")
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CacheStateError(
                f"unsupported checkpoint schema version {schema_version!r} (expected {CHECKPOINT_SCHEMA_VERSION!r})"
            )

        saved_layers = raw.get("layers")
        if not isinstance(saved_layers, list):
            raise CacheStateError("checkpoint missing 'layers' list")

        if len(saved_layers) != config.num_layers:
            raise CacheStateError(
                f"checkpoint has {len(saved_layers)} layers, but config expects {config.num_layers}"
            )

        saved_config = raw.get("config", {})
        if saved_config:
            # Validate static configuration fields
            if (
                saved_config.get("num_layers") is not None
                and saved_config["num_layers"] != config.num_layers
            ):
                raise CacheStateError(
                    f"checkpoint config num_layers ({saved_config['num_layers']}) != {config.num_layers}"
                )
            if (
                saved_config.get("num_kv_heads") is not None
                and saved_config["num_kv_heads"] != config.num_kv_heads
            ):
                raise CacheStateError(
                    f"checkpoint config num_kv_heads ({saved_config['num_kv_heads']}) != {config.num_kv_heads}"
                )
            if (
                saved_config.get("head_dim") is not None
                and saved_config["head_dim"] != config.head_dim
            ):
                raise CacheStateError(
                    f"checkpoint config head_dim ({saved_config['head_dim']}) != {config.head_dim}"
                )
            if (
                saved_config.get("batch_size") is not None
                and saved_config["batch_size"] != config.batch_size
            ):
                raise CacheStateError(
                    f"checkpoint config batch_size ({saved_config['batch_size']}) != {config.batch_size}"
                )
            if (
                saved_config.get("device") is not None
                and saved_config["device"] != config.device
                and map_location is None
            ):
                raise CacheStateError(
                    f"checkpoint config device ({saved_config['device']}) != {config.device}"
                )

        store = cls(config)
        for layer_storage, layer_dict in zip(store.layers, saved_layers):
            mapped_device = None
            if map_location is not None:
                mapped_device = layer_dict["metadata"]["positions"].device
            layer_storage.load_state_dict(layer_dict, config=config, target_device=mapped_device)

        _log.info("loaded KV cache checkpoint from %s (layers=%d)", target, len(store))
        return store


__all__ = ["CHECKPOINT_SCHEMA_VERSION", "KVStore", "LayerStorage"]
