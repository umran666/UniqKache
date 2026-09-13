"""Integer quantisation of KV tensors.

Prior work
----------
Quantising the KV cache to 8 bits, and the observation that keys and values
prefer *different* quantisation axes, is established work:

* **KIVI** — Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
  KV Cache", ICML 2024. arXiv:2402.02750. Recommends per-channel quantisation
  for keys (quantise along the head dimension) and per-token quantisation for
  values (quantise along the sequence dimension), because key outliers are
  concentrated in specific channels while value outliers spread across tokens.
* **KVQuant** — Hooper et al., "KVQuant: Towards 10 Million Context Length LLM
  Inference with KV Cache Quantization", NeurIPS 2024. arXiv:2401.18079.

What we reproduce
-----------------
A tuning-free uniform integer quantiser with a selectable reduction axis, used
at 8 bits by default, applied per-axis as KIVI recommends. We implement the
*symmetric* and *asymmetric* affine schemes directly.

Implementation differences from KIVI
-------------------------------------
* We do not implement the per-channel *residual* cache, the outlier
  pre-RoPE treatment, or the grouped-channel granularity tuning. Our
  reproduction is a single-level quantiser, so it is expected to be **worse**
  than published KIVI at equal bit width. Any comparison against KIVI numbers
  from the paper must state this.
* We do not support sub-8-bit widths in this milestone; the code path accepts a
  ``num_bits`` argument but only 8 is exercised by the test suite.

We make no novelty claim for anything in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniqkache.compression.base import BaseCompressor, CompressionResult
from uniqkache.utils.device import tensor_bytes
from uniqkache.utils.errors import UniqKacheError

# Guards against division by zero for all-zero slices.
_EPS = 1e-12


@dataclass
class QuantizedTensor:
    """An integer tensor plus the affine parameters needed to recover floats."""

    data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    axis: int
    num_bits: int
    symmetric: bool

    @property
    def original_shape(self) -> tuple[int, ...]:
        shape = list(self.data.shape)
        # ``scale`` is reduced along ``axis`` with keepdim, so the stored data
        # already carries the original shape.
        return tuple(shape)

    def dequantize(self) -> torch.Tensor:
        """Reconstruct a float tensor approximating the original."""
        floats = self.data.to(torch.float32)
        if self.symmetric:
            return floats * self.scale
        return (floats - self.zero_point.to(torch.float32)) * self.scale

    def bytes(self) -> int:
        """Storage footprint: quantised payload plus affine parameters."""
        return tensor_bytes(self.data) + tensor_bytes(self.scale) + tensor_bytes(self.zero_point)

    def num_tokens(self) -> int:
        """Sequence length of the quantised tensor (axis 2 convention)."""
        return int(self.data.shape[2])


def quantize(
    x: torch.Tensor,
    *,
    axis: int,
    num_bits: int = 8,
    symmetric: bool = True,
) -> QuantizedTensor:
    """Quantise ``x`` along ``axis`` with a uniform affine scheme.

    Parameters
    ----------
    x:
        Input tensor.
    axis:
        The axis **reduced over** to obtain the scale and zero-point. The
        reduction axis determines the granularity, and the two are easy to
        confuse, so for a KV tensor of shape ``[batch, heads, seq, head_dim]``:

        * ``axis=2`` reduces over the sequence. Every head-dimension channel
          gets one shared scale — this is **per-channel** granularity.
        * ``axis=3`` reduces over the head dimension. Every token gets its own
          scale — this is **per-token** granularity.

        Getting these backwards does not raise; it quietly produces a worse
        quantiser. See the module docstring for which one KIVI recommends where.
    num_bits:
        Bit width. Only 8 is supported in this milestone.
    symmetric:
        If True, the quantisation grid is centred on zero and no zero-point is
        used (zero-point is stored as zeros so the layout stays uniform).

    Returns
    -------
    QuantizedTensor

    Raises
    ------
    UniqKacheError
        If ``num_bits`` is not 8, or ``axis`` is out of range.
    """
    if num_bits != 8:
        raise UniqKacheError(
            f"only 8-bit quantisation is implemented in this milestone, got num_bits={num_bits}. "
            "Sub-8-bit paths are listed in the roadmap."
        )
    if not -x.dim() <= axis < x.dim():
        raise UniqKacheError(f"axis {axis} out of range for a {x.dim()}-D tensor")

    axis = axis % x.dim()
    x32 = x.to(torch.float32)
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))

    if symmetric:
        absmax = x32.abs().amax(dim=axis, keepdim=True)
        scale = (absmax / qmax).clamp_min(_EPS)
        zero_point = torch.zeros_like(scale)
        quantized = torch.round(x32 / scale).clamp(qmin, qmax)
    else:
        # The observed range must include zero, or the zero-point is not
        # representable and the scheme silently degrades.
        #
        # Why: the affine map is q = x/scale + zp with zp = qmin - xmin/scale.
        # If every value in a slice is positive, xmin > 0 and zp falls below
        # qmin. Storing zp as an int8 then clamps it, but the quantised data was
        # computed with the *unclamped* zp — so the two disagree and
        # dequantisation is wrong. Measured effect before the fix: max error
        # 0.115 for asymmetric versus 0.016 for symmetric, i.e. asymmetric was
        # several times worse than the scheme it is supposed to beat.
        #
        # Clamping the range to include zero makes zp land inside [qmin, qmax]
        # by construction, which is the standard formulation.
        xmin = x32.amin(dim=axis, keepdim=True).clamp_max(0.0)
        xmax = x32.amax(dim=axis, keepdim=True).clamp_min(0.0)
        scale = ((xmax - xmin) / (qmax - qmin)).clamp_min(_EPS)
        zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        quantized = torch.round(x32 / scale + zero_point).clamp(qmin, qmax)

    return QuantizedTensor(
        data=quantized.to(torch.int8),
        scale=scale,
        zero_point=zero_point.to(torch.int8),
        axis=axis,
        num_bits=num_bits,
        symmetric=symmetric,
    )


class Int8KVCompressor(BaseCompressor):
    """Per-axis int8 quantiser for K/V tensors.

    Defaults follow KIVI's granularity recommendation. Note carefully that the
    *reduction* axis is what is configured here, and it is the inverse of the
    granularity name:

    * keys → **per-channel**, i.e. one scale per head-dimension channel, shared
      across all tokens. That is ``key_axis=2`` (reduce over the sequence).
    * values → **per-token**, i.e. one scale per token, shared across channels.
      That is ``value_axis=3`` (reduce over the head dimension).

    The rationale in KIVI is that key outliers concentrate in a few persistent
    channels, so sharing a scale per channel keeps those channels accurate,
    whereas value outliers are scattered across tokens, so a per-token scale
    confines each outlier's damage to its own token.

    Parameters
    ----------
    key_axis, value_axis:
        Reduction axes for keys and values, as described above.
    num_bits:
        Bit width; 8 in this milestone.
    symmetric_keys, symmetric_values:
        Whether to use a symmetric grid. Keys default to symmetric. Values also
        default to symmetric, which is a **deviation** from KIVI's asymmetric
        recommendation and is recorded in every run's metadata so the difference
        is visible in the results rather than buried in the code.
    """

    name = "int8"
    lossy = True

    def __init__(
        self,
        *,
        key_axis: int = 2,
        value_axis: int = 3,
        num_bits: int = 8,
        symmetric_keys: bool = True,
        symmetric_values: bool = True,
    ) -> None:
        self.key_axis = key_axis
        self.value_axis = value_axis
        self.num_bits = num_bits
        self.symmetric_keys = symmetric_keys
        self.symmetric_values = symmetric_values

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> CompressionResult:
        self._check_inputs(keys, values)
        uncompressed = tensor_bytes(keys) + tensor_bytes(values)

        q_keys = quantize(
            keys, axis=self.key_axis, num_bits=self.num_bits, symmetric=self.symmetric_keys
        )
        q_values = quantize(
            values, axis=self.value_axis, num_bits=self.num_bits, symmetric=self.symmetric_values
        )
        compressed = q_keys.bytes() + q_values.bytes()

        return CompressionResult(
            keys=q_keys,
            values=q_values,
            num_tokens=int(keys.shape[2]),
            uncompressed_bytes=uncompressed,
            compressed_bytes=compressed,
        )

    def decompress(self, result: CompressionResult) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(result.keys, QuantizedTensor) or not isinstance(
            result.values, QuantizedTensor
        ):
            raise UniqKacheError(
                f"{type(self).__name__}.decompress expected QuantizedTensor payloads; "
                "the CompressionResult was not produced by this compressor."
            )
        return result.keys.dequantize(), result.values.dequantize()

    def state_dict(self) -> dict[str, object]:
        data = super().state_dict()
        data.update(
            {
                "key_axis": self.key_axis,
                "value_axis": self.value_axis,
                "num_bits": self.num_bits,
                "symmetric_keys": self.symmetric_keys,
                "symmetric_values": self.symmetric_values,
            }
        )
        return data


__all__ = ["Int8KVCompressor", "QuantizedTensor", "quantize"]
