"""A small, dependency-free dense-attention transformer for development.

Why this exists
---------------
Developing and testing a KV-cache manager requires running a model thousands of
times, on whatever machine the contributor has. Requiring a downloaded
checkpoint for every unit test makes the project unusable on constrained
hardware and makes CI slow and flaky.

This module provides a real transformer — grouped-query attention, RoPE,
RMSNorm, SwiGLU, a genuine KV cache path — with **deterministically initialised
random weights**. It needs no download, no tokenizer, and no network.

What its outputs are and are not good for
-----------------------------------------
* **Valid:** correctness. Incremental decoding through the cache must reproduce
  single-shot attention exactly; see
  ``tests/integration/test_incremental_matches_single_shot.py``.
* **Valid:** *relative* comparisons between cache policies. If policy A loses
  information that policy B keeps, A's perplexity on this model must be worse.
  That is a property of the cache, not of the model.
* **Invalid:** any absolute claim about model quality, language ability, or
  task accuracy. These are random weights. Perplexity on this model is a
  diagnostic, never a result. Any experiment reporting it must say so.

Supported model configurations
------------------------------
Presets are provided for consumer hardware. ``tiny`` is the default for tests
and runs in well under a second on CPU.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.utils.errors import BackendError
from uniqkache.utils.logging import get_logger
from uniqkache.utils.seed import set_seed

_log = get_logger(__name__)


@dataclass
class SyntheticConfig:
    """Shape of a :class:`SyntheticCausalLM`.

    Field names mirror the Hugging Face ``LlamaConfig`` names so that a reader
    who knows that ecosystem can map between them without a table.
    """

    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise BackendError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise BackendError(
                f"num_heads {self.num_heads} must be divisible by num_kv_heads "
                f"{self.num_kv_heads} (grouped-query attention)"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def cache_config(
        self,
        *,
        capacity: int | list[int] | None = None,
        attention_sinks: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        batch_size: int = 1,
    ) -> CacheConfig:
        """Build the matching :class:`CacheConfig` for this model."""
        return CacheConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
            capacity=capacity,
            attention_sinks=attention_sinks,
            batch_size=batch_size,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["head_dim"] = self.head_dim
        return data


PRESETS: dict[str, SyntheticConfig] = {
    # Default for unit tests. ~100k parameters; sub-second on CPU.
    "tiny": SyntheticConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        max_position_embeddings=2048,
    ),
    # Slightly larger: enough layers/heads for per-layer and per-head signal
    # studies to be non-degenerate.
    "small": SyntheticConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        max_position_embeddings=8192,
    ),
    # For long-context behaviour on a consumer GPU. Uses the GPU's memory, not
    # much of it.
    "medium": SyntheticConfig(
        vocab_size=8192,
        hidden_size=512,
        intermediate_size=1024,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,
        max_position_embeddings=32768,
    ),
}


def get_preset(name: str) -> SyntheticConfig:
    """Return a named preset configuration."""
    if name not in PRESETS:
        raise BackendError(
            f"unknown synthetic preset {name!r}. Available: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[name]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Apply rotary position embeddings to ``x`` of shape ``[B, H, T, D]``.

    ``positions`` holds the **absolute** position of each query/key. Absolute
    positions are essential here: once a cache evicts tokens, the surviving
    tokens' relative offsets change, and a policy that renumbered positions
    from zero would silently alter the model's behaviour. Keeping absolute
    positions makes the cache's effect on the model explicit and measurable.
    """
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise BackendError(f"RoPE requires an even head_dim, got {dim}")

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    positions = positions.to(device=x.device, dtype=torch.float32).reshape(-1)
    freqs = torch.outer(positions, inv_freq)  # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand ``[B, Hkv, T, D]`` to ``[B, Hkv*groups, T, D]`` for GQA."""
    if groups == 1:
        return x
    batch, heads, seq, dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, groups, seq, dim)
    return x.reshape(batch, heads * groups, seq, dim)


def causal_mask(query_positions: torch.Tensor, key_positions: torch.Tensor) -> torch.Tensor:
    """Boolean ``[Tq, Tk]`` mask, True where a query may attend to a key.

    Built from **absolute positions**, not from index arithmetic. This is what
    makes the mask correct for a cache holding a non-contiguous subset of the
    sequence after eviction — the case where an index-based mask silently
    permits attention to a token that appears earlier in the cache but belongs
    later in the sequence.
    """
    return key_positions.reshape(1, -1) <= query_positions.reshape(-1, 1)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class Attention(nn.Module):
    """Grouped-query causal self-attention driven by a :class:`KVCache`."""

    def __init__(self, config: SyntheticConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache | None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute attention, writing to and reading from ``cache``.

        Returns
        -------
        tuple
            ``(output, attention_weights)``. ``attention_weights`` is ``None``
            unless ``return_attention`` is True, and otherwise has shape
            ``[batch, num_heads, queries, keys]``.
        """
        batch, seq, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch, seq, self.num_kv_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = apply_rope(query, positions, self.config.rope_theta)
        key = apply_rope(key, positions, self.config.rope_theta)

        if cache is not None:
            # Write the new rows first, then read back the full resident set.
            # Reading back rather than reusing `key`/`value` locally is what
            # makes this a genuine test of the cache: if the cache drops or
            # corrupts a row, this path sees it.
            cache.append(self.layer_idx, key, value, positions=positions)
            all_keys, all_values = cache.get(self.layer_idx)
            key_positions = cache.store.layer(self.layer_idx).metadata.positions
        else:
            all_keys, all_values = key, value
            key_positions = positions

        all_keys = repeat_kv(all_keys, self.config.num_key_value_groups)
        all_values = repeat_kv(all_values, self.config.num_key_value_groups)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, all_keys.transpose(-1, -2)) * scale  # [B, H, Tq, Tk]

        mask = causal_mask(positions, key_positions).to(scores.device)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, all_values)
        output = output.transpose(1, 2).contiguous().view(batch, seq, -1)
        output = self.o_proj(output)

        return output, (weights if return_attention else None)


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, config: SyntheticConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer."""

    def __init__(self, config: SyntheticConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache | None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out, weights = self.self_attn(
            normed, positions, cache, return_attention=return_attention
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, weights


class SyntheticCausalLM(nn.Module):
    """A small causal language model with a real KV-cache path.

    Parameters
    ----------
    config:
        Model shape.
    seed:
        Weight-initialisation seed. Recorded so a run is reproducible.

    Notes
    -----
    Weights are initialised from ``seed`` deterministically. Two models built
    with the same config and seed are bit-identical, which is what makes the
    incremental-vs-single-shot correctness test exact rather than approximate.
    """

    def __init__(self, config: SyntheticConfig, seed: int = 0) -> None:
        super().__init__()
        self.config = config
        self.seed = seed

        with torch.random.fork_rng(devices=[]):
            set_seed(seed)
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList(
                [DecoderLayer(config, idx) for idx in range(config.num_layers)]
            )
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise weights the way trained transformers are initialised.

        This matters more than it looks. PyTorch's default ``nn.Embedding`` uses
        ``N(0, 1)``, which for a small hidden size produces logits of magnitude
        ~10^2 and a cross-entropy around 118 — a confidently-wrong random model
        whose perplexity saturates at ``inf``.

        A saturated diagnostic is useless here: the whole point of this model is
        that perplexity must *respond* when a cache policy discards information,
        and it cannot respond if it is already at infinity for the full-cache
        reference. Scaling initialisation to ``std=0.02``, as GPT-2 and Llama
        do, puts the untrained model near uniform prediction (loss ~= ln(vocab))
        and leaves the diagnostic with room to move.

        This is a deliberate engineering choice, and it is the reason the
        synthetic model's perplexity is a usable *relative* signal. It remains
        meaningless as an absolute quality number.
        """
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # -- introspection -----------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def weights_are_random(self) -> bool:
        """Always True. This model is never trained.

        Exposed as a property so that the quality evaluators and the benchmark
        record can detect it mechanically and refuse to present perplexity on
        this model as model quality. See
        :mod:`uniqkache.metrics.quality`.
        """
        return True

    def describe(self) -> str:
        c = self.config
        return (
            f"SyntheticCausalLM(vocab={c.vocab_size}, hidden={c.hidden_size}, "
            f"layers={c.num_layers}, heads={c.num_heads}, kv_heads={c.num_kv_heads}, "
            f"params={self.num_parameters:,}, seed={self.seed})"
        )

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        *,
        start_pos: int = 0,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run the model, optionally reading and writing a KV cache.

        Parameters
        ----------
        input_ids:
            ``[batch, seq]`` token ids.
        cache:
            When provided, K/V rows are appended to it and attention is computed
            against its full resident contents. When ``None``, attention is
            computed single-shot over ``input_ids`` with a standard causal mask.
        start_pos:
            Absolute position of the first token in ``input_ids``. Used when
            decoding incrementally: the prefill call uses 0, and each decode
            call passes the next absolute position.
        return_attention:
            Whether to return per-layer attention weights, needed by
            attention-based cache policies.

        Returns
        -------
        tuple
            ``(logits, attention_weights)`` where ``logits`` has shape
            ``[batch, seq, vocab_size]``.
        """
        if input_ids.dim() != 2:
            raise BackendError(
                f"input_ids must be [batch, seq], got shape {tuple(input_ids.shape)}"
            )

        _, seq = input_ids.shape
        positions = torch.arange(start_pos, start_pos + seq, device=input_ids.device)

        hidden_states = self.embed_tokens(input_ids)
        collected: list[torch.Tensor] | None = [] if return_attention else None

        for layer in self.layers:
            hidden_states, weights = layer(
                hidden_states, positions, cache, return_attention=return_attention
            )
            if collected is not None and weights is not None:
                collected.append(weights)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, collected


def build_model(
    preset: str | None = None,
    *,
    config: SyntheticConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SyntheticCausalLM:
    """Construct a synthetic model from a preset or an explicit config."""
    if config is None:
        config = get_preset(preset or "tiny")
    elif preset is not None:
        raise BackendError("pass either `preset` or `config`, not both")

    model = SyntheticCausalLM(config, seed=seed)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def build_cache_for_model(
    model: SyntheticCausalLM,
    *,
    capacity: int | None = None,
    attention_sinks: int = 0,
    policy_name: str = "full_cache",
    device: str = "cpu",
    batch_size: int = 1,
) -> KVCache:
    """Build a cache shaped for ``model``, with a policy from the registry."""
    from uniqkache.policies import build_policy

    config = model.config.cache_config(
        capacity=capacity,
        attention_sinks=attention_sinks,
        dtype=next(model.parameters()).dtype,
        device=device,
        batch_size=batch_size,
    )
    policy = None if capacity is None and policy_name == "full_cache" else build_policy(policy_name)
    return KVCache(config, policy=policy)


__all__ = [
    "PRESETS",
    "RMSNorm",
    "SyntheticCausalLM",
    "SyntheticConfig",
    "apply_rope",
    "build_cache_for_model",
    "build_model",
    "causal_mask",
    "get_preset",
    "repeat_kv",
]
