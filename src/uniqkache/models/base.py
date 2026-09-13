"""The language-model interface the runtime depends on.

UniqKache does not require any particular model library. It requires a model
that can (a) run a forward pass over token ids, (b) optionally read and write a
:class:`~uniqkache.cache.kv_cache.KVCache`, and (c) report absolute positions.

Anything satisfying :class:`LanguageModel` works — the built-in synthetic model,
a Hugging Face model through :mod:`uniqkache.models.hf_backend`, or a model a
contributor writes for a new architecture.

Adding a backend
----------------
Implement ``forward`` with the signature below and register it. Nothing else in
the framework needs to change: the generation engine, the benchmark runner and
the quality evaluators all consume this interface and nothing more.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from uniqkache.cache.kv_cache import KVCache


@runtime_checkable
class LanguageModel(Protocol):
    """Structural type for a model usable with UniqKache.

    Implementations must be deterministic given fixed weights and inputs. The
    correctness of every comparison in this repository depends on that: if two
    runs with identical inputs produce different logits, differences between
    cache policies become unmeasurable.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        *,
        start_pos: int = 0,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run a forward pass.

        Parameters
        ----------
        input_ids:
            ``[batch, seq]`` token ids.
        cache:
            Cache to read and write. ``None`` means single-shot attention over
            ``input_ids`` with no cache.
        start_pos:
            Absolute position of ``input_ids[:, 0]``.
        return_attention:
            Whether to also return per-layer attention weights of shape
            ``[batch, heads, queries, keys]``.

        Returns
        -------
        tuple
            ``(logits, attention_weights)`` with ``logits`` of shape
            ``[batch, seq, vocab_size]`` and ``attention_weights`` either
            ``None`` or a list with one entry per layer.
        """
        ...

    @property
    def num_layers(self) -> int:
        """Number of transformer layers that own a K/V pair."""
        ...


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface for text-level evaluation.

    Kept deliberately small: UniqKache only needs to turn text into ids and back
    so that a quality task can be expressed as text. Anything richer is the
    model library's business.
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        ...


__all__ = ["LanguageModel", "Tokenizer"]
