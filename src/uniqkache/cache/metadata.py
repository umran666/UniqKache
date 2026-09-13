"""Per-token bookkeeping for one cache layer.

Design note
-----------
Recency and importance statistics live **here, in the storage layer**, not
inside the policies. A policy is therefore a pure function
``PolicyState -> scores``: it reads signals and returns a ranking, and it never
mutates cache state. That separation is what makes it safe to swap policies, and
what lets an ablation study attribute a difference to the *decision rule* rather
than to bookkeeping side effects.

Every signal is stored as a tensor indexed by cache slot and is kept in sync
with evictions, so a policy can never observe a stale length.
"""

from __future__ import annotations

import torch

from uniqkache.utils.errors import CacheStateError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


class LayerMetadata:
    """Recency, importance and provenance signals for the tokens in one layer.

    Parameters
    ----------
    num_sinks:
        Number of leading tokens considered attention sinks (protected).
    device:
        Device the metadata tensors live on.
    """

    def __init__(self, num_sinks: int = 0, device: str | torch.device = "cpu") -> None:
        if num_sinks < 0:
            raise CacheStateError(f"num_sinks must be >= 0, got {num_sinks}")
        self._num_sinks = num_sinks
        self._device = torch.device(device)

        self._positions = torch.empty(0, dtype=torch.long, device=self._device)
        self._last_access = torch.empty(0, dtype=torch.long, device=self._device)
        self._cum_attention = torch.empty(0, dtype=torch.float32, device=self._device)
        self._hit_count = torch.empty(0, dtype=torch.long, device=self._device)

    # -- properties --------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        return int(self._positions.shape[0])

    @property
    def positions(self) -> torch.Tensor:
        return self._positions

    @property
    def last_access(self) -> torch.Tensor:
        return self._last_access

    @property
    def cum_attention(self) -> torch.Tensor:
        return self._cum_attention

    @property
    def hit_count(self) -> torch.Tensor:
        return self._hit_count

    @property
    def num_sinks(self) -> int:
        return self._num_sinks

    def is_sink(self) -> torch.Tensor:
        """Boolean mask marking protected leading tokens.

        Derived from absolute positions rather than slot index, so the mask
        stays correct after evictions shift slots around.
        """
        if self.num_tokens == 0:
            return torch.zeros(0, dtype=torch.bool, device=self._device)
        return self._positions < self._num_sinks

    # -- mutation ----------------------------------------------------------

    def append(
        self,
        num_new: int,
        *,
        positions: torch.Tensor | None = None,
        step: int = 0,
    ) -> None:
        """Extend bookkeeping for ``num_new`` freshly appended tokens.

        Parameters
        ----------
        num_new:
            How many tokens were appended.
        positions:
            Absolute position ids of the new tokens. When omitted, positions
            continue consecutively from the last recorded position.
        step:
            Current decode step; new tokens are marked as accessed at ``step``
            because attention has just been computed over them.
        """
        if num_new < 0:
            raise CacheStateError(f"num_new must be >= 0, got {num_new}")
        if num_new == 0:
            return

        if positions is None:
            start = int(self._positions[-1].item()) + 1 if self.num_tokens else 0
            new_positions = torch.arange(
                start, start + num_new, dtype=torch.long, device=self._device
            )
        else:
            new_positions = positions.to(device=self._device, dtype=torch.long).reshape(-1)
            if new_positions.shape[0] != num_new:
                raise CacheStateError(
                    f"positions has {new_positions.shape[0]} entries but num_new={num_new}"
                )

        self._positions = torch.cat([self._positions, new_positions])
        self._last_access = torch.cat(
            [
                self._last_access,
                torch.full((num_new,), int(step), dtype=torch.long, device=self._device),
            ]
        )
        self._cum_attention = torch.cat(
            [self._cum_attention, torch.zeros(num_new, dtype=torch.float32, device=self._device)]
        )
        self._hit_count = torch.cat(
            [self._hit_count, torch.zeros(num_new, dtype=torch.long, device=self._device)]
        )

    def note_attention(self, weights: torch.Tensor, step: int, threshold: float = 0.0) -> None:
        """Accumulate attention mass received by each cached token.

        Parameters
        ----------
        weights:
            A ``[num_tokens]`` vector of attention probabilities. The caller is
            responsible for reducing the model's full attention tensor down to
            one value per cached token (see
            :func:`uniqkache.cache.metadata.reduce_attention`).
        step:
            Current decode step, recorded as the access time of every token that
            received attention above ``threshold``.
        threshold:
            Attention mass below this value does not count as an access. A
            strictly positive threshold prevents numerically-negligible
            attention from resetting the recency of every token each step, which
            would make an LRU policy degenerate into "keep everything".
        """
        weights = weights.to(device=self._device, dtype=torch.float32).reshape(-1)
        if weights.shape[0] != self.num_tokens:
            raise CacheStateError(
                f"attention weights have length {weights.shape[0]} but the layer holds "
                f"{self.num_tokens} tokens. Reducing attention must target cached tokens only."
            )
        if self.num_tokens == 0:
            return

        accessed = weights > threshold
        self._cum_attention = self._cum_attention + weights
        self._hit_count = self._hit_count + accessed.to(torch.long)
        self._last_access = torch.where(
            accessed,
            torch.full_like(self._last_access, int(step)),
            self._last_access,
        )

    def note_access(self, indices: torch.Tensor, step: int) -> None:
        """Mark specific slots as accessed without recording attention mass.

        Used when a caller reads tokens explicitly (for example a prefetch or a
        re-computation) rather than through the attention path.
        """
        if self.num_tokens == 0:
            return
        indices = indices.to(device=self._device, dtype=torch.long).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.num_tokens):
            raise CacheStateError(
                f"note_access indices out of range [0, {self.num_tokens}): "
                f"got min={int(indices.min())}, max={int(indices.max())}"
            )
        self._last_access[indices] = int(step)
        self._hit_count[indices] = self._hit_count[indices] + 1

    def keep(self, indices: torch.Tensor) -> None:
        """Restrict bookkeeping to ``indices``, in the order given.

        Called by the store immediately after it gathers the surviving K/V rows,
        so metadata and tensors always describe the same slots.
        """
        indices = indices.to(device=self._device, dtype=torch.long).reshape(-1)
        self._positions = self._positions[indices]
        self._last_access = self._last_access[indices]
        self._cum_attention = self._cum_attention[indices]
        self._hit_count = self._hit_count[indices]

    def clear(self) -> None:
        """Drop all tokens, preserving configuration."""
        self._positions = torch.empty(0, dtype=torch.long, device=self._device)
        self._last_access = torch.empty(0, dtype=torch.long, device=self._device)
        self._cum_attention = torch.empty(0, dtype=torch.float32, device=self._device)
        self._hit_count = torch.empty(0, dtype=torch.long, device=self._device)

    def to(self, device: str | torch.device) -> LayerMetadata:
        """Move metadata tensors to ``device`` in place."""
        device = torch.device(device)
        self._positions = self._positions.to(device)
        self._last_access = self._last_access.to(device)
        self._cum_attention = self._cum_attention.to(device)
        self._hit_count = self._hit_count.to(device)
        self._device = device
        return self


def reduce_attention(
    attention: torch.Tensor,
    *,
    mode: str = "last_query",
    query_index: int | None = None,
) -> torch.Tensor:
    """Collapse a full attention tensor into one weight per cached key.

    Parameters
    ----------
    attention:
        Attention probabilities with shape ``[batch, heads, queries, keys]``.
    mode:
        * ``"last_query"`` — use only the final query row. This is the correct
          choice during autoregressive decoding, where only the newest token is
          being computed.
        * ``"all_queries"`` — average over every query row. Used during prefill,
          where all positions are genuine queries.
    query_index:
        Explicit query row to use. Overrides ``mode`` when provided.

    Returns
    -------
    torch.Tensor
        A ``[keys]`` vector of attention mass, averaged over batch and heads.

    Raises
    ------
    ValueError
        If the attention tensor is not 4-dimensional.
    """
    if attention.dim() != 4:
        raise ValueError(
            f"attention must be [batch, heads, queries, keys], got shape {tuple(attention.shape)}"
        )

    if query_index is not None:
        if not -attention.shape[2] <= query_index < attention.shape[2]:
            raise ValueError(
                f"query_index {query_index} out of range for {attention.shape[2]} queries"
            )
        selected = attention[:, :, query_index, :]
    elif mode == "last_query":
        selected = attention[:, :, -1, :]
    elif mode == "all_queries":
        selected = attention.mean(dim=2)
    else:
        raise ValueError(f"unknown reduce mode {mode!r}; expected 'last_query' or 'all_queries'")

    return selected.mean(dim=(0, 1))


__all__ = ["LayerMetadata", "reduce_attention"]
