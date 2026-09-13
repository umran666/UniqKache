"""Signal normalisation helpers shared by scoring policies.

Policies that combine several signals need them on a comparable scale before
weighting. These helpers are deliberately boring and explicit, because a
normalisation bug produces a policy that *looks* like it is using two signals
while actually being driven by whichever one happens to have the larger range.

Every helper handles the degenerate case (all values equal, or no tokens) by
returning zeros. It never returns ``NaN``, which would propagate into
``argsort`` and produce an arbitrary but plausible-looking eviction order.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def minmax(x: torch.Tensor) -> torch.Tensor:
    """Scale ``x`` to ``[0, 1]``. Returns zeros when the range is degenerate."""
    if x.numel() == 0:
        return x.to(torch.float32)
    x = x.to(torch.float32)
    lo = x.min()
    hi = x.max()
    span = hi - lo
    if float(span) < _EPS:
        return torch.zeros_like(x)
    return (x - lo) / span


def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale by rank rather than magnitude, in ``[0, 1]``.

    Useful for signals with heavy-tailed distributions — accumulated attention
    being the main one — where a single dominant token would otherwise compress
    every other token's min-max score to nearly zero.
    """
    if x.numel() == 0:
        return x.to(torch.float32)
    x = x.to(torch.float32)
    if x.numel() == 1:
        return torch.ones_like(x)
    order = torch.argsort(torch.argsort(x))  # rank, ascending
    return order.to(torch.float32) / (x.numel() - 1)


def weighted_sum(
    signals: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Combine normalised signals into one score vector.

    Parameters
    ----------
    signals:
        Already-normalised per-token signals, all of length ``num_cached``.
    weights:
        Weight per signal name. Signals absent from ``weights`` are ignored;
        weights naming an absent signal raise, because that means the caller
        believes a signal is active when it is not.

    Raises
    ------
    KeyError
        If ``weights`` names a signal that is not present.
    ValueError
        If the signals do not all share a length.
    """
    if not signals:
        raise ValueError("weighted_sum requires at least one signal")

    lengths = {name: int(sig.shape[0]) for name, sig in signals.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"all signals must share a length, got {lengths}")

    unknown = set(weights) - set(signals)
    if unknown:
        raise KeyError(
            f"weights reference signals that were not supplied: {sorted(unknown)}. "
            f"Available: {sorted(signals)}"
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError(f"weights must sum to a positive value, got {weights}")

    out = torch.zeros_like(next(iter(signals.values())), dtype=torch.float32)
    for name, weight in weights.items():
        out = out + float(weight) * signals[name].to(torch.float32)
    return out / total_weight


__all__ = ["minmax", "rank_normalize", "weighted_sum"]
