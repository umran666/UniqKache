"""Quality measurement.

The rule this module exists to enforce
--------------------------------------
A memory or latency improvement is only a *result* if the quality cost is
measured. "We reduced KV memory by 4x" is not a finding; "we reduced KV memory
by 4x and perplexity rose from 12.4 to 13.1" is. Every evaluator here returns a
:class:`QualityResult` that carries its own caveat, so a number cannot be lifted
out of context and quoted on its own.

On randomly-initialised models
------------------------------
The synthetic backend has random weights. Running perplexity on it produces a
number, and that number is **not** model quality. It is, however, a valid
*diagnostic of the cache*: if a policy discards information the model was using,
perplexity on that same random model must get worse. So the measurement is
useful for detecting that a policy destroys information, and useless as evidence
about language modelling.

:class:`QualityResult` marks this explicitly via ``is_interpretable``, and
:func:`uniqkache.metrics.record.validate_record` flags any record that reports
such a number as model quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from uniqkache.cache.kv_cache import KVCache
from uniqkache.models.base import LanguageModel
from uniqkache.utils.errors import BackendError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

RANDOM_WEIGHT_CAVEAT = (
    "measured on a randomly-initialised model. Valid as a diagnostic of cache "
    "behaviour (information loss must worsen it) but NOT valid as model quality. "
    "Do not quote as a language-modelling result."
)


def _absorb_attention(
    cache: KVCache,
    weights: list[torch.Tensor] | None,
    *,
    num_queries: int,
) -> None:
    """Feed per-layer attention weights into the cache's signal bookkeeping.

    Mirrors :meth:`uniqkache.runtime.generation.GenerationEngine._absorb_attention`,
    so a policy sees the same signals during the quality pass as during
    generation. If it did not, the quality number would describe a policy that
    was never run -- which is precisely what happened before this existed; see
    ``docs/research.md``, F13.

    ``mode`` follows the chunk: more than one query position is a prefill-like
    chunk and every query's attention counts, while a single query is a decode
    step and only the last query's attention is meaningful.
    """
    if weights is None:
        return
    mode = "all_queries" if num_queries > 1 else "last_query"
    for layer_idx, layer_weights in enumerate(weights):
        try:
            cache.note_attention(layer_idx, layer_weights, mode=mode)
        except Exception as exc:
            # Deliberately broad, and deliberately not silent: a length mismatch
            # means the policy's view of the cache and the attention tensor
            # disagree. That is a correctness problem, so it is reported rather
            # than continuing with stale signals.
            _log.warning(
                "could not record attention for layer %d during quality evaluation "
                "(%s); an attention-based policy will score this layer from stale "
                "signals and the quality number will not describe that policy",
                layer_idx,
                exc,
            )


@dataclass
class QualityResult:
    """A quality measurement together with what it does and does not mean."""

    metric: str
    value: float | None
    num_tokens: int
    is_interpretable: bool
    caveat: str = ""
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.value is None:
            return f"{self.metric}: not measured"
        flag = "" if self.is_interpretable else "  [NOT INTERPRETABLE AS MODEL QUALITY]"
        return f"{self.metric}={self.value:.4f} (n={self.num_tokens}){flag}"


def _model_has_random_weights(model: object) -> bool:
    """Whether a model advertises that its weights are random."""
    return bool(getattr(model, "weights_are_random", False))


def random_token_ids(vocab_size: int, length: int, seed: int = 0) -> torch.Tensor:
    """Deterministic pseudo-random token ids of shape ``[1, length]``.

    Used to build a workload of a given length without a dataset or a tokenizer.
    Deterministic from ``seed`` so that two cache policies are evaluated on
    byte-identical inputs, which is what makes their quality numbers comparable.
    """
    if vocab_size < 1:
        raise BackendError(f"vocab_size must be >= 1, got {vocab_size}")
    if length < 1:
        raise BackendError(f"length must be >= 1, got {length}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab_size, (1, length), generator=generator)


def perplexity(
    model: LanguageModel,
    input_ids: torch.Tensor,
    cache: KVCache | None = None,
    *,
    chunk_size: int = 1,
    device: str | torch.device | None = None,
) -> QualityResult:
    """Teacher-forced perplexity over ``input_ids``.

    Parameters
    ----------
    model:
        The model to evaluate.
    input_ids:
        ``[batch, seq]`` tokens. Scoring starts from the second token, since the
        first has no preceding context to predict it.
    cache:
        When provided, the sequence is fed through it incrementally so that the
        policy's eviction actually happens and its quality effect is measured.
        When ``None``, the sequence is scored in a single pass — the
        no-cache upper bound.
    chunk_size:
        Tokens per forward pass when a cache is used. ``1`` reproduces true
        streaming decode, which is the honest setting for a streaming policy.
        Larger values are faster but change when eviction occurs, so a result
        must record the value used.
    device:
        Device to synchronise against; informational only.

    Returns
    -------
    QualityResult
        ``metric="perplexity"``. Lower is better.
    """
    if input_ids.dim() != 2:
        raise BackendError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 2:
        raise BackendError("perplexity needs at least 2 tokens to score any prediction")
    if chunk_size < 1:
        raise BackendError(f"chunk_size must be >= 1, got {chunk_size}")

    # An attention-based policy must be fed attention here, exactly as it is
    # during generation. Without this, `cum_attention` stays all-zero for the
    # whole quality pass, and every attention-based policy silently scores every
    # token equally -- which means the quality number describes a policy that
    # was never actually run. See docs/research.md, F13.
    record_attention = bool(cache is not None and cache.policy is not None) and bool(
        cache.policy.uses_attention
    )

    vocab_size = int(input_ids.max().item()) + 1
    total_loss = 0.0
    total_scored = 0

    with torch.no_grad():
        if cache is None:
            logits, _ = model.forward(input_ids)
            vocab_size = logits.shape[-1]
            shifted_logits = logits[:, :-1, :].reshape(-1, vocab_size)
            targets = input_ids[:, 1:].reshape(-1)
            total_loss = float(F.cross_entropy(shifted_logits, targets, reduction="sum").item())
            total_scored = int(targets.numel())
        else:
            cache.reset()
            seq_len = int(input_ids.shape[1])
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = input_ids[:, start:end]
                logits, weights = model.forward(
                    chunk, cache=cache, start_pos=start, return_attention=record_attention
                )
                _absorb_attention(cache, weights, num_queries=end - start)
                vocab_size = logits.shape[-1]

                # Logit at absolute position p predicts the token at p+1. So this
                # chunk can score tokens start+1 .. end, with the token at `end`
                # scored by the chunk's final logit. Nothing is double-counted
                # because the next chunk starts scoring at end+1.
                num_targets = min(end, seq_len - 1) - start
                if num_targets <= 0:
                    continue
                shifted_logits = logits[:, :num_targets, :].reshape(-1, vocab_size)
                targets = input_ids[:, start + 1 : start + 1 + num_targets].reshape(-1)
                total_loss += float(
                    F.cross_entropy(shifted_logits, targets, reduction="sum").item()
                )
                total_scored += int(targets.numel())

    if total_scored == 0:
        return QualityResult(
            metric="perplexity",
            value=None,
            num_tokens=0,
            is_interpretable=False,
            caveat="no token could be scored; the sequence was too short",
        )

    mean_loss = total_loss / total_scored
    value = float(torch.exp(torch.tensor(mean_loss)).item())

    random_weights = _model_has_random_weights(model)
    return QualityResult(
        metric="perplexity",
        value=value,
        num_tokens=total_scored,
        is_interpretable=not random_weights,
        caveat=RANDOM_WEIGHT_CAVEAT if random_weights else "",
        details={
            "mean_cross_entropy": mean_loss,
            "chunk_size": chunk_size,
            "used_cache": cache is not None,
            "vocab_size": vocab_size,
        },
    )


def exact_match(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of positions where two equal-length token tensors agree.

    Returns a value in ``[0, 1]``. Raises on a shape mismatch rather than
    broadcasting, because silently comparing misaligned tensors is how a
    retrieval score becomes meaningless.
    """
    if prediction.shape != reference.shape:
        raise BackendError(
            f"exact_match requires identical shapes, got {tuple(prediction.shape)} "
            f"and {tuple(reference.shape)}"
        )
    if prediction.numel() == 0:
        return 0.0
    return float((prediction == reference).float().mean().item())


def needle_retrieval(
    model: LanguageModel,
    *,
    haystack_length: int,
    needle: torch.Tensor,
    vocab_size: int,
    depth: float = 0.5,
    seed: int = 0,
    cache: KVCache | None = None,
) -> QualityResult:
    """Measure whether the model can retrieve an inserted token sequence.

    A synthetic long-context retrieval task: filler tokens fill a context of
    ``haystack_length``, the ``needle`` is inserted at a relative ``depth``, and
    the model must reproduce the needle from the last position.

    Parameters
    ----------
    model:
        Model to evaluate.
    haystack_length:
        Total context length including the needle.
    needle:
        ``[1, k]`` token ids to insert and then attempt to recover.
    vocab_size:
        Vocabulary size for generating filler.
    depth:
        Where in the haystack to insert the needle, as a fraction in ``[0, 1]``.
        ``0.0`` is the very beginning, ``1.0`` the very end.
    seed:
        Filler seed. Fixed so two policies see identical contexts.
    cache:
        Optional cache; when supplied the policy's eviction is exercised, which
        is the point — a policy that evicts the needle should fail this task.

    Returns
    -------
    QualityResult
        ``metric="needle_retrieval"``. Higher is better, in ``[0, 1]``.

    Notes
    -----
    On a randomly-initialised model this measures chance-level behaviour and is
    flagged as not interpretable. Its value there is as a *negative control*: it
    demonstrates the harness runs and that a policy which evicts the needle
    cannot score well by accident.
    """
    if not 0.0 <= depth <= 1.0:
        raise BackendError(f"depth must be in [0, 1], got {depth}")
    needle_len = int(needle.shape[1])
    if needle_len < 1:
        raise BackendError("needle must contain at least one token")
    if haystack_length < needle_len + 2:
        raise BackendError(
            f"haystack_length {haystack_length} is too short for a {needle_len}-token "
            "needle plus context"
        )

    filler = random_token_ids(vocab_size, haystack_length, seed=seed)
    insert_at = int(depth * (haystack_length - needle_len))
    insert_at = max(0, min(insert_at, haystack_length - needle_len))
    context = torch.cat(
        [
            filler[:, :insert_at],
            needle.to(dtype=filler.dtype),
            filler[:, insert_at + needle_len :],
        ],
        dim=1,
    )

    with torch.no_grad():
        if cache is not None:
            cache.reset()
            logits, _ = model.forward(context, cache=cache, start_pos=0)
        else:
            logits, _ = model.forward(context)

        # Greedy continuation from the final position, for as many tokens as the
        # needle is long.
        generated = [logits[:, -1, :].argmax(dim=-1, keepdim=True)]
        position = context.shape[1]
        for _ in range(needle_len - 1):
            if cache is not None:
                step_logits, _ = model.forward(generated[-1], cache=cache, start_pos=position)
            else:
                raise BackendError(
                    "needle_retrieval without a cache can only score a 1-token needle; "
                    "supply a cache for multi-token needles"
                )
            generated.append(step_logits[:, -1, :].argmax(dim=-1, keepdim=True))
            position += 1

        prediction = torch.cat(generated, dim=1)

    score = exact_match(prediction, needle.to(dtype=prediction.dtype))
    random_weights = _model_has_random_weights(model)

    return QualityResult(
        metric="needle_retrieval",
        value=score,
        num_tokens=haystack_length,
        is_interpretable=not random_weights,
        caveat=(
            RANDOM_WEIGHT_CAVEAT
            if random_weights
            else (
                "single-sample retrieval. One sample cannot distinguish a policy that "
                "reliably loses the needle from one that loses it by chance; report a "
                "mean over many seeds before drawing conclusions."
            )
        ),
        details={
            "haystack_length": haystack_length,
            "needle_length": needle_len,
            "depth": depth,
            "seed": seed,
            "predicted": prediction.reshape(-1).tolist(),
            "expected": needle.reshape(-1).tolist(),
        },
    )


__all__ = [
    "RANDOM_WEIGHT_CAVEAT",
    "QualityResult",
    "exact_match",
    "needle_retrieval",
    "perplexity",
    "random_token_ids",
]
