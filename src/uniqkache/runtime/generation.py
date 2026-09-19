"""Generation engine: an explicit prefill + decode loop over a KV cache.

Why not use a library's ``generate``
-----------------------------------
Because this project's entire subject is *what happens to the cache during
generation*. A library generation loop hides the cache behind an opaque object
and decides for itself when to allocate, grow and free it. UniqKache needs:

* the ability to run a policy's decision at a known point in the loop;
* per-step timing, so TTFT and TPOT are measured rather than estimated;
* the ability to feed attention weights back into the cache for
  attention-based policies;
* determinism, so two policies can be compared on identical token streams.

So the loop is written out explicitly. It is short, and being able to read it is
the point.

Timing methodology
------------------
``TTFT`` is the wall-clock time of the prefill call, synchronised. ``TPOT`` is
the mean per-token time of the decode phase, synchronised per step. Device
synchronisation is performed before every timing boundary, because CUDA kernel
launches are asynchronous and timing without a barrier measures launch overhead
rather than work.

Both are reported in milliseconds. ``tokens/sec`` is the decode-phase throughput
over *generated* tokens; it deliberately excludes prefill so that a long prompt
does not flatter the number.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from uniqkache.cache.kv_cache import KVCache
from uniqkache.models.base import LanguageModel

if TYPE_CHECKING:
    from uniqkache.allocation.base import BaseAllocationStrategy
from uniqkache.utils.device import (
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    synchronize,
)
from uniqkache.utils.errors import BackendError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class GenerationConfig:
    """Decoding parameters.

    Defaults are chosen for **reproducibility**, not for interesting text: greedy
    decoding with no sampling means two runs on identical inputs produce
    identical tokens, which is a precondition for attributing a difference to
    the cache rather than to sampling noise.
    """

    max_new_tokens: int = 16
    temperature: float = 0.0
    seed: int = 0
    stop_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise BackendError(f"max_new_tokens must be >= 0, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise BackendError(f"temperature must be >= 0, got {self.temperature}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Timings, outputs and cache state from one generation run."""

    generated_ids: torch.Tensor
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    prefill_ms: float
    decode_ms: float
    total_ms: float
    tokens_per_second: float | None
    per_step_ms: list[float] = field(default_factory=list)
    peak_memory_bytes: int | None = None
    cache_stats: dict[str, Any] = field(default_factory=dict)
    hit_stop_token: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("generated_ids")
        data["generated_token_ids"] = self.generated_ids.reshape(-1).tolist()
        return data

    def summary(self) -> str:
        ttft = "n/a" if self.ttft_ms is None else f"{self.ttft_ms:.2f}ms"
        tpot = "n/a" if self.tpot_ms is None else f"{self.tpot_ms:.3f}ms"
        tps = "n/a" if self.tokens_per_second is None else f"{self.tokens_per_second:.1f}"
        peak = (
            "n/a"
            if self.peak_memory_bytes is None
            else f"{self.peak_memory_bytes / 1024**2:.1f}MiB"
        )
        return (
            f"prompt={self.prompt_tokens} gen={self.generated_tokens} "
            f"TTFT={ttft} TPOT={tpot} tok/s={tps} peak={peak}"
        )


class GenerationEngine:
    """Drives prefill and decode against a cache.

    Parameters
    ----------
    model:
        Any object satisfying :class:`~uniqkache.models.base.LanguageModel`.
    cache:
        The cache to populate. Its policy decides what survives.
    config:
        Decoding parameters.
    record_attention:
        Whether to compute and feed attention weights back to the cache. Enable
        only for policies that consume them (:attr:`uses_attention`), since
        materialising attention weights costs memory proportional to the context
        length and would otherwise distort the very measurements we take.
    """

    def __init__(
        self,
        model: LanguageModel,
        cache: KVCache,
        config: GenerationConfig | None = None,
        *,
        record_attention: bool | None = None,
        allocation_strategy: BaseAllocationStrategy | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.config = config or GenerationConfig()
        self.allocation_strategy = allocation_strategy

        if record_attention is None:
            policy = cache.policy
            record_attention = bool(
                (policy is not None and policy.uses_attention) or allocation_strategy is not None
            )
        self.record_attention = record_attention

        self.device = resolve_device(cache.config.device)
        if self.config.temperature > 0 and self.config.seed is not None:
            torch.manual_seed(self.config.seed)

    # ------------------------------------------------------------------

    def generate(self, input_ids: torch.Tensor) -> GenerationResult:
        """Run prefill then decode, returning timings and the token stream.

        Parameters
        ----------
        input_ids:
            ``[batch, seq]`` prompt token ids.

        Raises
        ------
        BackendError
            If the prompt is empty. An empty prompt would make TTFT undefined
            and is almost always a caller bug rather than an intentional case.
        """
        if input_ids.dim() != 2:
            raise BackendError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] == 0:
            raise BackendError("prompt is empty; prefill requires at least one token")

        self.cache.reset()
        reset_peak_memory(self.device)

        prompt_tokens = int(input_ids.shape[1])
        generated: list[torch.Tensor] = []
        per_step_ms: list[float] = []
        hit_stop = False

        # ---- prefill -----------------------------------------------------
        synchronize(self.device)
        t_start = time.perf_counter()
        logits, weights = self._forward(input_ids, start_pos=0, is_prefill=True)
        synchronize(self.device)
        prefill_ms = (time.perf_counter() - t_start) * 1000.0
        ttft_ms = prefill_ms
        self._absorb_attention(weights, mode="all_queries")

        if self.allocation_strategy is not None:
            total_budget = self.cache.config.total_capacity()
            if total_budget is not None:
                new_caps = self.allocation_strategy.allocate(
                    total_budget,
                    self.cache,
                    attention_sinks=self.cache.config.attention_sinks,
                )
                self.cache.set_capacity(new_caps)

        next_token = self._select_token(logits[:, -1, :])
        generated.append(next_token)
        self.cache.advance()

        # ---- decode ------------------------------------------------------
        decode_start = time.perf_counter()
        for step in range(1, self.config.max_new_tokens):
            position = prompt_tokens + step - 1
            synchronize(self.device)
            t_step = time.perf_counter()
            step_logits, step_weights = self._forward(
                next_token, start_pos=position, is_prefill=False
            )
            synchronize(self.device)
            per_step_ms.append((time.perf_counter() - t_step) * 1000.0)
            self._absorb_attention(step_weights, mode="last_query")

            next_token = self._select_token(step_logits[:, -1, :])
            generated.append(next_token)
            self.cache.advance()

            if self.config.stop_token_id is not None and bool(
                (next_token == self.config.stop_token_id).all()
            ):
                hit_stop = True
                break

        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        total_ms = prefill_ms + decode_ms

        generated_ids = torch.cat(generated, dim=1) if generated else input_ids[:, :0]
        generated_tokens = int(generated_ids.shape[1])
        # TPOT is the mean per-step decode time. Reported as None rather than 0
        # when nothing was decoded, because 0ms/token would read as infinitely
        # fast instead of "not measured".
        tpot_ms = (sum(per_step_ms) / len(per_step_ms)) if per_step_ms else None
        tokens_per_second = (
            generated_tokens / (decode_ms / 1000.0) if decode_ms > 0 and generated_tokens else None
        )

        return GenerationResult(
            generated_ids=generated_ids,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            tokens_per_second=tokens_per_second,
            per_step_ms=per_step_ms,
            peak_memory_bytes=peak_memory_bytes(self.device),
            cache_stats=self.cache.stats().to_dict(),
            hit_stop_token=hit_stop,
        )

    # ------------------------------------------------------------------

    def _forward(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run one forward pass, requesting attention only when it is used."""
        with torch.no_grad():
            logits, weights = self.model.forward(
                input_ids,
                cache=self.cache,
                start_pos=start_pos,
                return_attention=self.record_attention,
            )
        if is_prefill and logits.shape[1] != input_ids.shape[1]:
            raise BackendError(
                f"prefill returned {logits.shape[1]} positions for {input_ids.shape[1]} "
                "input tokens; the model backend is inconsistent"
            )
        return logits, weights

    def _absorb_attention(self, weights: list[torch.Tensor] | None, *, mode: str) -> None:
        """Feed per-layer attention weights into the cache's signal bookkeeping."""
        if weights is None:
            return
        for layer_idx, layer_weights in enumerate(weights):
            try:
                self.cache.note_attention(layer_idx, layer_weights, mode=mode)
            except Exception as exc:
                # A length mismatch means the policy's view of the cache and the
                # attention tensor disagree. That is a correctness problem, so we
                # report it rather than continuing with stale signals.
                _log.warning(
                    "could not record attention for layer %d (%s); attention-based "
                    "policies will score this layer from stale signals",
                    layer_idx,
                    exc,
                )

    def _select_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Pick the next token. Greedy by default for reproducibility."""
        if self.config.is_greedy:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits / self.config.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)


__all__ = ["GenerationConfig", "GenerationEngine", "GenerationResult"]
